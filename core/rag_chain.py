"""
rag_chain.py - RAG 核心问答链

对应 RAG 流程中的：
  Relevant Chunks → Augmented Query → LLM → Answer

负责：
  1. 将检索到的 Relevant Chunks 与用户 Query 合并，构造 Augmented Query
  2. 调用 LLM 生成最终 Answer
  3. 返回答案及引用的来源文档
"""
import logging
from typing import Dict, Any, List

import openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from core.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    RETRIEVAL_TOP_K,
    VECTOR_STORE_BACKEND,
    MILVUS_METRIC_TYPE,
)
from core.vector_store import VectorStoreManager
from core.llm_retry import retry_with_backoff

logger = logging.getLogger(__name__)


def _extract_token_usage(response) -> tuple:
    """
    从 LLM 响应中提取 Token 使用量。
    支持多种响应格式：LangChain AIMessage、OpenAI 原生响应、兼容 API 等。

    Returns:
        (input_tokens, output_tokens): 提取到的 Token 数量，提取失败返回 (0, 0)
    """
    input_tokens = 0
    output_tokens = 0

    try:
        usage = None

        # 路径1: LangChain AIMessage.usage_metadata（langchain-openai >= 0.1.0）
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            logger.debug("Token 提取路径1 (usage_metadata): %s", usage)

        # 路径2: response_metadata 中的 token_usage
        if not usage and hasattr(response, 'response_metadata'):
            rm = response.response_metadata or {}
            usage = rm.get("token_usage") or rm.get("usage")
            if usage:
                logger.debug("Token 提取路径2 (response_metadata): %s", usage)

        # 路径3: additional_kwargs 中的 usage
        if not usage and hasattr(response, 'additional_kwargs'):
            ak = response.additional_kwargs or {}
            usage = ak.get("usage") or ak.get("token_usage")
            if usage:
                logger.debug("Token 提取路径3 (additional_kwargs): %s", usage)

        # 路径4: 直接属性（某些兼容 API 的 AIMessage 子类）
        if not usage:
            for attr in ('usage', 'token_usage', 'llm_output'):
                val = getattr(response, attr, None)
                if val and isinstance(val, dict):
                    usage = val
                    logger.debug("Token 提取路径4 (属性 %s): %s", attr, usage)
                    break

        # 路径5: 遍历 response_metadata 中所有包含 "usage" 或 "token" 的 key
        if not usage and hasattr(response, 'response_metadata'):
            rm = response.response_metadata or {}
            for key in rm:
                if 'usage' in key.lower() or 'token' in key.lower():
                    val = rm[key]
                    if isinstance(val, dict):
                        usage = val
                        logger.debug("Token 提取路径5 (response_metadata.%s): %s", key, usage)
                        break

        # 路径6: 遍历 additional_kwargs 中所有包含 "usage" 或 "token" 的 key
        if not usage and hasattr(response, 'additional_kwargs'):
            ak = response.additional_kwargs or {}
            for key in ak:
                if 'usage' in key.lower() or 'token' in key.lower():
                    val = ak[key]
                    if isinstance(val, dict):
                        usage = val
                        logger.debug("Token 提取路径6 (additional_kwargs.%s): %s", key, usage)
                        break

        if usage and isinstance(usage, dict):
            # 尝试多种 key 名称
            input_tokens = (
                usage.get("input_tokens")
                or usage.get("prompt_tokens")
                or usage.get("input_token_count")
                or usage.get("prompt_token_count")
                or 0
            )
            output_tokens = (
                usage.get("output_tokens")
                or usage.get("completion_tokens")
                or usage.get("output_token_count")
                or usage.get("completion_token_count")
                or 0
            )

        if input_tokens == 0 and output_tokens == 0:
            logger.warning(
                "无法从 LLM 响应中提取 Token 使用量。"
                "usage_metadata=%s, response_metadata keys=%s, additional_kwargs keys=%s",
                getattr(response, 'usage_metadata', None),
                list(getattr(response, 'response_metadata', {}).keys()) if hasattr(response, 'response_metadata') else None,
                list(getattr(response, 'additional_kwargs', {}).keys()) if hasattr(response, 'additional_kwargs') else None,
            )

    except Exception as e:
        logger.error("提取 Token 使用量时出错：%s", e)

    return input_tokens, output_tokens


def _langchain_messages_to_openai(messages: List) -> List[Dict]:
    """
    将 LangChain 消息对象列表转换为 OpenAI API 格式的消息列表。

    LangChain 消息类型映射：
      SystemMessage → {"role": "system", "content": "..."}
      HumanMessage  → {"role": "user", "content": "..."}
      AIMessage     → {"role": "assistant", "content": "..."}
    """
    result = []
    for msg in messages:
        role = getattr(msg, 'type', 'user')
        if role == 'human':
            role = 'user'
        elif role == 'ai':
            role = 'assistant'
        result.append({"role": role, "content": msg.content})
    return result


def _calc_similarity(score: float, rtype: str) -> float:
    """
    根据检索类型计算相似度百分比 [0, 100]。

    - rerank: 分数已在 [0, 100] 范围（Reranker 输出 * 100），直接使用
    - vec: Milvus COSINE 后端返回余弦相似度 [0, 1]，直接乘 100；
           Chroma/Faiss 后端返回余弦距离 [0, 2]，需转换
    - 其他（bm25 / RRF 融合）: 使用 sigmoid 归一化到 [0, 100]
    """
    if rtype == "rerank":
        return max(0.0, min(100.0, score))
    elif rtype == "vec":
        if VECTOR_STORE_BACKEND == "milvus" and MILVUS_METRIC_TYPE == "COSINE":
            return max(0.0, min(100.0, score * 100.0))
        else:
            return max(0.0, min(100.0, (1.0 - score / 2.0) * 100.0))
    else:
        normalized = 1.0 / (1.0 + pow(2.71828, -score * 80.0))
        return max(0.0, min(100.0, normalized * 100.0))


def format_docs(docs: List[Document]) -> str:
    """
    将检索到的 Document 列表格式化为字符串上下文
    对应流程：Relevant Chunks → Augmented Query（Context 部分）
    """
    if not docs:
        return "（未找到相关参考文档）"

    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        source_name = source.split("\\")[-1].split("/")[-1]  # 只保留文件名
        formatted.append(f"【参考文档 {i}】来源：{source_name}\n{doc.page_content}")

    return "\n\n".join(formatted)


def format_docs_with_scores(docs_with_scores: List[tuple]) -> str:
    """
    将检索到的 Document 列表（带分数和检索类型）格式化为字符串上下文。
    支持两种格式：
      - 旧格式：List[tuple] → (Document, score)
      - 新格式：List[tuple] → (Document, score, retrieval_type)
    """
    if not docs_with_scores:
        return "（未找到相关参考文档）"

    formatted = []
    for i, item in enumerate(docs_with_scores, 1):
        doc = item[0]
        score = item[1]
        rtype = item[2] if len(item) > 2 else "vec"

        source = doc.metadata.get("source", "未知来源")
        source_name = source.split("\\")[-1].split("/")[-1]

        similarity = _calc_similarity(score, rtype)

        # 检索类型标注
        type_label = ""
        if rtype == "vec":
            type_label = "（语义检索）"
        elif rtype == "bm25":
            type_label = "（关键词检索）"
        elif rtype == "rerank":
            type_label = "（重排精筛）"
        else:
            type_label = f"（{rtype}）"

        formatted.append(
            f"【参考文档 {i}】来源：{source_name}（相关度：{similarity:.1f}%{type_label}）\n{doc.page_content}"
        )

    return "\n\n".join(formatted)


class RAGChain:
    """
    RAG 问答链
    完整流程：Query → Retriever → Relevant Chunks → Augmented Query → LLM → Answer
    """

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager
        self._llm = None
        self._chain = None
        self._raw_client = None

    @property
    def llm(self) -> ChatOpenAI:
        """懒加载 LLM"""
        if self._llm is None:
            logger.info(f"正在初始化 LLM：{OPENAI_MODEL}")
            self._llm = ChatOpenAI(
                model=OPENAI_MODEL,
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_BASE_URL,
                temperature=0.1,       # 低温度保证回答稳定、准确
                max_tokens=2048,
                streaming=True,        # 启用流式输出，提升体验
            )
        return self._llm

    @property
    def raw_client(self) -> openai.OpenAI:
        """懒加载原始 OpenAI 客户端（用于直接提取 response.usage）"""
        if self._raw_client is None:
            self._raw_client = openai.OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                timeout=60.0,
            )
        return self._raw_client

    def _build_chain(self):
        """
        构建 LCEL（LangChain Expression Language）RAG Chain

        完整流程：
          question → retriever → format_docs → prompt → llm → output_parser
        """
        retriever = self.vector_store_manager.get_retriever(top_k=RETRIEVAL_TOP_K)

        # 构造 ChatPrompt（System Prompt + 用户问题）
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(RAG_PROMPT_TEMPLATE),
        ])

        # 构建 LCEL Chain
        chain = (
            {
                # 对应流程：Query → Retriever → format_docs（生成 context）
                "context": retriever | RunnableLambda(format_docs),
                # 原始 question 直传
                "question": RunnablePassthrough(),
            }
            | prompt          # Augmented Query：将 context + question 注入 Prompt
            | self.llm        # LLM 生成 Answer
            | StrOutputParser()  # 解析输出为字符串
        )
        return chain

    @property
    def chain(self):
        """懒加载 Chain"""
        if self._chain is None:
            self._chain = self._build_chain()
        return self._chain

    def query(self, question: str, history: List[Dict] = None) -> Dict[str, Any]:
        """
        执行 RAG 问答（非流式）

        Args:
            question: 用户提问
            history: 对话历史 [{"role": "user"|"assistant", "content": str}]

        Returns:
            {
              "answer": str,
              "source_documents": [...],
              "source_scores": [...],
            }
        """
        logger.info(f"收到问题：{question}")

        # Step 1: 检索相关 Chunks（带分数和检索类型）
        docs_with_scores = self.vector_store_manager.similarity_search_with_scores(question)
        relevant_docs = [item[0] for item in docs_with_scores]

        # Step 2: 格式化 Context（包含历史对话）
        context = format_docs_with_scores(docs_with_scores)
        if history:
            history_text = self._format_history(history)
            context = f"【对话历史】\n{history_text}\n\n【本次检索结果】\n{context}"

        # Step 3: 构造完整 Augmented Query 并调用 LLM
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(RAG_PROMPT_TEMPLATE),
        ])

        messages = prompt.format_messages(context=context, question=question)

        openai_messages = _langchain_messages_to_openai(messages)

        response = retry_with_backoff(
            lambda msgs: self.raw_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=msgs,
                max_tokens=2048,
                temperature=0.1,
            ),
            openai_messages,
            max_retries=5,
        )

        answer = response.choices[0].message.content

        input_tokens = 0
        output_tokens = 0
        if response.usage is not None:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0
            logger.debug("原始客户端提取 Token: prompt_tokens=%s, completion_tokens=%s",
                         input_tokens, output_tokens)
        else:
            logger.warning("API 响应中 usage 为 None，无法提取 Token 使用量")

        logger.info(f"问答完成，答案长度：{len(answer)} 字符，Token: 输入={input_tokens} 输出={output_tokens}")

        return {
            "answer": answer,
            "source_documents": relevant_docs,
            "source_scores": docs_with_scores,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def _format_history(self, history: List[Dict]) -> str:
        """格式化对话历史"""
        lines = []
        for msg in history[-6:]:  # 保留最近 6 条
            role = "用户" if msg.get("role") == "user" else "助手"
            lines.append(f"{role}：{msg.get('content', '')}")
        return "\n".join(lines)

    def query_stream(self, question: str, history: List[Dict] = None):
        """
        执行 RAG 问答（流式输出，用于 SSE 接口）

        Args:
            question: 用户提问
            history: 对话历史

        Yields:
            str: 逐步生成的文本片段
        """
        logger.info(f"流式问答，问题：{question}")

        # 检索相关 Chunks（带分数）
        docs_with_scores = self.vector_store_manager.similarity_search_with_scores(question)
        context = format_docs_with_scores(docs_with_scores)

        # 注入历史
        if history:
            history_text = self._format_history(history)
            context = f"【对话历史】\n{history_text}\n\n【本次检索结果】\n{context}"

        # 构造 Prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(RAG_PROMPT_TEMPLATE),
        ])
        messages = prompt.format_messages(context=context, question=question)

        # 流式调用 LLM
        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content

    def get_source_info(self, question: str) -> List[Dict]:
        """
        仅返回检索到的来源文档信息（不调用 LLM）
        用于调试或展示引用来源
        """
        docs_with_scores = self.vector_store_manager.similarity_search_with_scores(question)
        result = []
        for item in docs_with_scores:
            doc = item[0]
            score = item[1]
            rtype = item[2] if len(item) > 2 else "vec"
            similarity = _calc_similarity(score, rtype)
            result.append({
                "source": doc.metadata.get("source", "未知"),
                "content_preview": doc.page_content[:200] + "...",
                "metadata": doc.metadata,
                "score": round(score, 4),
                "similarity": round(similarity, 1),
            })
        return result

    def query_with_profile(
        self,
        question: str,
        history: List[Dict] = None,
        user_profile_context: str = "",
        style: str = "detailed",
    ) -> Dict[str, Any]:
        """
        带用户画像的自适应问答（非流式）

        Args:
            question: 用户提问
            history: 对话历史
            user_profile_context: 用户画像上下文（注入 System Prompt）
            style: 回答风格 concise/detailed/technical/plain

        Returns:
            {
              "answer": str,
              "source_documents": [...],
              "source_scores": [...],
              "retrieval_details": {...},
            }
        """
        logger.info(f"自适应问答，问题：{question}，风格：{style}")

        # Step 1: 检索
        docs_with_scores = self.vector_store_manager.similarity_search_with_scores(question)
        relevant_docs = [item[0] for item in docs_with_scores]

        # 记录检索详情（用于溯源树）
        retrieval_details = {
            "method": "混合检索" if self.vector_store_manager.is_hybrid_search_enabled() else "向量检索",
            "candidate_count": len(docs_with_scores),
            "reranker_enabled": self.vector_store_manager.is_reranker_enabled(),
        }

        # Step 2: 格式化 Context
        context = format_docs_with_scores(docs_with_scores)
        if history:
            history_text = self._format_history(history)
            context = f"【对话历史】\n{history_text}\n\n【本次检索结果】\n{context}"

        # Step 3: 构建自适应 System Prompt
        adaptive_system_prompt = SYSTEM_PROMPT
        if user_profile_context:
            adaptive_system_prompt = (
                SYSTEM_PROMPT
                + "\n\n## 个性化适配指令\n"
                + user_profile_context
            )

        # Step 4: 构造 Prompt 并调用 LLM
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(adaptive_system_prompt),
            HumanMessagePromptTemplate.from_template(RAG_PROMPT_TEMPLATE),
        ])

        messages = prompt.format_messages(context=context, question=question)
        openai_messages = _langchain_messages_to_openai(messages)

        response = retry_with_backoff(
            lambda msgs: self.raw_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=msgs,
                max_tokens=2048,
                temperature=0.1,
            ),
            openai_messages,
            max_retries=5,
        )

        answer = response.choices[0].message.content

        input_tokens = 0
        output_tokens = 0
        if response.usage is not None:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0

        logger.info(f"自适应问答完成，答案长度：{len(answer)} 字符")

        return {
            "answer": answer,
            "source_documents": relevant_docs,
            "source_scores": docs_with_scores,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "retrieval_details": retrieval_details,
        }

    def query_stream_with_profile(
        self,
        question: str,
        history: List[Dict] = None,
        user_profile_context: str = "",
    ):
        """
        带用户画像的自适应流式问答

        Args:
            question: 用户提问
            history: 对话历史
            user_profile_context: 用户画像上下文

        Yields:
            str: 逐步生成的文本片段
        """
        logger.info(f"自适应流式问答，问题：{question}")

        docs_with_scores = self.vector_store_manager.similarity_search_with_scores(question)
        context = format_docs_with_scores(docs_with_scores)

        if history:
            history_text = self._format_history(history)
            context = f"【对话历史】\n{history_text}\n\n【本次检索结果】\n{context}"

        adaptive_system_prompt = SYSTEM_PROMPT
        if user_profile_context:
            adaptive_system_prompt = (
                SYSTEM_PROMPT
                + "\n\n## 个性化适配指令\n"
                + user_profile_context
            )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(adaptive_system_prompt),
            HumanMessagePromptTemplate.from_template(RAG_PROMPT_TEMPLATE),
        ])
        messages = prompt.format_messages(context=context, question=question)

        for chunk in self.llm.stream(messages):
            if chunk.content:
                yield chunk.content

    def get_retrieval_details(self, question: str) -> Dict:
        """获取检索详情（用于溯源树构建）"""
        docs_with_scores = self.vector_store_manager.similarity_search_with_scores(question)
        return {
            "method": "混合检索" if self.vector_store_manager.is_hybrid_search_enabled() else "向量检索",
            "candidate_count": len(docs_with_scores),
            "reranker_enabled": self.vector_store_manager.is_reranker_enabled(),
            "docs": [
                {
                    "source": item[0].metadata.get("source", "未知"),
                    "score": item[1],
                    "retrieval_type": item[2] if len(item) > 2 else "vec",
                }
                for item in docs_with_scores
            ],
        }
