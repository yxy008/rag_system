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

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    RETRIEVAL_TOP_K,
)
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


def _calc_similarity(score: float, rtype: str) -> float:
    """
    根据检索类型计算相似度百分比 [0, 100]。

    - rerank: 分数已在 [0, 100] 范围（Reranker 输出 * 100），直接使用
    - vec: Chroma 余弦距离 [0, 2]，0=完全相同，2=完全相反
    - 其他（bm25 / RRF 融合）: 使用 sigmoid 归一化到 [0, 100]
    """
    if rtype == "rerank":
        return max(0.0, min(100.0, score))
    elif rtype == "vec":
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

        # 调用 LLM
        messages = prompt.format_messages(context=context, question=question)
        response = self.llm.invoke(messages)
        answer = response.content

        logger.info(f"问答完成，答案长度：{len(answer)} 字符")

        return {
            "answer": answer,
            "source_documents": relevant_docs,
            "source_scores": docs_with_scores,
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
