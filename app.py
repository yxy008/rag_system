"""
app.py - Flask Web 服务入口

提供 REST API + SSE 流式接口 + 静态 Web 前端

接口列表：
  GET  /                      - 问答界面（HTML）
  POST /api/chat              - 普通问答（JSON 响应）
  POST /api/chat/stream       - 流式问答（SSE）
  GET  /api/status             - 系统状态
  POST /api/ingest             - 触发文档入库（API 方式）
  POST /api/upload             - 上传单个文件并入库
  GET  /api/history/<session>  - 获取对话历史
  DELETE /api/history/<session> - 清空指定对话历史
  GET  /api/config             - 获取系统配置（含混合检索状态）
  POST /api/config             - 更新系统配置（混合检索开关等）
  GET  /api/health             - 知识库健康检查（五层检查体系）
  GET  /api/health?quick=true  - 快速健康检查（跳过检索抽样）
  POST /api/evaluation/ragas/phase1       - RAGAS Phase 1 评估（4指标，无需 ground truth）
  POST /api/evaluation/ragas/phase2       - RAGAS Phase 2 评估（8指标，需 ground truth）
  GET  /api/evaluation/ragas/report       - 获取评估报告
  GET  /api/evaluation/ragas/trend        - 获取评估趋势
  GET  /api/evaluation/ragas/samples      - 获取评估样本预览
  POST /api/evaluation/ragas/ground-truth - 添加 ground truth
  GET  /api/evaluation/ragas/ground-truth - 查看所有 ground truth
  DELETE /api/evaluation/ragas/ground-truth/<id> - 删除 ground truth
"""
import json
import logging
import uuid
import os
import gc
import sqlite3
import tempfile
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, jsonify, render_template, Response, stream_with_context, g
from flask_cors import CORS

# ========== 设置 jieba 缓存目录（避免存储在 C 盘） ==========
# jieba 使用 tempfile.gettempdir() 确定缓存目录，默认在 C 盘
# 在导入 vector_store（内部 import jieba）之前设置 tempdir
_JIEBA_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(_JIEBA_CACHE_DIR, exist_ok=True)
tempfile.tempdir = _JIEBA_CACHE_DIR

from core.config import (
    FLASK_PORT, FLASK_DEBUG, DOCUMENTS_DIR,
    CACHE_ENABLED, CACHE_SIMILARITY_THRESHOLD, CACHE_COARSE_THRESHOLD,
    CACHE_RERANKER_THRESHOLD, CACHE_CANDIDATE_COUNT, CACHE_MAX_ENTRIES, CACHE_TTL_HOURS, OPENAI_MODEL,
    VECTOR_STORE_BACKEND, CHROMA_HOST, CHROMA_PORT,
    MILVUS_HOST, MILVUS_PORT, MILVUS_DIMENSION, MILVUS_INDEX_TYPE, MILVUS_METRIC_TYPE,
    MILVUS_CACHE_INDEX_TYPE,
    FAISS_PERSIST_DIR,
)
from core.document_processor import DocumentProcessor, SUPPORTED_EXTENSIONS
from core.vector_store import VectorStoreManager
from core.rag_chain import RAGChain, _calc_similarity
from core.semantic_cache import SemanticCache
from services.evaluation import evaluation_service
from storage.conversation_store import conversation_store
from core.rate_limiter import rate_limiter
from core.llm_retry import retry_with_backoff
from services.knowledge_health import KnowledgeHealthChecker
from services.user_profile import profile_manager
from services.confidence import confidence_evaluator
from services.knowledge_graph import knowledge_graph, ComparativeQA, ScenarioSimulator, DocumentSummarizer
from services.collaboration import answer_feedback, expert_router
from services.ragas_evaluation import ragas_evaluator

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ========== 初始化 Flask ==========
app = Flask(__name__)
CORS(app)

# ========== 初始化核心组件（单例） ==========
vs_manager = VectorStoreManager()
rag = RAGChain(vector_store_manager=vs_manager)

# ========== 初始化语义缓存 ==========
semantic_cache = SemanticCache(
    backend=VECTOR_STORE_BACKEND,
    chroma_host=CHROMA_HOST,
    chroma_port=CHROMA_PORT,
    milvus_host=MILVUS_HOST,
    milvus_port=int(MILVUS_PORT),
    milvus_dimension=MILVUS_DIMENSION,
    milvus_index_type=MILVUS_CACHE_INDEX_TYPE,
    milvus_metric_type=MILVUS_METRIC_TYPE,
    faiss_dimension=MILVUS_DIMENSION,
    faiss_persist_dir=FAISS_PERSIST_DIR,
    embedding_function=vs_manager.embeddings,
    enabled=CACHE_ENABLED,
    similarity_threshold=CACHE_SIMILARITY_THRESHOLD,
    coarse_threshold=CACHE_COARSE_THRESHOLD,
    reranker_threshold=CACHE_RERANKER_THRESHOLD,
    candidate_count=CACHE_CANDIDATE_COUNT,
    max_entries=CACHE_MAX_ENTRIES,
    ttl_hours=CACHE_TTL_HOURS,
)

# 注册文档变更回调：入库/清空时自动清空语义缓存
vs_manager.add_on_documents_changed(lambda: semantic_cache.invalidate())

# ========== 初始化知识库健康检查器 ==========
doc_processor = DocumentProcessor()
health_checker = KnowledgeHealthChecker(vs_manager, doc_processor)


def auto_warmup_on_startup():
    """启动时自动预热：从 cache_warmup.json 加载 FAQ 并预热"""
    import json as _json
    warmup_file = os.path.join(os.path.dirname(__file__), "data", "cache_warmup.json")
    try:
        if not os.path.exists(warmup_file):
            logger.info("未找到 cache_warmup.json，跳过自动预热")
            return

        with open(warmup_file, "r", encoding="utf-8") as f:
            entries = _json.load(f)

        if not entries:
            logger.info("cache_warmup.json 为空，跳过自动预热")
            return

        count = semantic_cache.warmup(entries)
        logger.info("启动自动预热完成：成功预热 %d 条 FAQ", count)

    except Exception as e:
        logger.warning("启动自动预热失败：%s", e)


auto_warmup_on_startup()


def start_cache_cleanup_task():
    """启动定时清理过期缓存的后台任务（每 30 分钟执行一次）"""
    def cleanup_loop():
        while True:
            time.sleep(1800)
            try:
                semantic_cache.cleanup_expired()
            except Exception as e:
                logger.error("定时清理缓存失败：%s", e)

    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()


# 启动缓存清理后台任务
start_cache_cleanup_task()


def start_gc_collect_task():
    """启动定期垃圾回收的后台任务（每 10 分钟执行一次）"""
    def gc_loop():
        while True:
            time.sleep(600)
            try:
                collected = gc.collect()
                if collected > 0:
                    logger.debug("GC 回收了 %d 个对象", collected)
            except Exception as e:
                logger.error("GC 回收失败：%s", e)

    thread = threading.Thread(target=gc_loop, daemon=True)
    thread.start()


start_gc_collect_task()

# ========== 对话历史存储（SQLite 持久化） ==========
def get_history(session_id: str, user_id: str = "") -> list:
    """获取指定 session 的对话历史，可按用户ID过滤"""
    return conversation_store.get_history(session_id, user_id)


def add_to_history(session_id: str, role: str, content: str, sources: list = None, user_id: str = ""):
    """向指定 session 添加对话记录，可选附带来源信息和用户ID"""
    conversation_store.save(session_id, role, content, sources, user_id)


def clear_history(session_id: str, user_id: str = ""):
    """清空指定 session 的对话历史，可按用户ID过滤"""
    conversation_store.delete_session(session_id, user_id)


def _get_auth_user_id() -> str:
    """从认证上下文获取当前用户ID"""
    user_id = getattr(g, 'current_user_id', None)
    return str(user_id) if user_id else ""


# ============================================================
# 路由定义
# ============================================================

@app.before_request
def check_rate_limit():
    """全局速率限制检查（仅对 /api/ 开头的请求生效）"""
    if not request.path.startswith("/api/"):
        return None

    # 使用客户端 IP 作为限流 key
    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
    # 取第一个 IP（如果有代理）
    client_ip = client_ip.split(",")[0].strip()

    allowed, retry_after = rate_limiter.is_allowed(client_ip)
    if not allowed:
        logger.warning("速率限制触发：IP=%s，需等待 %.1f 秒", client_ip, retry_after)
        return jsonify({
            "error": "请求过于频繁，请稍后再试",
            "retry_after_seconds": round(retry_after, 1),
        }), 429

    return None


@app.route("/")
def index():
    """返回前端聊天界面"""
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    普通问答接口（非流式）

    Request Body:
        {
          "question": "你的问题",
          "session_id": "可选的会话ID，不传则自动生成",
          "hybrid": true/false  // 可选，覆盖混合检索设置
        }

    Response:
        {
          "answer": "LLM 生成的答案",
          "session_id": "会话ID",
          "sources": [{"source": "...", "similarity": "...", "retrieval_type": "vec+bm25"}]
        }
    """
    data = request.get_json()
    start_time = time.time()
    if not data or "question" not in data:
        return jsonify({"error": "请提供 question 字段"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    session_id = data.get("session_id", str(uuid.uuid4()))
    hybrid = data.get("hybrid")
    reranker = data.get("reranker")
    style = data.get("style", "detailed")
    username = data.get("username")
    user_id = _get_auth_user_id()

    try:
        # 获取用户画像上下文
        user_profile_context = ""
        if username:
            try:
                user_profile_context = profile_manager.get_adaptive_prompt_context(username, style)
            except Exception as e:
                logger.warning(f"获取用户画像失败：{e}")

        # 检查知识库
        doc_count = vs_manager.get_document_count()
        if doc_count == 0:
            return jsonify({
                "error": "知识库为空，请先运行 python ingest.py 将文档入库"
            }), 400

        # 获取历史
        history = get_history(session_id, user_id)
        if history:
            logger.info(f"会话 {session_id} 有 {len(history)} 条历史记录")

        # ===== 两级缓存检查 =====
        logger.info("[app-常规] 开始缓存查询, history=%s", bool(history))
        cached = semantic_cache.lookup(question)
        if cached:
            logger.info("缓存命中（%s），跳过检索和 LLM 生成：query='%s' -> cached='%s'",
                        cached.get("match_type"), question, cached["query"])

            semantic_cache.record_hit(cached.get("match_type", "semantic"))

            sources = json.loads(cached["sources_json"])

            # 语义命中后将新查询也存入缓存，下次同样问题直接精确匹配
            if cached.get("match_type") == "semantic":
                semantic_cache.store(question, cached["answer"], sources)

            add_to_history(session_id, "user", question, user_id=user_id)
            add_to_history(session_id, "assistant", cached["answer"], sources, user_id)

            latency = (time.time() - start_time) * 1000
            evaluation_service.record_request(
                True, cached.get("match_type", "semantic"), latency, 0, len(sources),
                question=question,
            )

            # 可信度评估（缓存命中时也评估）
            confidence = None
            try:
                retrieval_details = {
                    "method": "缓存命中",
                    "candidate_count": len(sources),
                    "reranker_enabled": False,
                }
                confidence = confidence_evaluator.evaluate(
                    sources=sources,
                    answer=cached["answer"],
                    question=question,
                    retrieval_details=retrieval_details,
                )
                confidence_evaluator.save_provenance(
                    session_id=session_id,
                    question=question,
                    answer=cached["answer"],
                    sources=sources,
                    provenance_tree=confidence.get("provenance_tree", {}),
                    confidence=confidence,
                )
            except Exception as e:
                logger.warning(f"可信度评估失败：{e}")

            return jsonify({
                "answer": cached["answer"],
                "session_id": session_id,
                "sources": sources,
                "from_cache": True,
                "cached_query": cached["query"],
                "cache_score": f"{cached['score']:.4f}",
                "cache_match_type": cached.get("match_type", "semantic"),
                "confidence": confidence,
            })
        else:
            semantic_cache.record_miss()

        # 临时切换混合检索/Reranker/多查询 状态
        orig_hybrid = vs_manager.is_hybrid_search_enabled()
        orig_reranker = vs_manager.is_reranker_enabled()
        orig_multi_query = vs_manager.is_multi_query_enabled()
        if hybrid is not None:
            vs_manager.set_hybrid_search(bool(hybrid))
        if reranker is not None:
            vs_manager.set_reranker(bool(reranker))
        # 禁用多查询 LLM 改写，避免一次请求调用两次 LLM
        vs_manager.set_multi_query(False)

        retrieval_start = time.time()
        if user_profile_context:
            result = rag.query_with_profile(question, history=history, user_profile_context=user_profile_context, style=style)
        else:
            result = rag.query(question, history=history)
        retrieval_latency = (time.time() - retrieval_start) * 1000

        # 恢复原始状态
        vs_manager.set_hybrid_search(orig_hybrid)
        vs_manager.set_reranker(orig_reranker)
        vs_manager.set_multi_query(orig_multi_query)

        # 整理来源信息
        sources = []
        for item in result.get("source_scores", []):
            doc = item[0]
            score = item[1]
            rtype = item[2] if len(item) > 2 else "vec"
            source = doc.metadata.get("source", "未知")

            similarity = _calc_similarity(score, rtype)

            # 检索类型标注
            if rtype == "vec":
                type_label = "语义"
            elif rtype == "bm25":
                type_label = "关键词"
            elif "+" in rtype:
                type_label = "混合"
            else:
                type_label = "向量"

            sources.append({
                "source": source.split("\\")[-1].split("/")[-1],
                "preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                "full_content": doc.page_content,
                "score": round(score, 4),
                "similarity": f"{similarity:.1f}%",
                "similarity_value": round(similarity, 1),
                "retrieval_type": type_label,
                "page": doc.metadata.get("page"),
                "total_pages": doc.metadata.get("total_pages"),
            })

        # 按相似度降序排列
        sources.sort(key=lambda x: x["similarity_value"], reverse=True)

        # 保存对话历史（附带来源信息）
        add_to_history(session_id, "user", question, user_id=user_id)
        add_to_history(session_id, "assistant", result["answer"], sources, user_id)

        # ===== 存入语义缓存（缓存未命中/不可用则存入） =====
        logger.info("[app-常规] 缓存未命中，存入缓存, query='%s'", question[:80])
        semantic_cache.store(question, result["answer"], sources)

        total_latency = (time.time() - start_time) * 1000
        evaluation_service.record_request(
            False, None, total_latency, retrieval_latency, len(sources),
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
            question=question,
        )

        # 可信度评估
        confidence = None
        try:
            retrieval_details = result.get("retrieval_details", {})
            confidence = confidence_evaluator.evaluate(
                sources=sources,
                answer=result["answer"],
                question=question,
                retrieval_details=retrieval_details,
            )
            confidence_evaluator.save_provenance(
                session_id=session_id,
                question=question,
                answer=result["answer"],
                sources=sources,
                provenance_tree=confidence.get("provenance_tree", {}),
                confidence=confidence,
            )
        except Exception as e:
            logger.warning(f"可信度评估失败：{e}")

        return jsonify({
            "answer": result["answer"],
            "session_id": session_id,
            "sources": sources,
            "from_cache": False,
            "confidence": confidence,
        })

    except Exception as e:
        logger.error(f"问答失败：{e}", exc_info=True)
        return jsonify({"error": f"服务内部错误：{str(e)}"}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """
    流式问答接口（Server-Sent Events）

    Request Body:
        {
          "question": "你的问题",
          "session_id": "可选的会话ID",
          "hybrid": true/false  // 可选
        }

    Response: SSE 流
        data: {"type": "token", "content": "..."}
        data: {"type": "done", "session_id": "...", "sources": [...]}
        data: {"type": "error", "message": "..."}
    """
    logger.info(f"流式问答接口：{request.get_json()}")
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "请提供 question 字段"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    session_id = data.get("session_id", str(uuid.uuid4()))
    hybrid = data.get("hybrid")
    reranker = data.get("reranker")
    style = data.get("style", "detailed")
    username = data.get("username")
    user_id = _get_auth_user_id()

    def generate(_question=question, _session_id=session_id, _hybrid=hybrid,
                 _reranker=reranker, _style=style, _username=username, _user_id=user_id):
        # 将闭包捕获的变量解包为局部变量，避免生成器 free variable 报错
        question = _question
        session_id = _session_id
        hybrid = _hybrid
        reranker = _reranker
        style = _style
        username = _username
        user_id = _user_id

        stream_start_time = time.time()
        try:
            # 获取用户画像上下文
            user_profile_context = ""
            if username:
                try:
                    user_profile_context = profile_manager.get_adaptive_prompt_context(username, style)
                except Exception as e:
                    logger.warning(f"获取用户画像失败：{e}")

            doc_count = vs_manager.get_document_count()
            if doc_count == 0:
                yield f"data: {json.dumps({'type': 'error', 'message': '知识库为空，请先入库文档'}, ensure_ascii=False)}\n\n"
                return

            # 临时切换混合检索/Reranker/多查询 状态
            orig_hybrid = vs_manager.is_hybrid_search_enabled()
            orig_reranker = vs_manager.is_reranker_enabled()
            orig_multi_query = vs_manager.is_multi_query_enabled()
            if hybrid is not None:
                vs_manager.set_hybrid_search(bool(hybrid))
            if reranker is not None:
                vs_manager.set_reranker(bool(reranker))
            # 流式接口禁用多查询 LLM 改写，避免一次请求调用两次 LLM
            vs_manager.set_multi_query(False)

            # 获取历史
            history = get_history(session_id, user_id)
            logger.info(f"[app-流式] session={session_id[:16]}..., history_len={len(history)}, history_raw={history}")
            
            # ===== 两级缓存检查 =====
            logger.info("[app-流式] 开始缓存查询, history=%s", bool(history))
            cached = semantic_cache.lookup(question)
            if cached:
                logger.info("流式-缓存命中（%s）：query='%s' -> cached='%s'",
                            cached.get("match_type"), question, cached["query"])

                semantic_cache.record_hit(cached.get("match_type", "semantic"))

                sources = json.loads(cached["sources_json"])

                # 语义命中后将新查询也存入缓存，下次同样问题直接精确匹配
                if cached.get("match_type") == "semantic":
                    semantic_cache.store(question, cached["answer"], sources)

                # 先发送数据源事件
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

                # 模拟流式输出：按字符分块发送缓存答案
                answer = cached["answer"]
                chunk_size = 10
                for i in range(0, len(answer), chunk_size):
                    chunk = answer[i:i + chunk_size]
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"

                # 可信度评估（先于 done 事件，确保结果返回前端）
                confidence = None
                try:
                    retrieval_details = {
                        "method": "缓存命中",
                        "candidate_count": len(sources),
                        "reranker_enabled": False,
                    }
                    confidence = confidence_evaluator.evaluate(
                        sources=sources,
                        answer=answer,
                        question=question,
                        retrieval_details=retrieval_details,
                    )
                    confidence_evaluator.save_provenance(
                        session_id=session_id,
                        question=question,
                        answer=answer,
                        sources=sources,
                        provenance_tree=confidence.get("provenance_tree", {}),
                        confidence=confidence,
                    )
                except Exception as e:
                    logger.warning(f"可信度评估失败：{e}")

                yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'sources': sources, 'from_cache': True, 'confidence': confidence}, ensure_ascii=False)}\n\n"

                add_to_history(session_id, "user", question, user_id=user_id)
                add_to_history(session_id, "assistant", answer, sources, user_id)

                # 恢复原始状态
                vs_manager.set_hybrid_search(orig_hybrid)
                vs_manager.set_reranker(orig_reranker)
                vs_manager.set_multi_query(orig_multi_query)

                # 记录评估数据
                cache_latency = (time.time() - stream_start_time) * 1000
                evaluation_service.record_request(
                    True, cached.get("match_type", "semantic"),
                    cache_latency, 0, len(sources),
                    question=question,
                )
                return
            else:
                semantic_cache.record_miss()

            # 先检索（获取 sources），再流式生成
            retrieval_start = time.time()
            docs_with_scores = vs_manager.similarity_search_with_scores(question)
            retrieval_latency = (time.time() - retrieval_start) * 1000
            from core.rag_chain import format_docs_with_scores
            context = format_docs_with_scores(docs_with_scores)

            if history:
                history_text = rag._format_history(history)
                context = f"【对话历史】\n{history_text}\n\n【本次检索结果】\n{context}"

            # 构建 sources 并立即发送（先于答案流式输出）
            sources = []
            for item in docs_with_scores:
                doc = item[0]
                score = item[1]
                rtype = item[2] if len(item) > 2 else "vec"
                source = doc.metadata.get("source", "未知")
                similarity = _calc_similarity(score, rtype)
                if rtype == "vec":
                    type_label = "语义"
                elif rtype == "bm25":
                    type_label = "关键词"
                elif rtype == "rerank":
                    type_label = "重排"
                elif "+" in rtype:
                    type_label = "混合"
                else:
                    type_label = "向量"
                sources.append({
                    "source": source.split("\\")[-1].split("/")[-1],
                    "preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                    "full_content": doc.page_content,
                    "score": round(score, 4),
                    "similarity": f"{similarity:.1f}%",
                    "retrieval_type": type_label,
                    "page": doc.metadata.get("page"),
                    "total_pages": doc.metadata.get("total_pages"),
                })

            # 先发送数据源事件（前端立即渲染折叠的数据源）
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

            from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
            from core.config import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, OPENAI_MODEL

            adaptive_system_prompt = SYSTEM_PROMPT
            if user_profile_context:
                adaptive_system_prompt = SYSTEM_PROMPT + "\n\n## 个性化适配指令\n" + user_profile_context

            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(adaptive_system_prompt),
                HumanMessagePromptTemplate.from_template(RAG_PROMPT_TEMPLATE),
            ])
            messages = prompt.format_messages(context=context, question=question)

            from core.rag_chain import _langchain_messages_to_openai
            openai_messages = _langchain_messages_to_openai(messages)

            answer_parts = []
            input_tokens = 0
            output_tokens = 0

            stream = rag.raw_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=openai_messages,
                max_tokens=2048,
                temperature=0.1,
                stream=True,
                stream_options={"include_usage": True},
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    answer_parts.append(content)
                    yield f"data: {json.dumps({'type': 'token', 'content': content}, ensure_ascii=False)}\n\n"

                if chunk.usage is not None:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0
                    logger.debug("流式原始客户端提取 Token: prompt_tokens=%s, completion_tokens=%s",
                                 input_tokens, output_tokens)

            answer = "".join(answer_parts)

            # 保存历史（附带来源信息）
            add_to_history(session_id, "user", question, user_id=user_id)
            add_to_history(session_id, "assistant", answer, sources, user_id)

            # 恢复原始状态
            vs_manager.set_hybrid_search(orig_hybrid)
            vs_manager.set_reranker(orig_reranker)
            vs_manager.set_multi_query(orig_multi_query)

            # ===== 存入语义缓存（缓存未命中/不可用则存入） =====
            logger.info("[app-流式] 缓存未命中，存入缓存, query='%s'", question[:80])
            semantic_cache.store(question, answer, sources)

            # 记录评估数据
            total_latency = (time.time() - stream_start_time) * 1000
            evaluation_service.record_request(
                False, None, total_latency, retrieval_latency, len(sources),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                question=question,
            )

            # 可信度评估
            confidence = None
            try:
                retrieval_details = {
                    "method": "混合检索" if vs_manager.is_hybrid_search_enabled() else "向量检索",
                    "candidate_count": len(docs_with_scores),
                    "reranker_enabled": vs_manager.is_reranker_enabled(),
                }
                confidence = confidence_evaluator.evaluate(
                    sources=sources,
                    answer=answer,
                    question=question,
                    retrieval_details=retrieval_details,
                )
                confidence_evaluator.save_provenance(
                    session_id=session_id,
                    question=question,
                    answer=answer,
                    sources=sources,
                    provenance_tree=confidence.get("provenance_tree", {}),
                    confidence=confidence,
                )
            except Exception as e:
                logger.warning(f"可信度评估失败：{e}")

            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'sources': sources, 'from_cache': False, 'confidence': confidence}, ensure_ascii=False)}\n\n"

        except GeneratorExit:
            pass
        except Exception as e:
            logger.error(f"流式问答失败：{e}", exc_info=True)
            try:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            except Exception:
                pass

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/status", methods=["GET"])
def status():
    """
    系统状态接口

    Response:
        {
          "status": "ok",
          "doc_count": 1234,
          "embedding_model": "BAAI/bge-m3",
          "llm_model": "GLM-4.7",
          "hybrid_search": true,
          "vector_store_backend": "milvus / chroma / faiss",
          "active_sessions": 3,
        }
    """
    from core.config import EMBEDDING_MODEL_NAME, OPENAI_MODEL, get_model_display_name
    doc_count = vs_manager.get_document_count()
    return jsonify({
        "status": "ok",
        "doc_count": doc_count,
        "embedding_model": get_model_display_name(EMBEDDING_MODEL_NAME),
        "llm_model": OPENAI_MODEL,
        "knowledge_base_ready": doc_count > 0,
        "hybrid_search": vs_manager.is_hybrid_search_enabled(),
        "reranker": vs_manager.is_reranker_enabled(),
        "vector_store_backend": vs_manager.backend,
        "active_sessions": conversation_store.get_active_session_count(),
        "semantic_cache": semantic_cache.get_stats(),
    })


@app.route("/api/dashboard", methods=["GET"])
def dashboard():
    """
    仪表盘综合统计接口
    聚合系统状态、知识库、缓存、会话等全部统计信息
    """
    import platform
    import psutil

    doc_count = vs_manager.get_document_count()
    cache_stats = semantic_cache.get_stats()

    # 系统信息
    mem = psutil.virtual_memory()
    system_info = {
        "status": "running",
        "python_version": platform.python_version(),
        "os": platform.system(),
        "cpu_count": psutil.cpu_count(),
        "memory_total_mb": round(mem.total / (1024 * 1024), 1),
        "memory_used_mb": round(mem.used / (1024 * 1024), 1),
        "memory_free_mb": round(mem.available / (1024 * 1024), 1),
        "memory_percent": mem.percent,
    }

    # 知识库统计
    knowledge_base = {
        "document_count": doc_count,
        "ready": doc_count > 0,
    }

    # 模型信息
    from core.config import EMBEDDING_MODEL_NAME, OPENAI_MODEL, RERANKER_MODEL_NAME, get_model_display_name
    models = {
        "llm": OPENAI_MODEL,
        "embedding": get_model_display_name(EMBEDDING_MODEL_NAME),
        "reranker": get_model_display_name(RERANKER_MODEL_NAME) if vs_manager.is_reranker_enabled() else "未启用",
        "hybrid_search": vs_manager.is_hybrid_search_enabled(),
    }

    # 检索配置
    search_config = {
        "hybrid_search": vs_manager.is_hybrid_search_enabled(),
        "reranker": vs_manager.is_reranker_enabled(),
        "vector_store_backend": vs_manager.backend,
    }

    # 会话统计
    sessions_info = {
        "active_count": conversation_store.get_active_session_count(),
    }

    # 技术栈描述（动态，根据实际配置生成）
    backend_labels = {
        "chroma": "ChromaDB",
        "milvus": "Milvus",
        "faiss": "FAISS",
    }
    backend_label = backend_labels.get(vs_manager.backend, vs_manager.backend)
    tech_stack = f"Python + LangChain + {backend_label} + {OPENAI_MODEL}"
    if vs_manager.is_hybrid_search_enabled():
        tech_stack += " + BM25 混合检索"
    if vs_manager.is_reranker_enabled():
        tech_stack += " + BGE Reranker"

    return jsonify({
        "system": system_info,
        "knowledge_base": knowledge_base,
        "models": models,
        "search_config": search_config,
        "tech_stack": tech_stack,
        "semantic_cache": cache_stats,
        "cache_warmup": semantic_cache.get_warmup_status(entry_count=cache_stats.get("entry_count")),
        "evaluation": evaluation_service.get_stats(),
        "sessions": sessions_info,
        "rate_limiter": rate_limiter.get_stats(),
    })


@app.route("/api/health", methods=["GET"])
def knowledge_health():
    """
    知识库健康检查接口

    返回多维度健康检查报告：
      - 文档层：重复文档、空文档检测
      - 切片层：空切片、过短/过长切片、长度分布
      - 向量层：零向量、维度一致性、Embedding 模型状态
      - 检索层：BM25/向量库一致性、检索功能抽样测试
      - 索引层：向量库连接、BM25 索引状态

    Query Parameters:
      quick: 设为 "true" 执行快速检查（跳过检索质量抽样）
    """
    quick = request.args.get("quick", "false").lower() == "true"
    if quick:
        report = health_checker.quick_check()
    else:
        report = health_checker.full_check()
    return jsonify(report)


@app.route("/api/evaluation", methods=["GET"])
def get_evaluation():
    """获取评估统计信息"""
    return jsonify(evaluation_service.get_stats())


@app.route("/api/evaluation/reset", methods=["POST"])
def reset_evaluation():
    """重置评估统计"""
    evaluation_service.reset()
    return jsonify({"success": True, "message": "评估统计已重置"})


# ======================================================================
#  RAGAS 质量评估 API
# ======================================================================

@app.route("/api/evaluation/ragas/phase1", methods=["POST"])
def run_ragas_phase1():
    """
    执行 Phase 1 评估（无需 ground truth）

    评估指标：Faithfulness, Answer Relevancy, Context Precision, Context Relevancy

    Request Body (optional):
        {"sample_limit": 20}   - 评估样本数量，默认 20
    """
    data = request.get_json(silent=True) or {}
    sample_limit = int(data.get("sample_limit", 20))
    result = ragas_evaluator.run_phase1(sample_limit=sample_limit)
    return jsonify(result)


@app.route("/api/evaluation/ragas/phase2", methods=["POST"])
def run_ragas_phase2():
    """
    执行 Phase 2 评估（需要 ground truth）

    评估指标：Phase 1 四个指标 + Context Recall, Answer Correctness, Answer Semantic Similarity

    前提：需要先在 /api/evaluation/ragas/ground-truth 接口标注 ground truth

    Request Body (optional):
        {"sample_limit": 20}   - 评估样本数量，默认 20
    """
    data = request.get_json(silent=True) or {}
    sample_limit = int(data.get("sample_limit", 20))
    result = ragas_evaluator.run_phase2(sample_limit=sample_limit)
    return jsonify(result)


@app.route("/api/evaluation/ragas/report", methods=["GET"])
def get_ragas_report():
    """
    获取最近的 RAGAS 评估报告

    Query params:
        phase  - "phase1" / "phase2" / 不传则返回两者
        limit  - 返回条数，默认 10
    """
    phase = request.args.get("phase", None)
    limit = int(request.args.get("limit", 10))

    if phase:
        results = ragas_evaluator.get_recent_results(phase=phase, limit=limit)
    else:
        p1 = ragas_evaluator.get_recent_results(phase="phase1", limit=limit)
        p2 = ragas_evaluator.get_recent_results(phase="phase2", limit=limit)
        results = {"phase1": p1, "phase2": p2}

    return jsonify({"results": results, "ground_truth_count": ragas_evaluator.get_ground_truth_count()})


@app.route("/api/evaluation/ragas/trend", methods=["GET"])
def get_ragas_trend():
    """
    获取评估趋势数据（用于前端折线图）

    Query params:
        phase  - "phase1" / "phase2"，默认 "phase1"
        limit  - 返回条数，默认 30
    """
    phase = request.args.get("phase", "phase1")
    limit = int(request.args.get("limit", 30))
    trend = ragas_evaluator.get_trend(phase=phase, limit=limit)
    return jsonify({"phase": phase, "trend": trend})


@app.route("/api/evaluation/ragas/ground-truth", methods=["GET"])
def get_ground_truths():
    """获取所有 ground truth 条目"""
    entries = ragas_evaluator.get_all_ground_truths()
    return jsonify({"entries": entries, "count": len(entries)})


@app.route("/api/evaluation/ragas/ground-truth", methods=["POST"])
def add_ground_truth():
    """
    添加 ground truth（单条或批量）

    Request Body:
        单条: {"question": "...", "ground_truth": "..."}
        批量: {"entries": [{"question": "...", "ground_truth": "..."}, ...]}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "请求体为空"}), 400

    # 批量添加
    if "entries" in data:
        count = ragas_evaluator.batch_add_ground_truth(data["entries"])
        return jsonify({"success": True, "count": count})

    # 单条添加
    question = data.get("question", "").strip()
    ground_truth = data.get("ground_truth", "").strip()
    if not question or not ground_truth:
        return jsonify({"success": False, "error": "question 和 ground_truth 不能为空"}), 400

    success = ragas_evaluator.add_ground_truth(question, ground_truth)
    return jsonify({"success": success})


@app.route("/api/evaluation/ragas/ground-truth/<int:gt_id>", methods=["DELETE"])
def delete_ground_truth(gt_id: int):
    """删除一条 ground truth"""
    success = ragas_evaluator.delete_ground_truth(gt_id)
    return jsonify({"success": success})


@app.route("/api/evaluation/ragas/samples", methods=["GET"])
def get_ragas_sample_data():
    """
    获取可用于评估的样本预览（不实际执行评估）

    Query params:
        limit - 返回样本数，默认 10

    用于前端：展示评估数据来源、手动选择 ground truth 标注
    """
    limit = int(request.args.get("limit", 10))
    samples = ragas_evaluator.collect_eval_samples(limit=limit)
    # 截断长文本用于前端展示
    for s in samples:
        s["context_preview"] = s["contexts"][0][:200] + "..." if s["contexts"] else ""
        s["context_count"] = len(s["contexts"])
        s["answer_preview"] = s["answer"][:200] + "..." if len(s["answer"]) > 200 else s["answer"]
    return jsonify({"samples": samples, "total_collected": len(samples)})


@app.route("/api/cache/warmup", methods=["POST"])
def warmup_cache():
    """
    缓存预热：批量预加载 FAQ 问答对

    Request Body:
        {
            "entries": [
                {"question": "什么是RAG？", "answer": "RAG是检索增强生成..."},
                ...
            ]
        }
    """
    data = request.get_json()
    if not data or "entries" not in data:
        return jsonify({"success": False, "message": "请提供预热条目列表"}), 400

    entries = data["entries"]
    count = semantic_cache.warmup(entries)
    return jsonify({
        "success": True,
        "warmed_count": count,
        "total_entries": len(entries),
        "message": f"缓存预热完成，成功预热 {count} 条"
    })


@app.route("/api/cache/warmup/status", methods=["GET"])
def warmup_status():
    """获取缓存预热状态"""
    return jsonify(semantic_cache.get_warmup_status())


@app.route("/api/ingest", methods=["POST"])
def ingest():
    """
    通过 API 触发文档入库

    Request Body:
        { "clear": false, "dir": "可选的自定义目录" }

    Response:
        { "success": true, "chunks_added": 123 }
    """
    data = request.get_json() or {}
    clear = data.get("clear", False)
    custom_dir = data.get("dir")

    try:
        from core.config import DOCUMENTS_DIR
        processor = DocumentProcessor(embeddings=vs_manager.embeddings)

        if clear:
            vs_manager.clear_collection()

        target_dir = custom_dir or DOCUMENTS_DIR
        chunks = processor.load_and_split(target_dir)
        if not chunks:
            return jsonify({"error": f"目录 {target_dir} 中没有找到可入库的文档"}), 400

        count = vs_manager.add_documents(chunks)
        return jsonify({
            "success": True,
            "chunks_added": count,
            "total_in_db": vs_manager.get_document_count(),
        })

    except Exception as e:
        logger.error(f"文档入库失败：{e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """
    上传单个文件并自动入库

    Request: multipart/form-data
        file: 要上传的文件

    Response:
        {
          "success": true,
          "filename": "xxx.pdf",
          "chunks_added": 5,
          "total_in_db": 1234
        }
    """
    if "file" not in request.files:
        return jsonify({"error": "请选择要上传的文件"}), 400

    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return jsonify({"error": "请选择要上传的文件"}), 400

    # 安全处理文件名
    original_filename = file.filename
    filename = secure_filename(original_filename)
    if not filename:
        return jsonify({"error": "文件名不合法"}), 400

    # 检查文件扩展名
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
        return jsonify({
            "error": f"不支持的文件格式：{ext}。支持的格式：{supported}"
        }), 400

    # 保存文件到 documents 目录
    documents_path = Path(DOCUMENTS_DIR)
    documents_path.mkdir(parents=True, exist_ok=True)

    # 如果文件已存在，添加序号避免覆盖
    save_path = documents_path / filename
    if save_path.exists():
        stem = save_path.stem
        counter = 1
        while save_path.exists():
            save_path = documents_path / f"{stem}_{counter}{ext}"
            counter += 1

    try:
        file.save(str(save_path))
        logger.info(f"文件已保存：{save_path}")
    except Exception as e:
        logger.error(f"保存文件失败：{e}")
        return jsonify({"error": f"保存文件失败：{str(e)}"}), 500

    # 处理文件并入库
    try:
        processor = DocumentProcessor(embeddings=vs_manager.embeddings)
        docs = processor._load_single_file(str(save_path))
        if not docs:
            return jsonify({"error": f"文件内容为空或无法解析：{filename}"}), 400

        chunks = processor.split_documents(docs)
        if not chunks:
            return jsonify({"error": f"文件分割后无有效内容：{filename}"}), 400

        count = vs_manager.add_documents(chunks)
        logger.info(f"文件入库成功：{filename}，新增 {count} 个 Chunk")

        return jsonify({
            "success": True,
            "filename": filename,
            "chunks_added": count,
            "total_in_db": vs_manager.get_document_count(),
        })

    except Exception as e:
        logger.error(f"文件处理入库失败：{e}", exc_info=True)
        # 清理已保存的文件
        try:
            if save_path.exists():
                save_path.unlink()
        except Exception:
            pass
        return jsonify({"error": f"文件处理失败：{str(e)}"}), 500


@app.route("/api/history/<session_id>", methods=["GET"])
def get_history_api(session_id):
    """获取指定会话的历史记录（含溯源树和可信度）"""
    user_id = _get_auth_user_id()
    history = get_history(session_id, user_id)

    provenance_records = confidence_evaluator.get_provenance(session_id)
    provenance_map = {}
    for rec in provenance_records:
        key = rec.get("question", "") + "|||" + rec.get("answer", "")
        provenance_map[key] = {
            "provenance_tree": rec.get("provenance"),
            "confidence": rec.get("confidence"),
        }

    for idx, msg in enumerate(history):
        if msg.get("role") == "assistant":
            for i in range(idx - 1, -1, -1):
                if history[i].get("role") == "user":
                    key = history[i].get("content", "") + "|||" + msg.get("content", "")
                    if key in provenance_map:
                        msg["provenance_tree"] = provenance_map[key].get("provenance_tree")
                        msg["confidence"] = provenance_map[key].get("confidence")
                    break

    return jsonify({"session_id": session_id, "history": history, "count": len(history)})


@app.route("/api/history/<session_id>", methods=["DELETE"])
def clear_history_api(session_id):
    """清空指定会话的历史记录"""
    user_id = _get_auth_user_id()
    clear_history(session_id, user_id)
    return jsonify({"session_id": session_id, "message": "已清空"})


@app.route("/api/sessions", methods=["GET"])
def get_sessions():
    """获取当前用户的会话列表"""
    user_id = _get_auth_user_id()
    sessions = conversation_store.get_all_sessions(user_id)
    return jsonify({"sessions": sessions, "count": len(sessions)})


@app.route("/api/sessions/<session_id>/rename", methods=["POST"])
def rename_session(session_id):
    """重命名会话"""
    user_id = _get_auth_user_id()
    data = request.get_json() or {}
    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "会话名称不能为空"}), 400

    success = conversation_store.rename_session(session_id, new_name, user_id)
    if success:
        return jsonify({"success": True, "message": "重命名成功"})
    else:
        return jsonify({"error": "会话不存在"}), 404


@app.route("/api/config", methods=["GET"])
def get_config():
    """获取系统配置"""
    return jsonify({
        "hybrid_search": vs_manager.is_hybrid_search_enabled(),
        "reranker": vs_manager.is_reranker_enabled(),
        "multi_query": vs_manager.is_multi_query_enabled(),
    })


@app.route("/api/config", methods=["POST"])
def update_config():
    """更新系统配置"""
    data = request.get_json() or {}
    if "hybrid_search" in data:
        vs_manager.set_hybrid_search(bool(data["hybrid_search"]))
    if "reranker" in data:
        vs_manager.set_reranker(bool(data["reranker"]))
    if "multi_query" in data:
        vs_manager.set_multi_query(bool(data["multi_query"]))
    return jsonify({
        "hybrid_search": vs_manager.is_hybrid_search_enabled(),
        "reranker": vs_manager.is_reranker_enabled(),
        "multi_query": vs_manager.is_multi_query_enabled(),
        "message": "配置已更新",
    })


# ============================================================
# 语义缓存管理
# ============================================================

@app.route("/api/cache/clear", methods=["POST"])
def clear_cache():
    """清空语义缓存"""
    semantic_cache.invalidate()
    return jsonify({
        "success": True,
        "message": "语义缓存已清空",
        "stats": semantic_cache.get_stats(),
    })


@app.route("/api/cache/stats", methods=["GET"])
def cache_stats():
    """获取语义缓存统计"""
    return jsonify(semantic_cache.get_stats())


# ============================================================
# 用户认证模块
# ============================================================

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "conversations.db")

def get_db():
    """获取数据库连接"""
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA journal_mode=WAL")
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_auth_db():
    """初始化用户表"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            nickname TEXT DEFAULT '',
            status TEXT DEFAULT 'active',
            created_at TEXT NOT NULL,
            last_login_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def migrate_to_unified_db():
    """
    三库合一迁移：将 rag_system.db 和 data/eval_metrics.db 的数据合并到 conversations.db

    迁移策略：
      - 检查旧库文件是否存在，存在则复制数据到统一库
      - 迁移成功后，将旧库文件重命名为 .bak 备份，避免重复迁移
      - 迁移是幂等的，多次运行不会重复导入数据
    """
    import shutil

    base_dir = os.path.dirname(__file__)
    unified_path = DB_PATH  # conversations.db
    old_auth_path = os.path.join(base_dir, "rag_system.db")
    old_eval_path = os.path.join(base_dir, "data", "eval_metrics.db")

    # 迁移 rag_system.db 中的 users 表
    if os.path.exists(old_auth_path):
        try:
            logger.info("检测到旧数据库 rag_system.db，开始迁移 users 表...")
            old_conn = sqlite3.connect(old_auth_path)
            old_conn.row_factory = sqlite3.Row
            new_conn = sqlite3.connect(unified_path)

            # 确保目标表存在
            new_conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    nickname TEXT DEFAULT '',
                    status TEXT DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    last_login_at TEXT
                )
            """)

            rows = old_conn.execute("SELECT * FROM users").fetchall()
            migrated_count = 0
            for row in rows:
                try:
                    new_conn.execute(
                        """INSERT OR IGNORE INTO users
                           (id, username, password_hash, nickname, status, created_at, last_login_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (row["id"], row["username"], row["password_hash"],
                         row["nickname"], row["status"], row["created_at"], row["last_login_at"])
                    )
                    if new_conn.changes() > 0:
                        migrated_count += 1
                except sqlite3.IntegrityError:
                    pass  # 用户名已存在，跳过

            new_conn.commit()
            old_conn.close()
            new_conn.close()

            # 迁移成功，备份旧文件
            bak_path = old_auth_path + ".bak"
            shutil.move(old_auth_path, bak_path)
            logger.info(f"users 表迁移完成：{migrated_count} 条记录，旧文件已备份为 rag_system.db.bak")
        except Exception as e:
            logger.warning(f"迁移 rag_system.db 失败（不影响主流程）：{e}")

    # 迁移 data/eval_metrics.db 中的指标表
    if os.path.exists(old_eval_path):
        try:
            logger.info("检测到旧数据库 eval_metrics.db，开始迁移指标表...")
            old_conn = sqlite3.connect(old_eval_path)
            old_conn.row_factory = sqlite3.Row
            new_conn = sqlite3.connect(unified_path)

            # 确保目标表存在
            new_conn.execute("""
                CREATE TABLE IF NOT EXISTS eval_counters (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    total_requests INTEGER DEFAULT 0,
                    llm_requests INTEGER DEFAULT 0,
                    exact_cache_hits INTEGER DEFAULT 0,
                    semantic_cache_hits INTEGER DEFAULT 0,
                    cache_misses INTEGER DEFAULT 0,
                    total_input_tokens INTEGER DEFAULT 0,
                    total_output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_llm_latency_ms REAL DEFAULT 0.0,
                    total_retrieval_latency_ms REAL DEFAULT 0.0,
                    updated_at TEXT
                )
            """)
            new_conn.execute("""
                CREATE TABLE IF NOT EXISTS eval_recent_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    question TEXT DEFAULT '',
                    from_cache INTEGER DEFAULT 0,
                    cache_match_type TEXT,
                    latency_ms REAL DEFAULT 0,
                    retrieval_latency_ms REAL DEFAULT 0,
                    source_count INTEGER DEFAULT 0,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0
                )
            """)

            # 迁移计数器
            counter_row = old_conn.execute("SELECT * FROM eval_counters WHERE id = 1").fetchone()
            if counter_row:
                new_conn.execute(
                    """INSERT OR REPLACE INTO eval_counters
                       (id, total_requests, llm_requests, exact_cache_hits, semantic_cache_hits,
                        cache_misses, total_input_tokens, total_output_tokens, total_tokens,
                        total_llm_latency_ms, total_retrieval_latency_ms, updated_at)
                       VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (counter_row["total_requests"], counter_row["llm_requests"],
                     counter_row["exact_cache_hits"], counter_row["semantic_cache_hits"],
                     counter_row["cache_misses"], counter_row["total_input_tokens"],
                     counter_row["total_output_tokens"], counter_row["total_tokens"],
                     counter_row["total_llm_latency_ms"], counter_row["total_retrieval_latency_ms"],
                     counter_row["updated_at"])
                )

            # 迁移最近请求记录
            req_rows = old_conn.execute("SELECT * FROM eval_recent_requests ORDER BY id").fetchall()
            req_count = 0
            for row in req_rows:
                new_conn.execute(
                    """INSERT INTO eval_recent_requests
                       (timestamp, question, from_cache, cache_match_type, latency_ms,
                        retrieval_latency_ms, source_count, input_tokens, output_tokens, total_tokens)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (row["timestamp"], row["question"], row["from_cache"], row["cache_match_type"],
                     row["latency_ms"], row["retrieval_latency_ms"], row["source_count"],
                     row["input_tokens"], row["output_tokens"], row["total_tokens"])
                )
                req_count += 1

            new_conn.commit()
            old_conn.close()
            new_conn.close()

            # 迁移成功，备份旧文件
            bak_path = old_eval_path + ".bak"
            shutil.move(old_eval_path, bak_path)
            logger.info(f"指标表迁移完成：计数器 1 条，请求记录 {req_count} 条，旧文件已备份为 eval_metrics.db.bak")
        except Exception as e:
            logger.warning(f"迁移 eval_metrics.db 失败（不影响主流程）：{e}")


token_store = {}
user_token_map = {}

def auth_validate_token(token):
    return token_store.get(token)

def auth_get_user(user_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if row:
        return dict(row)
    return None

@app.before_request
def auth_before_request():
    """请求前验证 Token（排除认证相关路径和静态资源）"""
    path = request.path
    if (path.startswith("/api/auth/") or
        not path.startswith("/api/") or
        path.startswith("/static")):
        return
    token = request.headers.get("X-Auth-Token")
    user_id = auth_validate_token(token)
    if not user_id:
        return jsonify({"error": "未登录或Token已过期", "code": 401}), 401
    g.current_user_id = user_id


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    password = data.get("password", "")
    nickname = (data.get("nickname") or "").strip()

    if not username:
        return jsonify({"success": False, "error": "用户名不能为空"}), 400
    if len(username) < 3 or len(username) > 20:
        return jsonify({"success": False, "error": "用户名需要 3-20 个字符"}), 400
    if not password or len(password) < 6:
        return jsonify({"success": False, "error": "密码长度不能少于6位"}), 400

    conn = get_db()
    existing = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
    if existing:
        return jsonify({"success": False, "error": "用户名已被注册"}), 400

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pwd_hash = generate_password_hash(password)
    nick = nickname if nickname else username
    cursor = conn.execute(
        "INSERT INTO users (username, password_hash, nickname, created_at) VALUES (?, ?, ?, ?)",
        (username, pwd_hash, nick, now)
    )
    conn.commit()
    logger.info(f"用户注册成功：username={username}，id={cursor.lastrowid}")
    return jsonify({"success": True, "message": "注册成功，请登录"})


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    password = data.get("password", "")

    if not username:
        return jsonify({"success": False, "error": "用户名不能为空"}), 400
    if not password:
        return jsonify({"success": False, "error": "密码不能为空"}), 400

    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if not row:
        return jsonify({"success": False, "error": "用户名或密码错误"}), 401

    user = dict(row)
    if user["status"] != "active":
        return jsonify({"success": False, "error": "账号已被禁用"}), 401
    if not check_password_hash(user["password_hash"], password):
        return jsonify({"success": False, "error": "用户名或密码错误"}), 401

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("UPDATE users SET last_login_at=? WHERE id=?", (now, user["id"]))
    conn.commit()

    old_token = user_token_map.get(user["id"])
    if old_token and old_token in token_store:
        del token_store[old_token]

    token = uuid.uuid4().hex + str(int(time.time() * 1000))
    token_store[token] = user["id"]
    user_token_map[user["id"]] = token

    logger.info(f"用户登录成功：username={username}，id={user['id']}")
    return jsonify({
        "success": True, "message": "登录成功", "token": token,
        "user_id": user["id"], "username": user["username"],
        "nickname": user["nickname"],
        "created_at": user.get("created_at", ""),
        "last_login_at": now
    })


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    token = request.headers.get("X-Auth-Token", "")
    if token and token in token_store:
        user_id = token_store.pop(token)
        user_token_map.pop(user_id, None)
        logger.info(f"用户已注销：userId={user_id}")
    return jsonify({"success": True, "message": "已退出登录"})


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    token = request.headers.get("X-Auth-Token", "")
    user_id = auth_validate_token(token)
    if not user_id:
        return jsonify({"error": "未登录或Token已过期"}), 401
    user = auth_get_user(user_id)
    if not user:
        return jsonify({"error": "用户不存在"}), 401
    return jsonify({
        "user_id": user["id"], "username": user["username"],
        "nickname": user["nickname"], "created_at": user.get("created_at", ""),
        "last_login_at": user.get("last_login_at", "")
    })


@app.route("/api/auth/update-profile", methods=["PUT"])
def auth_update_profile():
    token = request.headers.get("X-Auth-Token", "")
    user_id = auth_validate_token(token)
    if not user_id:
        return jsonify({"error": "未登录或Token已过期"}), 401
    data = request.get_json() or {}
    nickname = (data.get("nickname") or "").strip()
    if not nickname:
        return jsonify({"success": False, "error": "昵称不能为空"}), 400
    conn = get_db()
    conn.execute("UPDATE users SET nickname=? WHERE id=?", (nickname, user_id))
    conn.commit()
    logger.info(f"用户修改昵称：userId={user_id}，新昵称={nickname}")
    return jsonify({"success": True, "message": "昵称修改成功", "nickname": nickname})


@app.route("/api/auth/change-password", methods=["PUT"])
def auth_change_password():
    token = request.headers.get("X-Auth-Token", "")
    user_id = auth_validate_token(token)
    if not user_id:
        return jsonify({"error": "未登录或Token已过期"}), 401
    data = request.get_json() or {}
    old_pwd = data.get("oldPassword", "")
    new_pwd = data.get("newPassword", "")
    if not old_pwd:
        return jsonify({"success": False, "error": "请输入当前密码"}), 400
    if not new_pwd or len(new_pwd) < 6:
        return jsonify({"success": False, "error": "新密码长度不能少于6位"}), 400
    conn = get_db()
    row = conn.execute("SELECT password_hash FROM users WHERE id=?", (user_id,)).fetchone()
    if not row or not check_password_hash(row["password_hash"], old_pwd):
        return jsonify({"success": False, "error": "当前密码不正确"}), 400
    new_hash = generate_password_hash(new_pwd)
    conn.execute("UPDATE users SET password_hash=? WHERE id=?", (new_hash, user_id))
    conn.commit()
    logger.info(f"用户修改密码：userId={user_id}")
    return jsonify({"success": True, "message": "密码修改成功"})


# ============================================================
# 方向1：千人千面 -- 个性化适配引擎
# ============================================================

@app.route("/api/profile/<username>", methods=["GET"])
def get_user_profile(username):
    """获取用户画像"""
    profile = profile_manager.get_profile(username)
    question_history = profile_manager.get_question_history(username, limit=20)
    profile["recent_questions"] = question_history
    return jsonify(profile)


@app.route("/api/profile/<username>", methods=["PUT"])
def update_user_profile(username):
    """更新用户画像"""
    data = request.get_json() or {}
    profile = profile_manager.update_profile(username, data)
    return jsonify({"success": True, "profile": profile})


@app.route("/api/profile/<username>/style", methods=["PUT"])
def update_answer_style(username):
    """更新回答风格偏好"""
    data = request.get_json() or {}
    style = data.get("style", "detailed")
    profile = profile_manager.update_style_preference(username, style)
    return jsonify({"success": True, "style": profile.get("style_preference")})


@app.route("/api/bookmarks/<username>", methods=["GET"])
def get_bookmarks(username):
    """获取用户收藏列表"""
    bookmarks = profile_manager.get_bookmarks(username)
    return jsonify({"bookmarks": bookmarks, "count": len(bookmarks)})


@app.route("/api/bookmarks/<username>", methods=["POST"])
def add_bookmark(username):
    """添加/取消收藏（toggle模式：已收藏则取消，未收藏则添加）"""
    data = request.get_json() or {}
    question = data.get("question", "")
    answer = data.get("answer", "")
    sources = data.get("sources")
    note = data.get("note", "")

    if not question or not answer:
        return jsonify({"error": "问题和答案不能为空"}), 400

    result = profile_manager.toggle_bookmark(username, question, answer, sources, note)
    return jsonify({"success": True, **result})


@app.route("/api/bookmarks/<username>/batch-check", methods=["POST"])
def batch_check_bookmarks(username):
    """批量检查多个问答是否已收藏"""
    data = request.get_json() or {}
    qa_pairs = data.get("qa_pairs", [])

    if not qa_pairs:
        return jsonify({"bookmark_map": {}})

    bookmark_map = profile_manager.get_bookmarks_batch_check(username, qa_pairs)
    return jsonify({"bookmark_map": bookmark_map})


@app.route("/api/bookmarks/<username>/<int:bookmark_id>/note", methods=["PUT"])
def update_bookmark_note(username, bookmark_id):
    """更新收藏笔记"""
    data = request.get_json() or {}
    note = data.get("note", "")
    profile_manager.update_bookmark_note(bookmark_id, username, note)
    return jsonify({"success": True})


@app.route("/api/bookmarks/<username>/<int:bookmark_id>", methods=["DELETE"])
def delete_bookmark(username, bookmark_id):
    """删除收藏"""
    profile_manager.delete_bookmark(bookmark_id, username)
    return jsonify({"success": True})


# ============================================================
# 方向2：知其所以然 -- 思维链与溯源可视化
# ============================================================

@app.route("/api/provenance/<session_id>", methods=["GET"])
def get_provenance(session_id):
    """获取会话的答案溯源记录"""
    records = confidence_evaluator.get_provenance(session_id)
    return jsonify({"session_id": session_id, "records": records, "count": len(records)})


@app.route("/api/confidence", methods=["POST"])
def evaluate_confidence():
    """评估答案可信度"""
    data = request.get_json() or {}
    sources = data.get("sources", [])
    answer = data.get("answer", "")
    question = data.get("question", "")
    retrieval_details = data.get("retrieval_details")

    result = confidence_evaluator.evaluate(sources, answer, question, retrieval_details)
    return jsonify(result)


# ============================================================
# 方向3：不只是问答 -- 知识探索与创作工具
# ============================================================

@app.route("/api/knowledge-graph", methods=["GET"])
def get_knowledge_graph():
    """获取知识图谱数据"""
    graph_data = knowledge_graph.get_graph_data()
    return jsonify(graph_data)


@app.route("/api/knowledge-graph/build", methods=["POST"])
def build_knowledge_graph():
    """从知识库文档构建知识图谱"""
    try:
        all_docs = vs_manager.get_all_documents()
        if not all_docs:
            return jsonify({"error": "知识库为空，请先入库文档"}), 400

        documents = []
        for doc in all_docs:
            documents.append({
                "source": doc.metadata.get("source", "未知"),
                "full_content": doc.page_content,
            })

        graph_data = knowledge_graph.build_from_documents(documents)
        return jsonify({"success": True, "graph": graph_data})
    except Exception as e:
        logger.error(f"构建知识图谱失败：{e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def compare_qa():
    """对比式问答"""
    data = request.get_json() or {}
    topic_a = data.get("topic_a", "")
    topic_b = data.get("topic_b", "")

    if not topic_a or not topic_b:
        return jsonify({"error": "请提供两个对比主题"}), 400

    try:
        docs_a = vs_manager.similarity_search_with_scores(topic_a)
        docs_b = vs_manager.similarity_search_with_scores(topic_b)

        from core.rag_chain import format_docs_with_scores
        context_a = format_docs_with_scores(docs_a)
        context_b = format_docs_with_scores(docs_b)

        prompt = ComparativeQA.build_comparison_prompt(topic_a, topic_b, context_a, context_b)

        response = rag.raw_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.1,
        )
        answer = response.choices[0].message.content

        return jsonify({"answer": answer, "topic_a": topic_a, "topic_b": topic_b})
    except Exception as e:
        logger.error(f"对比问答失败：{e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulate", methods=["POST"])
def simulate_qa():
    """场景模拟问答"""
    data = request.get_json() or {}
    scenario = data.get("scenario", "")

    if not scenario:
        return jsonify({"error": "请提供假设场景"}), 400

    try:
        docs = vs_manager.similarity_search_with_scores(scenario)
        from core.rag_chain import format_docs_with_scores
        context = format_docs_with_scores(docs)

        prompt = ScenarioSimulator.build_simulation_prompt(scenario, context)

        response = rag.raw_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.1,
        )
        answer = response.choices[0].message.content

        return jsonify({"answer": answer, "scenario": scenario})
    except Exception as e:
        logger.error(f"场景模拟问答失败：{e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/summarize", methods=["POST"])
def summarize_document():
    """智能文档摘要生成"""
    data = request.get_json() or {}
    content = data.get("content", "")
    level = data.get("level", "structured")

    if not content:
        return jsonify({"error": "请提供文档内容"}), 400

    valid_levels = ["one_line", "paragraph", "structured", "bullets", "actions"]
    if level not in valid_levels:
        level = "structured"

    try:
        prompt = DocumentSummarizer.build_summary_prompt(content, level)

        response = rag.raw_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1,
        )
        summary = response.choices[0].message.content

        return jsonify({"summary": summary, "level": level})
    except Exception as e:
        logger.error(f"文档摘要生成失败：{e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ============================================================
# 方向4：群体智慧 -- 协作式知识生态
# ============================================================

@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    """提交/更新答案反馈（toggle模式：再次点击相同评分则取消）"""
    data = request.get_json() or {}
    session_id = data.get("session_id", "")
    question = data.get("question", "")
    answer = data.get("answer", "")
    username = data.get("username", "anonymous")
    rating = data.get("rating", "")
    comment = data.get("comment", "")

    if not question or not answer:
        return jsonify({"error": "问题和答案不能为空"}), 400

    result = answer_feedback.upsert_feedback(
        session_id, question, answer, username, rating, comment
    )

    # 低评分自动触发专家路由
    if rating == "negative" and result.get("action") == "created" and confidence_evaluator:
        expert_router.route_question(question, 30.0)

    return jsonify({"success": True, **result})


@app.route("/api/feedback/batch-check", methods=["POST"])
def batch_check_feedback():
    """批量查询用户对多个问答的反馈状态"""
    data = request.get_json() or {}
    qa_pairs = data.get("qa_pairs", [])
    username = data.get("username", "anonymous")

    if not qa_pairs:
        return jsonify({"feedback_map": {}})

    feedback_map = answer_feedback.get_user_feedback_batch(qa_pairs, username)
    return jsonify({"feedback_map": feedback_map})


@app.route("/api/feedback", methods=["GET"])
def get_feedback():
    """获取反馈列表"""
    session_id = request.args.get("session_id")
    limit = int(request.args.get("limit", 50))
    feedback_list = answer_feedback.get_feedback(session_id, limit)
    return jsonify({"feedback": feedback_list, "count": len(feedback_list)})


@app.route("/api/feedback/stats", methods=["GET"])
def get_feedback_stats():
    """获取反馈统计"""
    stats = answer_feedback.get_feedback_stats()
    return jsonify(stats)


@app.route("/api/feedback/<int:feedback_id>", methods=["DELETE"])
def delete_feedback(feedback_id):
    """删除反馈"""
    success = answer_feedback.delete_feedback(feedback_id)
    if success:
        return jsonify({"success": True})
    return jsonify({"error": "反馈不存在"}), 404


@app.route("/api/feedback/<int:feedback_id>/comment", methods=["POST"])
def add_feedback_comment(feedback_id):
    """添加反馈评论"""
    data = request.get_json() or {}
    username = data.get("username", "anonymous")
    content = data.get("content", "")
    parent_id = data.get("parent_id")

    if not content:
        return jsonify({"error": "评论内容不能为空"}), 400

    comment_id = answer_feedback.add_comment(feedback_id, username, content, parent_id)
    return jsonify({"success": True, "comment_id": comment_id})


@app.route("/api/expert-routing/pending", methods=["GET"])
def get_pending_expert_questions():
    """获取待专家处理的问题"""
    questions = expert_router.get_pending_questions()
    return jsonify({"questions": questions, "count": len(questions)})


@app.route("/api/expert-routing/<int:routing_id>/resolve", methods=["POST"])
def resolve_expert_question(routing_id):
    """专家回答问题"""
    data = request.get_json() or {}
    expert_answer = data.get("answer", "")
    expert_name = data.get("expert_name", "")

    if not expert_answer:
        return jsonify({"error": "请提供专家回答"}), 400

    expert_router.resolve_question(routing_id, expert_answer, expert_name)
    return jsonify({"success": True})


@app.route("/api/expert-routing/stats", methods=["GET"])
def get_expert_routing_stats():
    """获取专家路由统计"""
    stats = expert_router.get_routing_stats()
    return jsonify(stats)


# ============================================================
# 方向5：AI 原生体验 -- 重新定义交互
# ============================================================

@app.route("/api/export/answer", methods=["POST"])
def export_answer():
    """
    将问答内容导出为文档

    Request Body:
        {
          "question": "用户问题",
          "answer": "AI 回答",
          "format": "markdown" | "text"
        }
    """
    data = request.get_json() or {}
    question = data.get("question", "")
    answer = data.get("answer", "")
    export_format = data.get("format", "markdown")

    if not answer:
        return jsonify({"error": "答案内容不能为空"}), 400

    if export_format == "markdown":
        content = f"""# 问答记录

## 问题
{question}

## 回答
{answer}

---
*由企业智能问答助手生成*
"""
    else:
        content = f"问题：{question}\n\n回答：{answer}\n\n---\n由企业智能问答助手生成"

    return jsonify({
        "success": True,
        "content": content,
        "format": export_format,
        "filename": f"qa_export.{'md' if export_format == 'markdown' else 'txt'}",
    })


@app.route("/api/batch-qa", methods=["POST"])
def batch_qa():
    """
    批量问答

    Request Body:
        {
          "questions": ["问题1", "问题2", ...]
        }
    """
    data = request.get_json() or {}
    questions = data.get("questions", [])

    if not questions:
        return jsonify({"error": "请提供问题列表"}), 400

    if len(questions) > 50:
        return jsonify({"error": "单次批量最多支持 50 个问题"}), 400

    results = []
    for i, question in enumerate(questions):
        try:
            docs_with_scores = vs_manager.similarity_search_with_scores(question)
            from core.rag_chain import format_docs_with_scores
            context = format_docs_with_scores(docs_with_scores)

            from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template(RAG_PROMPT_TEMPLATE),
            ])
            messages = prompt.format_messages(context=context, question=question)
            from core.rag_chain import _langchain_messages_to_openai
            openai_messages = _langchain_messages_to_openai(messages)

            response = rag.raw_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=openai_messages,
                max_tokens=1024,
                temperature=0.1,
            )
            answer = response.choices[0].message.content

            sources = []
            for item in docs_with_scores:
                doc = item[0]
                score = item[1]
                rtype = item[2] if len(item) > 2 else "vec"
                source = doc.metadata.get("source", "未知")
                similarity = _calc_similarity(score, rtype)
                sources.append({
                    "source": source.split("\\")[-1].split("/")[-1],
                    "similarity": f"{similarity:.1f}%",
                })

            results.append({
                "index": i,
                "question": question,
                "answer": answer,
                "sources": sources,
                "status": "success",
            })
        except Exception as e:
            logger.error(f"批量问答第 {i} 个问题失败：{e}")
            results.append({
                "index": i,
                "question": question,
                "answer": "",
                "sources": [],
                "status": "error",
                "error": str(e),
            })

    return jsonify({
        "results": results,
        "total": len(questions),
        "success_count": sum(1 for r in results if r["status"] == "success"),
        "error_count": sum(1 for r in results if r["status"] == "error"),
    })


# ============================================================
# 启动服务
# ============================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  RAG 智能问答系统启动中...")
    logger.info("=" * 60)

    init_auth_db()
    migrate_to_unified_db()
    doc_count = vs_manager.get_document_count()
    if doc_count == 0:
        logger.warning("⚠ 知识库为空！请先运行：python ingest.py")
    else:
        logger.info(f"✓ 知识库就绪，共 {doc_count} 个向量")
        # 优先从 pickle 缓存加载 BM25 索引（毫秒级），失败则从向量库重建
        if vs_manager._load_bm25_index():
            logger.info(f"✓ BM25 索引已从缓存加载，共 {len(vs_manager._bm25_corpus)} 篇")
        else:
            all_docs = vs_manager.get_all_documents()
            if all_docs:
                vs_manager._build_bm25_index(all_docs)
                logger.info(f"✓ BM25 索引已从向量库重建，共 {len(all_docs)} 篇")

    logger.info(f"✓ 服务地址：http://127.0.0.1:{FLASK_PORT}")
    logger.info(f"✓ 混合检索：{'已启用' if vs_manager.is_hybrid_search_enabled() else '已关闭'}")
    logger.info(f"✓ Reranker：{'已启用' if vs_manager.is_reranker_enabled() else '已关闭'}")
    logger.info(f"✓ 语义缓存：{'已启用' if CACHE_ENABLED else '已关闭'}（阈值={CACHE_SIMILARITY_THRESHOLD}，最大条目={CACHE_MAX_ENTRIES}，TTL={CACHE_TTL_HOURS}h）")
    logger.info("=" * 60)

    app.run(
        host="0.0.0.0",
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        threaded=True,
    )
