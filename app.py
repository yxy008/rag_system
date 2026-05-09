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
"""
import json
import logging
import uuid
import os
from collections import defaultdict
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS

from config import FLASK_PORT, FLASK_DEBUG, DOCUMENTS_DIR
from document_processor import DocumentProcessor, SUPPORTED_EXTENSIONS
from vector_store import VectorStoreManager
from rag_chain import RAGChain, _calc_similarity

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

# ========== 对话历史存储（内存，session_id -> List[Dict]） ==========
# 生产环境建议换 Redis 或 SQLite
MAX_HISTORY_PER_SESSION = 20  # 每个 session 最多保留对话轮数
sessions: dict = defaultdict(list)


def get_history(session_id: str) -> list:
    """获取指定 session 的对话历史"""
    return sessions.get(session_id, [])


def add_to_history(session_id: str, role: str, content: str):
    """向指定 session 添加对话记录"""
    sessions[session_id].append({"role": role, "content": content})
    # 限制历史长度，防止内存溢出
    if len(sessions[session_id]) > MAX_HISTORY_PER_SESSION:
        sessions[session_id] = sessions[session_id][-MAX_HISTORY_PER_SESSION:]


def clear_history(session_id: str):
    """清空指定 session 的对话历史"""
    if session_id in sessions:
        del sessions[session_id]


# ============================================================
# 路由定义
# ============================================================

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
    if not data or "question" not in data:
        return jsonify({"error": "请提供 question 字段"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    session_id = data.get("session_id", str(uuid.uuid4()))
    hybrid = data.get("hybrid")
    reranker = data.get("reranker")

    try:
        # 检查知识库
        doc_count = vs_manager.get_document_count()
        if doc_count == 0:
            return jsonify({
                "error": "知识库为空，请先运行 python ingest.py 将文档入库"
            }), 400

        # 获取历史
        history = get_history(session_id)
        if history:
            logger.info(f"会话 {session_id} 有 {len(history)} 条历史记录")

        # 临时切换混合检索/Reranker 状态
        orig_hybrid = vs_manager.is_hybrid_search_enabled()
        orig_reranker = vs_manager.is_reranker_enabled()
        if hybrid is not None:
            vs_manager.set_hybrid_search(bool(hybrid))
        if reranker is not None:
            vs_manager.set_reranker(bool(reranker))

        result = rag.query(question, history=history)

        # 保存对话历史
        add_to_history(session_id, "user", question)
        add_to_history(session_id, "assistant", result["answer"])

        # 恢复原始状态
        vs_manager.set_hybrid_search(orig_hybrid)
        vs_manager.set_reranker(orig_reranker)

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
                "retrieval_type": type_label,
                "page": doc.metadata.get("page"),
                "total_pages": doc.metadata.get("total_pages"),
            })

        return jsonify({
            "answer": result["answer"],
            "session_id": session_id,
            "sources": sources,
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
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "请提供 question 字段"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    session_id = data.get("session_id", str(uuid.uuid4()))
    hybrid = data.get("hybrid")
    reranker = data.get("reranker")

    def generate():
        try:
            doc_count = vs_manager.get_document_count()
            if doc_count == 0:
                yield f"data: {json.dumps({'type': 'error', 'message': '知识库为空，请先入库文档'}, ensure_ascii=False)}\n\n"
                return

            # 临时切换混合检索/Reranker 状态
            orig_hybrid = vs_manager.is_hybrid_search_enabled()
            orig_reranker = vs_manager.is_reranker_enabled()
            if hybrid is not None:
                vs_manager.set_hybrid_search(bool(hybrid))
            if reranker is not None:
                vs_manager.set_reranker(bool(reranker))

            # 获取历史
            history = get_history(session_id)

            # 先检索（获取 sources），再流式生成
            docs_with_scores = vs_manager.similarity_search_with_scores(question)
            from rag_chain import format_docs_with_scores
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
            from config import SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
                HumanMessagePromptTemplate.from_template(RAG_PROMPT_TEMPLATE),
            ])
            messages = prompt.format_messages(context=context, question=question)

            # 流式生成答案
            answer_parts = []
            for token in rag.llm.stream(messages):
                if token.content:
                    answer_parts.append(token.content)
                    yield f"data: {json.dumps({'type': 'token', 'content': token.content}, ensure_ascii=False)}\n\n"

            answer = "".join(answer_parts)

            # 保存历史
            add_to_history(session_id, "user", question)
            add_to_history(session_id, "assistant", answer)

            # 恢复原始状态
            vs_manager.set_hybrid_search(orig_hybrid)
            vs_manager.set_reranker(orig_reranker)

            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'sources': sources}, ensure_ascii=False)}\n\n"

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
          "active_sessions": 3,
        }
    """
    from config import EMBEDDING_MODEL_NAME, OPENAI_MODEL
    doc_count = vs_manager.get_document_count()
    return jsonify({
        "status": "ok",
        "doc_count": doc_count,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": OPENAI_MODEL,
        "knowledge_base_ready": doc_count > 0,
        "hybrid_search": vs_manager.is_hybrid_search_enabled(),
        "active_sessions": len(sessions),
    })


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
        from config import DOCUMENTS_DIR
        processor = DocumentProcessor()

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
        processor = DocumentProcessor()
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
    """获取指定会话的历史记录"""
    history = get_history(session_id)
    return jsonify({"session_id": session_id, "history": history, "count": len(history)})


@app.route("/api/history/<session_id>", methods=["DELETE"])
def clear_history_api(session_id):
    """清空指定会话的历史记录"""
    clear_history(session_id)
    return jsonify({"session_id": session_id, "message": "已清空"})


@app.route("/api/config", methods=["GET"])
def get_config():
    """获取系统配置"""
    return jsonify({
        "hybrid_search": vs_manager.is_hybrid_search_enabled(),
        "reranker": vs_manager.is_reranker_enabled(),
    })


@app.route("/api/config", methods=["POST"])
def update_config():
    """更新系统配置"""
    data = request.get_json() or {}
    if "hybrid_search" in data:
        vs_manager.set_hybrid_search(bool(data["hybrid_search"]))
    if "reranker" in data:
        vs_manager.set_reranker(bool(data["reranker"]))
    return jsonify({
        "hybrid_search": vs_manager.is_hybrid_search_enabled(),
        "reranker": vs_manager.is_reranker_enabled(),
        "message": "配置已更新",
    })


# ============================================================
# 启动服务
# ============================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  RAG 智能问答系统启动中...")
    logger.info("=" * 60)

    doc_count = vs_manager.get_document_count()
    if doc_count == 0:
        logger.warning("⚠ 知识库为空！请先运行：python ingest.py")
    else:
        logger.info(f"✓ 知识库就绪，共 {doc_count} 个向量")
        # 预热时重建 BM25 索引
        all_docs = vs_manager.get_all_documents()
        if all_docs:
            vs_manager._build_bm25_index(all_docs)
            logger.info(f"✓ BM25 索引已构建，共 {len(all_docs)} 篇")

    logger.info(f"✓ 服务地址：http://127.0.0.1:{FLASK_PORT}")
    logger.info(f"✓ 混合检索：{'已启用' if vs_manager.is_hybrid_search_enabled() else '已关闭'}")
    logger.info(f"✓ Reranker：{'已启用' if vs_manager.is_reranker_enabled() else '已关闭'}")
    logger.info("=" * 60)

    app.run(
        host="0.0.0.0",
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        threaded=True,
    )
