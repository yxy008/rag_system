"""
routes/openapi.py - OpenAPI 路由蓝图

提供外部系统接入的 OpenAPI 接口，包括：
  POST /api/open/chat        - 开放式问答接口（API Key 认证）
  GET  /api/open/keys        - 列出所有 API Key
  POST /api/open/keys        - 创建新 API Key
  PUT  /api/open/keys/<id>   - 更新 API Key
  DELETE /api/open/keys/<id> - 删除 API Key
  POST /api/open/keys/<id>/revoke   - 吊销 API Key
  POST /api/open/keys/<id>/activate - 重新激活 API Key

与现有 /api/chat 接口的区别：
  - 使用 X-API-Key 请求头认证（而非 X-Auth-Token）
  - 不受全局 Token 认证拦截
  - 返回格式适配测评平台（output + trace_id）
  - 不需要流式输出
"""

import json
import logging
import time
import uuid

from flask import Blueprint, request, jsonify, g

from core.api_key_manager import api_key_manager

logger = logging.getLogger(__name__)

# 创建蓝图
openapi_bp = Blueprint("openapi", __name__)


# ============================================================
# API Key 认证中间件（仅对本蓝图生效）
# ============================================================

@openapi_bp.before_request
def verify_api_key():
    """验证 X-API-Key 请求头"""
    # /api/open/keys 管理接口走 X-Auth-Token 认证（由全局 before_request 处理）
    # 只有 /api/open/chat 走 API Key 认证
    if not request.path.startswith("/api/open/chat"):
        return None

    api_key = request.headers.get("X-API-Key", "").strip()
    if not api_key:
        return jsonify({"error": "缺少 X-API-Key 请求头", "code": 401}), 401

    key_info = api_key_manager.validate_key(api_key)
    if not key_info:
        return jsonify({"error": "无效的 API Key", "code": 401}), 401

    # 将 Key 信息存入请求上下文
    g.api_key_info = key_info


# ============================================================
# 开放式问答接口
# ============================================================

@openapi_bp.route("/api/open/chat", methods=["POST"])
def open_chat():
    """
    开放式问答接口（供外部系统/测评平台调用）

    Headers:
        X-API-Key: rag-xxxx...

    Request Body:
        {
            "input": "用户问题",
            "context": "可选上下文"
        }

    Response:
        {
            "output": "完整回复",
            "trace_id": "追踪ID",
            "sources": [...],
            "from_cache": false,
            "confidence": {...}
        }
    """
    # 延迟导入，避免循环依赖，同时复用 app.py 中的全局组件
    from core.rag_chain import _calc_similarity

    data = request.get_json()
    start_time = time.time()
    trace_id = str(uuid.uuid4())

    if not data or "input" not in data:
        return jsonify({
            "error": "请提供 input 字段",
            "trace_id": trace_id,
        }), 400

    question = data["input"].strip()
    if not question:
        return jsonify({
            "error": "input 不能为空",
            "trace_id": trace_id,
        }), 400

    # 可选上下文（预留，暂不影响检索逻辑）
    extra_context = data.get("context", "")

    # 从 app 上下文获取核心组件（由 app.py 初始化）
    app = request.environ.get("werkzeug.request").environ.get("flask.app", None)
    if app is None:
        # 通过 current_app 获取
        from flask import current_app
        app_ref = current_app._get_current_object()
    else:
        app_ref = app

    # 获取 app.py 中初始化的全局组件
    rag = app_ref.config.get("RAG_CHAIN")
    vs_manager = app_ref.config.get("VS_MANAGER")
    semantic_cache = app_ref.config.get("SEMANTIC_CACHE")
    evaluation_service = app_ref.config.get("EVALUATION_SERVICE")
    confidence_evaluator = app_ref.config.get("CONFIDENCE_EVALUATOR")

    if not rag or not vs_manager:
        return jsonify({
            "error": "服务未就绪，核心组件未初始化",
            "trace_id": trace_id,
        }), 503

    try:
        # 检查知识库
        doc_count = vs_manager.get_document_count()
        if doc_count == 0:
            return jsonify({
                "error": "知识库为空，请先入库文档",
                "trace_id": trace_id,
            }), 400

        # 生成会话 ID（每次请求独立，不保留对话历史）
        session_id = str(uuid.uuid4())

        # ===== 语义缓存检查 =====
        cached = None
        if semantic_cache:
            cached = semantic_cache.lookup(question)
            if cached:
                logger.info("[OpenAPI] 缓存命中：%s", cached.get("match_type"))
                semantic_cache.record_hit(cached.get("match_type", "semantic"))
                sources = json.loads(cached["sources_json"])

                if cached.get("match_type") == "semantic":
                    semantic_cache.store(question, cached["answer"], sources)

                latency = (time.time() - start_time) * 1000
                if evaluation_service:
                    evaluation_service.record_request(
                        True, cached.get("match_type", "semantic"), latency, 0, len(sources),
                        question=question,
                    )

                # 可信度评估
                confidence = None
                if confidence_evaluator:
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
                    except Exception as e:
                        logger.warning("可信度评估失败：%s", e)

                return jsonify({
                    "output": cached["answer"],
                    "trace_id": trace_id,
                    "sources": sources,
                    "from_cache": True,
                    "confidence": confidence,
                })
            else:
                semantic_cache.record_miss()

        # ===== 执行检索 + LLM 生成 =====
        # rag.query() 内部已集成轻量动态检索（自动调节 alpha + 多查询判断）
        retrieval_start = time.time()
        result = rag.query(question, history=[])
        retrieval_latency = (time.time() - retrieval_start) * 1000

        # 整理来源信息（与 /api/chat 逻辑一致）
        sources = []
        for item in result.get("source_scores", []):
            doc = item[0]
            score = item[1]
            rtype = item[2] if len(item) > 2 else "vec"
            source = doc.metadata.get("source", "未知")

            similarity = _calc_similarity(score, rtype)

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

        sources.sort(key=lambda x: x["similarity_value"], reverse=True)

        # 存入语义缓存
        if semantic_cache:
            semantic_cache.store(question, result["answer"], sources)

        total_latency = (time.time() - start_time) * 1000
        if evaluation_service:
            evaluation_service.record_request(
                False, None, total_latency, retrieval_latency, len(sources),
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                question=question,
            )

        # 可信度评估
        confidence = None
        if confidence_evaluator:
            try:
                retrieval_details = result.get("retrieval_details", {})
                confidence = confidence_evaluator.evaluate(
                    sources=sources,
                    answer=result["answer"],
                    question=question,
                    retrieval_details=retrieval_details,
                )
            except Exception as e:
                logger.warning("可信度评估失败：%s", e)

        return jsonify({
            "output": result["answer"],
            "trace_id": trace_id,
            "sources": sources,
            "from_cache": False,
            "confidence": confidence,
        })

    except Exception as e:
        logger.error("[OpenAPI] 问答失败：%s", e, exc_info=True)
        return jsonify({
            "error": f"服务内部错误：{str(e)}",
            "trace_id": trace_id,
        }), 500


# ============================================================
# API Key 管理接口
# ============================================================

@openapi_bp.route("/api/open/keys", methods=["GET"])
def list_api_keys():
    """列出所有 API Key"""
    include_inactive = request.args.get("include_inactive", "0") == "1"
    keys = api_key_manager.list_keys(include_inactive=include_inactive)
    return jsonify({"keys": keys, "total": len(keys)})


@openapi_bp.route("/api/open/keys", methods=["POST"])
def create_api_key():
    """
    创建新的 API Key

    Request Body:
        {
            "name": "Key名称",
            "description": "Key描述（可选）",
            "rate_limit": 60
        }
    """
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "请提供 name 字段"}), 400

    description = data.get("description", "")
    rate_limit = data.get("rate_limit", 60)
    created_by = getattr(g, "current_user_id", "admin")

    result = api_key_manager.create_key(
        name=name,
        description=description,
        rate_limit=rate_limit,
        created_by=created_by,
    )

    return jsonify(result), 201


@openapi_bp.route("/api/open/keys/<int:key_id>", methods=["GET"])
def get_api_key(key_id):
    """获取单个 API Key 信息"""
    key_info = api_key_manager.get_key(key_id)
    if not key_info:
        return jsonify({"error": "API Key 不存在"}), 404
    return jsonify(key_info)


@openapi_bp.route("/api/open/keys/<int:key_id>", methods=["PUT"])
def update_api_key(key_id):
    """
    更新 API Key 信息

    Request Body:
        {
            "name": "新名称（可选）",
            "description": "新描述（可选）",
            "rate_limit": 60
        }
    """
    data = request.get_json() or {}
    success = api_key_manager.update_key(
        key_id=key_id,
        name=data.get("name"),
        description=data.get("description"),
        rate_limit=data.get("rate_limit"),
    )
    if not success:
        return jsonify({"error": "更新失败，API Key 可能不存在"}), 404
    return jsonify({"success": True, "message": "更新成功"})


@openapi_bp.route("/api/open/keys/<int:key_id>", methods=["DELETE"])
def delete_api_key(key_id):
    """永久删除 API Key"""
    success = api_key_manager.delete_key(key_id)
    if not success:
        return jsonify({"error": "删除失败，API Key 不存在"}), 404
    return jsonify({"success": True, "message": "已永久删除"})


@openapi_bp.route("/api/open/keys/<int:key_id>/revoke", methods=["POST"])
def revoke_api_key(key_id):
    """吊销 API Key（软删除）"""
    success = api_key_manager.revoke_key(key_id)
    if not success:
        return jsonify({"error": "吊销失败，API Key 不存在"}), 404
    return jsonify({"success": True, "message": "已吊销"})


@openapi_bp.route("/api/open/keys/<int:key_id>/activate", methods=["POST"])
def activate_api_key(key_id):
    """重新激活已吊销的 API Key"""
    success = api_key_manager.activate_key(key_id)
    if not success:
        return jsonify({"error": "激活失败，API Key 不存在"}), 404
    return jsonify({"success": True, "message": "已重新激活"})
