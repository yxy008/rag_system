"""rag_system MCP Server -- 基于官方 MCP Python SDK

将 agent1/rag_system 的核心能力（语义检索、RAG 问答、文档入库、知识图谱、
文档对比、文档摘要、健康检查）包装为 MCP 工具，通过标准 MCP 协议暴露给外部产品。

支持两种传输方式：
  - stdio 传输：python mcp_server.py
  - SSE/HTTP 传输：python mcp_server.py --sse --port 8765

外部产品通过 MCP Client + stdio/SSE 连接此进程，自动发现并调用所有工具。

参考规范：spec/tools/v2_enterprise_tools.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

# ========== 日志配置 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] MCP: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag_mcp_server")


def _setup_path() -> None:
    """设置 Python 路径，确保可以导入 rag_system 的核心组件。"""
    rag_dir = os.path.dirname(os.path.abspath(__file__))
    if rag_dir not in sys.path:
        sys.path.insert(0, rag_dir)
    os.chdir(rag_dir)


_setup_path()

# ========== MCP SDK 导入 ==========
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# ========== 延迟初始化（单例模式） ==========
_lazy_components: dict[str, Any] = {}
_initialized = False


def _get_rag_components() -> dict[str, Any]:
    """延迟初始化 RAG 系统核心组件。

    延迟初始化确保只有在工具被调用时才加载 BGE 模型等重量级资源，
    避免在 MCP 握手阶段就消耗大量内存。
    """
    global _initialized
    if _initialized:
        return _lazy_components

    logger.info("正在初始化 RAG 核心组件...")

    from core.config import (
        VECTOR_STORE_BACKEND, CHROMA_HOST, CHROMA_PORT,
        MILVUS_HOST, MILVUS_PORT, MILVUS_DIMENSION,
        MILVUS_INDEX_TYPE, MILVUS_METRIC_TYPE, FAISS_PERSIST_DIR,
        OPENAI_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL,
    )
    from core.vector_store import VectorStoreManager
    from core.rag_chain import RAGChain
    from core.document_processor import DocumentProcessor
    from services.knowledge_health import KnowledgeHealthChecker
    from services.knowledge_graph import (
        KnowledgeGraphBuilder, ComparativeQA, DocumentSummarizer
    )

    vs_manager = VectorStoreManager()
    rag = RAGChain(vector_store_manager=vs_manager)
    doc_processor = DocumentProcessor()
    health_checker = KnowledgeHealthChecker(vs_manager, doc_processor)
    kg_builder = KnowledgeGraphBuilder()

    _lazy_components["vs_manager"] = vs_manager
    _lazy_components["rag"] = rag
    _lazy_components["doc_processor"] = doc_processor
    _lazy_components["health_checker"] = health_checker
    _lazy_components["kg_builder"] = kg_builder
    _lazy_components["config"] = {
        "vector_backend": VECTOR_STORE_BACKEND,
        "chroma_host": CHROMA_HOST,
        "chroma_port": CHROMA_PORT,
        "milvus_host": MILVUS_HOST,
        "milvus_port": MILVUS_PORT,
        "milvus_dimension": MILVUS_DIMENSION,
        "milvus_index_type": MILVUS_INDEX_TYPE,
        "milvus_metric_type": MILVUS_METRIC_TYPE,
        "faiss_persist_dir": FAISS_PERSIST_DIR,
        "llm_model": OPENAI_MODEL,
    }
    _lazy_components["openai_config"] = {
        "api_key": OPENAI_API_KEY,
        "base_url": OPENAI_BASE_URL,
        "model": OPENAI_MODEL,
    }

    _initialized = True
    logger.info("RAG 核心组件初始化完成（后端=%s）", VECTOR_STORE_BACKEND)
    return _lazy_components


# ============================================================
# 工具处理函数
# ============================================================

async def _handle_rag_search(query: str, top_k: int = 6) -> dict[str, Any]:
    """语义搜索知识库文档，返回相关文档片段及元数据。"""
    comps = _get_rag_components()
    vs_manager = comps["vs_manager"]
    docs_with_scores = vs_manager.similarity_search_with_scores(query)
    results = []
    for item in docs_with_scores[:top_k]:
        doc = item[0]
        score = item[1]
        rtype = item[2] if len(item) > 2 else "vec"
        results.append({
            "content": doc.page_content[:800],
            "source": doc.metadata.get("source", ""),
            "score": round(score, 4),
            "retrieval_type": rtype,
            "page": doc.metadata.get("page"),
        })
    return {"status": "success", "results": results, "query": query, "result_count": len(results)}


async def _handle_rag_chat(query: str, session_id: str = "default") -> dict[str, Any]:
    """基于知识库的 RAG 问答，检索相关文档后生成回答。"""
    comps = _get_rag_components()
    rag = comps["rag"]
    result = rag.query(query)
    sources = []
    for item in result.get("source_scores", []):
        doc = item[0]
        score = item[1]
        rtype = item[2] if len(item) > 2 else "vec"
        sources.append({
            "source": doc.metadata.get("source", ""),
            "preview": doc.page_content[:300],
            "score": round(score, 4),
            "retrieval_type": rtype,
        })
    return {
        "status": "success",
        "answer": result["answer"],
        "sources": sources,
        "query": query,
        "session_id": session_id,
    }


async def _handle_rag_status() -> dict[str, Any]:
    """获取知识库当前状态：文档数、切片数、向量数据库后端类型等。"""
    comps = _get_rag_components()
    vs_manager = comps["vs_manager"]
    config = comps["config"]
    doc_count = vs_manager.get_document_count()
    return {
        "status": "success",
        "document_count": doc_count,
        "vector_store_backend": config["vector_backend"],
        "llm_model": config["llm_model"],
        "hybrid_search": vs_manager.is_hybrid_search_enabled(),
        "reranker": vs_manager.is_reranker_enabled(),
        "multi_query": vs_manager.is_multi_query_enabled(),
    }


async def _handle_rag_health() -> dict[str, Any]:
    """快速健康检查：验证向量库连接、Embedding 模型可用性、文档处理管道。"""
    comps = _get_rag_components()
    health_checker = comps["health_checker"]
    result = health_checker.quick_check()
    return {"status": "success", "health": result}


async def _handle_rag_ingest(file_path: str) -> dict[str, Any]:
    """将文档文件入库到知识库（分块 + 嵌入 + 存储）。"""
    comps = _get_rag_components()
    doc_processor = comps["doc_processor"]
    vs_manager = comps["vs_manager"]

    if not os.path.exists(file_path):
        return {"status": "error", "message": f"文件不存在: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()
    from core.document_processor import SUPPORTED_EXTENSIONS
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS.keys())
        return {"status": "error", "message": f"不支持的文件格式: {ext}，支持: {supported}"}

    try:
        docs = doc_processor._load_single_file(file_path)
        if not docs:
            return {"status": "error", "message": f"文件内容为空或无法解析: {file_path}"}

        chunks = doc_processor.split_documents(docs)
        if not chunks:
            return {"status": "error", "message": f"文件分割后无有效内容: {file_path}"}

        count = vs_manager.add_documents(chunks)
        logger.info("文件入库成功: %s，新增 %d 个 Chunk", file_path, count)
        return {"status": "success", "chunks_added": count, "file": file_path}
    except Exception as e:
        logger.error("文件入库失败: %s", e, exc_info=True)
        return {"status": "error", "message": f"入库失败: {str(e)}"}


async def _handle_rag_knowledge_graph(entity: str) -> dict[str, Any]:
    """查询知识图谱中的实体关系。如图谱未构建则自动构建。"""
    comps = _get_rag_components()
    kg_builder = comps["kg_builder"]
    vs_manager = comps["vs_manager"]

    graph_data = kg_builder.get_graph_data()
    if graph_data["node_count"] == 0:
        logger.info("知识图谱为空，自动从知识库文档构建...")
        all_docs = vs_manager.get_all_documents()
        if not all_docs:
            return {"status": "error", "message": "知识库为空，无法构建知识图谱"}
        documents = []
        for doc in all_docs:
            documents.append({
                "source": doc.metadata.get("source", ""),
                "full_content": doc.page_content,
            })
        graph_data = kg_builder.build_from_documents(documents)

    related_nodes = []
    related_edges = []
    for node in graph_data["nodes"]:
        if entity.lower() in node["name"].lower():
            related_nodes.append(node)
            node_id = node["id"]
            for edge in graph_data["edges"]:
                if edge["source"] == node_id or edge["target"] == node_id:
                    related_edges.append(edge)

    return {
        "status": "success",
        "entity": entity,
        "matched_nodes": related_nodes,
        "related_edges": related_edges,
        "total_nodes": graph_data["node_count"],
        "total_edges": graph_data["edge_count"],
    }


async def _handle_rag_compare(question1: str, question2: str) -> dict[str, Any]:
    """对比分析两个问题，基于知识库内容生成结构化对比结果。"""
    comps = _get_rag_components()
    vs_manager = comps["vs_manager"]
    rag = comps["rag"]
    openai_config = comps["openai_config"]

    docs1 = vs_manager.similarity_search_with_scores(question1)
    docs2 = vs_manager.similarity_search_with_scores(question2)

    from core.rag_chain import format_docs_with_scores
    context_a = format_docs_with_scores(docs1)
    context_b = format_docs_with_scores(docs2)

    prompt = ComparativeQA.build_comparison_prompt(question1, question2, context_a, context_b)

    try:
        import openai
        client = openai.OpenAI(
            api_key=openai_config["api_key"],
            base_url=openai_config["base_url"],
        )
        response = client.chat.completions.create(
            model=openai_config["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0.1,
        )
        answer = response.choices[0].message.content
        return {"status": "success", "comparison": answer, "topic_a": question1, "topic_b": question2}
    except Exception as e:
        logger.error("对比分析失败: %s", e, exc_info=True)
        return {"status": "error", "message": f"对比分析失败: {str(e)}"}


async def _handle_rag_summarize(doc_ids: list[str]) -> dict[str, Any]:
    """对知识库中的指定文档生成摘要。"""
    comps = _get_rag_components()
    vs_manager = comps["vs_manager"]
    openai_config = comps["openai_config"]

    all_docs = vs_manager.get_all_documents()
    target_contents = []
    for doc in all_docs:
        source = doc.metadata.get("source", "")
        source_name = source.split("\\")[-1].split("/")[-1]
        if source_name in doc_ids or source in doc_ids:
            target_contents.append(doc.page_content)

    if not target_contents:
        return {"status": "error", "message": f"未找到指定文档: {doc_ids}"}

    combined = "\n\n---\n\n".join(target_contents[:20])
    if len(combined) > 8000:
        combined = combined[:8000] + "\n...(内容过长已截断)"

    prompt = DocumentSummarizer.build_summary_prompt(combined, "structured")

    try:
        import openai
        client = openai.OpenAI(
            api_key=openai_config["api_key"],
            base_url=openai_config["base_url"],
        )
        response = client.chat.completions.create(
            model=openai_config["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1,
        )
        summary = response.choices[0].message.content
        return {"status": "success", "summary": summary, "doc_ids": doc_ids}
    except Exception as e:
        logger.error("文档摘要生成失败: %s", e, exc_info=True)
        return {"status": "error", "message": f"摘要生成失败: {str(e)}"}


async def _handle_rag_list_documents() -> dict[str, Any]:
    """列出知识库中所有已入库的文档。"""
    comps = _get_rag_components()
    vs_manager = comps["vs_manager"]
    all_docs = vs_manager.get_all_documents()
    doc_map: dict[str, dict] = {}
    for doc in all_docs:
        source = doc.metadata.get("source", "")
        source_name = source.split("\\")[-1].split("/")[-1]
        if source_name not in doc_map:
            doc_map[source_name] = {
                "source": source_name,
                "full_path": source,
                "chunk_count": 0,
            }
        doc_map[source_name]["chunk_count"] += 1
    return {
        "status": "success",
        "document_count": len(doc_map),
        "total_chunks": len(all_docs),
        "documents": list(doc_map.values()),
    }


# ============================================================
# 工具名称到处理函数的映射
# ============================================================
_TOOL_HANDLERS = {
    "rag_search": _handle_rag_search,
    "rag_chat": _handle_rag_chat,
    "rag_status": _handle_rag_status,
    "rag_health": _handle_rag_health,
    "rag_ingest": _handle_rag_ingest,
    "rag_knowledge_graph": _handle_rag_knowledge_graph,
    "rag_compare": _handle_rag_compare,
    "rag_summarize": _handle_rag_summarize,
    "rag_list_documents": _handle_rag_list_documents,
}


# ============================================================
# 创建 MCP Server
# ============================================================

def create_rag_mcp_server() -> Server:
    """创建并配置 RAG MCP Server，注册全部 9 个工具。"""
    from core.config import (
        MCP_SERVER_NAME, MCP_SERVER_VERSION,
    )

    server = Server(MCP_SERVER_NAME)

    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        return [
            Tool(
                name="rag_search",
                description="搜索 RAG 知识库中的文档。返回相关文档片段、元数据和相似度分数。"
                            "当你需要查找内部文档、政策、技术参考资料时使用。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索查询文本"},
                        "top_k": {
                            "type": "integer",
                            "description": "返回结果数量（默认 6）",
                            "default": 6,
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="rag_chat",
                description="基于 RAG 知识库内容回答问题。系统检索相关文档后生成回答。"
                            "用于需要基于知识库内容的事实性问答。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "要提问的问题"},
                        "session_id": {
                            "type": "string",
                            "description": "会话 ID，用于对话连续性",
                            "default": "default",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="rag_status",
                description="获取 RAG 知识库当前状态：文档数量、切片数量、向量数据库后端类型、"
                            "混合检索/Reranker/多查询融合开关状态。用于了解可用知识的范围和系统配置。",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="rag_health",
                description="对 RAG 知识库执行快速健康检查。检查向量库连接、Embedding 模型可用性、"
                            "文档处理管道状态。用于验证知识库是否正常运行。",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="rag_ingest",
                description="将文档文件入库到 RAG 知识库。文档将被分块、嵌入并存储到向量数据库。"
                            "支持 PDF、DOCX、TXT、MD、XLSX、CSV、HTML 等格式。"
                            "用于向知识库添加新文档。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "要入库的文档文件路径",
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            Tool(
                name="rag_knowledge_graph",
                description="查询知识图谱中的实体及其关系。如果图谱尚未构建，将自动从知识库文档构建。"
                            "返回关联实体、关系类型和图谱上下文。用于探索概念和文档之间的关联。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": "要在知识图谱中查询的实体名称或概念",
                        },
                    },
                    "required": ["entity"],
                },
            ),
            Tool(
                name="rag_compare",
                description="使用知识库对比分析两个问题或主题。返回结构化对比结果，"
                            "突出相似点、差异和关联关系。用于竞品分析或多因素评估。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question1": {"type": "string", "description": "第一个问题或主题"},
                        "question2": {"type": "string", "description": "第二个问题或主题"},
                    },
                    "required": ["question1", "question2"],
                },
            ),
            Tool(
                name="rag_summarize",
                description="为知识库中的一个或多个文档生成摘要。"
                            "用于获取长文档的精简概览。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "doc_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "要生成摘要的文档 ID 列表（文件名）",
                        },
                    },
                    "required": ["doc_ids"],
                },
            ),
            Tool(
                name="rag_list_documents",
                description="列出 RAG 知识库中当前存储的所有文档。"
                            "返回文档 ID、名称、切片数量等元数据。"
                            "用于在搜索前了解可用知识范围。",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        handler = _TOOL_HANDLERS.get(name)
        if handler is None:
            return [TextContent(type="text", text=json.dumps(
                {"status": "error", "message": f"未知工具: {name}"}, ensure_ascii=False
            ))]

        try:
            result = await handler(**arguments)
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        except Exception as e:
            logger.error("工具 %s 调用失败: %s", name, e, exc_info=True)
            return [TextContent(type="text", text=json.dumps(
                {"status": "error", "message": str(e)}, ensure_ascii=False
            ))]

    return server


# ============================================================
# 启动入口
# ============================================================

async def run_stdio() -> None:
    """通过 stdio 传输启动 MCP Server（用于子进程通信）。"""
    server = create_rag_mcp_server()
    logger.info("RAG MCP Server 启动（stdio 模式）")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


async def run_sse(host: str = "0.0.0.0", port: int = 8765) -> None:
    """通过 SSE/HTTP 传输启动 MCP Server（用于远程调用）。"""
    try:
        from starlette.applications import Starlette
        from starlette.routing import Route, Mount
        from starlette.responses import JSONResponse
        import uvicorn
    except ImportError as e:
        logger.error("SSE 模式需要额外依赖，请安装: pip install starlette uvicorn")
        raise

    from mcp.server.sse import SseServerTransport

    server = create_rag_mcp_server()
    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0], streams[1],
                server.create_initialization_options(),
            )

    async def handle_health(request):
        return JSONResponse({"status": "ok", "server": "rag-mcp-server"})

    starlette_app = Starlette(
        routes=[
            Route("/health", endpoint=handle_health),
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse_transport.handle_post_message),
        ],
    )

    logger.info("RAG MCP Server 启动（SSE 模式，http://%s:%d）", host, port)
    config = uvicorn.Config(starlette_app, host=host, port=port, log_level="info")
    http_server = uvicorn.Server(config)
    await http_server.serve()


async def main() -> None:
    """主入口：根据命令行参数选择传输方式。"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG MCP Server")
    parser.add_argument(
        "--sse", action="store_true",
        help="使用 SSE/HTTP 传输模式（默认使用 stdio 模式）",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="SSE 模式监听地址（默认 0.0.0.0）",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="SSE 模式监听端口（默认 8765）",
    )
    args = parser.parse_args()

    if args.sse:
        await run_sse(host=args.host, port=args.port)
    else:
        await run_stdio()


if __name__ == "__main__":
    asyncio.run(main())