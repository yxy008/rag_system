"""RAG MCP Server 独立启动脚本

与 Flask Web 服务（app.py）解耦，专门用于启动 MCP Server。

使用方式:
    # stdio 模式（默认，用于 MCP Client 子进程连接）
    .venv\Scripts\python.exe run_mcp_server.py

    # SSE/HTTP 模式（用于远程 HTTP 调用）
    .venv\Scripts\python.exe run_mcp_server.py --sse --port 8765
"""
import asyncio
import os
import sys

# 确保可以导入 rag_system 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server import main

if __name__ == "__main__":
    asyncio.run(main())