# RAG 智能问答系统

基于 **LangChain + Chroma + BGE 中文模型** 构建的企业级本地 RAG（检索增强生成）问答系统，支持混合检索与两阶段精排。

---

## 目录

- [系统架构](#系统架构)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [配置参考](#配置参考)
- [API 接口](#api-接口)
- [检索策略说明](#检索策略说明)
- [依赖说明](#依赖说明)

---

## 系统架构

```
                          +------------------+
                          |   Documents      |
                          | (txt/pdf/docx/   |
                          |  md/xlsx/csv/    |
                          |  html/sqlite)    |
                          +--------+---------+
                                   |
                                   v
                          +--------+---------+
                          | Document Loader  |
                          | (多格式加载器)    |
                          +--------+---------+
                                   |
                                   v
                          +--------+---------+
                          | Text Splitter    |
                          | (RecursiveChar   |
                          |  TextSplitter)   |
                          +--------+---------+
                                   |
                                   v
                          +--------+---------+
                          |    Chunks        |
                          +--------+---------+
                              |         |
                    +---------+         +---------+
                    v                             v
          +---------+---------+        +---------+---------+
          | Embedding Model   |        | BM25 Index        |
          | (bge-large-zh)    |        | (jieba 分词)       |
          +---------+---------+        +---------+---------+
                    |                             |
                    v                             v
          +---------+---------+        +---------+---------+
          | Vector DB (Chroma)|        | Keyword Index     |
          +---------+---------+        +---------+---------+
                    |                             |
                    +-------------+---------------+
                                  |
                    (用户提问)     v
                    +-------------+---------------+
                    |      混合检索 (RRF 融合)     |
                    |  向量检索 + BM25 关键词检索   |
                    |        粗筛: top 12          |
                    +-------------+---------------+
                                  |
                                  v
                    +-------------+---------------+
                    |   Reranker 精排              |
                    | (bge-reranker-large)         |
                    |   交叉编码器重排序             |
                    |        精筛: top 4           |
                    +-------------+---------------+
                                  |
                                  v
                    +-------------+---------------+
                    |   Augmented Query           |
                    | (Context + Question)        |
                    +-------------+---------------+
                                  |
                                  v
                    +-------------+---------------+
                    |        LLM                  |
                    |  (GPT-4o / Ollama / ...)    |
                    +-------------+---------------+
                                  |
                                  v
                    +-------------+---------------+
                    |        Answer               |
                    +-----------------------------+
```

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **两阶段检索** | 粗筛（混合检索 RRF 融合）→ 精筛（Reranker 交叉编码器），兼顾速度与精度 |
| **混合检索** | 向量语义检索 + BM25 关键词检索，RRF 算法融合排序 |
| **中文优化** | BGE 中文 Embedding 模型 + jieba 中文分词 + BGE 中文 Reranker |
| **GPU 加速** | Embedding 和 Reranker 均支持 CUDA GPU 推理 |
| **多格式支持** | txt / pdf / docx / md / xlsx / csv / html / sqlite |
| **流式输出** | SSE（Server-Sent Events）流式问答，逐字返回 |
| **多会话管理** | 支持多 session 对话历史，上下文连续对话 |
| **Chroma 服务模式** | Chroma 以独立 HTTP 服务运行，数据持久化，支持多客户端 |
| **动态配置** | 运行时可通过 API 切换混合检索 / Reranker 开关 |

---

## 快速开始

### 环境要求

- Python 3.9+
- CUDA GPU（可选，用于加速 Embedding 和 Reranker）
- Windows / Linux / macOS

### 第一步：安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 首次运行会自动下载 Embedding 模型（`BAAI/bge-m3`，约 2.2GB）和 Reranker 模型（`BAAI/bge-reranker-v2-m3`，约 2.2GB），请确保网络畅通且磁盘空间充足。也可提前下载模型放入 `models/` 目录。

### 第二步：配置环境变量

```bash
copy .env.example .env
```

编辑 `.env` 文件，填写以下必要配置：

```env
# LLM 配置（OpenAI 兼容接口）
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# Embedding 模型（HuggingFace ID 或本地路径）
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_DEVICE=cuda

# Chroma 服务配置
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION_NAME=rag_documents

# 检索配置
RETRIEVAL_TOP_K=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=50
HYBRID_SEARCH_ALPHA=0.6

# 多查询融合检索
MULTI_QUERY_ENABLED=true
MULTI_QUERY_COUNT=3

# 语义分块
SEMANTIC_CHUNKING_ENABLED=true
SEMANTIC_CHUNKING_PERCENTILE=90.0

# Reranker 配置
RERANKER_MODEL_NAME=BAAI/bge-reranker-v2-m3
RERANKER_TOP_K=4
RERANKER_CANDIDATE_K=12
RERANKER_ENABLED=true
```

**使用 Ollama 本地模型（免费）：**

1. 安装 [Ollama](https://ollama.ai/) 并拉取模型：`ollama pull llama3.2`
2. 修改 `.env`：
   ```env
   OPENAI_API_KEY=ollama
   OPENAI_BASE_URL=http://localhost:11434/v1
   OPENAI_MODEL=llama3.2
   ```

### 第三步：启动 Chroma 向量数据库服务

```bash
start_chroma.bat
```

或手动启动：

```bash
.venv\Scripts\chroma.exe run --path ./chroma_db --host localhost --port 8000
```

> Chroma 服务默认监听 `http://localhost:8000`，数据持久化在 `chroma_db/` 目录。

### 第四步：准备文档

将知识库文档放入 `documents/` 目录，支持以下格式：

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| 纯文本 | `.txt` | UTF-8 编码文本 |
| PDF | `.pdf` | PDF 文档 |
| Word | `.docx` | Word 文档 |
| Markdown | `.md` | Markdown 文件 |
| Excel | `.xlsx` / `.xls` | 表格数据（按行读取） |
| CSV | `.csv` | 逗号分隔值文件 |
| 网页 | URL | 通过 API 传入 URL 加载 |
| SQLite | `.db` / `.sqlite` | 数据库表内容 |

### 第五步：入库文档

```bash
python ingest.py
```

**可选参数：**

```bash
python ingest.py --dir ./my_docs    # 指定自定义文档目录
python ingest.py --clear            # 清空数据库后重新入库
```

### 第六步：启动问答服务

```bash
python app.py
```

或直接双击 `start.bat`（Windows 一键启动，含依赖检测）。

访问 **http://127.0.0.1:5000** 开始问答。

---

## 项目结构

```
rag_system/
├── documents/                # 知识库文档目录（不入库 git）
├── chroma_db/                # Chroma 向量数据库持久化目录（不入库 git）
├── models/                   # 本地模型存放目录（可选，不入库 git）
│
├── static/
│   ├── css/style.css         # 前端样式
│   └── js/app.js             # 前端交互逻辑
├── templates/
│   └── index.html            # 聊天界面模板
│
├── config.py                 # 全局配置管理（从 .env 读取）
├── document_processor.py     # 文档加载器 + 文本分割器（含语义分块）
├── vector_store.py           # Chroma 向量库 + BM25 混合检索 + RRF 融合
├── reranker.py               # Reranker 交叉编码器精排模块
├── rag_chain.py              # RAG 核心问答链（LCEL）
├── ingest.py                 # 文档入库脚本
├── app.py                    # Flask Web 服务入口
│
├── generate_test_docs.py     # 测试文档生成器（docx/txt）
├── generate_pdf_docs.py      # PDF 测试文档生成器
├── generate_complex_pdf.py   # 复杂 PDF 测试文档生成器
├── generate_test_data.py     # 测试数据生成器
├── download_reranker.py      # Reranker 模型下载脚本
├── copy_reranker.py          # Reranker 模型拷贝脚本
│
├── requirements.txt          # Python 依赖清单
├── .env.example              # 环境变量示例文件
├── .gitignore                # Git 忽略规则
├── start.bat                 # Windows 一键启动脚本
├── start_chroma.bat          # Chroma 服务启动脚本
└── README.md                 # 项目说明文档
```

### 模块职责

| 模块 | 职责 | 对应 RAG 流程 |
|------|------|--------------|
| `config.py` | 读取 `.env` 配置，提供统一配置入口 | 全局配置 |
| `document_processor.py` | 加载多格式文档 + 文本分割为 Chunks | Documents → Chunks |
| `vector_store.py` | 向量化存储 + 混合检索（向量+BM25）+ RRF 融合 | Chunks → Vector DB / 检索 |
| `reranker.py` | 交叉编码器对候选文档精排 | 粗筛结果 → 精筛结果 |
| `rag_chain.py` | 构造 Augmented Query + 调用 LLM 生成答案 | Context → Answer |
| `ingest.py` | 文档入库命令行工具 | 入库流程编排 |
| `app.py` | Flask Web 服务 + REST API + SSE 流式 | 对外服务 |

---

## 配置参考

### 完整环境变量列表

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `OPENAI_API_KEY` | - | LLM API 密钥（Ollama 填 `ollama`） |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | LLM API 地址 |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | LLM 模型名称 |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Embedding 模型路径或 HuggingFace ID |
| `EMBEDDING_DEVICE` | `cuda` | Embedding 推理设备（`cuda` / `cpu`） |
| `CHROMA_HOST` | `localhost` | Chroma 服务地址 |
| `CHROMA_PORT` | `8000` | Chroma 服务端口 |
| `CHROMA_COLLECTION_NAME` | `rag_documents` | Chroma 集合名称 |
| `DOCUMENTS_DIR` | `./documents` | 文档存放目录 |
| `CHUNK_SIZE` | `1200` | 文档分块大小（字符数） |
| `CHUNK_OVERLAP` | `200` | 相邻分块重叠大小（字符数） |
| `RETRIEVAL_TOP_K` | `4` | 最终返回的文档数量 |
| `HYBRID_SEARCH_ALPHA` | `0.6` | 混合检索向量权重（0=纯BM25，1=纯向量） |
| `MULTI_QUERY_ENABLED` | `true` | 是否启用多查询融合检索 |
| `MULTI_QUERY_COUNT` | `3` | 多查询融合生成的查询变体数量 |
| `SEMANTIC_CHUNKING_ENABLED` | `true` | 是否启用语义分块 |
| `SEMANTIC_CHUNKING_PERCENTILE` | `90.0` | 语义分块相似度分位数阈值 |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Reranker 模型路径或 HuggingFace ID |
| `RERANKER_TOP_K` | `4` | Reranker 精排后返回数量 |
| `RERANKER_CANDIDATE_K` | `12` | Reranker 粗筛候选数量 |
| `RERANKER_ENABLED` | `true` | 是否启用 Reranker |
| `FLASK_PORT` | `5000` | Web 服务端口 |
| `FLASK_DEBUG` | `false` | Flask 调试模式 |

### System Prompt 自定义

编辑 `config.py` 中的 `SYSTEM_PROMPT` 变量：

```python
SYSTEM_PROMPT = """你是公司的办事助手，有关员工的休假、报销、办公用品申请等各类事项都由你来负责答复，你工作时注意以下几点：

1、回答问题逻辑清晰、内容全面，按步骤讲解

2、与用户问题有关的信息，要全面的回复，不要有遗漏，比如有效期、提前多少天申请这类特别规定

3、只要遇到跟休假、报销、办公用品相关的问题，优先按照文件知识库中检索到的内容进行回复，要充分利用检索到的上下文，合理解读并综合回答，不要简单地说"没有规定"

4、如果检索到的参考文档中完全没有涉及用户提问的内容，才告知员工规定中暂未明确，并建议请示直属上级

5、如果用户询问的内容与休假、报销、办公用品都不相关，则不需要查询文件知识库
"""
```

---

## API 接口

### 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 聊天界面（HTML） |
| `POST` | `/api/chat` | 普通问答（JSON 响应） |
| `POST` | `/api/chat/stream` | 流式问答（SSE） |
| `GET` | `/api/status` | 系统状态查询 |
| `POST` | `/api/ingest` | 触发文档入库 |
| `GET` | `/api/history/<session_id>` | 获取对话历史 |
| `DELETE` | `/api/history/<session_id>` | 清空对话历史 |
| `GET` | `/api/config` | 获取系统配置 |
| `POST` | `/api/config` | 更新系统配置 |

### 接口详情

#### POST /api/chat

普通问答接口，返回完整 JSON 响应。

**请求体：**

```json
{
  "question": "年假怎么申请？",
  "session_id": "可选的会话ID，不传则自动生成",
  "hybrid": true,
  "reranker": true
}
```

**响应：**

```json
{
  "answer": "根据公司年假管理制度，员工申请年假需要...",
  "session_id": "abc123",
  "sources": [
    {
      "source": "年假管理制度.pdf",
      "preview": "第一条 年假天数...",
      "full_content": "第一条 年假天数...",
      "score": 0.1234,
      "similarity": "93.8%",
      "retrieval_type": "混合"
    }
  ]
}
```

#### POST /api/chat/stream

流式问答接口，通过 SSE 逐字返回答案。

**请求体：** 同 `/api/chat`

**SSE 事件类型：**

| 事件类型 | 说明 |
|----------|------|
| `token` | 逐字返回的答案片段 |
| `done` | 回答完成，附带 sources 和 session_id |
| `error` | 错误信息 |

#### GET /api/status

查询系统运行状态。

**响应：**

```json
{
  "status": "ok",
  "doc_count": 1234,
  "embedding_model": "./models/bge-large-zh-v1.5",
  "llm_model": "gpt-4o",
  "knowledge_base_ready": true,
  "hybrid_search": true,
  "active_sessions": 3
}
```

#### POST /api/ingest

通过 API 触发文档入库。

**请求体：**

```json
{
  "clear": false,
  "dir": "可选的自定义目录"
}
```

#### POST /api/config

运行时动态更新系统配置。

**请求体：**

```json
{
  "hybrid_search": true,
  "reranker": false
}
```

---

## 检索策略说明

### 两阶段检索流程

```
用户提问
    │
    ▼
┌──────────────────────────────────────┐
│  阶段一：粗筛（Hybrid Search）         │
│                                      │
│  向量检索 top_k=12                    │
│    +                                 │
│  BM25 关键词检索 top_k=12             │
│    ↓                                 │
│  RRF（Reciprocal Rank Fusion）融合    │
│    ↓                                 │
│  输出：12 条候选文档                   │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  阶段二：精筛（Reranker）              │
│                                      │
│  bge-reranker-large 交叉编码器        │
│  对 12 条候选逐一打分                  │
│    ↓                                 │
│  按相关性分数降序排列                  │
│    ↓                                 │
│  输出：top 4 条最终结果                │
└──────────────────────────────────────┘
```

### 混合检索（Hybrid Search）

结合两种互补的检索方式：

| 检索方式 | 优势 | 劣势 |
|----------|------|------|
| **向量检索**（语义） | 理解语义，同义词/近义词匹配 | 对专有名词、精确关键词不敏感 |
| **BM25 检索**（关键词） | 精确关键词匹配，专有名词敏感 | 不理解语义，同义词无法匹配 |

通过 RRF 算法融合两种检索结果，`HYBRID_SEARCH_ALPHA` 控制向量权重（推荐 0.6）。

### Reranker 精排

使用交叉编码器（Cross-Encoder）替代双编码器（Bi-Encoder）进行精排：

- **Bi-Encoder**（Embedding 模型）：分别编码 query 和 document，速度快但精度有限
- **Cross-Encoder**（Reranker）：将 (query, document) 一起送入模型，直接输出相关性分数，精度更高

Reranker 在粗筛结果上二次精排，兼顾速度与精度。

---

## 依赖说明

### 核心依赖

| 包 | 版本 | 用途 |
|----|------|------|
| `langchain` | 1.3.0a2 | RAG 框架核心 |
| `langchain-core` | 1.4.0a2 | LangChain 核心抽象 |
| `langchain-text-splitters` | 1.1.2 | 文本分割器 |
| `langchain-chroma` | 0.2.2 | Chroma 向量库集成 |
| `langchain-huggingface` | 1.2.2 | HuggingFace Embedding 集成 |
| `langchain-openai` | 1.2.1 | OpenAI 兼容 LLM 调用 |
| `openai` | 2.35.1 | OpenAI API 客户端 |
| `chromadb` | 0.6.3 | 向量数据库（HTTP 服务模式） |
| `sentence-transformers` | 5.1.0 | Embedding + Reranker 模型 |
| `flask` | 3.1.0 | Web 服务框架 |
| `flask-cors` | 5.0.0 | 跨域支持 |

### 文档处理

| 包 | 版本 | 用途 |
|----|------|------|
| `pypdf` | 5.1.0 | PDF 文档解析 |
| `pdfplumber` | 0.11.9 | PDF 表格提取 |
| `python-docx` | 1.1.2 | Word 文档解析 |
| `docx2txt` | 0.8 | Word 纯文本提取 |
| `unstructured` | 0.16.11 | 通用文档解析 |
| `openpyxl` | 3.1.5 | Excel 文件读取 |
| `beautifulsoup4` | 4.12.3 | HTML 网页解析 |
| `lxml` | >=5.0.0 | XML/HTML 解析加速 |

### 检索增强

| 包 | 版本 | 用途 |
|----|------|------|
| `rank-bm25` | 0.2.2 | BM25 关键词检索算法 |
| `jieba` | 0.42.1 | 中文分词（BM25 索引构建） |

### 工具类

| 包 | 版本 | 用途 |
|----|------|------|
| `python-dotenv` | 1.0.1 | 环境变量管理 |
| `pydantic` | 2.10.6 | 数据校验 |
| `tqdm` | 4.67.1 | 进度条显示 |
| `requests` | 2.32.3 | HTTP 请求（网页加载） |

---

## 常见问题

### Q1: 启动时报错 "Connection refused" 连接 Chroma 失败？

确保先启动了 Chroma 服务：双击 `start_chroma.bat` 或手动执行启动命令。Chroma 服务默认监听 `localhost:8000`。

### Q2: 首次运行下载模型很慢怎么办？

可以提前手动下载模型到 `models/` 目录，然后在 `.env` 中配置本地路径：
```env
EMBEDDING_MODEL_NAME=./models/bge-m3
RERANKER_MODEL_NAME=./models/bge-reranker-v2-m3
```

### Q3: 如何使用免费的本地 LLM（Ollama）？

1. 安装 [Ollama](https://ollama.ai/) 并拉取模型：`ollama pull llama3.2`
2. 修改 `.env`：
   ```env
   OPENAI_API_KEY=ollama
   OPENAI_BASE_URL=http://localhost:11434/v1
   OPENAI_MODEL=llama3.2
   ```

### Q4: 没有 GPU 可以使用吗？

可以。在 `.env` 中将 `EMBEDDING_DEVICE=cpu`，Reranker 也会自动使用 CPU。但推理速度会明显变慢。

### Q5: 如何切换检索策略？

- 通过 API：`POST /api/config` 传入 `{"hybrid_search": true/false, "reranker": true/false}`
- 通过 `.env`：设置 `RERANKER_ENABLED=false` 关闭精排，仅使用混合检索

### Q6: 文档入库后如何更新？

重新运行 `python ingest.py --clear` 会清空数据库后重新入库所有文档。

---

## 许可证

本项目采用 [MIT License](https://opensource.org/licenses/MIT) 开源协议。你可以自由使用、修改和分发本项目代码，但需保留原始版权声明。
