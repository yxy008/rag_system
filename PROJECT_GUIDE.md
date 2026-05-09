# RAG 智能问答系统 - 项目完全掌握指南

> **版本**: v2.0 (含 Reranker 两阶段检索)  
> **技术栈**: LangChain + Chroma(服务模式) + bge-large-zh-v1.5 + BAAI/bge-reranker-large + Flask  
> **LLM**: gpt-4o (OpenAI 兼容接口，可替换为 GLM-4 / Ollama 等任意兼容模型)

---

## 一、项目架构总览

### 1.1 核心架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户交互层                                      │
│                    Web UI (templates/index.html)                            │
│                    静态资源 (static/css/, static/js/)                       │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ HTTP/SSE
┌────────────────────────────────▼────────────────────────────────────────────┐
│                            Flask Web 服务                                    │
│                              (app.py)                                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │
│  │ /api/chat    │ │/api/chat/stream│ │ /api/config  │ │ /api/status  │    │
│  │ 普通问答      │ │ 流式问答(SSE)  │ │ 配置管理      │ │ 系统状态      │    │
│  │ /api/ingest  │ │ /api/history   │                                        │
│  │ 文档入库      │ │ 对话历史管理    │                                        │
│  └──────┬───────┘ └──────┬───────┘ └──────────────┘ └──────────────┘    │
└─────────┼─────────────────┼────────────────────────────────────────────────┘
          │                 │
┌─────────▼─────────────────▼────────────────────────────────────────────────┐
│                           RAG 核心层                                         │
│                                                                             │
│  ┌────────────────────┐    ┌────────────────────┐                          │
│  │   RAGChain         │    │  VectorStoreManager │                          │
│  │   (rag_chain.py)   │◄───│  (vector_store.py)  │                          │
│  │                    │    │                     │                          │
│  │ • Prompt 构造       │    │ • 向量检索           │                          │
│  │ • LLM 调用         │    │ • BM25 检索          │                          │
│  │ • 流式输出         │    │ • RRF 融合           │                          │
│  │ • 来源标注         │    │ • Reranker 精排      │                          │
│  └────────┬───────────┘    └────────┬───────────┘                          │
│           │                          │                                      │
│  ┌────────▼──────────────────────────▼───────────┐                          │
│  │              reranker.py                    │                          │
│  │           BAAI/bge-reranker-large              │                          │
│  │              交叉编码器精排                     │                          │
│  └────────────────────────────────────────────────┘                          │
│                                                                             │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│                          数据与模型层                                        │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │  Chroma 服务    │  │  BM25 索引      │  │  Embedding 模型 │              │
│  │  (向量数据库)    │  │  (内存中)       │  │  (bge-large)   │              │
│  │  localhost:8000 │  │                 │  │                 │              │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐                                   │
│  │  文档目录        │  │  LLM API        │                                   │
│  │  documents/     │  │  (gpt-4o 等)    │                                   │
│  └─────────────────┘  └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 两阶段检索流程详解

```
用户提问
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 阶段一：粗筛（速度优先）                                          │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐                          │
│  │ 向量检索      │    │ BM25 检索     │                          │
│  │ top_k=12     │    │ top_k=12     │                          │
│  └──────┬───────┘    └──────┬───────┘                          │
│         │                    │                                   │
│         └────────┬───────────┘                                   │
│                  ▼                                               │
│         ┌──────────────┐                                         │
│         │ RRF 融合     │  倒数排名融合                           │
│         │ α=0.6       │  公式: α/(k+rank_v+1)+(1-α)/(k+rank_b+1)│
│         └──────┬───────┘                                         │
│                ▼                                                 │
│         ┌──────────────┐                                         │
│         │ 12条候选文档  │                                         │
│         └──────────────┘                                         │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│ 阶段二：精筛（精度优先）                                          │
│                                                                 │
│  ┌──────────────┐                                               │
│  │ Reranker     │  BAAI/bge-reranker-large                     │
│  │ 交叉编码器    │  将 (query, doc) 一起输入                      │
│  └──────┬───────┘                                               │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │ 重排序输出    │                                               │
│  │ top_k=4      │                                               │
│  └──────────────┘                                               │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                          LLM 生成答案
```

---

## 二、代码文件详解

### 2.1 文件清单与职责

| 文件 | 类型 | 职责 | 重要程度 |
|------|------|------|----------|
| `config.py` | 配置 | 全局配置项（模型、路径、参数、Prompt） | ⭐⭐⭐ |
| `app.py` | 入口 | Flask Web 服务，API 路由，对话历史管理 | ⭐⭐⭐ |
| `vector_store.py` | 核心 | 向量检索 + BM25 + RRF + Reranker 两阶段检索 | ⭐⭐⭐ |
| `rag_chain.py` | 核心 | RAG 问答链，Prompt 构造，LLM 调用，流式输出 | ⭐⭐⭐ |
| `document_processor.py` | 处理 | 文档加载（多格式）+ 章节感知切分 | ⭐⭐ |
| `reranker.py` | 核心 | Reranker 懒加载与交叉编码器精排 | ⭐⭐ |
| `ingest.py` | 脚本 | 文档入库命令行工具 | ⭐⭐ |
| `requirements.txt` | 配置 | Python 依赖清单 | ⭐⭐ |
| `.env` / `.env.example` | 配置 | 环境变量配置（API Key、模型路径等） | ⭐⭐ |
| `start.bat` | 脚本 | Windows 一键启动脚本（含依赖检测） | ⭐ |
| `start_chroma.bat` | 脚本 | Chroma 向量数据库服务启动脚本 | ⭐ |
| `download_reranker.py` | 脚本 | 下载 bge-reranker-large 模型到本地 | ⭐ |
| `copy_reranker.py` | 脚本 | 从 HuggingFace 缓存复制 Reranker 模型 | ⭐ |
| `templates/index.html` | 前端 | 问答界面 HTML | ⭐ |
| `static/js/app.js` | 前端 | 前端交互逻辑（SSE 流式接收） | ⭐ |
| `static/css/style.css` | 前端 | 界面样式 | ⭐ |
| `models/` | 模型 | 本地 Embedding 模型目录（bge-large-zh-v1.5 等） | ⭐ |
| `chroma_db/` | 数据 | Chroma 向量数据库持久化目录 | ⭐ |
| `documents/` | 数据 | 待入库的原始文档目录 | ⭐ |

---

## 三、代码阅读顺序建议

### 3.1 推荐的阅读路径（从数据流视角）

```
第1步：理解配置层
  └─ config.py
     • 理解所有可配置项（七大配置块）
     • 理解模型的路径和选择
     • 理解 System Prompt 和 RAG Prompt 模板

第2步：理解数据层
  └─ document_processor.py
     • 理解文档如何被加载（12 种格式支持）
     • 理解 SectionAwareSplitter 的切分策略
     • 理解 SUPPORTED_EXTENSIONS 注册机制

第3步：理解检索层（核心）
  └─ vector_store.py
     • VectorStoreManager 类的初始化
     • BM25 检索实现 (simple_tokenize + BM25Okapi)
     • RRF 融合算法（含 alpha 权重）
     • 两阶段检索流程（粗筛 + 精筛）

第4步：理解精排层
  └─ reranker.py
     • CrossEncoder 的使用方式
     • 懒加载模式
     • rerank 与 rerank_with_sources 的区别

第5步：理解问答层
  └─ rag_chain.py
     • format_docs_with_scores 的格式化逻辑
     • RAGChain.query() 的完整流程
     • query_stream() 流式输出
     • LCEL Chain 构建
     • 对话历史注入机制

第6步：理解服务层
  └─ app.py
     • 9 个 API 端点的实现
     • SSE 流式输出机制
     • 对话历史管理（sessions）
     • 运行时配置切换（临时切换 + 自动恢复）
     • 启动预热流程

第7步：理解入库流程
  └─ ingest.py
     • 完整的入库 pipeline
     • --clear / --dir 参数

第8步：理解辅助脚本
  └─ start.bat / start_chroma.bat
     • Windows 环境下的启动流程
  └─ download_reranker.py / copy_reranker.py
     • Reranker 模型的下载和部署
```

### 3.2 核心类图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              VectorStoreManager                          │
├─────────────────────────────────────────────────────────────────────────┤
│ 属性:                                                                        │
│   _embeddings: HuggingFaceEmbeddings  (懒加载)                              │
│   _vector_store: Chroma                  (HTTP 客户端)                    │
│   _bm25_index: BM25Okapi                (内存索引)                        │
│   _bm25_corpus: List[Document]          (原始文档)                        │
│   _reranker_manager: RerankerManager    (懒加载)                         │
│   _hybrid_enabled: bool                 (混合检索开关)                    │
│   _reranker_enabled: bool               (Reranker 开关)                   │
├─────────────────────────────────────────────────────────────────────────┤
│ 方法:                                                                        │
│   add_documents(chunks) → int                  # 入库                      │
│   similarity_search(query, top_k) → List[Doc]  # 简单检索                 │
│   similarity_search_with_scores(query) → List[(Doc, score, type)] # 详情  │
│   _hybrid_search(query, top_k) → ...           # 混合检索                  │
│   _hybrid_search_with_scores(query, top_k) → ... # 混合检索(带分数)        │
│   _rrf_fusion(vec, bm25, alpha) → ...          # RRF 融合                 │
│   _bm25_search(query, top_k) → ...             # BM25 检索                │
│   _get_reranker() → RerankerManager             # 懒加载 Reranker           │
│   get_retriever(top_k) → Retriever              # LangChain Retriever      │
│   get_document_count() → int                    # 文档数量                  │
│   clear_collection() → bool                     # 清空数据库                │
│   set_hybrid_search(bool) / set_reranker(bool)  # 运行时切换               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ uses
┌─────────────────────────────────────────────────────────────────────────┐
│                              RerankerManager                             │
├─────────────────────────────────────────────────────────────────────────┤
│   rerank(query, docs, top_k) → List[(Doc, score)]                       │
│   rerank_with_sources(query, docs_with_scores, top_k) → List[(Doc, score, type)] │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ uses
┌─────────────────────────────────────────────────────────────────────────┐
│                         CrossEncoder (sentence-transformers)            │
│                         BAAI/bge-reranker-large                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 四、关键模块详解

### 4.1 config.py - 配置管理中心

```python
# 七大配置块

LLM_CONFIG = {
    OPENAI_API_KEY,      # LLM API 密钥
    OPENAI_BASE_URL,     # API 地址（兼容 OpenAI 格式）
    OPENAI_MODEL,        # 模型名（默认 gpt-4o，可替换为 GLM-4 / Ollama 等）
}

EMBEDDING_CONFIG = {
    EMBEDDING_MODEL_NAME,   # ./models/bge-large-zh-v1.5
    EMBEDDING_DEVICE,       # cuda / cpu
}

CHROMA_CONFIG = {
    CHROMA_HOST,             # Chroma 服务地址（默认 localhost）
    CHROMA_PORT,             # Chroma 服务端口（默认 8000）
    CHROMA_COLLECTION_NAME,  # 集合名称（默认 rag_documents）
}

DOCUMENT_CONFIG = {
    DOCUMENTS_DIR,           # 文档目录路径（默认 documents/）
}

RETRIEVAL_CONFIG = {
    RETRIEVAL_TOP_K,         # 最终返回数量（默认 4）
    HYBRID_SEARCH_ALPHA,     # 向量权重（默认 0.6）
    CHUNK_SIZE,              # 切片大小（默认 800）
    CHUNK_OVERLAP,           # 重叠大小（默认 150）
}

RERANKER_CONFIG = {
    RERANKER_MODEL_NAME,    # BAAI/bge-reranker-large
    RERANKER_TOP_K,         # 精排后数量（默认 4）
    RERANKER_CANDIDATE_K,    # 粗筛候选数（默认 12）
    RERANKER_ENABLED,       # 默认启用
}

FLASK_CONFIG = {
    FLASK_PORT,             # Flask 服务端口（默认 5000）
    FLASK_DEBUG,            # 调试模式（默认 false）
}

# Prompt 配置（也在 config.py 中）
SYSTEM_PROMPT = """你是公司的办事助手..."""
RAG_PROMPT_TEMPLATE = """以下是从知识库中检索到的..."""
```

### 4.2 vector_store.py - 两阶段检索

#### 阶段一：粗筛
```python
# 向量检索：Bi-Encoder 编码查询和文档，计算余弦相似度
vec_results = vector_store.similarity_search_with_score(query, k=candidate_k)

# BM25 检索：基于词频的关键词检索
bm25_results = _bm25_search(query, candidate_k)

# RRF 融合：综合两种检索结果
# 公式: RRF(d) = alpha * 1/(k + rank_vec(d) + 1) + (1-alpha) * 1/(k + rank_bm25(d) + 1)
# 其中 k=60（平滑参数），alpha=0.6（向量权重）
fused = _rrf_fusion(vec_results, bm25_results, alpha=0.6)
```

#### 阶段二：精筛
```python
# Reranker：Cross-Encoder 同时编码 query+doc，精度更高
if self._reranker_enabled:
    docs = [doc for doc, _, _ in fused]  # 从 RRF 融合结果中提取文档
    reranked = reranker.rerank(query, docs, top_k=top_k)
    # 返回格式：(doc, rerank_score * 100, "rerank")
    return [(doc, score * 100, "rerank") for doc, score in reranked]

# 未启用 Reranker：直接返回 RRF 融合结果
return [(doc, score, rtype) for doc, score, rtype in fused[:top_k]]
```

#### 关键方法说明

| 方法 | 说明 |
|------|------|
| `similarity_search(query, top_k)` | 简单检索（支持纯向量/混合检索切换） |
| `similarity_search_with_scores(query, top_k)` | 带分数的检索，支持两阶段（粗筛+精筛） |
| `_hybrid_search(query, top_k)` | 混合检索：向量 + BM25 → RRF 融合 |
| `_hybrid_search_with_scores(query, top_k)` | 混合检索（带分数）：粗筛 → RRF → Reranker 精筛 |
| `_rrf_fusion(vec, bm25, alpha)` | RRF 倒数排名融合算法 |
| `_bm25_search(query, top_k)` | BM25 关键词检索 |
| `get_retriever(top_k)` | 获取 LangChain Retriever（用于 LCEL Chain） |
| `set_hybrid_search(bool)` / `set_reranker(bool)` | 运行时切换检索策略 |
| `clear_collection()` | 清空向量数据库和 BM25 索引 |

### 4.3 rag_chain.py - RAG 问答链

#### 核心方法

| 方法 | 说明 |
|------|------|
| `query(question, history)` | 非流式问答，返回完整答案 + 来源文档 |
| `query_stream(question, history)` | 流式问答（生成器），逐步 yield 文本片段 |
| `get_source_info(question)` | 仅返回检索来源信息，不调用 LLM（调试用） |
| `_format_history(history)` | 将对话历史格式化为文本（保留最近 6 条） |
| `_build_chain()` | 构建 LCEL Chain（LangChain Expression Language） |

#### LCEL Chain 构建

```python
def _build_chain(self):
    retriever = self.vector_store_manager.get_retriever(top_k=RETRIEVAL_TOP_K)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(RAG_PROMPT_TEMPLATE),
    ])
    
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | self.llm
        | StrOutputParser()
    )
    return chain
```

#### 完整调用链（query 方法）

```python
def query(question, history):
    # 1. 检索（两阶段：粗筛 + Reranker 精筛）
    docs_with_scores = vs_manager.similarity_search_with_scores(question)
    
    # 2. 格式化 Context（含相关度分数和检索类型标注）
    context = format_docs_with_scores(docs_with_scores)
    if history:
        history_text = self._format_history(history)
        context = f"【对话历史】\n{history_text}\n\n【本次检索结果】\n{context}"
    
    # 3. 构造 Prompt
    messages = prompt.format_messages(context=context, question=question)
    
    # 4. 调用 LLM
    response = llm.invoke(messages)
    
    return {
        "answer": response.content,
        "source_documents": [...],
        "source_scores": docs_with_scores,
    }
```

#### 辅助函数

```python
def format_docs(docs: List[Document]) -> str:
    """将 Document 列表格式化为字符串上下文（无分数）"""

def format_docs_with_scores(docs_with_scores: List[tuple]) -> str:
    """将带分数的检索结果格式化为字符串上下文
       支持格式：(Document, score) 或 (Document, score, retrieval_type)
       输出示例：【参考文档 1】来源：企业报销制度.docx（相关度：85.3%（重排精筛））
    """
```

### 4.4 document_processor.py - 文档处理

#### 支持的格式
| 格式 | 加载器 | 切分方式 |
|------|--------|----------|
| txt, md, pdf, docx, doc | LangChain 内置 | SectionAwareSplitter |
| xlsx, xls | ExcelLoader（自定义，pandas） | 按行拆分 |
| csv | CSVLoader（自定义，pandas） | 按行拆分 |
| html, htm | UnstructuredHTMLLoader | SectionAwareSplitter |
| sqlite, db | SQLiteLoader（自定义） | 按行拆分 |
| URL | WebPageLoader（自定义） | BeautifulSoup 解析 |

#### DocumentProcessor 核心方法

| 方法 | 说明 |
|------|------|
| `load_documents(dir)` | 从目录加载所有支持格式的文档 |
| `load_url(url)` | 从 URL 加载网页内容 |
| `load_sqlite(db_path, table, query)` | 从 SQLite 数据库加载内容 |
| `split_documents(docs)` | 将文档按章节结构切分为 Chunks |
| `load_and_split(dir)` | 一步完成加载 + 分割（便捷方法） |

#### SUPPORTED_EXTENSIONS 注册表
```python
SUPPORTED_EXTENSIONS = {
    ".txt":   ("文本文件", None),           # TextLoader
    ".pdf":   ("PDF 文件", None),           # PyPDFLoader
    ".docx":  ("Word 文件", None),          # Docx2txtLoader
    ".doc":   ("Word 文件", None),          # Docx2txtLoader
    ".md":    ("Markdown 文件", None),       # UnstructuredMarkdownLoader
    ".xlsx":  ("Excel 文件", ExcelLoader),   # 自定义 pandas 加载器
    ".xls":   ("Excel 文件", ExcelLoader),
    ".csv":   ("CSV 文件", CSVLoader),       # 自定义 pandas 加载器
    ".html":  ("HTML 文件", None),           # UnstructuredHTMLLoader
    ".htm":   ("HTML 文件", None),
    ".sqlite":("SQLite 数据库", SQLiteLoader), # 自定义加载器
    ".db":    ("SQLite 数据库", SQLiteLoader),
}
```

#### SectionAwareSplitter 切分策略
```python
# 优先按章节结构切分
SECTION_PATTERNS = [
    r"^(第?[一二三四五六七八九十百零\d]+[章节条款]...)",  # 一级标题
    r"^\d+\.\d+\.?\s+.+",    # 二级标题 1.1
    r"^\d+\.\d+\.\d+\.?\s+", # 三级标题 1.1.1
    r"^#{1,3}\s+",           # Markdown 标题
]
```

---

## 五、Flask Web 服务详解

### 5.1 app.py - API 端点一览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 返回前端聊天界面（HTML） |
| `/api/chat` | POST | 普通问答（JSON 响应），支持 `hybrid`/`reranker` 参数 |
| `/api/chat/stream` | POST | 流式问答（SSE），支持 `hybrid`/`reranker` 参数 |
| `/api/status` | GET | 系统状态（文档数、模型信息、混合检索状态） |
| `/api/ingest` | POST | 通过 API 触发文档入库，支持 `clear`/`dir` 参数 |
| `/api/history/<session_id>` | GET | 获取指定会话的对话历史 |
| `/api/history/<session_id>` | DELETE | 清空指定会话的对话历史 |
| `/api/config` | GET | 获取当前配置（混合检索/Reranker 开关状态） |
| `/api/config` | POST | 更新配置（运行时切换混合检索/Reranker） |

### 5.2 对话历史管理

```python
# 内存存储（生产环境建议换 Redis 或 SQLite）
sessions: dict = defaultdict(list)  # session_id -> List[{"role", "content"}]
MAX_HISTORY_PER_SESSION = 20        # 每个 session 最多保留 20 条

def get_history(session_id) -> list      # 获取历史
def add_to_history(session_id, role, content)  # 追加记录（自动截断）
def clear_history(session_id)            # 清空历史
```

### 5.3 运行时配置切换

```python
# /api/chat 和 /api/chat/stream 都支持临时切换检索策略
# 请求体示例：
{
    "question": "年假怎么申请？",
    "session_id": "abc123",
    "hybrid": true,     # 可选：临时启用/关闭混合检索
    "reranker": true    # 可选：临时启用/关闭 Reranker
}

# 切换是临时的：请求处理完后自动恢复原始状态
orig_hybrid = vs_manager.is_hybrid_search_enabled()
orig_reranker = vs_manager.is_reranker_enabled()
# ... 处理请求 ...
vs_manager.set_hybrid_search(orig_hybrid)  # 恢复
vs_manager.set_reranker(orig_reranker)     # 恢复
```

### 5.4 启动流程

```python
if __name__ == "__main__":
    # 1. 检查知识库是否为空
    doc_count = vs_manager.get_document_count()
    
    # 2. 预热：重建 BM25 索引
    all_docs = vs_manager.get_all_documents()
    vs_manager._build_bm25_index(all_docs)
    
    # 3. 启动 Flask（多线程模式）
    app.run(host="0.0.0.0", port=FLASK_PORT, threaded=True)
```

---

## 六、数据流完整解析

### 6.1 文档入库流程

```
documents/ 目录
     │
     ▼
DocumentProcessor.load_documents()
     │ 加载各种格式 → Document 对象列表
     ▼
DocumentProcessor.split_documents()
     │ SectionAwareSplitter 按章节切分
     ▼
VectorStoreManager.add_documents(chunks)
     │
     ├── Chroma.add_documents()
     │     │ 每个 chunk 调用 bge-large-zh-v1.5 编码 → 向量
     │     ▼
     │     Chroma 服务（localhost:8000）
     │
     └── _build_bm25_index()
           │ 所有文档构建 BM25 倒排索引
           ▼
           BM25Okapi (内存中)
```

### 6.2 问答请求流程

```
用户提问 → Flask API
     │
     ▼
rag.query(question, history)
     │
     ├── VectorStoreManager.similarity_search_with_scores(question)
     │     │
     │     ├── 阶段一：粗筛
     │     │     ├── 向量检索 top_k=12
     │     │     ├── BM25 检索 top_k=12
     │     │     └── RRF 融合 → 12 条候选
     │     │
     │     └── 阶段二：精筛
     │           └── Reranker.rerank() → top_k=4
     │
     ├── format_docs_with_scores()
     │     格式化: 【参考文档1】来源：xxx（相关度：85%）
     │
     └── LLM.invoke(prompt)
           │
           ▼
     返回 answer + sources
```

---

## 七、API 接口一览

| 方法 | 路径 | 说明 | 请求体 |
|------|------|------|--------|
| GET | `/` | 问答界面 | - |
| POST | `/api/chat` | 普通问答 | `{question, session_id?, hybrid?, reranker?}` |
| POST | `/api/chat/stream` | 流式问答(SSE) | 同上 |
| GET | `/api/status` | 系统状态 | - |
| POST | `/api/ingest` | 触发入库 | `{clear?, dir?}` |
| GET | `/api/history/<session>` | 获取历史 | - |
| DELETE | `/api/history/<session>` | 清空历史 | - |
| GET | `/api/config` | 获取配置 | - |
| POST | `/api/config` | 更新配置 | `{hybrid_search?, reranker?}` |

---

## 八、环境变量与配置

```bash
# .env 文件
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://xxx/api/paas/v4
OPENAI_MODEL=gpt-4o

EMBEDDING_MODEL_NAME=./models/bge-large-zh-v1.5
EMBEDDING_DEVICE=cuda

CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_COLLECTION_NAME=rag_documents

DOCUMENTS_DIR=documents

RETRIEVAL_TOP_K=4
HYBRID_SEARCH_ALPHA=0.6
CHUNK_SIZE=800
CHUNK_OVERLAP=150

RERANKER_MODEL_NAME=BAAI/bge-reranker-large
RERANKER_TOP_K=4
RERANKER_CANDIDATE_K=12
RERANKER_ENABLED=true

FLASK_PORT=5000
FLASK_DEBUG=false
```

---

## 九、启动与使用

### 9.1 启动顺序

```bash
# 1. 启动 Chroma 向量数据库服务（使用项目内置的 chroma.exe）
.\start_chroma.bat
# 等价于: .\.venv\Scripts\chroma.exe run --path ./chroma_db --host localhost --port 8000

# 2. 文档入库（首次需要）
python ingest.py                    # 默认 documents/ 目录
python ingest.py --dir ./my_docs    # 自定义目录
python ingest.py --clear            # 清空后重新入库

# 3. 启动 Flask 服务（或使用一键启动脚本）
python app.py
# 或: .\start.bat

# 4. 访问
http://127.0.0.1:5000
```

### 9.2 目录结构

```
rag_system/
├── app.py                    # Flask 入口
├── config.py                 # 配置
├── ingest.py                 # 入库脚本
├── document_processor.py     # 文档处理
├── vector_store.py           # 向量检索核心
├── rag_chain.py              # RAG 问答链
├── reranker.py               # Reranker 精排
├── requirements.txt          # 依赖
├── .env / .env.example       # 环境变量
├── start.bat                 # 一键启动脚本
├── start_chroma.bat          # Chroma 启动脚本
├── download_reranker.py      # Reranker 模型下载
├── copy_reranker.py          # Reranker 模型复制
│
├── documents/                # 源文档目录
│   ├── 企业报销制度.docx
│   ├── 公司假期制度.docx
│   └── 办公用品申领制度.docx
│
├── chroma_db/                # Chroma 数据目录（服务模式持久化）
│
├── models/                   # 本地模型
│   ├── bge-large-zh-v1.5/    # Embedding 模型
│   └── all-MiniLM-L6-v2/     # 备用轻量 Embedding 模型
│
├── templates/
│   └── index.html            # 前端界面
│
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## 十、扩展指南

### 10.1 添加新的文档格式

```python
# document_processor.py

# 1. 创建自定义 Loader
class MyFormatLoader:
    def load(self, file_path) -> List[Document]:
        # 实现加载逻辑
        return documents

# 2. 注册到 SUPPORTED_EXTENSIONS
SUPPORTED_EXTENSIONS = {
    # ...existing...
    ".myext": ("My Format", MyFormatLoader),
}
```

### 10.2 添加新的检索策略

```python
# vector_store.py

def _new_search(self, query: str, top_k: int) -> List[Document]:
    """添加新检索方法"""
    # 实现新检索逻辑
    pass

def similarity_search_with_scores(self, query: str, top_k: int):
    # 在此处调用新方法
    return self._new_search(query, top_k)
```

### 10.3 更换 LLM

```python
# config.py
OPENAI_BASE_URL = "https://your-llm-api/v4"
OPENAI_MODEL = "your-model-name"

# rag_chain.py - 如需调整 temperature
self._llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0.1,  # 调整生成随机性
)
```

---

## 十一、调试与排错

### 11.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 知识库为空 | 未运行 ingest.py | `python ingest.py` |
| Chroma 连接失败 | 服务未启动 | 运行 `.\start_chroma.bat` |
| 模型下载慢 | HuggingFace 网络 | 配置镜像或使用本地模型 |
| 检索结果差 | chunk_size 不合适 | 调小 CHUNK_SIZE |
| LLM 调用失败 | API 密钥问题 | 检查 OPENAI_API_KEY |

### 11.2 调试技巧

```python
# 在关键位置添加日志
import logging
logger = logging.getLogger(__name__)
logger.info(f"检索返回 {len(results)} 条结果")
logger.debug(f"Context: {context[:200]}")
```

### 11.3 测试检索质量

```python
# 在 Python 中直接测试
from vector_store import VectorStoreManager

vs = VectorStoreManager()
results = vs.similarity_search_with_scores("你的问题")
for doc, score, rtype in results:
    print(f"[{rtype}] {score:.3f}: {doc.page_content[:100]}...")
```

---

## 十二、架构亮点总结

1. **两阶段检索**: 粗筛（速度快）+ 精筛（精度高），兼顾效率和效果
2. **混合检索**: 向量语义 + BM25 关键词 + RRF 融合，覆盖不同检索场景
3. **懒加载**: Embedding 模型和 Reranker 按需加载，加快启动速度
4. **流式输出**: SSE 实现打字机效果，提升用户体验
5. **配置热更新**: API 实时调整混合检索和 Reranker 参数，无需重启
6. **多格式支持**: PDF/Word/Excel/CSV/HTML/SQLite/URL 统一处理
7. **章节感知切分**: 按文档结构切分，保留语义完整性
8. **Chroma 服务模式**: 独立进程管理向量数据，支持持久化和多客户端
9. **对话历史管理**: 基于 session_id 的多轮对话支持，自动截断防溢出
10. **OpenAI 兼容接口**: LLM 层使用标准 OpenAI API，可无缝替换为任意兼容模型
