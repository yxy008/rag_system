# RAG 智能问答系统 · Python 版

基于 **LangChain + Chroma + BGE 中文模型** 构建的企业级本地 RAG（检索增强生成）智能问答系统，支持混合检索、两阶段精排、语义缓存、多会话管理等完整功能。

> 另有 **Java 版**（Spring AI + Elasticsearch + ONNX Runtime），见 [rag_system_springai](../rag_system_springai)。

---

## 目录

- [系统架构](#系统架构)
- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [API 接口](#api-接口)
- [配置参考](#配置参考)
- [技术亮点](#技术亮点)
- [Java 版对比](#java-版对比)

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
                          | (组合拳分块策略)   |
                          | 章节感知/语义/递归 |
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
          | (BAAI/bge-m3)     |        | (jieba 分词)       |
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
                    | (bge-reranker-v2-m3)        |
                    |   交叉编码器重排序             |
                    |        精筛: top 4           |
                    +-------------+---------------+
                                  |
                                  v
                    +-------------+---------------+
                    |   两级语义缓存               |
                    | 精确匹配(MD5) + 语义匹配     |
                    | (Chroma 向量相似度)          |
                    +-------------+---------------+
                                  |
                    (缓存未命中时)  v
                    +-------------+---------------+
                    |   Augmented Query           |
                    | (Context + History + Q)     |
                    +-------------+---------------+
                                  |
                                  v
                    +-------------+---------------+
                    |        LLM                  |
                    |  (GPT-4o / DeepSeek /       |
                    |   Qwen / Ollama 本地模型)    |
                    +-------------+---------------+
                                  |
                                  v
                    +-------------+---------------+
                    |   Answer (SSE 流式输出)      |
                    +-----------------------------+
```

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **两阶段检索** | 粗筛（混合检索 RRF 融合）→ 精筛（Reranker 交叉编码器），兼顾速度与精度 |
| **混合检索** | 向量语义检索 + BM25 关键词检索，RRF 算法融合排序 |
| **中文深度优化** | BGE-M3 中文 Embedding + jieba 中文分词 + BGE-Reranker-v2-M3 中文精排 |
| **组合拳分块** | 章节感知切分 + 语义边界切分 + 递归字符切分，根据文档特征自动路由 |
| **两级语义缓存** | 精确匹配（MD5）+ 语义匹配（向量相似度），大幅降低 LLM 调用成本 |
| **GPU 加速** | Embedding 和 Reranker 均支持 CUDA GPU 推理 |
| **多格式支持** | txt / pdf / docx / md / xlsx / csv / html / sqlite |
| **SSE 流式输出** | Server-Sent Events 流式问答，逐字返回，体验流畅 |
| **多会话管理** | 支持多 session 对话历史，上下文连续对话，会话重命名 |
| **用户认证** | 注册/登录/个人中心/修改密码，Token 认证 |
| **仪表盘** | 系统状态、知识库统计、缓存命中率、评估指标一站式展示 |
| **速率限制** | 基于 IP 的请求频率控制，防止滥用 |
| **LLM 重试** | 指数退避自动重试，应对 API 限流和临时故障 |
| **动态配置** | 运行时可通过 API 切换混合检索 / Reranker 开关 |
| **缓存预热** | 启动时自动加载 FAQ 预热缓存，支持 API 批量预热 |

---

## 技术栈

| 层级 | 技术 | 版本 | 说明 |
|------|------|------|------|
| **RAG 框架** | LangChain | 1.3 | 链式调用、文档加载、文本分割 |
| **Web 框架** | Flask | 3.1 | REST API + SSE 流式 + 模板渲染 |
| **向量数据库** | Chroma | 0.6 | HTTP 服务模式，持久化存储 |
| **Embedding** | BGE-M3 (BAAI) | - | 1024 维，100+ 语言，HuggingFace 推理 |
| **Reranker** | BGE-Reranker-v2-M3 | - | 交叉编码器，8192 tokens，HuggingFace 推理 |
| **关键词检索** | BM25 (rank-bm25) | 0.2 | Okapi BM25 算法 |
| **中文分词** | jieba | 0.42 | 精确模式分词，HMM 新词发现 |
| **LLM** | OpenAI 兼容接口 | - | 支持 GPT-4o / DeepSeek / Qwen / Ollama |
| **文档解析** | PyPDF / pdfplumber / python-docx / unstructured / openpyxl / BeautifulSoup | - | 多格式文档加载 |
| **用户认证** | Werkzeug Security + SQLite | - | 密码哈希 + Token 认证 |
| **对话存储** | SQLite | - | 会话历史持久化 |
| **前端** | 原生 HTML/CSS/JS | - | 响应式设计，侧边栏导航 |

---

## 快速开始

### 环境要求

- Python 3.9+
- CUDA GPU（可选，用于加速 Embedding 和 Reranker）
- Windows / Linux / macOS

### 第一步：安装依赖

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 首次运行会自动下载 Embedding 模型（`BAAI/bge-m3`，约 2.2GB）和 Reranker 模型（`BAAI/bge-reranker-v2-m3`，约 2.2GB），请确保网络畅通且磁盘空间充足。

### 第二步：配置环境变量

```bash
copy .env.example .env
```

编辑 `.env` 文件，填写必要配置：

```env
# LLM 配置（OpenAI 兼容接口）
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# Embedding 模型
EMBEDDING_MODEL_NAME=BAAI/bge-m3
EMBEDDING_DEVICE=cuda

# Chroma 服务配置
CHROMA_HOST=localhost
CHROMA_PORT=8000

# 检索配置
RETRIEVAL_TOP_K=4
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
```

**使用 Ollama 本地模型（免费）：**

```env
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.2
```

### 第三步：启动 Chroma 向量数据库

```bash
# 使用项目自带的 chroma
.venv\Scripts\chroma.exe run --path ./chroma_db --host localhost --port 8000
```

### 第四步：准备文档并入库

将知识库文档放入 `documents/` 目录，然后运行：

```bash
python ingest.py
```

### 第五步：启动服务

```bash
python app.py
```

访问 http://127.0.0.1:5000 即可使用。

---

## 项目结构

```
rag_system/
├── app.py                    # Flask Web 服务入口，所有 API 路由
├── config.py                 # 全局配置管理（.env 读取）
├── rag_chain.py              # RAG 核心链：检索 + 生成
├── vector_store.py           # 向量数据库管理（Chroma + BM25 + Reranker）
├── document_processor.py     # 文档加载 + 组合拳分块策略
├── semantic_cache.py         # 两级语义缓存（精确 + 语义匹配）
├── reranker.py               # BGE-Reranker 交叉编码器精排
├── conversation_store.py     # 对话历史 SQLite 持久化
├── evaluation.py             # 请求评估统计（命中率、延迟、Token）
├── rate_limiter.py           # 基于 IP 的速率限制
├── llm_retry.py              # LLM 调用指数退避重试
├── ingest.py                 # 命令行文档入库脚本
├── requirements.txt          # Python 依赖清单
├── .env.example              # 环境变量模板
├── cache_warmup.json         # 缓存预热 FAQ 数据
├── static/
│   ├── css/style.css         # 前端样式
│   └── js/app.js             # 前端交互逻辑
├── templates/
│   └── index.html            # 前端页面模板
├── docs/
│   └── RAG 智能问答系统.md    # 项目技术文档
└── documents/                # 知识库文档目录
```

---

## API 接口

### 问答

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 普通问答（JSON 响应） |
| POST | `/api/chat/stream` | 流式问答（SSE） |

### 知识库管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/ingest` | 触发文档入库 |
| POST | `/api/upload` | 上传单个文件并入库 |

### 对话历史

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/history/{session_id}` | 获取对话历史 |
| DELETE | `/api/history/{session_id}` | 清空对话历史 |
| GET | `/api/sessions` | 获取所有会话列表 |
| POST | `/api/sessions/{session_id}/rename` | 重命名会话 |

### 系统状态与配置

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/status` | 系统状态 |
| GET | `/api/dashboard` | 仪表盘综合统计 |
| GET | `/api/config` | 获取配置 |
| POST | `/api/config` | 更新配置（混合检索/Reranker 开关） |

### 语义缓存

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/cache/clear` | 清空缓存 |
| GET | `/api/cache/stats` | 缓存统计 |
| POST | `/api/cache/warmup` | 缓存预热 |
| GET | `/api/cache/warmup/status` | 预热状态 |

### 评估

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/evaluation` | 评估统计 |
| POST | `/api/evaluation/reset` | 重置评估 |

### 用户认证

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/register` | 用户注册 |
| POST | `/api/auth/login` | 用户登录 |
| POST | `/api/auth/logout` | 用户注销 |
| GET | `/api/auth/me` | 获取当前用户信息 |
| PUT | `/api/auth/update-profile` | 修改个人资料 |
| PUT | `/api/auth/change-password` | 修改密码 |

---

## 配置参考

完整配置项见 `.env.example`，关键配置说明：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `OPENAI_MODEL` | gpt-3.5-turbo | LLM 模型名称 |
| `EMBEDDING_MODEL_NAME` | BAAI/bge-m3 | Embedding 模型 |
| `RETRIEVAL_TOP_K` | 4 | 最终返回文档数 |
| `CHUNK_SIZE` | 1200 | 分块大小（字符） |
| `CHUNK_OVERLAP` | 200 | 分块重叠（字符） |
| `HYBRID_SEARCH_ALPHA` | 0.6 | 混合检索向量权重 |
| `RERANKER_ENABLED` | true | 是否启用 Reranker |
| `MULTI_QUERY_ENABLED` | true | 多查询融合检索 |
| `SEMANTIC_CHUNKING_ENABLED` | true | 语义分块 |
| `CACHE_ENABLED` | true | 语义缓存 |
| `CACHE_SIMILARITY_THRESHOLD` | 0.85 | 语义匹配相似度阈值 |

---

## 技术亮点

### 1. 组合拳文档分块策略

根据文档特征自动选择最优分块器：

| 文档类型 | 判断条件 | 分块器 |
|----------|----------|--------|
| 表格类 | 文件后缀（xlsx/csv/sqlite） | 按行保留 |
| 制度/合同 | 检测"第X条"等标题模式 | 章节感知切分 |
| 网页/报告 | 长文本无章节结构 | Embedding 语义切分 |
| 短文本 | 兜底 | 递归字符切分 |

### 2. 两阶段检索架构

```
粗筛（Hybrid Search）          精筛（Reranker）
  向量 top_k=12                 交叉编码器逐对打分
    +              RRF 融合  →    ↓
  BM25 top_k=12                 输出 top 4
```

### 3. 两级语义缓存

```
用户提问
  → 第一级：MD5 精确匹配（O(1)，始终可用）
  → 第二级：向量语义匹配（kNN，仅无历史时使用）
  → 缓存命中 → 跳过检索和 LLM，直接返回
```

### 4. 中文深度优化

- **BGE-M3**：BAAI 第三代多语言 Embedding，1024 维，中文效果 SOTA
- **jieba 分词**：精确模式 + HMM 新词发现，BM25 检索精度远超字符 n-gram
- **BGE-Reranker-v2-M3**：交叉编码器，逐对计算 query-document 相关性

### 5. LLM 指数退避重试

```python
# 自动重试 5 次，间隔 1s → 2s → 4s → 8s → 16s
llm_stream = retry_with_backoff(
    lambda msgs: rag.llm.stream(msgs),
    messages,
    max_retries=5,
)
```

---

## Java 版对比

| 维度 | Python 版 | Java 版 |
|------|-----------|---------|
| **框架** | Flask + LangChain | Spring Boot 3.3 + Spring AI |
| **向量数据库** | Chroma | Elasticsearch 8.15 |
| **Embedding** | BGE-M3 (HuggingFace) | BGE-M3 (ONNX Runtime) |
| **Reranker** | BGE-Reranker-v2-M3 (HF) | BGE-Reranker-v2-M3 (ONNX) |
| **关键词检索** | BM25 (rank-bm25) | BM25 (ES 内置) |
| **中文分词** | jieba | ES 标准分词器 |
| **数据库** | SQLite | MySQL + JPA |
| **前端** | Flask 模板 | Thymeleaf |
| **API 接口** | 完全一致 | 完全一致 |
| **功能** | 完全一致 | 完全一致 |

两个版本功能完全对等，API 接口一致，前端共用同一套 HTML/CSS/JS。选择建议：

- **Python 版**：适合 AI/ML 团队，Python 生态丰富，模型加载方便
- **Java 版**：适合企业级 Java 团队，Spring 生态成熟，ONNX 推理性能更优

---

## License

MIT License