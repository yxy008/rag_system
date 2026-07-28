# RAG 智能问答系统 · Python 版

基于 **LangChain + Chroma / Faiss / Milvus + BGE 中文模型** 构建的企业级本地 RAG（检索增强生成）智能问答系统，支持多查询融合检索、轻量动态检索、两阶段精排、分层上下文架构、语义缓存、多会话管理等完整功能。

> 另有 **Java 版**（Spring AI + Elasticsearch + ONNX Runtime），见 [rag_system_springai](https://github.com/yxy008/rag_system_springai.git)。

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
          | Vector DB          |        | Keyword Index     |
          | Chroma/Faiss/Milvus|        | (pickle 持久化)    |
          +---------+---------+        +---------+---------+
                    |                             |
                    +-------------+---------------+
                                  |
                    (用户提问)     v
                    +-------------+---------------+
                    |   轻量动态检索               |
                    | 根据问题特征动态调整 alpha     |
                    | (精确术语→偏BM25, 口语→偏向量)|
                    +-------------+---------------+
                                  |
                                  v
                    +-------------+---------------+
                    |   多查询融合检索（可选）       |
                    | LLM 生成 2~3 个查询变体      |
                    | 分别检索 → RRF 融合去重      |
                    +-------------+---------------+
                                  |
                                  v
                    +-------------+---------------+
                    |   混合检索 (RRF 融合)         |
                    | 向量检索 + BM25 关键词检索    |
                    |  COSINE 最低相似度过滤       |
                    |  粗筛: max(candidate_k,      |
                    |        top_k*2) 条候选       |
                    +-------------+---------------+
                                  |
                                  v
                    +-------------+---------------+
                    |   Reranker 精排              |
                    | (bge-reranker-v2-m3)        |
                    |   交叉编码器重排序             |
                    |   相关度阈值过滤 + 回退机制    |
                    |   精筛: top_k 条             |
                    +-------------+---------------+
                                  |
                                  v
                    +-------------+---------------+
                    |   检索结果缓存（LRU）          |
                    | (query, top_k, hybrid,       |
                    |  reranker) 缓存 key          |
                    +-------------+---------------+
                                  |
                                  v
                    +-------------+---------------+
                    |   两级语义缓存               |
                    | 精确匹配(MD5) + 语义匹配     |
                    | (向量相似度, COSINE 阈值)    |
                    +-------------+---------------+
                                  |
                    (缓存未命中时)  v
                    +-------------+---------------+
                    |   分层上下文构建             |
                    | L1: System Prompt           |
                    | L2: 检索结果(system role)   |
                    | L3: 对话历史                |
                    | L4: 当前问题                |
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
| **两阶段检索** | 粗筛（混合检索 RRF 融合，含 COSINE 最低相似度过滤）→ 精筛（Reranker 交叉编码器 + 相关度阈值过滤 + 回退机制） |
| **多查询融合检索** | LLM 生成 2~3 个查询变体，分别检索后 RRF 融合去重，大幅提升复杂语义问题的召回率 |
| **轻量动态检索** | 零额外 LLM 调用，根据问题特征自动调整混合检索权重 alpha（精确术语偏 BM25，口语化偏向量） |
| **混合检索** | 向量语义检索 + BM25 关键词检索，RRF 算法融合排序，alpha 动态可调 |
| **中文深度优化** | BGE-M3 中文 Embedding + jieba 中文分词 + BGE-Reranker-v2-M3 中文精排 |
| **组合拳分块** | 章节感知切分 + 语义边界切分 + 递归字符切分，根据文档特征自动路由 |
| **两级语义缓存** | 精确匹配（MD5）+ 语义匹配（向量相似度 + COSINE 阈值过滤），大幅降低 LLM 调用成本 |
| **分层上下文架构** | System Prompt → 检索结果 → 对话历史 → 当前问题，分层注入，LLM 可区分参考信息与用户输入 |
| **多向量数据库** | 支持 Chroma / Faiss / Milvus 三种后端，通过环境变量一键切换 |
| **GPU 加速** | Embedding 和 Reranker 均支持 CUDA GPU 推理 |
| **多格式支持** | txt / pdf / docx / md / xlsx / csv / html / sqlite |
| **SSE 流式输出** | Server-Sent Events 流式问答，逐字返回，体验流畅 |
| **多会话管理** | 支持多 session 对话历史，上下文连续对话，会话重命名 |
| **用户认证** | 注册/登录/个人中心/修改密码，Token 认证 |
| **仪表盘** | 系统状态、知识库统计、缓存命中率、评估指标一站式展示 |
| **速率限制** | 基于 IP 的请求频率控制，防止滥用 |
| **LLM 重试** | 指数退避自动重试，应对 API 限流和临时故障 |
| **动态配置** | 运行时可通过 API 切换混合检索 / Reranker / 多查询融合 开关 |
| **缓存预热** | 启动时自动加载 FAQ 预热缓存，支持 API 批量预热 |
| **检索缓存（LRU）** | 对相同检索参数缓存结果，入库/清空时自动失效，减少重复计算 |

---

## 技术栈

| 层级 | 技术 | 版本 | 说明 |
|------|------|------|------|
| **RAG 框架** | LangChain | 1.3 | 链式调用、文档加载、文本分割 |
| **Web 框架** | Flask | 3.1 | REST API + SSE 流式 + 模板渲染 |
| **向量数据库** | Chroma / Faiss / Milvus | - | 多后端支持，Chroma HTTP 服务 / Faiss 本地内存 / Milvus 分布式 |
| **Embedding** | BGE-M3 (BAAI) | - | 1024 维，100+ 语言，8192 tokens，HuggingFace 推理 |
| **Reranker** | BGE-Reranker-v2-M3 | - | 交叉编码器，8192 tokens，HuggingFace 推理 |
| **关键词检索** | BM25 (rank-bm25) | 0.2 | Okapi BM25 算法，支持 pickle 本地持久化 |
| **中文分词** | jieba | 0.42 | 精确模式分词，HMM 新词发现 |
| **LLM** | OpenAI 兼容接口 | - | 支持 GPT-4o / DeepSeek / Qwen / Ollama |
| **文档解析** | PyPDF / pdfplumber / python-docx / openpyxl / BeautifulSoup | - | 多格式文档加载 |
| **用户认证** | Werkzeug Security + SQLite | - | 密码哈希 + Token 认证 |
| **对话存储** | SQLite | - | 会话历史持久化 |
| **前端** | 原生 HTML/CSS/JS | - | 响应式设计，侧边栏导航 |

---

## 快速开始

### 环境要求

- Python 3.9+
- CUDA GPU（可选，用于加速 Embedding 和 Reranker）
- Windows / Linux / macOS
- 向量数据库后端三选一：
  - **Chroma**（默认）：需启动 Chroma 服务
  - **Faiss**：纯本地内存索引，无需外部服务，适合小规模快速验证
  - **Milvus**：需启动 Docker Milvus 服务，适合大规模生产环境

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

# 向量数据库后端（chroma / faiss / milvus）
VECTOR_STORE_BACKEND=chroma

# Chroma 服务配置
CHROMA_HOST=localhost
CHROMA_PORT=8000

# 检索配置
RETRIEVAL_TOP_K=6
CHUNK_SIZE=600
CHUNK_OVERLAP=200

# 多查询融合检索
MULTI_QUERY_ENABLED=true
MULTI_QUERY_COUNT=3

# Reranker 配置
RERANKER_ENABLED=true
RERANKER_CANDIDATE_K=40
RERANKER_RELEVANCE_THRESHOLD=75

# 缓存配置
CACHE_SIMILARITY_THRESHOLD=0.95
CACHE_COARSE_THRESHOLD=0.70
```

**使用 Ollama 本地模型（免费）：**

```env
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.2
```

### 第三步：启动向量数据库

根据 `VECTOR_STORE_BACKEND` 选择对应的启动方式：

**Chroma（默认）：**

```bash
# 使用项目自带的 chroma
.venv\Scripts\chroma.exe run --path ./chroma_db --host localhost --port 8000
```

**Faiss（无需外部服务）：**

设置环境变量即可，无需额外启动服务：
```env
VECTOR_STORE_BACKEND=faiss
```

**Milvus（Docker）：**

```bash
docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
```
然后配置：
```env
VECTOR_STORE_BACKEND=milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530
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
├── app.py                        # Flask Web 服务入口，所有 API 路由
├── config.py                     # 全局配置管理（.env 读取）
├── core/                         # 核心模块
│   ├── config.py                 # 配置管理模块
│   ├── vector_store.py           # 向量数据库管理（Chroma / Faiss / Milvus + BM25 + Reranker）
│   ├── rag_chain.py              # RAG 核心链：检索 + 生成（分层上下文 + 动态检索）
│   ├── document_processor.py     # 文档加载 + 组合拳分块策略
│   ├── semantic_cache.py         # 两级语义缓存（精确 + 语义匹配，支持多后端）
│   ├── reranker.py               # BGE-Reranker-v2-M3 交叉编码器精排 + 阈值过滤
│   ├── llm_retry.py              # LLM 调用指数退避重试
│   ├── rate_limiter.py           # 基于 IP 的速率限制
│   └── api_key_manager.py        # API Key 管理
├── services/                     # 业务服务层
│   ├── evaluation.py             # 请求评估统计（命中率、延迟、Token）
│   ├── ragas_evaluation.py       # RAGAS 评估
│   ├── confidence.py             # 回答置信度评估
│   ├── knowledge_graph.py        # 知识图谱
│   ├── knowledge_health.py       # 知识库健康检查
│   ├── user_profile.py           # 用户画像
│   └── collaboration.py          # 协作功能
├── storage/                      # 存储层
│   └── conversation_store.py     # 对话历史 SQLite 持久化
├── routes/                       # API 路由
│   └── openapi.py                # OpenAPI 路由定义
├── scripts/                      # 工具脚本
│   ├── ingest.py                 # 命令行文档入库脚本
│   ├── download_reranker.py      # Reranker 模型下载
│   ├── generate_test_data.py     # 测试数据生成
│   └── ...
├── data/                         # 数据文件
│   ├── cache_warmup.json         # 缓存预热 FAQ 数据
│   └── evaluation_dataset.json   # 评估数据集
├── static/                       # 前端静态资源
│   ├── css/style.css             # 前端样式
│   └── js/app.js                 # 前端交互逻辑
├── templates/
│   └── index.html                # 前端页面模板
├── requirements.txt              # Python 依赖清单
├── .env.example                  # 环境变量模板
└── documents/                    # 知识库文档目录
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
| `VECTOR_STORE_BACKEND` | chroma | 向量数据库后端：chroma / faiss / milvus |
| `RETRIEVAL_TOP_K` | 6 | 最终返回文档数 |
| `CHUNK_SIZE` | 600 | 分块大小（字符） |
| `CHUNK_OVERLAP` | 200 | 分块重叠（字符） |
| `HYBRID_SEARCH_ALPHA` | 0.6 | 混合检索向量权重（运行时根据问题特征动态调整） |
| `VECTOR_COSINE_MIN_THRESHOLD` | 0.55 | 向量粗筛 COSINE 最低相似度阈值（低于此值过滤） |
| `MULTI_QUERY_ENABLED` | true | 是否启用多查询融合检索（LLM 生成查询变体） |
| `MULTI_QUERY_COUNT` | 3 | 生成的查询变体数量 |
| `RERANKER_ENABLED` | true | 是否启用 Reranker 精排 |
| `RERANKER_CANDIDATE_K` | 40 | Reranker 粗筛候选数 |
| `RERANKER_TOP_K` | 6 | Reranker 精排后返回数量 |
| `RERANKER_RELEVANCE_THRESHOLD` | 75 | Reranker 相关度阈值（0~100，低于此值过滤，极端情况回退） |
| `SEMANTIC_CHUNKING_ENABLED` | true | 语义分块 |
| `CACHE_ENABLED` | true | 语义缓存 |
| `CACHE_SIMILARITY_THRESHOLD` | 0.95 | 语义缓存总相似度阈值（主要用于向后兼容） |
| `CACHE_COARSE_THRESHOLD` | 0.70 | 缓存语义匹配粗筛 COSINE 阈值（实际生效的匹配阈值） |
| `CACHE_MAX_ENTRIES` | 1000 | 最大缓存条目数 |
| `CACHE_TTL_HOURS` | 24 | 缓存有效期（小时） |

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

### 2. 多阶段检索架构

```
动态检索（Dynamic Alpha）        多查询融合（Multi-Query）
  问题特征分析                       LLM 生成 2~3 个查询变体
  精确术语 → 偏 BM25                 分别检索 → RRF 融合去重
  口语化   → 偏向量                  ↓
  alpha ∈ [0.15, 0.9]           提升复杂语义召回率

粗筛（Hybrid Search）          精筛（Reranker）
  向量检索                        交叉编码器逐对打分
    +              RRF 融合  →    相关度阈值过滤（<75分淘汰）
  BM25 关键词                      ↓
  COSINE 最低相似度过滤            极端情况回退（全低于阈值）
  candidate_k=max(40, top_k*2)    输出 top_k 条
```

### 3. 两级语义缓存

```
用户提问
  → 第一级：MD5 精确匹配（O(1)，始终可用）
  → 第二级：向量语义匹配（kNN + COSINE 粗筛阈值 0.70）
  → 缓存命中 → 跳过检索和 LLM，直接返回
  → 缓存策略：LFU 淘汰 + TTL 过期 + 入库自动清空
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

### 6. 轻量动态检索（Dynamic Alpha）

零额外 LLM 调用，根据问题特征自动调节混合检索权重：

| 问题特征 | 调整方向 | 幅度 |
|----------|----------|------|
| 精确术语/数字（如"A01"、"500元"） | 偏 BM25 | -0.25 |
| 引号/书名号精确引用 | 偏 BM25 | -0.15 |
| 口语化/语义查询（"能不能"、"怎么"） | 偏向量 | +0.10 |
| 问题长度 > 30 字 | 偏向量（语义理解） | +0.08 |
| 问题长度 < 8 字 | 偏 BM25（精确查询） | -0.10 |

alpha 动态范围：[0.15, 0.9]

### 7. 分层上下文架构（Layered Context）

与传统将 context + history 拼接为单一字符串不同，采用分层消息结构：

```
L1 [System]     角色规则        ← 最稳定，缓存命中率最高
L2 [System]     检索结果        ← 以 system role 注入，LLM 可区分参考信息 vs 用户输入
L3 [History]    对话历史        ← 原生对话格式（HumanMessage/AIMessage）
L4 [Human]      当前问题        ← 最不稳定，放最后
```

### 8. 多后端向量数据库

通过 `VECTOR_STORE_BACKEND` 环境变量一键切换：

| 后端 | 适用场景 | 特点 |
|------|----------|------|
| **Chroma** | 默认，中等规模 | HTTP 服务模式，持久化存储 |
| **Faiss** | 小规模快速验证 | 纯本地内存索引，无需外部服务 |
| **Milvus** | 大规模生产环境 | 分布式高性能，十亿级向量检索 |

---

## Java 版对比

| 维度 | Python 版 | Java 版 |
|------|-----------|---------|
| **框架** | Flask + LangChain | Spring Boot 3.3 + Spring AI |
| **向量数据库** | Chroma / Faiss / Milvus | Elasticsearch 8.15 |
| **Embedding** | BGE-M3 (HuggingFace) | BGE-M3 (ONNX Runtime) |
| **Reranker** | BGE-Reranker-v2-M3 (HF) | BGE-Reranker-v2-M3 (ONNX) |
| **关键词检索** | BM25 (rank-bm25) | BM25 (ES 内置) |
| **中文分词** | jieba | ES 标准分词器 |
| **数据库** | SQLite | MySQL + JPA |
| **前端** | Flask 模板 | Thymeleaf |
| **API 接口** | 完全一致 | 完全一致 |
| **功能** | 基本一致（Python 版功能更丰富） | 基本一致 |

两个版本功能完全对等，API 接口一致，前端共用同一套 HTML/CSS/JS。选择建议：

- **Python 版**：适合 AI/ML 团队，Python 生态丰富，模型加载方便
- **Java 版**：适合企业级 Java 团队，Spring 生态成熟，ONNX 推理性能更优

---

## License

MIT License
