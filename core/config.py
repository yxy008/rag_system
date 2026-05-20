"""
config.py - 全局配置管理
读取 .env 文件中的配置项，提供统一的配置入口
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ========== 项目根目录 ==========
BASE_DIR = Path(__file__).parent.parent

# ========== LLM 配置 ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# ========== RAGAS 评估专用 LLM 配置 ==========
# 评估使用轻量模型降低成本（gpt-4o-mini 足够胜任评判任务）
EVAL_API_KEY = os.getenv("EVAL_API_KEY", "")
EVAL_BASE_URL = os.getenv("EVAL_BASE_URL", "https://api.openai.com/v1")
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "deepseek-v4-flash")
EVAL_EMBEDDING_MODEL = os.getenv("EVAL_EMBEDDING_MODEL", "BAAI/bge-m3")

# ========== Embedding 模型配置 ==========
# BGE-M3: BAAI 第三代多语言 Embedding 模型
#   - 支持 100+ 语言，8192 tokens 输入长度
#   - 原生支持稠密+稀疏混合向量
#   - 可通过 HuggingFace ID 自动下载，或指定本地路径
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")  # 若 PyTorch 为 CPU 版则自动回退

# ========== 向量数据库后端配置 ==========
# 支持的后端：chroma（默认）, faiss, milvus
# Chroma：需要先启动 chroma 服务（.venv\Scripts\chroma.exe run --path ./chroma_db）
# Faiss：纯内存/本地索引，无需外部服务，适合小规模数据快速验证
# Milvus：高性能分布式向量数据库，适合大规模生产环境
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "chroma")

# ========== Chroma 向量数据库配置（服务模式） ==========
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

# ========== Milvus 向量数据库配置 ==========
# 启动方式：docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_documents")
MILVUS_DIMENSION = int(os.getenv("MILVUS_DIMENSION", "1024"))
# 索引类型：IVF_FLAT（推荐，适合百万级）, HNSW（高性能，内存消耗大）, FLAT（精确搜索，适合 < 10万）
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")
# 相似度度量：COSINE（余弦相似度）, IP（内积）, L2（欧氏距离）
MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "COSINE")

# ========== Faiss 向量数据库配置（本地内存索引） ==========
FAISS_PERSIST_DIR = os.getenv("FAISS_PERSIST_DIR", None)

# ========== 文档目录 ==========
DOCUMENTS_DIR = str(BASE_DIR / os.getenv("DOCUMENTS_DIR", "documents"))

# ========== 检索配置 ==========
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "6"))
# HYBRID_SEARCH_ALPHA: 混合检索中向量权重（0.0=纯BM25, 1.0=纯向量）
# 推荐值 0.5~0.7（向量权重略高，因为语义理解更强）
HYBRID_SEARCH_ALPHA = float(os.getenv("HYBRID_SEARCH_ALPHA", "0.6"))
# CHUNK_SIZE: 每个切片最大字符数（BGE-M3 支持 8192 tokens，可适当增大）
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
# CHUNK_OVERLAP: 相邻切片重叠字符数（200 防止信息被硬切断）
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ========== 向量检索粗筛配置 ==========
# 向量检索最低 COSINE 相似度阈值（0~1），低于此值的文档将被过滤
# Milvus COSINE 度量下 search 返回的是余弦相似度 [0, 1]，直接与阈值比较
# 0.55 确保过滤掉明显不相关的结果，同时不会杀太狠导致候选不足
VECTOR_COSINE_MIN_THRESHOLD = float(os.getenv("VECTOR_COSINE_MIN_THRESHOLD", "0.55"))

# ========== 多查询融合检索配置 ==========
# 是否启用多查询融合检索（LLM 生成多个查询变体，合并检索结果）
MULTI_QUERY_ENABLED = os.getenv("MULTI_QUERY_ENABLED", "true").lower() == "true"
# 生成的查询变体数量（建议 2~3，太多会增加延迟）
MULTI_QUERY_COUNT = int(os.getenv("MULTI_QUERY_COUNT", "3"))

# ========== 语义分块配置 ==========
# 是否启用语义分块（基于 Embedding 相似度切分，替代规则分块）
SEMANTIC_CHUNKING_ENABLED = os.getenv("SEMANTIC_CHUNKING_ENABLED", "true").lower() == "true"
# 语义分块相似度分位数阈值（越低切得越细，推荐 85~95）
SEMANTIC_CHUNKING_PERCENTILE = float(os.getenv("SEMANTIC_CHUNKING_PERCENTILE", "90.0"))

# ========== Reranker 配置 ==========
# 模型：BAAI/bge-reranker-v2-m3（可通过 HuggingFace ID 自动下载，或指定本地路径）
# bge-reranker-v2-m3: 与 BGE-M3 Embedding 配套，8192 tokens，100+ 语言
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
# Reranker 精排后返回的数量（通常与 RETRIEVAL_TOP_K 一致）
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "6"))
# Reranker 粗筛候选数：精排前多取一些，确保不漏掉好结果（建议 8~30）
RERANKER_CANDIDATE_K = int(os.getenv("RERANKER_CANDIDATE_K", "40"))
# 是否默认启用 Reranker（True=两阶段检索，False=纯混合检索）
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
# Reranker 相关度阈值（0~100），低于此分数的结果将被过滤
# BGE-reranker 分数分布：>80 高度相关，60~80 中度相关，<60 弱相关
# 设为 75 可过滤掉部分相关的结果，确保传给 LLM 的都是高质量上下文
# 极端情况：若所有结果均低于阈值，则回退到不过滤，返回 top_k 条
RERANKER_RELEVANCE_THRESHOLD = float(os.getenv("RERANKER_RELEVANCE_THRESHOLD", "75"))

# ========== Flask 服务器配置 ==========
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

# ========== System Prompt ==========
SYSTEM_PROMPT = """你是公司的办事助手，有关员工的休假、报销、办公用品申请等各类事项都由你来负责答复。

## 工作原则

1. 回答逻辑清晰、内容全面，按步骤讲解
2. 与用户问题有关的信息要全面回复，不要有遗漏（如有效期、提前天数、所需材料等特别规定）
3. 优先按照文件知识库中检索到的内容回复，充分利用检索到的上下文，合理解读并综合回答
4. 只有当参考文档中**完全没有涉及**用户提问的内容时，才告知规定中暂未明确，并建议请示直属上级
5. 如果用户询问的内容与休假、报销、办公用品都不相关，则不需要查询文件知识库

## 回答格式要求

1. **必须标注引用来源**：每个关键信息点后面用方括号标注来源，格式为 `【来源：文件名】`
2. **结构化输出**：按以下结构组织回答：
   - 先给出核心结论（一句话总结）
   - 再分点详述具体规定
   - 最后给出操作建议或注意事项
3. **不确定时明确说明**：如果某条信息在文档中没有明确规定，要明确指出，不要编造

## 回答示例

用户问："年假怎么申请？"

正确回答格式：
```
【核心结论】员工申请年假需要提前3个工作日提交申请，经直属上级审批通过后方可休假。

【具体规定】
1. 申请时间：需提前3个工作日提交申请【来源：休假管理制度.pdf】
2. 申请方式：通过OA系统提交年假申请单【来源：休假管理制度.pdf】
3. 审批流程：直属上级审批 → 部门负责人审批（5天以上）【来源：休假管理制度.pdf】
4. 年假天数：入职满1年享5天，每增加1年增加1天，上限15天【来源：员工手册.pdf】

【注意事项】
- 年假最小请假单位为半天
- 当年度未休完的年假可延期至次年3月31日
```
"""

# ========== RAG Prompt 模板 ==========
RAG_PROMPT_TEMPLATE = """以下是从知识库中检索到的参考文档：

{context}

---
【重要指令】
1. 请充分利用上述参考文档中提供的信息来回答问题
2. 即使没有完全匹配的条款，也要综合相关上下文给出合理的回答
3. 每个关键信息点后面必须用方括号标注来源文件名，格式为 `【来源：文件名】`
4. 按"核心结论 → 具体规定 → 注意事项"的结构组织回答
5. 只有当参考文档中完全没有涉及问题时，才说明"规定中暂未明确，建议请示直属上级"

请回答问题：
{question}
"""

# ========== 语义缓存配置 ==========
# 是否启用语义缓存（True=启用，False=关闭）
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
# 语义相似度阈值（0.0~1.0，越高越严格，推荐 0.92~0.97）
CACHE_SIMILARITY_THRESHOLD = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))
# 最大缓存条目数（超出后触发 LFU 淘汰）
CACHE_MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "1000"))
# 缓存有效期（小时，超时自动失效）
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))
# 缓存向量粗筛相似度阈值（COSINE，用于向量检索阶段过滤候选）
# 0.70 在确保不遗漏语义近似候选的同时，控制精排阶段的候选数量
CACHE_COARSE_THRESHOLD = float(os.getenv("CACHE_COARSE_THRESHOLD", "0.70"))
# 缓存 Reranker 精排阈值（已废弃，缓存已改为单阶段向量相似度匹配，不再使用 Reranker）
# 保留此配置项仅为向后兼容，实际缓存匹配仅使用 CACHE_COARSE_THRESHOLD
CACHE_RERANKER_THRESHOLD = float(os.getenv("CACHE_RERANKER_THRESHOLD", "0.70"))
# 缓存向量检索候选数（向量检索取 N 条候选，取相似度最高者作为命中）
CACHE_CANDIDATE_COUNT = int(os.getenv("CACHE_CANDIDATE_COUNT", "10"))
# 缓存 Collection 专用索引类型（独立于主知识库）
# HNSW 对小规模数据（< 2000 条）召回率极高，适合缓存场景
MILVUS_CACHE_INDEX_TYPE = os.getenv("MILVUS_CACHE_INDEX_TYPE", "HNSW")


def get_model_display_name(model_path: str) -> str:
    """
    从模型路径/ID 中提取可读的模型名称

    支持多种格式：
      - 本地相对路径：./models/bge-m3 → bge-m3
      - 本地绝对路径：E:\\models\\bge-m3 → bge-m3
      - HuggingFace ID：BAAI/bge-m3 → bge-m3
      - 纯模型名：text-embedding-3-small → text-embedding-3-small
    """
    if not model_path:
        return "未知"
    normalized = model_path.replace("\\", "/")
    name = normalized.rstrip("/").split("/")[-1]
    return name if name else model_path
