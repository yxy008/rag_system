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
BASE_DIR = Path(__file__).parent

# ========== LLM 配置 ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# ========== Embedding 模型配置 ==========
# BGE-M3: BAAI 第三代多语言 Embedding 模型
#   - 支持 100+ 语言，8192 tokens 输入长度
#   - 原生支持稠密+稀疏混合向量
#   - 可通过 HuggingFace ID 自动下载，或指定本地路径
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda")

# ========== Chroma 向量数据库配置（服务模式） ==========
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")

# ========== 文档目录 ==========
DOCUMENTS_DIR = str(BASE_DIR / os.getenv("DOCUMENTS_DIR", "documents"))

# ========== 检索配置 ==========
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
# HYBRID_SEARCH_ALPHA: 混合检索中向量权重（0.0=纯BM25, 1.0=纯向量）
# 推荐值 0.5~0.7（向量权重略高，因为语义理解更强）
HYBRID_SEARCH_ALPHA = float(os.getenv("HYBRID_SEARCH_ALPHA", "0.6"))
# CHUNK_SIZE: 每个切片最大字符数（BGE-M3 支持 8192 tokens，可适当增大）
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
# CHUNK_OVERLAP: 相邻切片重叠字符数（200 防止信息被硬切断）
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

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
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "4"))
# Reranker 粗筛候选数：精排前多取一些，确保不漏掉好结果（建议 8~16）
RERANKER_CANDIDATE_K = int(os.getenv("RERANKER_CANDIDATE_K", "12"))
# 是否默认启用 Reranker（True=两阶段检索，False=纯混合检索）
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"

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
