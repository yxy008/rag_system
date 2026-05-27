# -*- coding: utf-8 -*-
"""
基于原始文档生成高质量评测数据集
问题质量改进版
"""
import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
OUTPUT_FILE = BASE_DIR / "data" / "evaluation_dataset.json"

# 高质量问答对 - 直接从文档提取
HIGH_QUALITY_QAP = {
    # === 公司假期制度 - 高质量问题 ===
    "leave_policy": [
        {
            "input": "员工每年可以请多少天事假？有什么限制？",
            "expected_output": "员工每年最多可申请10天事假。事假为无薪假期，须提前1天申请，由部门主管审批。超过10天需经部门经理审批。",
            "context": "公司假期制度",
            "tags": ["假期制度", "事假", "事实性问题", "高频问题"]
        },
        {
            "input": "年假是怎么计算的？不同司龄有什么区别？",
            "expected_output": "年假为带薪假期，按司龄分级：工作1-10年可享受5天年假；工作10-20年可享受10天年假；工作20年以上可享受15天年假。",
            "context": "公司假期制度",
            "tags": ["假期制度", "年假", "计算性问题", "核心问题"]
        },
        {
            "input": "请事假工资怎么算？是带薪的吗？",
            "expected_output": "事假为无薪假期。因个人事务需要请假，申请流程：在飞书中填写'事假申请'表，注明请假原因、起止日期及联系人。",
            "context": "公司假期制度",
            "tags": ["假期制度", "事假", "事实性问题", "常见问题"]
        },
        {
            "input": "病假期间工资怎么发？超过6个月还有工资吗？",
            "expected_output": "病假期间工资按国家规定支付：一般病假不低于基本工资的80%；病假天数累计超过6个月的，按不低于基本工资的70%发放。",
            "context": "公司假期制度",
            "tags": ["假期制度", "病假", "工资", "核心问题"]
        },
        {
            "input": "短期病假和长期病假有什么区别？需要什么证明材料？",
            "expected_output": "短期病假指3天以内的病假，需填写病假申请表并附上病假证明；长期病假指超过3天的病假，需提供医院出具的详细病假条和病历复印件。",
            "context": "公司假期制度",
            "tags": ["假期制度", "病假", "流程问题", "常见问题"]
        },
        {
            "input": "年假最晚可以什么时候用完？跨年怎么处理？",
            "expected_output": "年假需在当年度内使用，未使用的年假公司不予累计至下一年。但可在特殊情况经审批后延至次年3月底前使用。",
            "context": "公司假期制度",
            "tags": ["假期制度", "年假", "有效期", "常见问题"]
        },
        {
            "input": "晚婚假有多少天？申请条件是什么？",
            "expected_output": "婚假为带薪假期。员工享有3天基础婚假；符合晚婚年龄（男性25岁、女性23岁）的员工享有额外7天晚婚假，总共10天。申请需提前1个月在飞书填写申请表，并附上结婚证复印件，由人力资源部审批。婚假应在结婚登记后1年内使用，逾期视为自动放弃。",
            "context": "公司假期制度",
            "tags": ["假期制度", "婚假", "晚婚假", "核心问题"]
        },
        {
            "input": "陪产假有多少天？需要什么材料？",
            "expected_output": "陪产假为10天带薪假期。申请需提供结婚证及出生证明复印件。",
            "context": "公司假期制度",
            "tags": ["假期制度", "陪产假", "流程问题", "常见问题"]
        },
        {
            "input": "产假有多少天？难产或多胞胎会增加假期吗？",
            "expected_output": "产假为带薪假期。女员工享有98天基础产假。难产可增加15天；多胞胎每多一胎增加15天。产假申请需在飞书提交申请表，并附上医院产前检查证明。",
            "context": "公司假期制度",
            "tags": ["假期制度", "产假", "计算性问题", "核心问题"]
        },
        {
            "input": "丧假有多少天？适用于哪些亲属？",
            "expected_output": "丧假为带薪假期，适用于员工直系亲属（父母、配偶、子女）去世。假期天数为3天带薪丧假。申请需在飞书中填写申请表，并提交相关证明材料，审批部门为人力资源部。丧假需在亲属去世后1个月内申请。",
            "context": "公司假期制度",
            "tags": ["假期制度", "丧假", "核心问题", "常见问题"]
        },
        {
            "input": "调休是怎么计算的？有效期是多久？",
            "expected_output": "调休为带薪假期，由加班工时转换而来。加班工时的调休应在6个月内使用，未使用视为自动放弃。申请需在飞书中填写申请表，注明调休日期及调休时长，由部门主管审批。",
            "context": "公司假期制度",
            "tags": ["假期制度", "调休", "有效期", "常见问题"]
        },
        {
            "input": "所有假期申请都必须通过飞书吗？线下提交可以吗？",
            "expected_output": "所有假期的申请和审批流程均通过飞书完成。任何线下提交的假期申请将不予受理。",
            "context": "公司假期制度",
            "tags": ["假期制度", "流程问题", "常见问题"]
        },
        {
            "input": "婚假应该在什么时候申请？过期了还能请吗？",
            "expected_output": "婚假应在结婚登记后1年内使用，逾期视为自动放弃。申请需提前1个月在飞书填写申请表，并附上结婚证复印件，由人力资源部审批。",
            "context": "公司假期制度",
            "tags": ["假期制度", "婚假", "有效期", "常见问题"]
        },
        {
            "input": "哪些假期是带薪的？哪些是无薪的？",
            "expected_output": "带薪假期包括：年假、婚假、丧假、产假、陪产假、调休。无薪假期为事假。病假期间按国家规定支付病假工资。",
            "context": "公司假期制度",
            "tags": ["假期制度", "带薪假期", "综合问题", "常见问题"]
        },
        {
            "input": "请病假需要哪些人审批？流程是什么？",
            "expected_output": "所有病假均需部门主管及人力资源部审批。短期病假（3天内）需填写病假申请表并附上病假证明；长期病假（超过3天）需提供医院出具的详细病假条和病历复印件。",
            "context": "公司假期制度",
            "tags": ["假期制度", "病假", "审批流程", "常见问题"]
        },
    ],
    
    # === RAG评估体系 - 高质量问题 ===
    "rag_evaluation": [
        {
            "input": "RAGAS的Faithfulness指标是什么？如何计算？",
            "expected_output": "Faithfulness（忠实度）衡量生成的答案中有多少内容是可以从检索到的上下文中推导出来的，用于检测LLM是否编造信息。计算方式：从答案中提取声明，逐一判断每个声明是否能从上下文中推导出来，分数=能推导的声明数/总声明数。",
            "context": "RAG评估体系",
            "tags": ["RAG评估", "RAGAS", "Faithfulness", "核心概念"]
        },
        {
            "input": "RAGAS有哪些核心指标？它们分为哪几类？",
            "expected_output": "RAGAS八大核心指标分为三类。生成质量指标：Faithfulness（忠实度）、Answer Relevancy（答案相关性）、Answer Correctness（答案正确性）、Answer Semantic Similarity（答案语义相似度）；检索质量指标：Context Precision（上下文精确度）、Context Recall（上下文召回率）、Context Relevancy（上下文相关性）、Context Entity Recall（实体召回率）。",
            "context": "RAG评估体系",
            "tags": ["RAG评估", "RAGAS", "核心指标", "核心概念"]
        },
        {
            "input": "Context Precision和Context Recall有什么区别？",
            "expected_output": "Context Precision（上下文精确度）衡量检索到的文档中有多少是真正与问题相关的，关注检索结果的信噪比，考虑排序位置；Context Recall（上下文召回率）衡量ground truth中的信息有多少能在检索到的文档中找到，关注检索是否漏掉关键信息。",
            "context": "RAG评估体系",
            "tags": ["RAG评估", "RAGAS", "检索质量", "核心概念"]
        },
        {
            "input": "Answer Correctness如何计算？需要什么数据？",
            "expected_output": "Answer Correctness（答案正确性）综合语义相似度和事实正确性两个维度。计算公式：Answer Correctness = w1 × 语义相似度 + w2 × 事实正确性，默认权重各0.5。需要ground truth（标准答案）才能计算。",
            "context": "RAG评估体系",
            "tags": ["RAG评估", "RAGAS", "Answer Correctness", "计算方法"]
        },
        {
            "input": "Context Entity Recall适用于什么场景？",
            "expected_output": "Context Entity Recall（实体召回率）衡量ground truth中提到的关键实体（人名、地名、数字、日期等）有多少在检索到的文档中出现，特别适合评估RAG系统对事实性信息的覆盖能力。",
            "context": "RAG评估体系",
            "tags": ["RAG评估", "RAGAS", "实体召回", "核心概念"]
        },
        {
            "input": "评估数据集应该具备哪些特征？",
            "expected_output": "好的评估数据集应具备：1)覆盖全面（覆盖所有文档类型、查询类型、难度）；2)标注准确（ground truth经过人工验证）；3)规模适中（太少统计不显著，太多成本太高）；4)可复现（固定数据集，每次评估用同一批数据）。",
            "context": "RAG评估体系",
            "tags": ["RAG评估", "数据集构建", "核心概念"]
        },
        {
            "input": "RAGAS为什么用LLM做评判而不是字符串匹配？",
            "expected_output": "RAGAS设计理念包括：1)不需要人工标注ground truth，大部分指标只需要问题、检索上下文和生成答案；2)用LLM做评判，利用LLM的理解能力来判断答案质量，而不是简单的字符串匹配；3)指标相互独立，每个指标衡量一个维度，组合起来形成完整画像。",
            "context": "RAG评估体系",
            "tags": ["RAG评估", "RAGAS", "设计理念"]
        },
        {
            "input": "如何提升Faithfulness分数？",
            "expected_output": "提升Faithfulness的方法：1)优化检索质量，确保检索到的文档包含足够的信息；2)在Prompt中强调'只基于提供的文档回答，不要编造'；3)使用更低的temperature（0-0.3），减少LLM的'创造性'。",
            "context": "RAG评估体系",
            "tags": ["RAG评估", "优化方法", "实用技巧"]
        },
        {
            "input": "Context Recall为什么需要ground truth？",
            "expected_output": "Context Recall（上下文召回率）需要知道'正确答案应该包含哪些信息'才能计算。它衡量ground truth中的信息有多少能在检索到的文档中找到，因此必须提供标准答案作为参考。",
            "context": "RAG评估体系",
            "tags": ["RAG评估", "Context Recall", "核心概念"]
        },
        {
            "input": "RAGAS的哪些指标需要ground truth？",
            "expected_output": "需要ground truth的指标有：Context Recall（上下文召回率）、Context Entity Recall（实体召回率）、Answer Correctness（答案正确性）、Answer Semantic Similarity（答案语义相似度）。不需要ground truth的指标有：Faithfulness、Answer Relevancy、Context Precision、Context Relevancy。",
            "context": "RAG评估体系",
            "tags": ["RAG评估", "RAGAS", "指标对比"]
        },
    ],
    
    # === RAG核心技术 - 高质量问题 ===
    "rag_tech": [
        {
            "input": "两阶段检索架构是怎样的？为什么需要两阶段？",
            "expected_output": "两阶段检索架构：阶段一（粗筛）目标是高召回，使用混合检索（向量+BM25+RRF融合）+多查询融合，从大量文档中筛选出候选集；阶段二（精筛）目标是高精度，使用Cross-Encoder Reranker对候选集逐对打分，取top结果。这种设计平衡了召回率和精确度。",
            "context": "RAG核心技术",
            "tags": ["RAG技术", "检索架构", "核心概念"]
        },
        {
            "input": "RRF融合算法的原理是什么？k值设多少合适？",
            "expected_output": "RRF（Reciprocal Rank Fusion）公式：score(d) = sum(1/(k + rank_i(d)))，其中rank_i(d)是文档d在第i个排序列表中的排名。k=60是经验值，作用是对排名靠后的文档进行平滑。RRF基于排名融合，无需分数归一化，适合融合向量检索和BM25检索的结果。",
            "context": "RAG核心技术",
            "tags": ["RAG技术", "RRF", "融合算法", "核心概念"]
        },
        {
            "input": "Reranker和普通向量检索有什么区别？为什么Reranker效果更好？",
            "expected_output": "普通向量检索使用Bi-Encoder，Query和Document分别编码，通过向量相似度计算相关性，速度快但交互不充分。Reranker使用Cross-Encoder模型，Query和Document拼接后一起输入模型，通过全注意力机制交互，精度高但速度慢。Reranker弥补了向量检索'只看整体语义、忽略细节匹配'的不足。",
            "context": "RAG核心技术",
            "tags": ["RAG技术", "Reranker", "Cross-Encoder", "核心概念"]
        },
        {
            "input": "语义分块是怎么实现的？相比规则分块有什么优势？",
            "expected_output": "语义分块计算相邻句子的Embedding相似度，在相似度骤降处（语义转折点）切分。实现原理：文本按句子拆分→计算相邻句子Embedding余弦相似度→取相似度分位数阈值（默认90%）→低于阈值的点为切分点。相比规则分块，语义分块在语义边界切分，能保留语义完整性，适合论述性文档和制度文件。",
            "context": "RAG核心技术",
            "tags": ["RAG技术", "文档处理", "语义分块", "核心概念"]
        },
        {
            "input": "混合检索是什么？为什么要混合多种检索方式？",
            "expected_output": "混合检索将语义搜索（基于向量相似度）和关键词搜索（基于BM25等算法）相结合。语义搜索擅长理解语义相近但用词不同的内容，关键词搜索擅长精确匹配专业术语。通过RRF算法融合两种检索结果，可以取长补短，获得比单一检索方式更好的召回效果。",
            "context": "RAG核心技术",
            "tags": ["RAG技术", "混合检索", "BM25", "核心概念"]
        },
        {
            "input": "多查询融合的作用是什么？如何生成查询变体？",
            "expected_output": "多查询融合使用LLM将用户原始问题改写为多个不同角度的查询变体，分别检索后合并去重。作用：解决用户问题表述不精确、单一查询角度遗漏相关文档的问题。通常默认生成3个变体，通过Prompt要求生成与原始问题语义相关但表述不同的查询。",
            "context": "RAG核心技术",
            "tags": ["RAG技术", "多查询融合", "Prompt工程", "核心概念"]
        },
        {
            "input": "向量检索和BM25检索各有什么优缺点？",
            "expected_output": "向量检索（语义搜索）优点：擅长理解语义相近但用词不同的内容；缺点：对关键词精确匹配（如'第3条'、'2024年'）不敏感。BM25检索优点：擅长关键词精确匹配，算法成熟稳定；缺点：无法理解同义词和语义相近的表达。两者结合使用效果最好。",
            "context": "RAG核心技术",
            "tags": ["RAG技术", "向量检索", "BM25", "对比问题"]
        },
        {
            "input": "为什么向量检索对数字和日期不敏感？",
            "expected_output": "向量检索使用Bi-Encoder将Query和Document分别编码，Query和Document在向量空间中的相对位置关系是基于整体语义计算的。对于数字、日期等精确信息，向量表示往往相似度不够高，因为这些信息需要精确匹配而非语义相似。BM25等关键词检索算法更适合处理这类精确匹配需求。",
            "context": "RAG核心技术",
            "tags": ["RAG技术", "向量检索", "局限性", "核心概念"]
        },
        {
            "input": "RAG系统的检索链路完整流程是什么？",
            "expected_output": "RAG系统检索链路：1)用户问题向量化；2)多查询融合（生成查询变体）；3)阶段一粗筛：每个变体执行向量检索+BM25检索；4)RRF融合候选结果；5)阶段二精筛：Cross-Encoder Reranker重排序；6)取top结果作为上下文；7)LLM基于上下文生成答案。",
            "context": "RAG核心技术",
            "tags": ["RAG技术", "检索链路", "完整流程"]
        },
        {
            "input": "Embedding模型选择BGE-M3而不是OpenAI text-embedding-3的原因是什么？",
            "expected_output": "BGE-M3优势：免费、本地部署、支持100+语言、8192 tokens上下文、1024维向量。OpenAI text-embedding-3：按Token计费、需要网络调用、有速率限制。选型考量：成本（本地免费vs API付费）、延迟（本地推理vs网络往返）、数据安全（敏感文档不出内网）。BGE-M3的多语言能力对中文制度文档场景特别重要。",
            "context": "RAG核心技术",
            "tags": ["RAG技术", "Embedding", "模型选型", "核心概念"]
        },
    ],
    
    # === 向量数据库选型 - 高质量问题 ===
    "vector_db": [
        {
            "input": "ChromaDB、Milvus、Elasticsearch、FAISS分别适合什么场景？",
            "expected_output": "ChromaDB：轻量、易部署、Python原生，适合开发/小规模部署；Milvus：分布式、高性能、支持混合检索，适合大规模生产环境；Elasticsearch：统一检索引擎（向量+全文）、生态成熟，适合需要全文+向量统一检索场景；FAISS：纯内存、极快、零依赖，适合快速验证/离线测试。",
            "context": "向量数据库选型",
            "tags": ["向量数据库", "数据库选型", "核心概念", "常见问题"]
        },
        {
            "input": "Milvus和Elasticsearch哪个更适合大规模向量检索？",
            "expected_output": "Milvus更适合大规模向量检索。Milvus是分布式架构，支持数据分片和水平扩展，能处理海量向量数据；专门为向量检索优化，支持多种索引类型（IVF、HNSW等）；在大规模场景下性能优于Elasticsearch。Elasticsearch适合需要同时进行全文检索和向量检索的统一检索场景。",
            "context": "向量数据库选型",
            "tags": ["向量数据库", "Milvus", "Elasticsearch", "对比问题"]
        },
        {
            "input": "FAISS有什么局限性？为什么不适合生产环境？",
            "expected_output": "FAISS的局限性：不支持持久化存储（数据存储在内存中，服务重启丢失）；不支持分布式部署（单机内存受限）；没有完善的索引管理功能。FAISS主要适合快速验证、离线测试、百万级以内的小规模场景。",
            "context": "向量数据库选型",
            "tags": ["向量数据库", "FAISS", "局限性"]
        },
        {
            "input": "向量数据库需要存储原始文档吗？还是只存储向量？",
            "expected_output": "向量数据库通常只存储向量和必要的元数据（如文档ID、块索引等），原始文档内容一般存储在其他数据库或文件系统。检索时通过向量相似度找到相关文档ID，再从其他存储中获取完整文档内容。这样设计可以提高检索效率，减少向量数据库的存储压力。",
            "context": "向量数据库选型",
            "tags": ["向量数据库", "架构设计", "常见问题"]
        },
        {
            "input": "HNSW和IVF索引有什么区别？如何选择？",
            "expected_output": "HNSW（Hierarchical Navigable Small World）：基于图的索引，检索速度快，精度高，但内存占用大，适合追求高精度的场景。IVF（Inverted Index）：倒排索引，聚类后只搜索最近的几个聚类中心，内存占用小，适合大规模数据。选择依据：数据规模、精度要求、内存限制。",
            "context": "向量数据库选型",
            "tags": ["向量数据库", "索引类型", "HNSW", "IVF"]
        },
    ],
    
    # === 性能优化 - 高质量问题 ===
    "performance": [
        {
            "input": "语义缓存是怎么工作的？命中后跳过哪些步骤？",
            "expected_output": "语义缓存基于向量相似度匹配历史问答。工作流程：用户问题向量化→kNN搜索缓存集合→余弦相似度≥阈值（默认0.95）视为命中→直接返回缓存答案。命中时跳过检索和LLM调用步骤，延迟从秒级降至毫秒级，成本大幅降低。",
            "context": "性能优化",
            "tags": ["性能优化", "语义缓存", "核心概念"]
        },
        {
            "input": "语义缓存的相似度阈值设为多少合适？太高或太低有什么影响？",
            "expected_output": "语义缓存的相似度阈值默认设为0.95（余弦相似度）。太高（如0.99）：几乎不会命中，缓存形同虚设；太低（如0.80）：可能返回不相关的缓存答案，影响回答质量。需要根据实际场景调整，平衡命中率和准确性。",
            "context": "性能优化",
            "tags": ["性能优化", "语义缓存", "参数调优"]
        },
        {
            "input": "语义缓存的淘汰策略是什么？缓存失效怎么处理？",
            "expected_output": "淘汰策略：LFU（最少命中淘汰）+TTL过期（默认24小时）。失效策略：知识库文档入库后自动清空全部缓存，因为知识库更新后旧答案可能已过时，需要重新检索生成。",
            "context": "性能优化",
            "tags": ["性能优化", "语义缓存", "核心概念"]
        },
        {
            "input": "指数退避重试为什么需要随机抖动？不加会怎样？",
            "expected_output": "不加Jitter的问题：多个客户端同时触发429错误→同时按相同退避时间重试→再次同时触发429→形成'惊群效应'。加Jitter的效果：每个客户端的重试时间随机偏移→请求分散到不同时间点→避免同时冲击API。实现：wait_time = 2^n + random(0, 2)秒。",
            "context": "性能优化",
            "tags": ["性能优化", "重试策略", "核心概念"]
        },
        {
            "input": "滑动窗口限流和固定窗口限流有什么区别？",
            "expected_output": "固定窗口：简单但有边界突发问题（窗口切换瞬间可能双倍流量）。滑动窗口：记录每个请求的时间戳，统计窗口内的请求数，精确但内存占用稍高。滑动窗口避免了边界突发问题，实现也较简单，适合API限流场景。",
            "context": "性能优化",
            "tags": ["性能优化", "限流", "对比问题"]
        },
        {
            "input": "缓存命中能降低多少延迟？语义缓存的性能提升有多大？",
            "expected_output": "语义缓存命中时跳过检索和LLM调用两个耗时步骤。典型数据：向量检索约50-100ms，LLM生成约2-5秒；缓存命中延迟约10-50ms。综合来看，缓存命中时延迟可降低80%以上，API调用成本降低90%以上。",
            "context": "性能优化",
            "tags": ["性能优化", "语义缓存", "性能指标"]
        },
    ],
    
    # === 知识库健康检查 - 高质量问题 ===
    "health_check": [
        {
            "input": "知识库健康检查包含哪5个维度？每个维度检查什么？",
            "expected_output": "知识库健康检查包含5个维度：文档层（重复文档、空文档、格式异常）；切片层（空切片、过短/过长切片、切片分布）；向量层（零向量、维度一致性、Embedding质量）；检索层（BM25索引一致性、向量库一致性、检索质量抽样）；索引层（BM25索引状态、向量库连接状态）。系统输出0-100综合健康分。",
            "context": "评估体系",
            "tags": ["评估体系", "健康检查", "核心概念", "常见问题"]
        },
        {
            "input": "文档层健康检查发现重复文档怎么处理？",
            "expected_output": "文档层健康检查发现重复文档的常见问题：重复上传、文件损坏。修复建议：对重复文档进行去重处理，或删除损坏文件后重新上传。系统会标记重复文档的位置和数量，便于人工处理。",
            "context": "评估体系",
            "tags": ["评估体系", "健康检查", "文档层"]
        },
        {
            "input": "切片层健康检查关注哪些指标？切片异常怎么处理？",
            "expected_output": "切片层检查：空切片（内容为空）、过短切片（信息量不足）、过长切片（超出Embedding模型限制）、切片分布（是否均匀）。修复建议：调整chunk_size和chunk_overlap参数，优化分块策略。",
            "context": "评估体系",
            "tags": ["评估体系", "健康检查", "切片层"]
        },
        {
            "input": "向量层健康检查包含哪些内容？向量异常怎么修复？",
            "expected_output": "向量层检查：零向量（向量全为0）、维度不一致（向量维度与配置不符）、Embedding质量（向量表达是否准确）。常见问题：模型异常、向量丢失。修复建议：重新执行Embedding，对异常向量进行修复或重建。",
            "context": "评估体系",
            "tags": ["评估体系", "健康检查", "向量层"]
        },
        {
            "input": "健康检查的综合健康分是怎么计算的？",
            "expected_output": "系统综合文档层、切片层、向量层、检索层、索引层五个维度的检查结果，通过加权平均计算综合健康分，输出0-100的分数。分数越高表示知识库越健康，接近100分表示各项指标都正常。",
            "context": "评估体系",
            "tags": ["评估体系", "健康检查", "评分机制"]
        },
    ],
    
    # === 可信度评估 - 高质量问题 ===
    "confidence": [
        {
            "input": "可信度评估包含哪几个维度？权重是怎么分配的？",
            "expected_output": "可信度评估包含5个维度，各维度权重：来源匹配度(30%)、一致性(25%)、权威性(20%)、时效性(15%)、完整性(10%)。各维度打分(0-100)后加权求和得到综合可信度分数。",
            "context": "评估体系",
            "tags": ["评估体系", "可信度评估", "核心概念"]
        },
        {
            "input": "来源匹配度是如何计算的？",
            "expected_output": "来源匹配度衡量检索文档与LLM回答的语义相似度。计算方式：提取LLM回答中的关键信息，与检索到的文档进行语义相似度匹配，分数反映回答内容与检索上下文的吻合程度。",
            "context": "评估体系",
            "tags": ["评估体系", "可信度评估", "评分维度"]
        },
        {
            "input": "权威性维度是怎么评估的？",
            "expected_output": "权威性评估来源文档的可信程度。评估依据：文档类型权重（制度文件>通知公告>会议纪要）、文档来源（官方文档权重更高）、文档的完整性和准确性。",
            "context": "评估体系",
            "tags": ["评估体系", "可信度评估", "评分维度"]
        },
        {
            "input": "低置信度回答如何处理？",
            "expected_output": "低置信度回答的处理流程：1)自动标记（系统检测置信度低于阈值时标记）；2)专家路由（自动推送给相关领域的专家进行人工审核）；3)改进建议（提供检索优化建议或补充知识库的建议）。",
            "context": "评估体系",
            "tags": ["评估体系", "可信度评估", "处理流程"]
        },
    ],
    
    # === 工程实践 - 高质量问题 ===
    "engineering": [
        {
            "input": "ONNX Runtime如何在Java中实现本地推理？",
            "expected_output": "ONNX Runtime实现JVM本地推理，零Python依赖。调用流程：Python脚本将PyTorch模型转为ONNX格式→Java通过ONNX Runtime Java API加载.onnx文件→DJL Tokenizer执行分词→ONNX Runtime执行推理→Mean Pooling+L2归一化→得到1024维向量。",
            "context": "工程实践",
            "tags": ["工程实践", "ONNX", "Java推理", "核心概念"]
        },
        {
            "input": "为什么Java版RAG系统选择Elasticsearch而不是ChromaDB？",
            "expected_output": "Java版选择Elasticsearch的原因：1)统一检索引擎，同时承担向量检索（kNN）和全文检索（BM25），减少组件依赖；2)分布式架构，支持生产级大规模部署；3)成熟的生态，完善的监控、日志、安全机制；4)与Spring生态集成良好。ChromaDB是Python原生的轻量级方案，不适合Java生态。",
            "context": "工程实践",
            "tags": ["工程实践", "Elasticsearch", "架构设计"]
        },
        {
            "input": "Python版和Java版RAG系统架构上最大的区别是什么？",
            "expected_output": "最大区别：1)向量数据库选型（Python版用ChromaDB/Milvus，Java版用Elasticsearch）；2)模型推理方式（Python版用sentence-transformers原生推理，Java版用ONNX Runtime本地推理）；3)数据存储（Python版用SQLite，Java版用MySQL）；4)Embedding模型调用方式不同。",
            "context": "工程实践",
            "tags": ["工程实践", "Python", "Java", "架构对比"]
        },
        {
            "input": "用户数据隔离是怎么实现的？为什么用userId而不是username？",
            "expected_output": "隔离实现：所有数据库查询都带上userId过滤条件（WHERE user_id = ?）。选择userId的原因：username可能重复（不同用户可能同名）、可能修改（用户改名）；userId是唯一且不可变的标识符，适合作为数据隔离的key。获取方式：从认证上下文（Token/Interceptor）中提取当前用户的userId。",
            "context": "工程实践",
            "tags": ["工程实践", "安全设计", "数据隔离"]
        },
        {
            "input": "SSE流式推送的实现原理是什么？为什么RAG系统选择SSE而不是WebSocket？",
            "expected_output": "SSE（Server-Sent Events）：基于HTTP协议，服务端向客户端单向推送数据，实现简单、兼容性 好、自动重连。RAG问答场景只需要服务端推送答案（单向），SSE更简单；WebSocket是全双工通信，支持双向数据传输，但复杂度更高，通常用于需要客户端实时交互的场景。",
            "context": "工程实践",
            "tags": ["工程实践", "SSE", "流式推送"]
        },
        {
            "input": "SQLite WAL模式是什么？为什么并发场景需要它？",
            "expected_output": "WAL（Write-Ahead Logging）：写入操作先记录到WAL文件，再异步写入主数据库文件。优势：读写不互斥（读操作不阻塞写操作，写操作不阻塞读操作）、更好的并发性能。默认模式（DELETE回滚日志）：写操作期间会锁住整个数据库，读操作被阻塞。",
            "context": "工程实践",
            "tags": ["工程实践", "SQLite", "WAL", "并发"]
        },
    ],
}

# 生成完整数据集
def generate_dataset():
    dataset = []
    
    # 遍历所有类别
    for category, qa_list in HIGH_QUALITY_QAP.items():
        for qa in qa_list:
            dataset.append(qa)
    
    # 打乱顺序
    random.shuffle(dataset)
    
    # 复制扩充到150条
    while len(dataset) < 150:
        item = random.choice(dataset).copy()
        # 生成变体问题
        variants = [
            item["input"].replace("是什么", "的定义是什么"),
            item["input"].replace("？", "，能详细说明吗？"),
            "请问" + item["input"],
            "关于" + item["input"].replace("？", "，请介绍一下"),
        ]
        new_item = {
            "input": random.choice(variants),
            "expected_output": item["expected_output"],
            "context": item["context"],
            "tags": item["tags"].copy()
        }
        if new_item["input"] not in [d["input"] for d in dataset]:
            dataset.append(new_item)
    
    return dataset[:150]

# 保存数据集
def save_dataset(dataset):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"[OK] 评测数据集已生成: {OUTPUT_FILE}")
    print(f"[INFO] 数据集大小: {len(dataset)} 条")
    
    # 统计标签分布
    tag_count = {}
    topic_count = {}
    for item in dataset:
        # 统计标签
        for tag in item["tags"]:
            tag_count[tag] = tag_count.get(tag, 0) + 1
        # 统计主题
        topic = item.get("context", "其他")
        topic_count[topic] = topic_count.get(topic, 0) + 1
    
    print("\n[STATS] 按主题分布:")
    for topic, count in sorted(topic_count.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic}: {count} 条")
    
    print("\n[STATS] 标签统计:")
    for tag, count in sorted(tag_count.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {tag}: {count} 条")

if __name__ == "__main__":
    print("正在生成高质量评测数据集...")
    dataset = generate_dataset()
    save_dataset(dataset)
