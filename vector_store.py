"""
vector_store.py - 向量数据库模块

对应 RAG 流程中的：
  Chunks + Embedding Model → Vector Database (Chroma)
  Query + Embedding Model → Retriever → Relevant Chunks

负责：
  1. 初始化本地 Embedding 模型（BGE-M3）
  2. 将 Chunks 向量化并存入 Chroma 服务（HTTP 模式）
  3. 根据用户查询，从 Chroma 中检索最相关的 Chunks
  4. 混合检索：BM25 关键词检索 + 向量检索，RRF 融合
  5. Reranker 精排：bge-reranker-large 交叉编码器对候选文档重排序

检索链路（两阶段）：
  阶段一（粗筛）：向量检索 top_k=12 + BM25 top_k=12 → RRF 融合 → 12 条候选
  阶段二（精筛）：Reranker 对 12 条候选重排序 → top_k=4 条

Chroma 服务模式说明：
  启动命令：.venv\Scripts\chroma.exe run --path ./chroma_db --host localhost --port 8000
  Python 通过 chromadb.HttpClient 连接，数据持久化由服务进程管理
"""
import hashlib
import logging
import re
from functools import lru_cache
from typing import List, Optional, Tuple

import chromadb
import jieba
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    CHROMA_HOST,
    CHROMA_PORT,
    CHROMA_COLLECTION_NAME,
    RETRIEVAL_TOP_K,
    BASE_DIR,
    HYBRID_SEARCH_ALPHA,
    RERANKER_ENABLED,
    RERANKER_CANDIDATE_K,
    MULTI_QUERY_ENABLED,
    MULTI_QUERY_COUNT,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)

logger = logging.getLogger(__name__)


def jieba_tokenize(text: str) -> List[str]:
    """
    使用 jieba 进行中文分词，精度远高于字符 n-gram。
    适用于 BM25 关键词检索的精确分词。
    """
    tokens = jieba.lcut(text)
    stop_chars = set("，。、！？：；""'''（）【】《》 \t\n\r")
    return [t for t in tokens if t not in stop_chars and len(t.strip()) > 0]


class HybridRetriever(BaseRetriever):
    """
    自定义 Retriever，支持混合检索 + Reranker。
    替代 Chroma 原生的 as_retriever()，使 LCEL Chain 也能享受完整的检索能力。
    """

    def __init__(self, vs_manager: "VectorStoreManager", top_k: int = 4):
        super().__init__()
        self._vs_manager = vs_manager
        self._top_k = top_k

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = self._vs_manager.similarity_search_with_scores(query, top_k=self._top_k)
        return [item[0] for item in results]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


class VectorStoreManager:
    """
    向量数据库管理器（Chroma 服务模式）
    对应流程：
      构建阶段：Chunks → Embedding Model → Vector Database
      查询阶段：Query → Embedding Model → Retriever → Relevant Chunks

    两阶段检索：
      阶段一（粗筛）：向量检索 + BM25 → RRF 融合 → 候选文档
      阶段二（精筛）：Reranker 对候选文档重排序 → 最终结果
    """

    def __init__(self):
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vector_store: Optional[Chroma] = None
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_corpus: List[Tuple[str, Document]] = []
        self._hybrid_enabled: bool = True
        self._reranker_manager = None
        self._reranker_enabled: bool = RERANKER_ENABLED

    def _get_reranker(self):
        """懒加载 Reranker"""
        if self._reranker_manager is None:
            from reranker import RerankerManager
            self._reranker_manager = RerankerManager()
        return self._reranker_manager

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """懒加载 Embedding 模型（首次使用时才加载，避免启动过慢）"""
        if self._embeddings is None:
            model_path = EMBEDDING_MODEL_NAME
            if model_path.startswith("./") or model_path.startswith(".\\"):
                model_path = str(BASE_DIR / model_path)

            logger.info(f"正在加载 Embedding 模型：{model_path}（设备：{EMBEDDING_DEVICE}）")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={"device": EMBEDDING_DEVICE},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding 模型加载完成（BGE-M3，向量已归一化）")
        return self._embeddings

    def _get_chroma_client(self) -> chromadb.HttpClient:
        """创建 Chroma HTTP 客户端，连接到本地 chroma run 服务"""
        return chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
        )

    def get_or_create_vector_store(self) -> Chroma:
        """获取或创建 Chroma 向量数据库实例（HTTP 服务模式）"""
        if self._vector_store is None:
            client = self._get_chroma_client()
            self._vector_store = Chroma(
                client=client,
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=self.embeddings,
            )
            logger.info(f"已连接到 Chroma 服务：http://{CHROMA_HOST}:{CHROMA_PORT}，集合：{CHROMA_COLLECTION_NAME}")
        return self._vector_store

    def _build_bm25_index(self, documents: List[Document]):
        """
        为当前所有文档构建 BM25 索引。
        对应流程：Chunks → BM25 Index（关键词倒排索引）
        """
        if not documents:
            self._bm25_index = None
            self._bm25_corpus = []
            return

        # 保存原始文档（用于后续检索）
        self._bm25_corpus = [(doc.page_content, doc) for doc in documents]

        # 分词
        tokenized_corpus = [jieba_tokenize(doc.page_content) for doc in documents]
        self._bm25_index = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 索引构建完成：{len(documents)} 篇文档")

    def add_documents(self, chunks: List[Document]) -> int:
        """
        将 Chunks 向量化并写入 Chroma，同时增量更新 BM25 索引。
        对应流程：Chunks + Embedding Model → Vector Database
        """
        if not chunks:
            logger.warning("没有可写入的 Chunks")
            return 0

        vector_store = self.get_or_create_vector_store()

        logger.info(f"正在将 {len(chunks)} 个 Chunks 向量化并写入 Chroma...")
        vector_store.add_documents(chunks)
        logger.info(f"写入完成：{len(chunks)} 个 Chunks 已存储至 Chroma 服务")

        # 增量更新 BM25 索引（追加新文档，而非全量重建）
        for chunk in chunks:
            self._bm25_corpus.append((chunk.page_content, chunk))

        tokenized = [jieba_tokenize(content) for content, _ in self._bm25_corpus]
        self._bm25_index = BM25Okapi(tokenized)
        logger.info(f"BM25 索引增量更新完成：当前共 {len(self._bm25_corpus)} 篇文档")

        # 入库后清空检索缓存
        self._cached_search_with_scores.cache_clear()

        return len(chunks)

    def get_all_documents(self) -> List[Document]:
        """获取向量库中的所有文档（用于 BM25 索引重建）"""
        vector_store = self.get_or_create_vector_store()
        try:
            results = vector_store.similarity_search("", k=vector_store._collection.count())
            return results
        except Exception:
            return []

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """
        BM25 关键词检索。

        Returns:
            List[(Document, bm25_score)]: 相关文档及 BM25 分数
        """
        if self._bm25_index is None or not self._bm25_corpus:
            return []

        tokens = jieba_tokenize(query)
        scores = self._bm25_index.get_scores(tokens)

        # 取 top_k
        doc_scores = list(enumerate(scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in doc_scores[:top_k]:
            if score > 0:
                results.append((self._bm25_corpus[idx][1], score))
        return results

    def _rrf_fusion(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
        alpha: float = 0.5,
        k: int = 60,
    ) -> List[Tuple[Document, float, str]]:
        """
        RRF（Reciprocal Rank Fusion）- 倒数排名融合。

        将向量检索和 BM25 检索的结果按排名融合，避免单一检索方式的偏差。

        Args:
            vector_results: 向量检索结果 [(doc, score)]
            bm25_results: BM25 检索结果 [(doc, score)]
            alpha: 向量权重（0.5 表示各占一半）
            k: RRF 公式参数（通常 60）

        Returns:
            List[(doc, fused_score, retrieval_type)]: 融合后的结果
        """
        doc_ranks: dict = {}  # content_md5 -> (doc, score, type)

        # 向量检索排名
        for rank, (doc, score) in enumerate(vector_results):
            key = hashlib.md5(doc.page_content.encode()).hexdigest()
            rrf_score = alpha * (1 / (k + rank + 1))
            if key in doc_ranks:
                existing_doc, existing_score, existing_type = doc_ranks[key]
                doc_ranks[key] = (existing_doc, existing_score + rrf_score, existing_type + "+vec")
            else:
                doc_ranks[key] = (doc, rrf_score, "vec")

        # BM25 排名
        for rank, (doc, score) in enumerate(bm25_results):
            key = hashlib.md5(doc.page_content.encode()).hexdigest()
            rrf_score = (1 - alpha) * (1 / (k + rank + 1))
            if key in doc_ranks:
                existing_doc, existing_score, existing_type = doc_ranks[key]
                doc_ranks[key] = (existing_doc, existing_score + rrf_score, existing_type + "+bm25")
            else:
                doc_ranks[key] = (doc, rrf_score, "bm25")

        # 按融合分数排序
        sorted_results = sorted(doc_ranks.items(), key=lambda x: x[1][1], reverse=True)
        return [(doc, score, retrieval_type) for (_, (doc, score, retrieval_type)) in sorted_results]

    def _generate_query_variants(self, query: str) -> List[str]:
        """
        使用 LLM 生成多个查询变体，提高召回率。

        将用户的口语化/模糊问题改写为 2~3 个不同角度的检索查询，
        分别从关键词、语义、同义词等角度覆盖，合并后召回更全面。
        """
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model="gpt-4o",
                openai_api_key=OPENAI_API_KEY,
                openai_api_base=OPENAI_BASE_URL,
                temperature=0.3,
                max_tokens=200,
            )

            prompt = f"""你是一个查询改写助手。请将用户的问题改写为 {MULTI_QUERY_COUNT} 个不同角度的检索查询，每个查询一行，不要编号。

要求：
- 从不同角度表述同一个问题
- 使用文档中可能出现的正式用语
- 保持简洁，每个查询不超过 30 个字

用户问题：{query}

改写结果："""

            response = llm.invoke(prompt)
            variants = [line.strip() for line in response.content.strip().split("\n") if line.strip()]
            # 去重并过滤空行，保留原始查询
            seen = {query}
            result = [query]
            for v in variants:
                if v not in seen:
                    seen.add(v)
                    result.append(v)
            logger.info(f"查询改写完成：原始='{query}' → 变体={result[1:]}")
            return result
        except Exception as e:
            logger.warning(f"查询改写失败，回退到原始查询：{e}")
            return [query]

    def _multi_query_search_with_scores(
        self,
        query: str,
        top_k: int,
        alpha: float = None,
    ) -> List[Tuple[Document, float, str]]:
        """
        多查询融合检索：生成多个查询变体 → 分别检索 → RRF 融合。

        流程：
          1. LLM 生成 2~3 个查询变体
          2. 每个变体独立执行混合检索（向量 + BM25）
          3. 所有结果用 RRF 融合去重
          4. Reranker 精排
        """
        variants = self._generate_query_variants(query)

        if len(variants) == 1:
            return self._hybrid_search_with_scores(query, top_k, alpha)

        # 每个变体独立检索
        all_results: List[Tuple[Document, float, str]] = []
        for variant in variants:
            variant_results = self._hybrid_search_with_scores_no_rerank(variant, top_k * 2, alpha)
            all_results.extend(variant_results)

        if not all_results:
            return []

        # 多查询结果 RRF 融合去重
        fused = self._rrf_fusion_multi_query(all_results)
        logger.info(f"多查询融合：{len(variants)} 个变体 → {len(all_results)} 条候选 → {len(fused)} 条融合结果")

        # Reranker 精排
        if self._reranker_enabled:
            reranker = self._get_reranker()
            docs = [doc for doc, _, _ in fused]
            reranked = reranker.rerank(query, docs, top_k=top_k)
            return [(doc, score * 100, "rerank") for doc, score in reranked]

        return [(doc, score, rtype) for doc, score, rtype in fused[:top_k]]

    def _hybrid_search_with_scores_no_rerank(
        self,
        query: str,
        top_k: int,
        alpha: float = None,
    ) -> List[Tuple[Document, float, str]]:
        """
        混合检索（不带 Reranker），用于多查询融合的中间步骤。
        """
        if alpha is None:
            alpha = HYBRID_SEARCH_ALPHA

        vector_store = self.get_or_create_vector_store()
        candidate_k = max(RERANKER_CANDIDATE_K, top_k * 2)
        vec_results = vector_store.similarity_search_with_score(query, k=candidate_k)
        bm25_results = self._bm25_search(query, candidate_k)
        fused = self._rrf_fusion(vec_results, bm25_results, alpha=alpha)
        return [(doc, score, rtype) for doc, score, rtype in fused[:top_k]]

    def _rrf_fusion_multi_query(
        self,
        all_results: List[Tuple[Document, float, str]],
        k: int = 60,
    ) -> List[Tuple[Document, float, str]]:
        """
        多查询结果的 RRF 融合。

        将来自不同查询变体的检索结果按排名融合，
        被多个变体同时命中的文档得分更高。
        """
        doc_scores: dict = {}  # content_md5 -> (doc, total_score, type)

        for rank, (doc, score, rtype) in enumerate(all_results):
            key = hashlib.md5(doc.page_content.encode()).hexdigest()
            rrf_score = 1 / (k + rank + 1)
            if key in doc_scores:
                existing_doc, existing_score, existing_type = doc_scores[key]
                doc_scores[key] = (existing_doc, existing_score + rrf_score, existing_type)
            else:
                doc_scores[key] = (doc, rrf_score, rtype)

        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1][1], reverse=True)
        return [(doc, score, rtype) for _, (doc, score, rtype) in sorted_results]

    def similarity_search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        hybrid: bool = None,
    ) -> List[Document]:
        """
        根据用户查询检索最相关的 Chunks。

        Args:
            query: 用户查询
            top_k: 返回数量
            hybrid: 是否使用混合检索（None=使用全局配置）
        """
        if hybrid is None:
            hybrid = self._hybrid_enabled

        if not hybrid:
            vector_store = self.get_or_create_vector_store()
            return vector_store.similarity_search(query, k=top_k)

        # 混合检索：向量 + BM25
        return self._hybrid_search(query, top_k)

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        alpha: float = None,
    ) -> List[Document]:
        """
        混合检索核心实现：向量检索 + BM25 → RRF 融合。
        """
        if alpha is None:
            alpha = HYBRID_SEARCH_ALPHA

        vector_store = self.get_or_create_vector_store()

        # Step 1: 向量检索（扩大范围，取 2*top_k）
        vec_results = vector_store.similarity_search_with_score(query, k=top_k * 2)
        logger.info(f"向量检索返回 {len(vec_results)} 条结果")

        # Step 2: BM25 检索
        bm25_results = self._bm25_search(query, top_k * 2)
        logger.info(f"BM25 检索返回 {len(bm25_results)} 条结果")

        # Step 3: RRF 融合
        fused = self._rrf_fusion(vec_results, bm25_results, alpha=alpha)
        logger.info(f"RRF 融合后返回 {min(top_k, len(fused))} 条结果")

        # 返回前 top_k
        docs = [doc for doc, score, _ in fused[:top_k]]
        return docs

    def similarity_search_with_scores(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        hybrid: bool = None,
    ) -> List[tuple]:
        """
        根据用户查询检索最相关的 Chunks，同时返回相似度分数和检索类型。

        Returns:
            List[(Document, score, retrieval_type)]: 文档、分数、检索类型
        """
        if hybrid is None:
            hybrid = self._hybrid_enabled

        if not hybrid:
            vector_store = self.get_or_create_vector_store()
            vec_results = vector_store.similarity_search_with_score(query, k=top_k)

            # 纯向量检索也支持 Reranker 精排
            if self._reranker_enabled and len(vec_results) > 0:
                docs = [doc for doc, _ in vec_results]
                reranked = self._get_reranker().rerank(query, docs, top_k=top_k)
                return [(doc, score * 100, "rerank") for doc, score in reranked]

            # 不启用 Reranker：直接返回向量检索结果
            return [(doc, score, "vec") for doc, score in vec_results]

        return self._hybrid_search_with_scores(query, top_k)

    @lru_cache(maxsize=128)
    def _cached_search_with_scores(
        self,
        query: str,
        top_k: int,
        hybrid: bool,
        reranker: bool,
    ) -> tuple:
        """
        带缓存的检索方法（内部使用）。
        缓存 key 由 (query, top_k, hybrid, reranker) 组成。
        入库或清空数据库时会自动清空缓存。
        """
        # 临时保存当前状态
        saved_hybrid = self._hybrid_enabled
        saved_reranker = self._reranker_enabled

        self._hybrid_enabled = hybrid
        self._reranker_enabled = reranker

        try:
            if not hybrid:
                vector_store = self.get_or_create_vector_store()
                vec_results = vector_store.similarity_search_with_score(query, k=top_k)
                if reranker and len(vec_results) > 0:
                    docs = [doc for doc, _ in vec_results]
                    reranked = self._get_reranker().rerank(query, docs, top_k=top_k)
                    result = tuple((doc, score * 100, "rerank") for doc, score in reranked)
                else:
                    result = tuple((doc, score, "vec") for doc, score in vec_results)
            else:
                result = tuple(self._hybrid_search_with_scores(query, top_k))
        finally:
            self._hybrid_enabled = saved_hybrid
            self._reranker_enabled = saved_reranker

        return result

    def cached_similarity_search_with_scores(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        hybrid: bool = None,
    ) -> List[Tuple[Document, float, str]]:
        """
        带缓存的检索（对外接口）。
        高频相同查询直接返回缓存结果，避免重复计算。
        """
        if hybrid is None:
            hybrid = self._hybrid_enabled
        reranker = self._reranker_enabled

        cached = self._cached_search_with_scores(query, top_k, hybrid, reranker)
        return list(cached)

    def _hybrid_search_with_scores(
        self,
        query: str,
        top_k: int,
        alpha: float = None,
    ) -> List[Tuple[Document, float, str]]:
        """
        混合检索（带分数和类型标注）。

        两阶段：
          阶段一（粗筛）：多查询融合 / 向量检索 + BM25 → RRF 融合 → 候选
          阶段二（精筛）：Reranker 对候选重排序 → 最终结果
        """
        # 多查询融合检索（可选）
        if MULTI_QUERY_ENABLED:
            return self._multi_query_search_with_scores(query, top_k, alpha)

        if alpha is None:
            alpha = HYBRID_SEARCH_ALPHA

        vector_store = self.get_or_create_vector_store()

        # 阶段一：粗筛（候选数 > 最终返回数，留出精排空间）
        candidate_k = max(RERANKER_CANDIDATE_K, top_k * 2)
        vec_results = vector_store.similarity_search_with_score(query, k=candidate_k)
        bm25_results = self._bm25_search(query, candidate_k)
        fused = self._rrf_fusion(vec_results, bm25_results, alpha=alpha)

        # 阶段二：精筛（Reranker 重排序）
        if self._reranker_enabled:
            reranker = self._get_reranker()
            docs = [doc for doc, _, _ in fused]
            reranked = reranker.rerank(query, docs, top_k=top_k)
            return [(doc, score * 100, "rerank") for doc, score in reranked]

        # 未启用 Reranker：直接返回 RRF 融合结果
        return [(doc, score, rtype) for doc, score, rtype in fused[:top_k]]

    def get_retriever(self, top_k: int = RETRIEVAL_TOP_K):
        """获取支持混合检索 + Reranker 的 LangChain Retriever（用于 RAG Chain 集成）"""
        return HybridRetriever(vs_manager=self, top_k=top_k)

    def get_document_count(self) -> int:
        """获取当前向量数据库中存储的文档数量"""
        try:
            vector_store = self.get_or_create_vector_store()
            return vector_store._collection.count()
        except Exception:
            return 0

    def clear_collection(self) -> bool:
        """清空向量数据库中的所有数据（谨慎使用）"""
        try:
            vector_store = self.get_or_create_vector_store()
            vector_store._collection.delete(where={"$exists": True})
            self._bm25_index = None
            self._bm25_corpus = []
            self._cached_search_with_scores.cache_clear()
            logger.info("向量数据库已清空，BM25 索引已重置，检索缓存已清空")
            return True
        except Exception as e:
            logger.error(f"清空数据库失败：{e}")
            return False

    def set_hybrid_search(self, enabled: bool):
        """开关混合检索"""
        self._hybrid_enabled = enabled
        logger.info(f"混合检索已{'启用' if enabled else '关闭'}")

    def is_hybrid_search_enabled(self) -> bool:
        """查询混合检索状态"""
        return self._hybrid_enabled

    def set_reranker(self, enabled: bool):
        """开关 Reranker 重排序"""
        self._reranker_enabled = enabled
        logger.info(f"Reranker 已{'启用' if enabled else '关闭'}")

    def is_reranker_enabled(self) -> bool:
        """查询 Reranker 状态"""
        return self._reranker_enabled
