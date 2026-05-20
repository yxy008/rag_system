"""
vector_store.py - 向量数据库模块

对应 RAG 流程中的：
  Chunks + Embedding Model → Vector Database (Chroma / Faiss / Milvus)
  Query + Embedding Model → Retriever → Relevant Chunks

负责：
  1. 初始化本地 Embedding 模型（BGE-M3）
  2. 将 Chunks 向量化并存入向量数据库（支持 Chroma / Faiss / Milvus 切换）
  3. 根据用户查询，从向量数据库中检索最相关的 Chunks
  4. 混合检索：BM25 关键词检索 + 向量检索，RRF 融合
  5. Reranker 精排：bge-reranker-large 交叉编码器对候选文档重排序

检索链路（两阶段）：
  阶段一（粗筛）：向量检索 top_k=12 + BM25 top_k=12 → RRF 融合 → 12 条候选
  阶段二（精筛）：Reranker 对 12 条候选重排序 → top_k=4 条

多向量数据库支持：
  通过 VECTOR_STORE_BACKEND 环境变量切换后端：
  - chroma（默认）：需要先启动 chroma 服务
  - faiss：纯本地索引，无需外部服务，适合小规模数据快速验证
  - milvus：高性能分布式向量数据库，适合大规模生产环境

Chroma 服务模式说明：
  启动命令：.venv\Scripts\chroma.exe run --path ./chroma_db --host localhost --port 8000
  Python 通过 chromadb.HttpClient 连接，数据持久化由服务进程管理

Milvus 服务模式说明：
  启动命令：docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
  Python 通过 pymilvus 连接，数据持久化由 Milvus 服务进程管理
"""
import hashlib
import logging
import os
import pickle
import re
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Optional, Tuple

import chromadb
import jieba
import numpy as np
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from core.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    CHROMA_HOST,
    CHROMA_PORT,
    CHROMA_COLLECTION_NAME,
    MILVUS_HOST,
    MILVUS_PORT,
    MILVUS_COLLECTION_NAME,
    MILVUS_DIMENSION,
    MILVUS_INDEX_TYPE,
    MILVUS_METRIC_TYPE,
    RETRIEVAL_TOP_K,
    BASE_DIR,
    HYBRID_SEARCH_ALPHA,
    RERANKER_ENABLED,
    RERANKER_CANDIDATE_K,
    MULTI_QUERY_ENABLED,
    MULTI_QUERY_COUNT,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    VECTOR_STORE_BACKEND,
    VECTOR_COSINE_MIN_THRESHOLD,
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


# ============================================================
# 多向量数据库统一抽象接口
# ============================================================

class BaseVectorStore(ABC):
    """
    向量数据库统一抽象接口

    所有向量数据库后端（Chroma、Faiss 等）必须实现此接口，
    上层 VectorStoreManager 通过此接口与具体后端解耦。
    """

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到向量数据库，返回文档 ID 列表"""
        ...

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """文本相似度检索，返回最相关的 k 个文档"""
        ...

    @abstractmethod
    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """文本相似度检索（带分数），返回 (文档, 相似度分数) 列表"""
        ...

    @abstractmethod
    def count(self) -> int:
        """返回向量数据库中存储的文档总数"""
        ...

    @abstractmethod
    def clear(self) -> None:
        """清空向量数据库中的所有数据"""
        ...

    @abstractmethod
    def get_all_documents(self) -> List[Document]:
        """获取向量数据库中的所有文档"""
        ...


class ChromaVectorStore(BaseVectorStore):
    """
    Chroma 向量数据库实现

    基于 LangChain 的 Chroma 封装，连接外部 Chroma 服务（HTTP 模式）。
    数据持久化由 Chroma 服务进程管理，支持大规模数据。
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "rag_documents",
        embedding_function=None,
    ):
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        self._store: Optional[Chroma] = None

    def _get_store(self) -> Chroma:
        """懒加载 Chroma 实例"""
        if self._store is None:
            client = chromadb.HttpClient(host=self._host, port=self._port)
            self._store = Chroma(
                client=client,
                collection_name=self._collection_name,
                embedding_function=self._embedding_function,
            )
            logger.info(
                "已连接到 Chroma 服务：http://%s:%s，集合：%s",
                self._host, self._port, self._collection_name,
            )
        return self._store

    def add_documents(self, documents: List[Document]) -> List[str]:
        return self._get_store().add_documents(documents)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return self._get_store().similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        return self._get_store().similarity_search_with_score(query, k=k)

    def count(self) -> int:
        try:
            return self._get_store()._collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        try:
            self._get_store()._collection.delete(where={"$exists": True})
            logger.info("Chroma 集合已清空")
        except Exception as e:
            logger.error("清空 Chroma 集合失败：%s", e)

    def get_all_documents(self) -> List[Document]:
        try:
            store = self._get_store()
            total = store._collection.count()
            if total == 0:
                return []
            return store.similarity_search("", k=total)
        except Exception:
            return []


class FaissVectorStore(BaseVectorStore):
    """
    Faiss 向量数据库实现

    基于 Facebook AI Similarity Search (Faiss) 的本地向量索引。
    向量存储在内存中，文档文本存储在本地字典中。
    无需外部服务，适合小规模数据快速验证和开发测试。

    支持的索引类型：
      - flat：暴力搜索（精确但慢，适合 < 10万 向量）
      - hnsw：分层可导航小世界图（快速近似搜索，适合大规模数据）
    """

    def __init__(
        self,
        embedding_function,
        dimension: int = 1024,
        index_type: str = "hnsw",
        persist_dir: Optional[str] = None,
    ):
        import faiss

        self._embedding_function = embedding_function
        self._dimension = dimension
        self._index_type = index_type
        self._persist_dir = persist_dir

        if index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(dimension, 32)
            self._index.hnsw.efConstruction = 200
            self._index.hnsw.efSearch = 64
        elif index_type == "flat":
            self._index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"不支持的 Faiss 索引类型: {index_type}")

        self._documents: List[Document] = []
        self._id_to_idx: dict = {}

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._load_index()

    def add_documents(self, documents: List[Document]) -> List[str]:
        if not documents:
            return []

        texts = [doc.page_content for doc in documents]
        embeddings = self._embedding_function.embed_documents(texts)
        vectors = np.array(embeddings, dtype=np.float32)

        if self._index_type == "flat":
            faiss.normalize_L2(vectors)

        start_idx = len(self._documents)
        self._index.add(vectors)

        ids = []
        for i, doc in enumerate(documents):
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()[:16]
            ids.append(doc_id)
            self._id_to_idx[doc_id] = start_idx + i

        self._documents.extend(documents)
        logger.info(
            "Faiss 索引添加完成：新增 %d 条，当前共 %d 条，索引类型=%s",
            len(documents), len(self._documents), self._index_type,
        )
        return ids

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        results = self.similarity_search_with_score(query, k)
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        if not self._documents:
            return []

        query_embedding = self._embedding_function.embed_query(query)
        qv = np.array([query_embedding], dtype=np.float32)

        if self._index_type == "flat":
            faiss.normalize_L2(qv)

        actual_k = min(k, len(self._documents))
        distances, indices = self._index.search(qv, actual_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self._documents):
                continue
            doc = self._documents[idx]
            if self._index_type == "flat":
                score = float(dist)
            else:
                score = 1.0 / (1.0 + float(dist))
            results.append((doc, score))

        return results

    def count(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        import faiss

        if self._index_type == "hnsw":
            self._index = faiss.IndexHNSWFlat(self._dimension, 32)
            self._index.hnsw.efConstruction = 200
            self._index.hnsw.efSearch = 64
        else:
            self._index = faiss.IndexFlatIP(self._dimension)

        self._documents.clear()
        self._id_to_idx.clear()
        logger.info("Faiss 索引已清空")

    def get_all_documents(self) -> List[Document]:
        return list(self._documents)

    def _load_index(self):
        """从磁盘加载持久化的 Faiss 索引"""
        import faiss

        index_path = os.path.join(self._persist_dir, "faiss_index.bin")
        docs_path = os.path.join(self._persist_dir, "faiss_docs.json")

        if os.path.exists(index_path) and os.path.exists(docs_path):
            try:
                self._index = faiss.read_index(index_path)
                import json
                with open(docs_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._documents = [
                    Document(page_content=item["content"], metadata=item.get("metadata", {}))
                    for item in data
                ]
                for i, item in enumerate(data):
                    self._id_to_idx[item["id"]] = i
                logger.info("已从磁盘加载 Faiss 索引：%d 条文档", len(self._documents))
            except Exception as e:
                logger.warning("加载 Faiss 索引失败，将使用空索引：%s", e)

    def save_index(self):
        """将 Faiss 索引持久化到磁盘"""
        if not self._persist_dir:
            return

        import faiss
        import json

        try:
            os.makedirs(self._persist_dir, exist_ok=True)
            index_path = os.path.join(self._persist_dir, "faiss_index.bin")
            docs_path = os.path.join(self._persist_dir, "faiss_docs.json")

            faiss.write_index(self._index, index_path)

            data = []
            for i, doc in enumerate(self._documents):
                doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()[:16]
                data.append({
                    "id": doc_id,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                })

            with open(docs_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info("Faiss 索引已持久化到磁盘：%s", self._persist_dir)
        except Exception as e:
            logger.error("持久化 Faiss 索引失败：%s", e)


class MilvusVectorStore(BaseVectorStore):
    """
    Milvus 向量数据库实现

    基于 pymilvus 的 MilvusClient 新 API 连接 Milvus 服务（Docker 或云服务）。
    Milvus 是专为向量检索设计的高性能分布式数据库，支持：
      - 十亿级向量检索
      - 多种索引类型（IVF_FLAT、HNSW、IVF_PQ 等）
      - 标量过滤与混合查询
      - 数据持久化与副本机制

    适用场景：生产环境、大规模知识库、高并发检索

    启动方式：
      docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
    """

    def __init__(
        self,
        embedding_function,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "rag_documents",
        dimension: int = 1024,
        index_type: str = "IVF_FLAT",
        metric_type: str = "COSINE",
    ):
        self._embedding_function = embedding_function
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._dimension = dimension
        self._index_type = index_type
        self._metric_type = metric_type
        self._client = None

    def _get_client(self):
        """懒加载 MilvusClient 并确保集合存在且已加载到内存"""
        if self._client is not None:
            return self._client

        from pymilvus import MilvusClient

        uri = f"http://{self._host}:{self._port}"
        self._client = MilvusClient(uri=uri)

        if not self._client.has_collection(self._collection_name):
            self._client.create_collection(
                collection_name=self._collection_name,
                dimension=self._dimension,
                metric_type=self._metric_type,
                vector_field_name="embedding",
                auto_id=False,
                primary_field_name="id",
                id_type="string",
                max_length=64,
                varchar_max_length=65535,
            )
            logger.info(
                "Milvus 集合已创建并加载：%s，维度=%d，索引类型=%s，度量=%s",
                self._collection_name, self._dimension, self._index_type, self._metric_type,
            )
        else:
            self._client.load_collection(self._collection_name)
            stats = self._client.get_collection_stats(self._collection_name)
            row_count = stats.get("row_count", 0)
            logger.info(
                "已连接到 Milvus 集合：%s，当前实体数=%d",
                self._collection_name, row_count,
            )

        return self._client

    def _ensure_index(self):
        """确保集合已创建索引（首次写入时自动创建）"""
        client = self._get_client()

        try:
            index_info = client.describe_index(self._collection_name, "embedding")
            if index_info:
                return
        except Exception:
            pass

        from pymilvus.milvus_client.index import IndexParams

        index_params = IndexParams()

        extra_params = {}
        if self._index_type == "IVF_FLAT":
            extra_params = {"nlist": 128}
        elif self._index_type == "HNSW":
            extra_params = {"M": 32, "efConstruction": 400}

        index_params.add_index(
            field_name="embedding",
            index_type=self._index_type,
            metric_type=self._metric_type,
            params=extra_params,
        )

        client.create_index(
            collection_name=self._collection_name,
            index_params=index_params,
        )
        logger.info("Milvus 索引已创建：类型=%s，度量=%s", self._index_type, self._metric_type)

    def add_documents(self, documents: List[Document]) -> List[str]:
        if not documents:
            return []

        client = self._get_client()

        texts = [doc.page_content for doc in documents]
        embeddings = self._embedding_function.embed_documents(texts)

        ids = []
        insert_data = []
        for i, doc in enumerate(documents):
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()[:16]
            ids.append(doc_id)
            insert_data.append({
                "id": doc_id,
                "text": doc.page_content,
                "embedding": embeddings[i],
            })

        client.insert(
            collection_name=self._collection_name,
            data=insert_data,
        )
        self._ensure_index()

        stats = client.get_collection_stats(self._collection_name)
        row_count = stats.get("row_count", 0)
        logger.info(
            "Milvus 写入完成：新增 %d 条，当前共 %d 条",
            len(documents), row_count,
        )
        return ids

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        results = self.similarity_search_with_score(query, k)
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        client = self._get_client()

        stats = client.get_collection_stats(self._collection_name)
        if stats.get("row_count", 0) == 0:
            return []

        query_embedding = self._embedding_function.embed_query(query)

        search_params = {"metric_type": self._metric_type}
        if self._index_type == "HNSW":
            search_params["params"] = {"ef": 128}
        elif self._index_type == "IVF_FLAT":
            search_params["params"] = {"nprobe": 32}

        results = client.search(
            collection_name=self._collection_name,
            data=[query_embedding],
            anns_field="embedding",
            search_params=search_params,
            limit=k,
            output_fields=["text"],
        )

        output = []
        for hits in results:
            for hit in hits:
                doc = Document(
                    page_content=hit.get("text", "") or hit.get("entity", {}).get("text", ""),
                    metadata={"id": hit.get("id", "")},
                )
                output.append((doc, float(hit.get("distance", 0))))

        for i, (doc, dist) in enumerate(output):
            if self._metric_type == "COSINE":
                sim = float(dist)
                logger.info("[检索-向量粗筛] #%d id=%s, COSINE相似度=%.1f%%, 内容='%s'",
                            i + 1, doc.metadata.get("id", "")[:16], sim * 100, doc.page_content[:80])
            else:
                logger.info("[检索-向量粗筛] #%d id=%s, distance=%.4f, 内容='%s'",
                            i + 1, doc.metadata.get("id", "")[:16], dist, doc.page_content[:80])

        # COSINE 最低相似度过滤已移至 _hybrid_search_with_scores 的 RRF 融合之前执行，
        # 确保向量粗筛和 BM25 独立完成后再统一过滤，避免过早丢弃向量结果导致混合检索退化

        logger.info(
            "[检索-向量粗筛] query='%s', 索引=%s, k=%d, 返回 %d 条",
            query[:80], self._index_type, k, len(output)
        )

        return output

    def count(self) -> int:
        try:
            client = self._get_client()
            stats = client.get_collection_stats(self._collection_name)
            return stats.get("row_count", 0)
        except Exception:
            return 0

    def clear(self) -> None:
        try:
            client = self._get_client()
            # 先 release collection，否则无法 drop
            try:
                client.release_collection(self._collection_name)
            except Exception:
                pass
            # 直接 drop 整个 collection（清空数据 + 索引）
            client.drop_collection(self._collection_name)
            self._client = None
            logger.info("Milvus 集合已删除（清空）：%s，下次入库自动重建", self._collection_name)
        except Exception as e:
            logger.error("清空 Milvus 集合失败：%s", e)

    def get_all_documents(self) -> List[Document]:
        try:
            client = self._get_client()
            stats = client.get_collection_stats(self._collection_name)
            total = stats.get("row_count", 0)
            if total == 0:
                return []

            all_results = []
            batch_size = 8000
            offset = 0
            while offset < total:
                batch = client.query(
                    collection_name=self._collection_name,
                    filter="id != ''",
                    output_fields=["text"],
                    limit=batch_size,
                    offset=offset,
                )
                if not batch:
                    break
                all_results.extend(batch)
                offset += len(batch)

            return [
                Document(page_content=item.get("text", ""), metadata={"id": item.get("id", "")})
                for item in all_results
            ]
        except Exception:
            return []


def create_vector_store(
    backend: str,
    embedding_function,
    **kwargs,
) -> BaseVectorStore:
    """
    向量数据库工厂函数

    根据 backend 参数创建对应的向量数据库实例。

    Args:
        backend: 后端类型，支持 'chroma', 'faiss', 'milvus'
        embedding_function: Embedding 函数（HuggingFaceEmbeddings 实例）
        **kwargs: 各后端的特定参数
            chroma: host, port, collection_name
            faiss: dimension, index_type, persist_dir
            milvus: host, port, collection_name, dimension, index_type, metric_type

    Returns:
        BaseVectorStore 实例
    """
    if backend == "chroma":
        return ChromaVectorStore(
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 8000),
            collection_name=kwargs.get("collection_name", "rag_documents"),
            embedding_function=embedding_function,
        )
    elif backend == "faiss":
        return FaissVectorStore(
            embedding_function=embedding_function,
            dimension=kwargs.get("dimension", 1024),
            index_type=kwargs.get("index_type", "hnsw"),
            persist_dir=kwargs.get("persist_dir", None),
        )
    elif backend == "milvus":
        return MilvusVectorStore(
            embedding_function=embedding_function,
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", "19530"),
            collection_name=kwargs.get("collection_name", "rag_documents"),
            dimension=kwargs.get("dimension", 1024),
            index_type=kwargs.get("index_type", "IVF_FLAT"),
            metric_type=kwargs.get("metric_type", "COSINE"),
        )
    else:
        raise ValueError(f"不支持的向量数据库后端: {backend}，可选值: chroma, faiss, milvus")


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
    向量数据库管理器（支持多后端切换）

    通过 VECTOR_STORE_BACKEND 环境变量切换后端：
      - chroma（默认）：Chroma 服务模式，数据持久化由服务进程管理
      - faiss：纯本地索引，无需外部服务

    对应流程：
      构建阶段：Chunks → Embedding Model → Vector Database
      查询阶段：Query → Embedding Model → Retriever → Relevant Chunks

    两阶段检索：
      阶段一（粗筛）：向量检索 + BM25 → RRF 融合 → 候选文档
      阶段二（精筛）：Reranker 对候选文档重排序 → 最终结果
    """

    def __init__(self, backend: str = None):
        if backend is None:
            backend = VECTOR_STORE_BACKEND
        self._backend = backend
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vector_store: Optional[BaseVectorStore] = None
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_corpus: List[str] = []
        self._hybrid_enabled: bool = True
        self._reranker_manager = None
        self._reranker_enabled: bool = RERANKER_ENABLED
        self._multi_query_enabled: bool = MULTI_QUERY_ENABLED
        self._on_documents_changed_callbacks: List[callable] = []
        self._doc_count_cache_value: int = 0
        self._doc_count_cache_time: float = 0
        self._search_cache: OrderedDict = OrderedDict()
        self._search_cache_maxsize: int = 128

    @property
    def _bm25_pickle_path(self) -> str:
        """BM25 索引序列化文件路径"""
        return os.path.join(str(BASE_DIR), "data", "bm25_index.pkl")

    def _save_bm25_index(self):
        """将 BM25 索引序列化到本地 pickle 文件"""
        if self._bm25_index is None:
            return
        try:
            data = {
                "corpus": self._bm25_corpus,
                "index": self._bm25_index,
            }
            with open(self._bm25_pickle_path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug("BM25 索引已保存到 %s（%d 篇文档）", self._bm25_pickle_path, len(self._bm25_corpus))
        except Exception as e:
            logger.warning("BM25 索引保存失败：%s", e)

    def _load_bm25_index(self) -> bool:
        """
        从本地 pickle 文件加载 BM25 索引。

        Returns:
            True 表示加载成功，False 表示文件不存在或加载失败
        """
        pickle_path = self._bm25_pickle_path
        if not os.path.exists(pickle_path):
            logger.info("BM25 索引缓存文件不存在，将从向量库重建")
            return False

        try:
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)

            corpus = data.get("corpus", [])
            index = data.get("index")

            if index is None or not corpus:
                logger.warning("BM25 索引缓存文件为空或损坏，将从向量库重建")
                return False

            self._bm25_corpus = corpus
            self._bm25_index = index
            logger.info("BM25 索引已从缓存加载：%d 篇文档", len(corpus))
            return True

        except Exception as e:
            logger.warning("BM25 索引加载失败（%s），将从向量库重建", e)
            return False

    def _delete_bm25_pickle(self):
        """删除 BM25 索引 pickle 文件"""
        pickle_path = self._bm25_pickle_path
        if os.path.exists(pickle_path):
            try:
                os.remove(pickle_path)
                logger.debug("BM25 索引缓存文件已删除")
            except Exception as e:
                logger.warning("BM25 索引缓存文件删除失败：%s", e)

    @property
    def backend(self) -> str:
        """当前使用的向量数据库后端"""
        return self._backend

    def _get_reranker(self):
        """懒加载 Reranker"""
        if self._reranker_manager is None:
            from core.reranker import RerankerManager
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
            try:
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=model_path,
                    model_kwargs={"device": EMBEDDING_DEVICE},
                    encode_kwargs={"normalize_embeddings": True},
                )
            except (AssertionError, RuntimeError) as e:
                if "CUDA" in str(e) or "cuda" in str(e).lower():
                    logger.warning("CUDA 不可用（%s），改用 CPU 加载模型", e)
                    self._embeddings = HuggingFaceEmbeddings(
                        model_name=model_path,
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"normalize_embeddings": True},
                    )
                else:
                    raise
            logger.info("Embedding 模型加载完成（BGE-M3，向量已归一化）")
        return self._embeddings

    def get_or_create_vector_store(self) -> BaseVectorStore:
        """获取或创建向量数据库实例（根据 backend 配置自动选择后端）"""
        if self._vector_store is None:
            self._vector_store = create_vector_store(
                backend=self._backend,
                embedding_function=self.embeddings,
                host=CHROMA_HOST if self._backend == "chroma" else MILVUS_HOST,
                port=CHROMA_PORT if self._backend == "chroma" else MILVUS_PORT,
                collection_name=CHROMA_COLLECTION_NAME if self._backend == "chroma" else MILVUS_COLLECTION_NAME,
                dimension=MILVUS_DIMENSION,
                index_type=MILVUS_INDEX_TYPE,
                metric_type=MILVUS_METRIC_TYPE,
                persist_dir=str(BASE_DIR / "faiss_data") if self._backend == "faiss" else None,
            )
            logger.info(
                "向量数据库已初始化：后端=%s，集合=%s",
                self._backend,
                CHROMA_COLLECTION_NAME if self._backend == "chroma" else (
                    MILVUS_COLLECTION_NAME if self._backend == "milvus" else "faiss_data"
                ),
            )
        return self._vector_store

    def _build_bm25_index(self, documents: List[Document]):
        """
        为当前所有文档构建 BM25 索引，并自动序列化到本地 pickle 文件。
        对应流程：Chunks → BM25 Index（关键词倒排索引）
        """
        if not documents:
            self._bm25_index = None
            self._bm25_corpus = []
            self._delete_bm25_pickle()
            return

        self._bm25_corpus = [doc.page_content for doc in documents]

        tokenized_corpus = [jieba_tokenize(text) for text in self._bm25_corpus]
        self._bm25_index = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 索引构建完成：{len(documents)} 篇文档")

        self._save_bm25_index()

    def add_documents(self, chunks: List[Document]) -> int:
        """
        将 Chunks 向量化并写入向量数据库，同时增量更新 BM25 索引。
        对应流程：Chunks + Embedding Model → Vector Database
        """
        if not chunks:
            logger.warning("没有可写入的 Chunks")
            return 0

        vector_store = self.get_or_create_vector_store()

        backend_label = self._backend.upper()
        logger.info("正在将 %d 个 Chunks 向量化并写入 %s...", len(chunks), backend_label)
        vector_store.add_documents(chunks)
        logger.info("写入完成：%d 个 Chunks 已存储至 %s", len(chunks), backend_label)

        # 增量更新 BM25 索引（只存文本，不存完整 Document 对象以节省内存）
        for chunk in chunks:
            self._bm25_corpus.append(chunk.page_content)

        tokenized = [jieba_tokenize(text) for text in self._bm25_corpus]
        self._bm25_index = BM25Okapi(tokenized)
        logger.info(f"BM25 索引增量更新完成：当前共 {len(self._bm25_corpus)} 篇文档")

        self._save_bm25_index()

        # 入库后清空检索缓存
        self._search_cache.clear()

        # 使文档计数缓存失效
        self._doc_count_cache_time = 0

        # 触发文档变更回调（如清空语义缓存）
        self._notify_documents_changed()

        return len(chunks)

    def add_on_documents_changed(self, callback: callable):
        """注册文档变更回调（入库/清空时触发）"""
        self._on_documents_changed_callbacks.append(callback)

    def _notify_documents_changed(self):
        """触发所有文档变更回调"""
        for cb in self._on_documents_changed_callbacks:
            try:
                cb()
            except Exception as e:
                logger.error("文档变更回调执行失败：%s", e)

    def get_all_documents(self) -> List[Document]:
        """获取向量库中的所有文档（用于 BM25 索引重建）"""
        try:
            return self.get_or_create_vector_store().get_all_documents()
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

        doc_scores = list(enumerate(scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in doc_scores[:top_k]:
            if score > 0:
                doc = Document(page_content=self._bm25_corpus[idx])
                results.append((doc, score))
        return results

    def _apply_cosine_filter(
        self,
        vec_results: List[Tuple[Document, float]],
        log_tag: str = "",
    ) -> List[Tuple[Document, float]]:
        """对向量粗筛结果应用 COSINE 最低相似度阈值过滤，BM25 结果不受此影响。"""
        if not (MILVUS_METRIC_TYPE == "COSINE" and VECTOR_COSINE_MIN_THRESHOLD > 0):
            return vec_results

        min_similarity = VECTOR_COSINE_MIN_THRESHOLD
        before = len(vec_results)
        vec_results = [(doc, dist) for doc, dist in vec_results if dist >= min_similarity]
        if before != len(vec_results):
            tag_suffix = f"({log_tag})" if log_tag else ""
            logger.info(
                "[检索-向量粗筛-过滤%s] COSINE 阈值 %.2f: %d 条 -> %d 条（BM25 不受影响）",
                tag_suffix, VECTOR_COSINE_MIN_THRESHOLD, before, len(vec_results)
            )
        return vec_results

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
            from config import OPENAI_MODEL

            llm = ChatOpenAI(
                model=OPENAI_MODEL,
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

        # Reranker 精排 + 阈值过滤
        if self._reranker_enabled:
            reranker = self._get_reranker()
            return reranker.rerank_with_sources(query, fused, top_k=top_k)

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

        vec_results = self._apply_cosine_filter(vec_results, log_tag="多查询")

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

        vec_results = self._apply_cosine_filter(vec_results, log_tag="纯检索")

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

            # 纯向量检索也支持 Reranker 精排 + 阈值过滤
            if self._reranker_enabled and len(vec_results) > 0:
                docs_with_scores = [(doc, score, "vec") for doc, score in vec_results]
                return self._get_reranker().rerank_with_sources(query, docs_with_scores, top_k=top_k)

            # 不启用 Reranker：应用 COSINE 粗筛过滤后直接返回
            vec_results = self._apply_cosine_filter(vec_results, log_tag="纯向量")
            return [(doc, score, "vec") for doc, score in vec_results]

        return self._hybrid_search_with_scores(query, top_k)

    def _cached_search_with_scores(
        self,
        query: str,
        top_k: int,
        hybrid: bool,
        reranker: bool,
    ) -> tuple:
        """
        带缓存的检索方法（内部使用）。
        使用 OrderedDict 实现 LRU 淘汰，避免 @lru_cache 在实例方法上
        因 self 参与缓存 key 而导致的内存引用无法释放问题。
        缓存 key 由 (query, top_k, hybrid, reranker) 组成。
        入库或清空数据库时会自动清空缓存。
        """
        cache_key = (query, top_k, hybrid, reranker)

        if cache_key in self._search_cache:
            self._search_cache.move_to_end(cache_key)
            return self._search_cache[cache_key]

        saved_hybrid = self._hybrid_enabled
        saved_reranker = self._reranker_enabled

        self._hybrid_enabled = hybrid
        self._reranker_enabled = reranker

        try:
            if not hybrid:
                vector_store = self.get_or_create_vector_store()
                vec_results = vector_store.similarity_search_with_score(query, k=top_k)
                if reranker and len(vec_results) > 0:
                    docs_with_scores = [(doc, score, "vec") for doc, score in vec_results]
                    reranked = self._get_reranker().rerank_with_sources(query, docs_with_scores, top_k=top_k)
                    result = tuple(reranked)
                else:
                    vec_results = self._apply_cosine_filter(vec_results, log_tag="缓存纯向量")
                    result = tuple((doc, score, "vec") for doc, score in vec_results)
            else:
                result = tuple(self._hybrid_search_with_scores(query, top_k))
        finally:
            self._hybrid_enabled = saved_hybrid
            self._reranker_enabled = saved_reranker

        if len(self._search_cache) >= self._search_cache_maxsize:
            self._search_cache.popitem(last=False)

        self._search_cache[cache_key] = result
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
        if self._multi_query_enabled:
            return self._multi_query_search_with_scores(query, top_k, alpha)

        if alpha is None:
            alpha = HYBRID_SEARCH_ALPHA

        vector_store = self.get_or_create_vector_store()

        # 阶段一：粗筛（候选数 > 最终返回数，留出精排空间）
        candidate_k = max(RERANKER_CANDIDATE_K, top_k * 2)
        logger.info("[检索-粗筛开始] query='%s', top_k=%d, candidate_k=%d, alpha=%.2f",
                    query[:80], top_k, candidate_k, alpha if alpha else HYBRID_SEARCH_ALPHA)

        vec_results = vector_store.similarity_search_with_score(query, k=candidate_k)
        logger.info("[检索-向量粗筛] query='%s', 返回 %d 条", query[:80], len(vec_results))

        vec_results = self._apply_cosine_filter(vec_results, log_tag="主流程")

        bm25_results = self._bm25_search(query, candidate_k)
        logger.info("[检索-BM25粗筛] query='%s', 返回 %d 条", query[:80], len(bm25_results))
        for i, (doc, score) in enumerate(bm25_results):
            logger.info("[检索-BM25粗筛] #%d score=%.4f, 内容='%s'",
                        i + 1, score, doc.page_content[:80])

        fused = self._rrf_fusion(vec_results, bm25_results, alpha=alpha)
        logger.info("[检索-RRF融合] 向量 %d 条 + BM25 %d 条 -> 融合 %d 条",
                    len(vec_results), len(bm25_results), len(fused))
        for i, (doc, score, rtype) in enumerate(fused):
            logger.info("[检索-RRF融合] #%d score=%.6f, type=%s, 内容='%s'",
                        i + 1, score, rtype, doc.page_content[:80])

        # 阶段二：精筛（Reranker 重排序 + 阈值过滤）
        if self._reranker_enabled:
            logger.info("[检索-精筛开始] Reranker 对 %d 条候选重排序", len(fused))
            reranker = self._get_reranker()
            rerank_results = reranker.rerank_with_sources(query, fused, top_k=top_k)
            logger.info("[检索-精筛完成] 最终返回 %d 条", len(rerank_results))
            for i, (doc, score, rtype) in enumerate(rerank_results):
                logger.info("[检索-精筛结果] #%d score=%.1f%%, type=%s, 内容='%s'",
                            i + 1, score, rtype, doc.page_content[:80])
            return rerank_results

        # 未启用 Reranker：直接返回 RRF 融合结果
        return [(doc, score, rtype) for doc, score, rtype in fused[:top_k]]

    def get_retriever(self, top_k: int = RETRIEVAL_TOP_K):
        """获取支持混合检索 + Reranker 的 LangChain Retriever（用于 RAG Chain 集成）"""
        return HybridRetriever(vs_manager=self, top_k=top_k)

    def get_document_count(self) -> int:
        """获取当前向量数据库中存储的文档数量（带5秒缓存，避免频繁远程调用）"""
        now = time.time()
        if self._doc_count_cache_time > 0 and (now - self._doc_count_cache_time) < 5:
            return self._doc_count_cache_value
        try:
            count = self.get_or_create_vector_store().count()
            self._doc_count_cache_value = count
            self._doc_count_cache_time = now
            return count
        except Exception:
            return self._doc_count_cache_value

    def clear_collection(self) -> bool:
        """清空向量数据库中的所有数据（谨慎使用）"""
        try:
            self.get_or_create_vector_store().clear()
            self._bm25_index = None
            self._bm25_corpus = []
            self._delete_bm25_pickle()
            self._search_cache.clear()
            self._doc_count_cache_time = 0
            self._notify_documents_changed()
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

    def set_multi_query(self, enabled: bool):
        """开关多查询融合检索（会额外调用 LLM 生成查询变体）"""
        self._multi_query_enabled = enabled
        logger.info(f"多查询融合检索已{'启用' if enabled else '关闭'}")

    def is_multi_query_enabled(self) -> bool:
        """查询多查询融合检索状态"""
        return self._multi_query_enabled
