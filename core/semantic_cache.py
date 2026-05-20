"""
semantic_cache.py - 语义缓存模块

基于向量相似度的问答缓存，支持 ChromaDB 和 Milvus 两种后端存储。
当用户提出语义相似的问题时，直接返回缓存的答案，跳过检索和 LLM 生成。

核心机制：
  1. 缓存查询：将用户问题向量化，在缓存集合中 kNN 搜索最相似的历史问题
  2. 相似度阈值：余弦相似度 >= 阈值（默认 0.95）才视为命中
  3. LFU 淘汰：缓存条目数超过上限时，淘汰命中次数最少的条目
  4. TTL 过期：缓存条目超过有效期后自动失效
  5. 入库失效：知识库更新后自动清空所有缓存
"""
import hashlib
import json
import logging
import os
import socket
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import chromadb

logger = logging.getLogger(__name__)

CACHE_COLLECTION_NAME = "rag_semantic_cache"


class LangChainEmbeddingAdapter(chromadb.EmbeddingFunction):
    """
    LangChain Embedding 适配器

    将 LangChain 的 HuggingFaceEmbeddings 接口适配为 Chroma 要求的
    EmbeddingFunction 接口（__call__(self, input: List[str]) -> List[List[float]]）。

    解决 Chroma 0.4.16+ 版本中 EmbeddingFunction 签名检查不兼容的问题。
    """

    def __init__(self, lc_embeddings):
        self._lc_embeddings = lc_embeddings

    def __call__(self, input):
        return self._lc_embeddings.embed_documents(input)


class CacheStorageBackend(ABC):
    """缓存存储后端抽象基类，定义统一的缓存 CRUD 接口"""

    @abstractmethod
    def initialize(self) -> bool:
        """初始化存储连接和集合，返回是否成功"""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """检查存储是否就绪可用"""
        ...

    @abstractmethod
    def count(self) -> int:
        """获取缓存条目总数"""
        ...

    @abstractmethod
    def get_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        根据 ID 获取单条缓存
        返回: {"id": str, "document": str, "metadata": dict} 或 None
        """
        ...

    @abstractmethod
    def get_all_metadata(self) -> List[Dict]:
        """
        获取所有缓存条目的元数据
        返回: [{"id": str, "metadata": dict}, ...]
        """
        ...

    @abstractmethod
    def search_similar(self, query_text: str, n_results: int = 1) -> List[Dict]:
        """
        语义搜索相似缓存
        返回: [{"id": str, "document": str, "metadata": dict, "distance": float}, ...]
        """
        ...

    @abstractmethod
    def add_entry(self, doc_id: str, question: str, answer: str, sources_json: str, metadata: Dict) -> bool:
        """添加缓存条目，返回是否成功"""
        ...

    @abstractmethod
    def update_metadata(self, doc_id: str, metadata: Dict) -> bool:
        """更新缓存条目元数据，返回是否成功"""
        ...

    @abstractmethod
    def delete_entries(self, ids: List[str]) -> bool:
        """删除缓存条目，返回是否成功"""
        ...

    @abstractmethod
    def clear_all(self) -> bool:
        """清空所有缓存，返回是否成功"""
        ...


class ChromaCacheStorage(CacheStorageBackend):
    """基于 ChromaDB 的缓存存储后端"""

    def __init__(self, host: str = "localhost", port: int = 8000, embedding_function=None):
        self._host = host
        self._port = port
        self._embedding_function = embedding_function
        self._client: Optional[chromadb.HttpClient] = None
        self._collection: Optional[chromadb.Collection] = None

    def _check_reachable(self) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((self._host, self._port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def initialize(self) -> bool:
        if not self._check_reachable():
            logger.warning("ChromaDB 服务不可达（%s:%s），语义缓存不可用", self._host, self._port)
            return False
        try:
            self._client = chromadb.HttpClient(host=self._host, port=self._port)
            self._collection = self._client.get_or_create_collection(
                name=CACHE_COLLECTION_NAME,
                embedding_function=self._embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("语义缓存后端：ChromaDB（%s:%s）", self._host, self._port)
            return True
        except Exception as e:
            logger.warning("ChromaDB 缓存集合初始化失败：%s", e)
            return False

    def is_ready(self) -> bool:
        return self._collection is not None

    def count(self) -> int:
        try:
            return self._collection.count() if self._collection else 0
        except Exception:
            return 0

    def get_by_id(self, doc_id: str) -> Optional[Dict]:
        try:
            result = self._collection.get(ids=[doc_id], include=["metadatas", "documents"])
            if not result["ids"]:
                return None
            return {
                "id": result["ids"][0],
                "document": result["documents"][0] if result["documents"] else "",
                "metadata": result["metadatas"][0] if result["metadatas"] else {},
            }
        except Exception as e:
            logger.debug("ChromaDB get_by_id 失败：%s", e)
            return None

    def get_all_metadata(self) -> List[Dict]:
        try:
            result = self._collection.get(include=["metadatas"])
            if not result["ids"]:
                return []
            return [
                {"id": rid, "metadata": meta}
                for rid, meta in zip(result["ids"], result["metadatas"])
            ]
        except Exception as e:
            logger.warning("ChromaDB get_all_metadata 失败：%s", e)
            return []

    def search_similar(self, query_text: str, n_results: int = 1) -> List[Dict]:
        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["metadatas", "documents", "distances"],
            )
            if not results["ids"] or not results["ids"][0]:
                return []

            items = []
            for i in range(len(results["ids"][0])):
                items.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i] if results["documents"] and results["documents"][0] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i],
                })
            return items
        except Exception as e:
            logger.error("ChromaDB search_similar 失败：%s", e)
            return []

    def add_entry(self, doc_id: str, question: str, answer: str, sources_json: str, metadata: Dict) -> bool:
        try:
            self._collection.add(
                ids=[doc_id],
                documents=[answer],
                metadatas=[metadata],
            )
            return True
        except Exception as e:
            logger.error("ChromaDB add_entry 失败：%s", e)
            return False

    def update_metadata(self, doc_id: str, metadata: Dict) -> bool:
        try:
            self._collection.update(ids=[doc_id], metadatas=[metadata])
            return True
        except Exception as e:
            logger.debug("ChromaDB update_metadata 失败：%s", e)
            return False

    def delete_entries(self, ids: List[str]) -> bool:
        try:
            self._collection.delete(ids=ids)
            return True
        except Exception as e:
            logger.error("ChromaDB delete_entries 失败：%s", e)
            return False

    def clear_all(self) -> bool:
        try:
            self._client.delete_collection(CACHE_COLLECTION_NAME)
            self._collection = None
            return True
        except Exception as e:
            logger.error("ChromaDB clear_all 失败：%s", e)
            return False


class MilvusCacheStorage(CacheStorageBackend):
    """基于 Milvus 的缓存存储后端

    索引策略：
      - 默认 HNSW（M=32, efConstruction=400），适合缓存的小规模高精度场景
      - 支持通过参数切换为 IVF_FLAT 或 FLAT
      - 在首次写入数据后自动创建索引
    """

    def __init__(self, host: str = "localhost", port: int = 19530, embedding_function=None, dimension: int = 1024,
                 index_type: str = "HNSW", metric_type: str = "COSINE"):
        self._host = host
        self._port = port
        self._embedding_function = embedding_function
        self._dimension = dimension
        self._index_type = index_type
        self._metric_type = metric_type
        self._client = None
        self._collection_ready = False

    def _check_reachable(self) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((self._host, self._port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def initialize(self) -> bool:
        if not self._check_reachable():
            logger.warning("Milvus 服务不可达（%s:%s），语义缓存不可用", self._host, self._port)
            return False
        try:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=f"http://{self._host}:{self._port}")

            if self._client.has_collection(CACHE_COLLECTION_NAME):
                stats = self._client.get_collection_stats(CACHE_COLLECTION_NAME)
                existing_count = stats.get("row_count", 0)
                logger.info("[缓存-初始化] 已有集合: %s, 实体数=%d, 开始检查schema...",
                            CACHE_COLLECTION_NAME, existing_count)
                if not self._check_collection_schema():
                    logger.info("[缓存-初始化] schema不匹配, 删除重建(旧数据=%d条将丢失)", existing_count)
                    self._client.drop_collection(CACHE_COLLECTION_NAME)
                    self._create_cache_collection()
                else:
                    self._client.load_collection(CACHE_COLLECTION_NAME)
                    logger.info("[缓存-初始化] schema通过, 已加载, 实体数=%d", existing_count)
                    self._ensure_cache_index()
            else:
                logger.info("[缓存-初始化] 集合不存在, 新建: %s", CACHE_COLLECTION_NAME)
                self._create_cache_collection()

            self._collection_ready = True
            logger.info("语义缓存后端：Milvus（%s:%s）", self._host, self._port)
            return True
        except Exception as e:
            logger.warning("Milvus 缓存集合初始化失败：%s", e)
            return False

    def _create_cache_collection(self):
        """创建缓存集合并加载到内存（此时不创建索引，等首次写入数据后再创建）"""
        try:
            self._client.create_collection(
                collection_name=CACHE_COLLECTION_NAME,
                dimension=self._dimension,
                metric_type=self._metric_type,
                auto_id=False,
                id_type="string",
                max_length=64,
                varchar_max_length=65535,
                enable_dynamic_field=True,
            )
        except TypeError:
            # 旧版 pymilvus 不支持 enable_dynamic_field 参数，使用显式 schema
            from pymilvus import CollectionSchema, FieldSchema, DataType as dt

            fields = [
                FieldSchema(name="id", dtype=dt.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="vector", dtype=dt.FLOAT_VECTOR, dim=self._dimension),
            ]
            # 显式定义所有标量字段，确保 schema 包含它们
            scalar_fields = [
                ("text", dt.VARCHAR, 65535),
                ("query", dt.VARCHAR, 65535),
                ("sources_json", dt.VARCHAR, 65535),
                ("hit_count", dt.INT64),
                ("created_at", dt.VARCHAR, 128),
                ("last_hit_at", dt.VARCHAR, 128),
            ]
            for name, dtype, *args in scalar_fields:
                fields.append(FieldSchema(name=name, dtype=dtype, **({"max_length": args[0]} if args else {})))

            schema = CollectionSchema(fields=fields)
            self._client.create_collection(
                collection_name=CACHE_COLLECTION_NAME,
                dimension=self._dimension,
                metric_type=self._metric_type,
                schema=schema,
            )
            logger.info("Milvus 使用显式 schema 创建缓存集合（旧版 pymilvus）")

        self._client.load_collection(CACHE_COLLECTION_NAME)
        logger.info("Milvus 缓存集合已创建并加载：%s（维度=%d，索引类型=%s，度量=%s）",
                    CACHE_COLLECTION_NAME, self._dimension, self._index_type, self._metric_type)

    def _check_collection_schema(self) -> bool:
        """检查已有集合的 schema 是否满足缓存需求（VARCHAR 主键 + 动态字段启用）"""
        try:
            info = self._client.describe_collection(CACHE_COLLECTION_NAME)
            logger.info("[缓存-schema检查] describe_collection keys=%s", list(info.keys()))

            # 1) 检查动态字段是否启用
            dynamic_enabled = (
                info.get("enable_dynamic_field")
                or info.get("enable_dynamic")
                or info.get("enableDynamicField", False)
            )
            logger.info("[缓存-schema检查] enable_dynamic_field=%s, enable_dynamic=%s, enableDynamicField=%s -> 结果=%s",
                        info.get("enable_dynamic_field"), info.get("enable_dynamic"),
                        info.get("enableDynamicField"), dynamic_enabled)
            if not dynamic_enabled:
                logger.warning("[缓存-schema检查] 失败: 未启用动态字段")
                return False

            # 2) 检查主键类型是否为 VARCHAR
            _varchar_types = {21}  # DataType.VARCHAR 的值
            try:
                from pymilvus import DataType
                _varchar_types.add(int(DataType.VARCHAR))
            except Exception:
                pass

            for field in info.get("fields", []):
                if field.get("is_primary") or field.get("primary_key"):
                    field_type = field.get("type", "")

                    if isinstance(field_type, int):
                        if field_type in _varchar_types:
                            return True
                        logger.warning(
                            "缓存集合 %s 的主键类型为 DataType.%s，需要 VARCHAR(21)，将删除重建",
                            CACHE_COLLECTION_NAME, field_type,
                        )
                        return False

                    field_type_str = str(field_type).upper()
                    if "VARCHAR" in field_type_str or "STRING" in field_type_str:
                        return True
                    logger.warning(
                        "缓存集合 %s 的主键类型为 %s，需要 VARCHAR，将删除重建",
                        CACHE_COLLECTION_NAME, field_type_str,
                    )
                    return False

            # 没有找到主键字段，认为 schema 正常（可能是旧版本格式）
            return True
        except Exception as e:
            logger.warning("检查缓存集合 schema 失败：%s，将删除重建", e)
            return False

    def _ensure_cache_index(self):
        """确保缓存集合已创建索引

        索引创建策略：
          - HNSW（默认）：M=32, efConstruction=400，高精度图构建，适合缓存小规模场景
          - IVF_FLAT：nlist=128，适合百万级数据
          - 若数据量不足以训练 IVF_FLAT（< nlist），自动回退为 FLAT 暴力搜索
        """
        client = self._client

        try:
            index_info = client.describe_index(CACHE_COLLECTION_NAME, "vector")
            if index_info:
                return
        except Exception:
            pass

        from pymilvus.milvus_client.index import IndexParams

        index_params = IndexParams()

        # 检查当前数据量，不足 nlist=128 时回退为 FLAT
        actual_index_type = self._index_type
        extra_params = {}

        if self._index_type == "IVF_FLAT":
            stats = client.get_collection_stats(CACHE_COLLECTION_NAME)
            row_count = stats.get("row_count", 0)
            if row_count < 128:
                logger.info(
                    "缓存数据量 %d < nlist(128)，IVF_FLAT 训练数据不足，回退为 FLAT 暴力搜索",
                    row_count,
                )
                actual_index_type = "FLAT"
            else:
                extra_params = {"nlist": 128}
        elif self._index_type == "HNSW":
            extra_params = {"M": 32, "efConstruction": 400}

        index_params.add_index(
            field_name="vector",
            index_type=actual_index_type,
            metric_type=self._metric_type,
            params=extra_params,
        )

        client.create_index(
            collection_name=CACHE_COLLECTION_NAME,
            index_params=index_params,
        )
        logger.info("缓存集合索引已创建：类型=%s，度量=%s，参数=%s",
                    actual_index_type, self._metric_type, extra_params)

    def is_ready(self) -> bool:
        return self._collection_ready and self._client is not None

    def count(self) -> int:
        try:
            stats = self._client.get_collection_stats(CACHE_COLLECTION_NAME)
            return stats.get("row_count", 0)
        except Exception:
            return 0

    def get_by_id(self, doc_id: str) -> Optional[Dict]:
        try:
            result = self._client.get(
                collection_name=CACHE_COLLECTION_NAME,
                ids=[doc_id],
                output_fields=["text", "query", "sources_json", "hit_count", "created_at", "last_hit_at"],
            )
            logger.debug("Milvus get_by_id: doc_id=%s, result=%s", doc_id, result)
            if not result:
                logger.info("Milvus get_by_id 未找到：doc_id=%s", doc_id)
                return None
            item = result[0]
            return {
                "id": item.get("id", ""),
                "document": item.get("text", ""),
                "metadata": {
                    "query": item.get("query", ""),
                    "sources_json": item.get("sources_json", "[]"),
                    "hit_count": item.get("hit_count", 0),
                    "created_at": item.get("created_at", ""),
                    "last_hit_at": item.get("last_hit_at", ""),
                },
            }
        except Exception as e:
            logger.debug("Milvus get_by_id 失败：%s", e)
            return None

    def get_all_metadata(self) -> List[Dict]:
        try:
            stats = self._client.get_collection_stats(CACHE_COLLECTION_NAME)
            total = stats.get("row_count", 0)
            if total == 0:
                return []

            all_results = []
            batch_size = 8000
            offset = 0
            while offset < total:
                batch = self._client.query(
                    collection_name=CACHE_COLLECTION_NAME,
                    filter="id != ''",
                    output_fields=["query", "sources_json", "hit_count", "created_at", "last_hit_at"],
                    limit=batch_size,
                    offset=offset,
                )
                if not batch:
                    break
                all_results.extend(batch)
                offset += len(batch)

            return [
                {
                    "id": item.get("id", ""),
                    "metadata": {
                        "query": item.get("query", ""),
                        "sources_json": item.get("sources_json", "[]"),
                        "hit_count": item.get("hit_count", 0),
                        "created_at": item.get("created_at", ""),
                        "last_hit_at": item.get("last_hit_at", ""),
                    },
                }
                for item in all_results
            ]
        except Exception as e:
            logger.warning("Milvus get_all_metadata 失败：%s", e)
            return []

    def search_similar(self, query_text: str, n_results: int = 1) -> List[Dict]:
        try:
            query_embedding = self._embedding_function.embed_documents([query_text])[0]

            logger.info("[缓存-诊断] query_embedding 维度=%d, 前5个值=%.6f,%.6f,%.6f,%.6f,%.6f, 集合维度配置=%d",
                        len(query_embedding),
                        query_embedding[0], query_embedding[1], query_embedding[2], query_embedding[3], query_embedding[4],
                        self._dimension)

            # 构建搜索参数
            search_params = {"metric_type": self._metric_type}
            if self._index_type == "HNSW":
                search_params["params"] = {"ef": 128}
            elif self._index_type == "IVF_FLAT":
                search_params["params"] = {"nprobe": 16}

            results = self._client.search(
                collection_name=CACHE_COLLECTION_NAME,
                data=[query_embedding],
                anns_field="vector",
                search_params=search_params,
                limit=n_results,
                output_fields=["text", "query", "sources_json", "hit_count", "created_at", "last_hit_at"],
            )

            logger.info("Milvus search raw: results_type=%s, len=%s, results[0] type=%s",
                        type(results).__name__, len(results) if results else 0,
                        type(results[0]).__name__ if results and results[0] else "N/A")

            if not results or not results[0]:
                logger.info("Milvus search_similar 无结果：collection=%s", CACHE_COLLECTION_NAME)
                return []

            items = []
            for hit in results[0]:
                # 打印 hit 的所有 key 以确认字段名
                logger.info("Milvus hit keys=%s, id=%s, distance=%s, score=%s",
                            list(hit.keys()) if isinstance(hit, dict) else type(hit),
                            hit.get("id", "N/A") if isinstance(hit, dict) else "N/A",
                            hit.get("distance", "N/A") if isinstance(hit, dict) else "N/A",
                            hit.get("score", "N/A") if isinstance(hit, dict) else "N/A")
                entity = hit.get("entity", {})
                items.append({
                    "id": hit.get("id", ""),
                    "document": entity.get("text", ""),
                    "metadata": {
                        "query": entity.get("query", ""),
                        "sources_json": entity.get("sources_json", "[]"),
                        "hit_count": entity.get("hit_count", 0),
                        "created_at": entity.get("created_at", ""),
                        "last_hit_at": entity.get("last_hit_at", ""),
                    },
                    "distance": hit.get("distance", hit.get("score", 1.0)),
                })
            return items
        except Exception as e:
            logger.error("Milvus search_similar 失败：%s", e, exc_info=True)
            return []

    def add_entry(self, doc_id: str, question: str, answer: str, sources_json: str, metadata: Dict) -> bool:
        try:
            embedding = self._embedding_function.embed_documents([question])[0]

            data = [{
                "id": doc_id,
                "vector": embedding,
                "text": answer,
                "query": metadata.get("query", question),
                "sources_json": sources_json,
                "hit_count": metadata.get("hit_count", 0),
                "created_at": metadata.get("created_at", ""),
                "last_hit_at": metadata.get("last_hit_at", ""),
            }]

            self._client.insert(collection_name=CACHE_COLLECTION_NAME, data=data)
            try:
                self._client.flush(CACHE_COLLECTION_NAME)
            except Exception:
                pass

            # 验证写入：立即用 get_by_id 确认数据可见
            verify = self._client.get(
                collection_name=CACHE_COLLECTION_NAME,
                ids=[doc_id],
                output_fields=["query"],
            )
            if not verify or not verify[0].get("query"):
                logger.error("Milvus insert 后验证失败：doc_id=%s 写入后不可见", doc_id)
                return False

            # 写入成功后确保索引存在（与主知识库 add_documents 后调用 _ensure_index 策略一致）
            self._ensure_cache_index()

            logger.info("Milvus insert 验证成功：doc_id=%s, query=%s", doc_id, verify[0].get("query", "")[:40])
            return True
        except Exception as e:
            logger.error("Milvus add_entry 失败：%s", e, exc_info=True)
            return False

    def update_metadata(self, doc_id: str, metadata: Dict) -> bool:
        try:
            existing = self._client.get(
                collection_name=CACHE_COLLECTION_NAME,
                ids=[doc_id],
                output_fields=["text", "query", "sources_json"],
            )
            if not existing:
                return False

            item = existing[0]

            # 尝试获取向量（pymilvus 3.0 中 get 可能不返回向量字段）
            vector = item.get("vector", None)
            if vector is None or (isinstance(vector, list) and len(vector) == 0):
                # 无法获取向量，使用 delete + insert 方式更新
                # 先通过 search 获取向量
                try:
                    query_embedding = self._embedding_function.embed_documents([item.get("query", "")])[0]
                    search_results = self._client.search(
                        collection_name=CACHE_COLLECTION_NAME,
                        data=[query_embedding],
                        limit=1,
                        output_fields=["text", "query", "sources_json"],
                    )
                    if search_results and search_results[0]:
                        # 找到了，用 delete + insert 更新
                        self._client.delete(collection_name=CACHE_COLLECTION_NAME, ids=[doc_id])
                        # 重新插入（使用原始文本生成向量）
                        text = item.get("text", "")
                        new_embedding = self._embedding_function.embed_documents([text])[0]
                        data = [{
                            "id": doc_id,
                            "vector": new_embedding,
                            "text": text,
                            "query": metadata.get("query", item.get("query", "")),
                            "sources_json": metadata.get("sources_json", item.get("sources_json", "[]")),
                            "hit_count": metadata.get("hit_count", item.get("hit_count", 0)),
                            "created_at": metadata.get("created_at", item.get("created_at", "")),
                            "last_hit_at": metadata.get("last_hit_at", item.get("last_hit_at", "")),
                        }]
                        self._client.insert(collection_name=CACHE_COLLECTION_NAME, data=data)
                        return True
                except Exception:
                    pass
                # 无法安全更新，跳过
                logger.debug("Milvus update_metadata 跳过：无法获取向量字段")
                return False

            data = [{
                "id": doc_id,
                "vector": vector,
                "text": item.get("text", ""),
                "query": metadata.get("query", item.get("query", "")),
                "sources_json": metadata.get("sources_json", item.get("sources_json", "[]")),
                "hit_count": metadata.get("hit_count", item.get("hit_count", 0)),
                "created_at": metadata.get("created_at", item.get("created_at", "")),
                "last_hit_at": metadata.get("last_hit_at", item.get("last_hit_at", "")),
            }]

            self._client.upsert(collection_name=CACHE_COLLECTION_NAME, data=data)
            return True
        except Exception as e:
            logger.debug("Milvus update_metadata 失败：%s", e)
            return False

    def delete_entries(self, ids: List[str]) -> bool:
        try:
            self._client.delete(collection_name=CACHE_COLLECTION_NAME, ids=ids)
            return True
        except Exception as e:
            logger.error("Milvus delete_entries 失败：%s", e)
            return False

    def clear_all(self) -> bool:
        try:
            self._client.drop_collection(CACHE_COLLECTION_NAME)
            self._collection_ready = False
            return True
        except Exception as e:
            logger.error("Milvus clear_all 失败：%s", e)
            return False


class FaissCacheStorage(CacheStorageBackend):
    """基于 Faiss 的本地缓存存储后端（无需外部服务，数据存于内存）"""

    def __init__(self, embedding_function=None, dimension: int = 1024, persist_dir: str = None):
        self._embedding_function = embedding_function
        self._dimension = dimension
        self._persist_dir = persist_dir
        self._index = None
        self._ids: List[str] = []
        self._documents: List[str] = []
        self._metadata_list: List[Dict] = []
        self._faiss_id_to_idx: Dict[int, int] = {}

    def initialize(self) -> bool:
        try:
            import faiss
            import numpy as np

            base_index = faiss.IndexFlatIP(self._dimension)
            self._index = faiss.IndexIDMap(base_index)

            if self._persist_dir:
                os.makedirs(self._persist_dir, exist_ok=True)
                self._load_from_disk()

            logger.info("语义缓存后端：Faiss（本地内存，维度=%d）", self._dimension)
            return True
        except Exception as e:
            logger.warning("Faiss 缓存初始化失败：%s", e)
            return False

    def is_ready(self) -> bool:
        return self._index is not None

    def count(self) -> int:
        return len(self._ids)

    def _find_idx(self, doc_id: str) -> int:
        """根据字符串 ID 查找在列表中的位置"""
        try:
            return self._ids.index(doc_id)
        except ValueError:
            return -1

    def get_by_id(self, doc_id: str) -> Optional[Dict]:
        idx = self._find_idx(doc_id)
        if idx < 0:
            return None
        return {
            "id": self._ids[idx],
            "document": self._documents[idx],
            "metadata": self._metadata_list[idx],
        }

    def get_all_metadata(self) -> List[Dict]:
        return [
            {"id": rid, "metadata": meta}
            for rid, meta in zip(self._ids, self._metadata_list)
        ]

    def search_similar(self, query_text: str, n_results: int = 1) -> List[Dict]:
        try:
            import faiss
            import numpy as np

            if not self._ids:
                return []

            query_embedding = self._embedding_function.embed_documents([query_text])[0]
            qv = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(qv)

            actual_k = min(n_results, len(self._ids))
            distances, indices = self._index.search(qv, actual_k)

            items = []
            for dist, faiss_id in zip(distances[0], indices[0]):
                if faiss_id == -1:
                    continue
                idx = self._find_idx_by_faiss_id(int(faiss_id))
                if idx < 0:
                    continue
                items.append({
                    "id": self._ids[idx],
                    "document": self._documents[idx],
                    "metadata": self._metadata_list[idx],
                    "distance": float(dist),
                })
            return items
        except Exception as e:
            logger.error("Faiss search_similar 失败：%s", e)
            return []

    def _find_idx_by_faiss_id(self, faiss_id: int) -> int:
        """根据 Faiss 内部整数 ID 查找在列表中的位置（O(1)）"""
        return self._faiss_id_to_idx.get(faiss_id, -1)

    @staticmethod
    def _string_to_int_id(doc_id: str) -> int:
        """将字符串 ID 转为 Faiss 兼容的整数 ID"""
        return int(hashlib.md5(doc_id.encode()).hexdigest()[:15], 16) % (2 ** 63 - 1)

    def add_entry(self, doc_id: str, question: str, answer: str, sources_json: str, metadata: Dict) -> bool:
        try:
            import faiss
            import numpy as np

            embedding = self._embedding_function.embed_documents([question])[0]
            vec = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(vec)

            int_id = self._string_to_int_id(doc_id)

            existing_idx = self._find_idx(doc_id)
            if existing_idx >= 0:
                self._documents[existing_idx] = answer
                self._metadata_list[existing_idx] = metadata
                self._index.remove_ids(np.array([int_id], dtype=np.int64))
            else:
                existing_idx = len(self._ids)
                self._ids.append(doc_id)
                self._documents.append(answer)
                self._metadata_list.append(metadata)

            self._index.add_with_ids(vec, np.array([int_id], dtype=np.int64))
            self._faiss_id_to_idx[int_id] = existing_idx
            return True
        except Exception as e:
            logger.error("Faiss add_entry 失败：%s", e)
            return False

    def update_metadata(self, doc_id: str, metadata: Dict) -> bool:
        idx = self._find_idx(doc_id)
        if idx < 0:
            return False
        self._metadata_list[idx] = metadata
        return True

    def delete_entries(self, ids: List[str]) -> bool:
        try:
            import numpy as np

            int_ids = []
            for doc_id in ids:
                idx = self._find_idx(doc_id)
                if idx >= 0:
                    int_id = self._string_to_int_id(doc_id)
                    int_ids.append(int_id)
                    self._ids[idx] = None
                    self._documents[idx] = None
                    self._metadata_list[idx] = None
                    self._faiss_id_to_idx.pop(int_id, None)

            if int_ids:
                self._index.remove_ids(np.array(int_ids, dtype=np.int64))

            self._ids = [x for x in self._ids if x is not None]
            self._documents = [x for x in self._documents if x is not None]
            self._metadata_list = [x for x in self._metadata_list if x is not None]

            self._faiss_id_to_idx = {
                self._string_to_int_id(rid): i
                for i, rid in enumerate(self._ids)
            }

            return True
        except Exception as e:
            logger.error("Faiss delete_entries 失败：%s", e)
            return False

    def clear_all(self) -> bool:
        try:
            import faiss

            base_index = faiss.IndexFlatIP(self._dimension)
            self._index = faiss.IndexIDMap(base_index)
            self._ids.clear()
            self._documents.clear()
            self._metadata_list.clear()
            self._faiss_id_to_idx.clear()
            return True
        except Exception as e:
            logger.error("Faiss clear_all 失败：%s", e)
            return False

    def _load_from_disk(self):
        """从磁盘加载持久化的缓存数据"""
        import faiss

        index_path = os.path.join(self._persist_dir, "cache_faiss_index.bin")
        meta_path = os.path.join(self._persist_dir, "cache_faiss_meta.json")

        if os.path.exists(index_path) and os.path.exists(meta_path):
            try:
                self._index = faiss.read_index(index_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._ids = data.get("ids", [])
                self._documents = data.get("documents", [])
                self._metadata_list = data.get("metadata_list", [])
                self._faiss_id_to_idx = {
                    self._string_to_int_id(rid): i
                    for i, rid in enumerate(self._ids)
                }
                logger.info("已从磁盘加载 Faiss 缓存：%d 条", len(self._ids))
            except Exception as e:
                logger.warning("加载 Faiss 缓存失败，将使用空缓存：%s", e)

    def save_to_disk(self):
        """将缓存数据持久化到磁盘"""
        if not self._persist_dir:
            return
        try:
            import faiss

            index_path = os.path.join(self._persist_dir, "cache_faiss_index.bin")
            meta_path = os.path.join(self._persist_dir, "cache_faiss_meta.json")

            faiss.write_index(self._index, index_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "ids": self._ids,
                    "documents": self._documents,
                    "metadata_list": self._metadata_list,
                }, f, ensure_ascii=False)
            logger.debug("Faiss 缓存已持久化：%d 条", len(self._ids))
        except Exception as e:
            logger.warning("Faiss 缓存持久化失败：%s", e)


def _create_cache_storage(
    backend: str,
    embedding_function,
    chroma_host: str = "localhost",
    chroma_port: int = 8000,
    milvus_host: str = "localhost",
    milvus_port: int = 19530,
    milvus_dimension: int = 1024,
    milvus_index_type: str = "HNSW",
    milvus_metric_type: str = "COSINE",
    faiss_dimension: int = 1024,
    faiss_persist_dir: str = None,
) -> CacheStorageBackend:
    """缓存存储后端工厂函数"""
    if backend == "milvus":
        return MilvusCacheStorage(
            host=milvus_host,
            port=milvus_port,
            embedding_function=embedding_function,
            dimension=milvus_dimension,
            index_type=milvus_index_type,
            metric_type=milvus_metric_type,
        )
    elif backend == "faiss":
        return FaissCacheStorage(
            embedding_function=embedding_function,
            dimension=faiss_dimension,
            persist_dir=faiss_persist_dir,
        )
    else:
        chroma_ef = embedding_function
        if not isinstance(chroma_ef, chromadb.EmbeddingFunction):
            chroma_ef = LangChainEmbeddingAdapter(embedding_function)
        return ChromaCacheStorage(
            host=chroma_host,
            port=chroma_port,
            embedding_function=chroma_ef,
        )


class SemanticCache:
    """
    语义缓存管理器

    使用独立的向量集合存储缓存条目，每条记录包含：
      - id: 唯一标识（基于问题内容的 MD5）
      - query: 原始用户问题
      - answer: LLM 生成的回答
      - query_embedding: 问题的向量表示
      - sources_json: 来源文档的 JSON 字符串
      - hit_count: 命中次数（用于 LFU 淘汰）
      - created_at: 创建时间（ISO 格式，用于 TTL 过期）
      - last_hit_at: 最后命中时间

    缓存检索采用单阶段策略：
      向量检索 top_k=N -> COSINE 相似度阈值过滤（默认 0.70）
      直接取相似度最高的候选作为命中结果，不再使用 Reranker 精排
    """

    def __init__(
        self,
        backend: str = "chroma",
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        milvus_dimension: int = 1024,
        milvus_index_type: str = "HNSW",
        milvus_metric_type: str = "COSINE",
        faiss_dimension: int = 1024,
        faiss_persist_dir: str = None,
        embedding_function=None,
        enabled: bool = True,
        similarity_threshold: float = 0.95,
        coarse_threshold: float = 0.88,
        reranker_threshold: float = 0.85,
        candidate_count: int = 10,
        max_entries: int = 1000,
        ttl_hours: int = 24,
    ):
        self.backend = backend
        self.enabled = enabled
        self.similarity_threshold = similarity_threshold
        self.coarse_threshold = coarse_threshold
        self.reranker_threshold = reranker_threshold
        self.candidate_count = candidate_count
        self.max_entries = max_entries
        self.ttl_hours = ttl_hours

        self._chroma_host = chroma_host
        self._chroma_port = chroma_port
        self._milvus_host = milvus_host
        self._milvus_port = milvus_port
        self._milvus_dimension = milvus_dimension
        self._milvus_index_type = milvus_index_type
        self._milvus_metric_type = milvus_metric_type
        self._faiss_dimension = faiss_dimension
        self._faiss_persist_dir = faiss_persist_dir

        self._storage: Optional[CacheStorageBackend] = None
        self._embedding_function = embedding_function

        self._hit_count = 0
        self._miss_count = 0
        self._exact_hit_count = 0
        self._semantic_hit_count = 0
        self._lock = threading.Lock()
        self._stats_restored = False

        if self.enabled:
            self._storage = _create_cache_storage(
                backend=backend,
                embedding_function=embedding_function,
                chroma_host=chroma_host,
                chroma_port=chroma_port,
                milvus_host=milvus_host,
                milvus_port=milvus_port,
                milvus_dimension=milvus_dimension,
                milvus_index_type=milvus_index_type,
                milvus_metric_type=milvus_metric_type,
                faiss_dimension=faiss_dimension,
                faiss_persist_dir=faiss_persist_dir,
            )
            if not self._storage.initialize():
                self.enabled = False
                self._storage = None
                logger.warning("语义缓存后端初始化失败，缓存已自动禁用")

    @property
    def storage(self) -> Optional[CacheStorageBackend]:
        """获取缓存存储后端"""
        if not self.enabled or self._storage is None:
            return None
        if not self._stats_restored:
            self._restore_stats()
        return self._storage

    def _should_skip_cache(self, query_text: str) -> bool:
        creative_keywords = [
            "写一首", "创作", "编一个", "画一幅", "生成一个故事",
            "写首诗", "写个故事", "写篇", "写一段", "写个",
            "编故事", "作诗", "作一首", "写诗", "写歌",
            "生成一篇", "创作一篇", "写一篇",
        ]
        if any(kw in query_text for kw in creative_keywords):
            return True

        realtime_keywords = [
            "现在几点", "今天日期", "今天天气", "当前时间",
            "最新新闻", "实时股价", "实时汇率", "刚刚发生",
        ]
        if any(kw in query_text for kw in realtime_keywords):
            return True

        time_sensitive_keywords = [
            "今天天气", "明天天气", "昨天新闻",
        ]
        if any(kw in query_text for kw in time_sensitive_keywords):
            return True

        return False

    def lookup(self, question: str) -> Optional[Dict]:
        logger.info("[缓存-查询入口] query='%s', enabled=%s", question[:80], self.enabled)
        if not self.enabled:
            logger.info("[缓存-查询结束] 缓存未启用，跳过")
            return None

        if self._should_skip_cache(question):
            logger.info("[缓存-查询结束] 关键词跳过缓存")
            return None

        exact_result = self._exact_match_lookup(question)
        if exact_result:
            logger.info("[缓存-查询结束] 精确匹配命中，match_type=%s, score=%.2f",
                        exact_result.get("match_type"), exact_result.get("score", 0))
            return exact_result

        semantic_result = self._semantic_match_lookup(question)
        if semantic_result:
            logger.info("[缓存-查询结束] 语义匹配命中，match_type=%s, score=%.2f",
                        semantic_result.get("match_type"), semantic_result.get("score", 0))
            return semantic_result

        logger.info("[缓存-查询结束] 未命中（精确匹配+语义匹配均失败）")
        return None

    def _exact_match_lookup(self, question: str) -> Optional[Dict]:
        try:
            storage = self.storage
            if storage is None:
                logger.info("[缓存-精确匹配] storage不可用，跳过")
                return None

            doc_id = self._generate_id(question)
            logger.info("[缓存-精确匹配] 查询 doc_id=%s, query='%s'", doc_id, question[:80])
            result = storage.get_by_id(doc_id)
            if result is None:
                logger.info("[缓存-精确匹配-未命中] doc_id=%s 在storage中不存在", doc_id)
                return None

            logger.info("[缓存-精确匹配-已找到] doc_id=%s, metadata_keys=%s", doc_id, list(result.get("metadata", {}).keys()))

            metadata = result["metadata"]

            created_at_str = metadata.get("created_at", "")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    expire_at = created_at + timedelta(hours=self.ttl_hours)
                    if datetime.now() > expire_at:
                        logger.debug("精确匹配-缓存已过期：created_at=%s", created_at_str)
                        storage.delete_entries([doc_id])
                        return None
                except ValueError:
                    pass

            hit_count = metadata.get("hit_count", 0) + 1
            now_iso = datetime.now().isoformat()
            storage.update_metadata(doc_id, {
                **metadata,
                "hit_count": hit_count,
                "last_hit_at": now_iso,
            })

            cached_query = metadata.get("query", question)
            answer = result.get("document", "")
            sources_json = metadata.get("sources_json", "[]")

            logger.info("精确匹配命中：query='%s' -> cached='%s'", question, cached_query)

            return {
                "query": cached_query,
                "answer": answer,
                "sources_json": sources_json,
                "score": 1.0,
                "match_type": "exact",
            }

        except Exception as e:
            logger.debug("精确匹配查询失败，回退到语义匹配：%s", e)
            return None

    def _semantic_match_lookup(self, question: str) -> Optional[Dict]:
        """单阶段语义匹配：向量检索 -> COSINE 阈值过滤（不使用 Reranker）"""
        try:
            storage = self.storage
            if storage is None:
                logger.debug("语义匹配-跳过：storage 不可用")
                return None

            results = storage.search_similar(question, n_results=self.candidate_count)
            if not results:
                logger.info("语义匹配-未命中：search_similar 返回空")
                return None

            best_item = None
            best_similarity = -1.0
            logger.info("[缓存-语义匹配] 共 %d 条候选，阈值=%.2f", len(results), self.coarse_threshold)

            for item in results:
                distance = item["distance"]
                metadata = item["metadata"]

                if self.backend == "milvus":
                    similarity = float(distance)
                else:
                    similarity = 1.0 - distance

                cached_query = metadata.get("query", "")[:50]
                logger.info("[缓存-语义匹配] id=%s, 相似度=%.4f(%.1f%%), 阈值=%.2f, cached_query='%s'",
                            item["id"], similarity, similarity * 100,
                            self.coarse_threshold, cached_query)

                if similarity < self.coarse_threshold:
                    logger.info("[缓存-语义匹配-跳过] id=%s 相似度 %.1f%% < 阈值 %.1f%%",
                                item["id"], similarity * 100, self.coarse_threshold * 100)
                    continue

                created_at_str = metadata.get("created_at", "")
                expired = False
                if created_at_str:
                    try:
                        created_at = datetime.fromisoformat(created_at_str)
                        expire_at = created_at + timedelta(hours=self.ttl_hours)
                        if datetime.now() > expire_at:
                            logger.debug("语义匹配-跳过(过期)：id=%s, created_at=%s", item["id"], created_at_str)
                            storage.delete_entries([item["id"]])
                            expired = True
                    except ValueError:
                        pass

                if not expired and similarity > best_similarity:
                    best_similarity = similarity
                    best_item = item

            if best_item is None:
                logger.info("语义匹配-未命中：无候选通过阈值（共 %d 条粗筛结果）", len(results))
                return None

            logger.info(
                "语义匹配-命中：%d 条候选 -> 最佳相似度=%.4f(%.1f%%)",
                len(results), best_similarity, best_similarity * 100
            )

            item = best_item
            cache_id = item["id"]
            metadata = item["metadata"]

            hit_count = metadata.get("hit_count", 0) + 1
            now_iso = datetime.now().isoformat()
            storage.update_metadata(cache_id, {
                **metadata,
                "hit_count": hit_count,
                "last_hit_at": now_iso,
            })

            cached_query = metadata.get("query", question)
            answer = item.get("document", "")
            sources_json = metadata.get("sources_json", "[]")

            logger.info(
                "语义匹配命中：query='%s' -> cached='%s'，相似度=%.4f",
                question, cached_query, best_similarity
            )

            return {
                "query": cached_query,
                "answer": answer,
                "sources_json": sources_json,
                "score": best_similarity,
                "match_type": "semantic",
            }

        except Exception as e:
            logger.error("语义匹配查询失败：%s", e)
            return None

    def store(self, question: str, answer: str, sources: List[Dict]):
        logger.info("[缓存-存储入口] query='%s', enabled=%s", question[:80], self.enabled)
        if not self.enabled:
            logger.info("[缓存-存储] 缓存未启用，跳过")
            return

        if self._should_skip_cache(question):
            logger.info("[缓存-存储] 关键词跳过缓存")
            return

        try:
            storage = self.storage
            if storage is None:
                logger.warning("缓存存储跳过：缓存存储后端不可用")
                return

            current_count = storage.count()
            logger.info("缓存存储开始：query='%s'，当前缓存条目数=%d，最大条目数=%d",
                        question[:50], current_count, self.max_entries)

            if current_count >= self.max_entries:
                self._evict_by_lfu()

            sources_json = json.dumps(sources, ensure_ascii=False)
            now_iso = datetime.now().isoformat()

            metadata = {
                "query": question,
                "sources_json": sources_json,
                "hit_count": 1,
                "created_at": now_iso,
                "last_hit_at": now_iso,
            }

            doc_id = self._generate_id(question)
            logger.info("[缓存-写入] doc_id=%s, query='%s', 开始add_entry", doc_id, question[:80])
            success = storage.add_entry(doc_id, question, answer, sources_json, metadata)

            if success:
                verify_count = storage.count()
                logger.info("[缓存-写入-成功] doc_id=%s, query='%s', 当前缓存条目数=%d",
                            doc_id, question[:80], verify_count)
            else:
                logger.warning("[缓存-写入-失败] add_entry返回False, doc_id=%s, query='%s'",
                               doc_id, question[:80])

        except Exception as e:
            logger.error("缓存存储失败：%s", e, exc_info=True)

    def invalidate(self):
        if not self.enabled:
            return

        try:
            storage = self.storage
            if storage is None:
                return

            count = storage.count()
            if count > 0:
                storage.clear_all()
                self._storage = None
                self._stats_restored = False
                if self.enabled:
                    self._storage = _create_cache_storage(
                        backend=self.backend,
                        embedding_function=self._embedding_function,
                        chroma_host=self._chroma_host,
                        chroma_port=self._chroma_port,
                        milvus_host=self._milvus_host,
                        milvus_port=self._milvus_port,
                        milvus_dimension=self._milvus_dimension,
                        milvus_index_type=self._milvus_index_type,
                        milvus_metric_type=self._milvus_metric_type,
                        faiss_dimension=self._faiss_dimension,
                        faiss_persist_dir=self._faiss_persist_dir,
                    )
                    if self._storage and self._storage.initialize():
                        logger.info("语义缓存已清空并重建：删除了 %d 条缓存", count)
                    else:
                        logger.warning("语义缓存清空后重建失败")
                        self._storage = None
            with self._lock:
                self._hit_count = 0
                self._miss_count = 0
                self._exact_hit_count = 0
                self._semantic_hit_count = 0
        except Exception as e:
            logger.error("清空缓存失败：%s", e)

    def warmup(self, entries: List[Dict]) -> int:
        if not self.enabled or not entries:
            logger.info("缓存预热跳过：enabled=%s, entries=%d", self.enabled, len(entries) if entries else 0)
            return 0

        storage = self.storage
        if storage is None:
            logger.warning("缓存预热跳过：缓存存储不可用")
            return 0

        success_count = 0
        skip_count = 0

        for entry in entries:
            try:
                question = entry.get("question", "")
                answer = entry.get("answer", "")
                sources = entry.get("sources", [])

                if not question or not answer:
                    continue

                doc_id = self._generate_id(question)
                existing = storage.get_by_id(doc_id)
                if existing is not None:
                    skip_count += 1
                    continue

                sources_json = json.dumps(sources, ensure_ascii=False)
                now_iso = datetime.now().isoformat()

                metadata = {
                    "query": question,
                    "sources_json": sources_json,
                    "hit_count": 0,
                    "created_at": now_iso,
                    "last_hit_at": now_iso,
                }

                storage.add_entry(doc_id, question, answer, sources_json, metadata)
                success_count += 1

            except Exception as e:
                logger.warning("缓存预热条目失败：question='%s', error=%s", question, e)

        logger.info("缓存预热完成：成功=%d, 跳过(已存在)=%d, 总数=%d", success_count, skip_count, len(entries))
        return success_count

    def get_warmup_status(self, entry_count: int = None) -> Dict:
        if entry_count is None:
            try:
                storage = self.storage
                entry_count = storage.count() if storage else 0
            except Exception:
                entry_count = 0

        return {
            "enabled": self.enabled,
            "entry_count": entry_count,
            "max_entries": self.max_entries,
            "ready": entry_count > 0,
        }

    def _restore_stats(self):
        self._stats_restored = True
        if not self.enabled:
            return

        try:
            storage = self.storage
            if storage is None:
                return

            all_meta = storage.get_all_metadata()
            if not all_meta:
                logger.debug("缓存统计恢复：无缓存条目，跳过")
                return

            total_hits = 0
            for item in all_meta:
                metadata = item.get("metadata", {})
                hit_count = metadata.get("hit_count", 0)
                if hit_count > 0:
                    total_hits += (hit_count - 1)

            with self._lock:
                self._hit_count = total_hits

            logger.info(
                "缓存统计已从 %s 恢复：总命中=%d，缓存条目数=%d",
                self.backend.upper(), total_hits, len(all_meta)
            )

        except Exception as e:
            logger.warning("恢复缓存统计失败：%s", e)

    def get_stats(self) -> Dict:
        with self._lock:
            total = self._hit_count + self._miss_count
            hit_rate = (self._hit_count / total * 100) if total > 0 else 0.0

        try:
            storage = self.storage
            entry_count = storage.count() if storage else 0
        except Exception:
            entry_count = 0

        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "entry_count": entry_count,
            "max_entries": self.max_entries,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "exact_hit_count": self._exact_hit_count,
            "semantic_hit_count": self._semantic_hit_count,
            "hit_rate": f"{hit_rate:.1f}%",
            "similarity_threshold": self.similarity_threshold,
            "ttl_hours": self.ttl_hours,
        }

    def record_hit(self, match_type: str):
        with self._lock:
            self._hit_count += 1
            if match_type == "exact":
                self._exact_hit_count += 1
            else:
                self._semantic_hit_count += 1

    def record_miss(self):
        with self._lock:
            self._miss_count += 1

    def cleanup_expired(self):
        if not self.enabled:
            return

        try:
            storage = self.storage
            if storage is None:
                return

            all_meta = storage.get_all_metadata()
            if not all_meta:
                return

            expired_ids = []
            now = datetime.now()
            for item in all_meta:
                metadata = item.get("metadata", {})
                created_at_str = metadata.get("created_at", "")
                if created_at_str:
                    try:
                        created_at = datetime.fromisoformat(created_at_str)
                        if now > created_at + timedelta(hours=self.ttl_hours):
                            expired_ids.append(item["id"])
                    except ValueError:
                        pass

            if expired_ids:
                storage.delete_entries(expired_ids)
                logger.info("清理了 %d 条过期缓存", len(expired_ids))

        except Exception as e:
            logger.error("清理过期缓存失败：%s", e)

    def _evict_by_lfu(self):
        try:
            storage = self.storage
            if storage is None:
                return

            all_meta = storage.get_all_metadata()
            if not all_meta:
                return

            entries = []
            for item in all_meta:
                metadata = item.get("metadata", {})
                hit_count = metadata.get("hit_count", 0)
                entries.append((item["id"], hit_count))

            entries.sort(key=lambda x: x[1])

            evict_count = max(1, len(entries) // 10)
            evict_ids = [e[0] for e in entries[:evict_count]]

            storage.delete_entries(evict_ids)
            logger.info("LFU 淘汰：删除了 %d 条低频缓存", len(evict_ids))

        except Exception as e:
            logger.error("LFU 淘汰失败：%s", e)

    @staticmethod
    def _generate_id(question: str) -> str:
        return hashlib.md5(question.encode("utf-8")).hexdigest()[:16]