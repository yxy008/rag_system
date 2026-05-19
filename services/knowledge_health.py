"""
knowledge_health.py - 知识库健康检查模块

对知识库进行多维度健康检查，覆盖文档层、切片层、向量层、检索层、索引层，
帮助及时发现和诊断知识库中的问题。

检查维度：
  1. 文档层：重复文档、空文档、格式异常
  2. 切片层：空切片、过短切片、过长切片、切片分布
  3. 向量层：零向量检测、向量维度一致性、Embedding 质量
  4. 检索层：BM25 索引一致性、向量库一致性、检索质量抽样
  5. 索引层：BM25 索引状态、向量库连接状态

使用方式：
  from knowledge_health import KnowledgeHealthChecker

  checker = KnowledgeHealthChecker(vs_manager, doc_processor)
  report = checker.full_check()
  print(report["overall_score"])  # 0-100 综合健康分
"""
import hashlib
import logging
import re
import time
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document

from core.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_STORE_BACKEND,
    EMBEDDING_MODEL_NAME,
)

logger = logging.getLogger(__name__)


class KnowledgeHealthChecker:
    """
    知识库健康检查器

    对知识库进行全面体检，生成健康报告，包含：
      - 各维度得分（0-100）
      - 发现的问题列表
      - 修复建议
      - 综合健康评分
    """

    def __init__(self, vs_manager, doc_processor):
        """
        初始化健康检查器。

        Args:
            vs_manager: VectorStoreManager 实例
            doc_processor: DocumentProcessor 实例
        """
        self.vs_manager = vs_manager
        self.doc_processor = doc_processor
        self._warnings: List[Dict] = []
        self._suggestions: List[Dict] = []

    # ============================================================
    # 公共接口
    # ============================================================

    def full_check(self) -> Dict:
        """
        执行完整健康检查，返回综合报告。

        Returns:
            {
                "timestamp": "2024-01-01T00:00:00",
                "overall_score": 85,
                "overall_status": "healthy",
                "documents": {...},
                "chunks": {...},
                "vectors": {...},
                "retrieval": {...},
                "index": {...},
                "warnings": [...],
                "suggestions": [...],
            }
        """
        self._warnings = []
        self._suggestions = []

        start_time = time.time()

        documents = self._check_documents()
        chunks = self._check_chunks()
        vectors = self._check_vectors()
        retrieval = self._check_retrieval()
        index = self._check_index()

        overall_score = self._compute_overall_score(
            documents, chunks, vectors, retrieval, index
        )

        elapsed = round(time.time() - start_time, 2)

        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_score": overall_score,
            "overall_status": self._score_to_status(overall_score),
            "check_duration_seconds": elapsed,
            "documents": documents,
            "chunks": chunks,
            "vectors": vectors,
            "retrieval": retrieval,
            "index": index,
            "warnings": self._warnings,
            "suggestions": self._suggestions,
        }

        logger.info(
            "知识库健康检查完成：综合得分=%d，状态=%s，耗时=%.2fs，警告=%d条",
            overall_score, report["overall_status"], elapsed, len(self._warnings),
        )
        return report

    def quick_check(self) -> Dict:
        """
        快速健康检查（仅检查关键指标，不执行检索质量抽样）。

        Returns:
            简化的健康报告
        """
        self._warnings = []
        self._suggestions = []

        documents = self._check_documents()
        chunks = self._check_chunks()
        index = self._check_index()

        overall_score = self._compute_overall_score(
            documents, chunks, {"score": 100, "status": "healthy"}, {"score": 100, "status": "healthy"}, index
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_score": overall_score,
            "overall_status": self._score_to_status(overall_score),
            "documents": documents,
            "chunks": chunks,
            "index": index,
            "warnings": self._warnings,
            "suggestions": self._suggestions,
        }

    # ============================================================
    # 一、文档层检查
    # ============================================================

    def _check_documents(self) -> Dict:
        """
        检查文档层健康状态。

        检查项：
          - 文档总数
          - 重复文档检测（基于内容 MD5）
          - 空文档检测
          - 文档来源分布
        """
        try:
            all_docs = self._get_all_documents()
        except Exception as e:
            logger.error("获取文档列表失败：%s", e)
            self._warnings.append({
                "level": "error",
                "category": "documents",
                "message": f"无法获取文档列表：{e}",
            })
            return {"score": 0, "status": "error", "error": str(e)}

        total = len(all_docs)
        if total == 0:
            self._warnings.append({
                "level": "warning",
                "category": "documents",
                "message": "知识库中没有文档，请先导入文档",
            })
            self._suggestions.append({
                "category": "documents",
                "message": "运行 python ingest.py 导入文档，或通过 Web 界面上传文档",
            })
            return {
                "score": 0,
                "status": "empty",
                "total_documents": 0,
                "duplicate_count": 0,
                "empty_count": 0,
                "source_distribution": {},
            }

        # 重复文档检测
        content_hashes = {}
        duplicates = []
        empty_docs = []

        for doc in all_docs:
            content = doc.page_content.strip()
            if not content:
                empty_docs.append(doc)
                continue

            content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
            if content_hash in content_hashes:
                duplicates.append({
                    "doc_id": doc.metadata.get("id", "unknown"),
                    "source": doc.metadata.get("source", "unknown"),
                    "duplicate_of": content_hashes[content_hash],
                })
            else:
                content_hashes[content_hash] = doc.metadata.get("id", "unknown")

        # 来源分布
        source_dist = Counter(
            doc.metadata.get("source", "unknown") for doc in all_docs
        )

        # 计算得分
        score = 100
        dup_ratio = len(duplicates) / total if total > 0 else 0
        empty_ratio = len(empty_docs) / total if total > 0 else 0

        if dup_ratio > 0.2:
            score -= 30
            self._warnings.append({
                "level": "warning",
                "category": "documents",
                "message": f"重复文档比例过高（{dup_ratio:.1%}），共 {len(duplicates)} 个重复",
            })
            self._suggestions.append({
                "category": "documents",
                "message": "建议清理重复文档，避免检索结果冗余",
            })
        elif dup_ratio > 0.05:
            score -= 10
            self._warnings.append({
                "level": "info",
                "category": "documents",
                "message": f"存在少量重复文档（{dup_ratio:.1%}），共 {len(duplicates)} 个",
            })

        if empty_ratio > 0.1:
            score -= 20
            self._warnings.append({
                "level": "warning",
                "category": "documents",
                "message": f"空文档比例过高（{empty_ratio:.1%}），共 {len(empty_docs)} 个",
            })
        elif empty_docs:
            score -= 5

        return {
            "score": max(0, score),
            "status": self._score_to_status(score),
            "total_documents": total,
            "duplicate_count": len(duplicates),
            "duplicate_ratio": round(dup_ratio, 3),
            "empty_count": len(empty_docs),
            "empty_ratio": round(empty_ratio, 3),
            "unique_sources": len(source_dist),
            "source_distribution": dict(source_dist.most_common(10)),
            "duplicates": duplicates[:10],
        }

    # ============================================================
    # 二、切片层检查
    # ============================================================

    def _check_chunks(self) -> Dict:
        """
        检查切片层健康状态。

        检查项：
          - 切片总数
          - 空切片检测
          - 过短切片（< 50 字符）
          - 过长切片（> CHUNK_SIZE * 1.5）
          - 切片长度分布统计
          - 分块策略分布
        """
        try:
            all_docs = self._get_all_documents()
        except Exception as e:
            return {"score": 0, "status": "error", "error": str(e)}

        if not all_docs:
            return {
                "score": 0,
                "status": "empty",
                "total_chunks": 0,
                "empty_chunks": 0,
                "too_short_chunks": 0,
                "too_long_chunks": 0,
            }

        total = len(all_docs)
        empty_chunks = []
        too_short = []
        too_long = []
        lengths = []
        chunk_methods = Counter()

        for doc in all_docs:
            content = doc.page_content.strip()
            length = len(content)

            if not content:
                empty_chunks.append(doc)
            elif length < 50:
                too_short.append({"length": length, "source": doc.metadata.get("source", "unknown")})
            elif length > CHUNK_SIZE * 1.5:
                too_long.append({"length": length, "source": doc.metadata.get("source", "unknown")})

            lengths.append(length)
            method = doc.metadata.get("chunk_method", "unknown")
            chunk_methods[method] += 1

        # 统计
        lengths_arr = np.array(lengths) if lengths else np.array([0])
        stats = {
            "mean": round(float(np.mean(lengths_arr)), 1),
            "median": round(float(np.median(lengths_arr)), 1),
            "min": int(np.min(lengths_arr)),
            "max": int(np.max(lengths_arr)),
            "std": round(float(np.std(lengths_arr)), 1),
            "p25": round(float(np.percentile(lengths_arr, 25)), 1),
            "p75": round(float(np.percentile(lengths_arr, 75)), 1),
            "p95": round(float(np.percentile(lengths_arr, 95)), 1),
        }

        # 计算得分
        score = 100
        empty_ratio = len(empty_chunks) / total if total > 0 else 0
        short_ratio = len(too_short) / total if total > 0 else 0
        long_ratio = len(too_long) / total if total > 0 else 0

        if empty_ratio > 0.05:
            score -= 25
            self._warnings.append({
                "level": "warning",
                "category": "chunks",
                "message": f"空切片比例过高（{empty_ratio:.1%}），共 {len(empty_chunks)} 个",
            })
        elif empty_chunks:
            score -= 10

        if short_ratio > 0.15:
            score -= 15
            self._warnings.append({
                "level": "warning",
                "category": "chunks",
                "message": f"过短切片（<50字符）比例过高（{short_ratio:.1%}），共 {len(too_short)} 个，可能影响检索质量",
            })
            self._suggestions.append({
                "category": "chunks",
                "message": "过短切片缺乏上下文，建议增大 CHUNK_SIZE 或调整分块策略",
            })

        if long_ratio > 0.1:
            score -= 10
            self._warnings.append({
                "level": "info",
                "category": "chunks",
                "message": f"过长切片（>{CHUNK_SIZE * 1.5}字符）比例 {long_ratio:.1%}，共 {len(too_long)} 个，可能导致 Embedding 表示稀释",
            })

        # 切片长度分布是否合理（标准差过大说明切分不均匀）
        if stats["std"] > CHUNK_SIZE * 0.5:
            score -= 10
            self._warnings.append({
                "level": "info",
                "category": "chunks",
                "message": f"切片长度标准差过大（{stats['std']:.0f}），切分不够均匀",
            })

        return {
            "score": max(0, score),
            "status": self._score_to_status(score),
            "total_chunks": total,
            "empty_chunks": len(empty_chunks),
            "too_short_chunks": len(too_short),
            "too_long_chunks": len(too_long),
            "length_stats": stats,
            "chunk_method_distribution": dict(chunk_methods),
            "too_short_samples": too_short[:5],
            "too_long_samples": too_long[:5],
        }

    # ============================================================
    # 三、向量层检查
    # ============================================================

    def _check_vectors(self) -> Dict:
        """
        检查向量层健康状态。

        检查项：
          - 向量维度一致性
          - 零向量检测（抽样）
          - Embedding 模型状态
          - 向量库文档数与 BM25 索引一致性
        """
        score = 100
        details = {}

        # Embedding 模型状态
        try:
            embeddings = self.vs_manager.embeddings
            details["embedding_model"] = EMBEDDING_MODEL_NAME
            details["embedding_model_loaded"] = True
        except Exception as e:
            score -= 30
            details["embedding_model_loaded"] = False
            details["embedding_model_error"] = str(e)
            self._warnings.append({
                "level": "error",
                "category": "vectors",
                "message": f"Embedding 模型加载失败：{e}",
            })
            return {"score": max(0, score), "status": self._score_to_status(score), **details}

        # 向量维度一致性（抽样检查）
        try:
            all_docs = self._get_all_documents()
            if all_docs:
                sample_size = min(5, len(all_docs))
                sample_docs = all_docs[:sample_size]
                sample_texts = [doc.page_content for doc in sample_docs]

                sample_vectors = embeddings.embed_documents(sample_texts)
                dimensions = [len(v) for v in sample_vectors]

                if len(set(dimensions)) > 1:
                    score -= 25
                    self._warnings.append({
                        "level": "error",
                        "category": "vectors",
                        "message": f"向量维度不一致：{set(dimensions)}",
                    })
                else:
                    details["vector_dimension"] = dimensions[0]
                    details["vector_dimension_consistent"] = True

                # 零向量检测
                zero_count = 0
                for i, vec in enumerate(sample_vectors):
                    vec_norm = np.linalg.norm(vec)
                    if vec_norm < 1e-6:
                        zero_count += 1

                if zero_count > 0:
                    score -= 20
                    self._warnings.append({
                        "level": "warning",
                        "category": "vectors",
                        "message": f"抽样检测发现 {zero_count}/{sample_size} 个零向量，可能是空文本导致",
                    })

                details["sample_size"] = sample_size
                details["zero_vector_count"] = zero_count
                details["avg_vector_norm"] = round(
                    float(np.mean([np.linalg.norm(v) for v in sample_vectors])), 4
                )
        except Exception as e:
            logger.warning("向量抽样检查失败：%s", e)
            details["vector_check_error"] = str(e)

        return {
            "score": max(0, score),
            "status": self._score_to_status(score),
            **details,
        }

    # ============================================================
    # 四、检索层检查
    # ============================================================

    def _check_retrieval(self) -> Dict:
        """
        检查检索层健康状态。

        检查项：
          - BM25 索引与向量库文档数一致性
          - 检索功能可用性（抽样测试）
          - 混合检索配置状态
          - Reranker 配置状态
        """
        score = 100
        details = {}

        # BM25 与向量库一致性
        try:
            vs_count = self.vs_manager.get_document_count()
            bm25_count = len(self.vs_manager._bm25_corpus) if self.vs_manager._bm25_corpus else 0

            details["vector_store_count"] = vs_count
            details["bm25_index_count"] = bm25_count

            if vs_count > 0 and bm25_count == 0:
                score -= 30
                self._warnings.append({
                    "level": "error",
                    "category": "retrieval",
                    "message": "BM25 索引为空但向量库有数据，混合检索将退化为纯向量检索",
                })
                self._suggestions.append({
                    "category": "retrieval",
                    "message": "重新入库以重建 BM25 索引，或检查 BM25 索引构建逻辑",
                })
            elif vs_count != bm25_count:
                score -= 15
                self._warnings.append({
                    "level": "warning",
                    "category": "retrieval",
                    "message": f"BM25 索引（{bm25_count}）与向量库（{vs_count}）文档数不一致",
                })
                self._suggestions.append({
                    "category": "retrieval",
                    "message": "重新入库以确保 BM25 索引与向量库同步",
                })
        except Exception as e:
            logger.warning("检索一致性检查失败：%s", e)
            details["consistency_error"] = str(e)

        # 检索功能抽样测试
        if details.get("vector_store_count", 0) > 0:
            try:
                test_queries = [
                    "年假怎么申请",
                    "报销流程",
                    "办公用品申领",
                ]
                retrieval_tests = []
                for query in test_queries:
                    try:
                        results = self.vs_manager.search(query, top_k=3)
                        retrieval_tests.append({
                            "query": query,
                            "result_count": len(results),
                            "has_results": len(results) > 0,
                        })
                    except Exception as e:
                        retrieval_tests.append({
                            "query": query,
                            "error": str(e),
                            "has_results": False,
                        })

                details["retrieval_tests"] = retrieval_tests
                failed_tests = [t for t in retrieval_tests if not t.get("has_results")]
                if len(failed_tests) == len(retrieval_tests):
                    score -= 30
                    self._warnings.append({
                        "level": "error",
                        "category": "retrieval",
                        "message": "所有检索测试均失败，检索功能可能不可用",
                    })
                elif failed_tests:
                    score -= 10
                    self._warnings.append({
                        "level": "warning",
                        "category": "retrieval",
                        "message": f"{len(failed_tests)}/{len(retrieval_tests)} 个检索测试无结果",
                    })
            except Exception as e:
                logger.warning("检索抽样测试失败：%s", e)
                details["retrieval_test_error"] = str(e)

        # 混合检索配置
        details["hybrid_search_enabled"] = self.vs_manager.is_hybrid_search_enabled()
        details["reranker_enabled"] = self.vs_manager.is_reranker_enabled()
        details["multi_query_enabled"] = self.vs_manager.is_multi_query_enabled()

        if not details["hybrid_search_enabled"]:
            self._warnings.append({
                "level": "info",
                "category": "retrieval",
                "message": "混合检索已关闭，仅使用向量检索",
            })

        return {
            "score": max(0, score),
            "status": self._score_to_status(score),
            **details,
        }

    # ============================================================
    # 五、索引层检查
    # ============================================================

    def _check_index(self) -> Dict:
        """
        检查索引层健康状态。

        检查项：
          - 向量数据库连接状态
          - BM25 索引状态
          - 向量数据库后端类型
          - 集合/索引信息
        """
        score = 100
        details = {}

        details["vector_store_backend"] = VECTOR_STORE_BACKEND

        # 向量数据库连接检查
        try:
            vs = self.vs_manager.get_or_create_vector_store()
            doc_count = vs.count()
            details["vector_store_connected"] = True
            details["vector_store_doc_count"] = doc_count
        except Exception as e:
            score -= 40
            details["vector_store_connected"] = False
            details["vector_store_error"] = str(e)
            self._warnings.append({
                "level": "error",
                "category": "index",
                "message": f"向量数据库连接失败：{e}",
            })
            self._suggestions.append({
                "category": "index",
                "message": f"检查 {VECTOR_STORE_BACKEND} 服务是否正常运行",
            })

        # BM25 索引状态
        try:
            bm25 = self.vs_manager._bm25_index
            corpus = self.vs_manager._bm25_corpus
            details["bm25_index_ready"] = bm25 is not None
            details["bm25_corpus_size"] = len(corpus) if corpus else 0

            if bm25 is None and doc_count > 0:
                score -= 20
                self._warnings.append({
                    "level": "warning",
                    "category": "index",
                    "message": "BM25 索引未构建，混合检索将退化为纯向量检索",
                })
        except Exception as e:
            logger.warning("BM25 索引检查失败：%s", e)
            details["bm25_check_error"] = str(e)

        # 后端特定检查
        if VECTOR_STORE_BACKEND == "chroma":
            try:
                collection = vs._get_store()._collection
                details["collection_name"] = collection.name
                details["collection_count"] = collection.count()
            except Exception:
                pass
        elif VECTOR_STORE_BACKEND == "milvus":
            try:
                details["collection_name"] = self.vs_manager._vector_store._collection_name
            except Exception:
                pass

        return {
            "score": max(0, score),
            "status": self._score_to_status(score),
            **details,
        }

    # ============================================================
    # 辅助方法
    # ============================================================

    def _get_all_documents(self) -> List[Document]:
        """获取向量库中的所有文档"""
        try:
            vs = self.vs_manager.get_or_create_vector_store()
            return vs.get_all_documents()
        except Exception as e:
            logger.error("获取文档列表失败：%s", e)
            return []

    def _compute_overall_score(
        self,
        documents: Dict,
        chunks: Dict,
        vectors: Dict,
        retrieval: Dict,
        index: Dict,
    ) -> int:
        """
        计算综合健康评分。

        权重分配：
          - 文档层：15%
          - 切片层：20%
          - 向量层：20%
          - 检索层：25%
          - 索引层：20%
        """
        weights = {
            "documents": 0.15,
            "chunks": 0.20,
            "vectors": 0.20,
            "retrieval": 0.25,
            "index": 0.20,
        }

        scores = {
            "documents": documents.get("score", 0),
            "chunks": chunks.get("score", 0),
            "vectors": vectors.get("score", 0),
            "retrieval": retrieval.get("score", 0),
            "index": index.get("score", 0),
        }

        weighted_score = sum(
            scores[dim] * weights[dim] for dim in weights
        )

        return round(weighted_score)

    def _score_to_status(self, score: int) -> str:
        """将分数转换为状态标签"""
        if score >= 90:
            return "healthy"
        elif score >= 70:
            return "warning"
        elif score >= 50:
            return "degraded"
        else:
            return "critical"