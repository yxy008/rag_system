"""
reranker.py - Reranker 重排序模块

使用 BAAI/bge-reranker-large 交叉编码器对检索结果进行精排。
交叉编码器将 (query, document) 一起送入模型，直接输出相关性分数，
比向量检索的 Bi-Encoder 精度更高，适合精筛阶段。

工作流程：
  粗筛：向量检索 + BM25 → top_k=12 条候选（速度快）
  精筛：Reranker 对 12 条候选重排序 → top_k=4 条（精度高）
"""
import logging
from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL_NAME, RERANKER_TOP_K, BASE_DIR

logger = logging.getLogger(__name__)


class RerankerManager:
    """
    Reranker 管理器：加载模型 + 执行重排序

    模型选择：BAAI/bge-reranker-large
      - 566M 参数，中文 BGE 系列 reranker
      - 与 BGE-M3 Embedding 模型配套使用效果最佳
      - 输入：(query, document) → 输出：相关性分数 [0, 1]
    """

    def __init__(self):
        self._model = None

    @property
    def model(self) -> CrossEncoder:
        """懒加载 Reranker 模型（首次使用时加载）"""
        if self._model is None:
            model_path = RERANKER_MODEL_NAME
            if model_path.startswith("./") or model_path.startswith(".\\"):
                model_path = str(BASE_DIR / model_path)

            logger.info(f"正在加载 Reranker 模型：{model_path}")
            self._model = CrossEncoder(
                model_name=model_path,
                max_length=512,
                device="cuda",
            )
            logger.info("Reranker 模型加载完成（GPU 加速）")
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None,
    ) -> List[Tuple[Document, float]]:
        """
        对候选文档进行重排序。

        Args:
            query: 用户查询
            documents: 候选文档列表（来自向量检索/BM25 的粗筛结果）
            top_k: 返回的最终结果数量（None=使用配置默认值）

        Returns:
            List[(Document, rerank_score)]: 排序后的结果（分数越高越相关）
        """
        if top_k is None:
            top_k = RERANKER_TOP_K

        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs, show_progress_bar=False)

        doc_scores = list(zip(documents, scores.tolist()))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Reranker 完成：从 {len(documents)} 条候选中选出 top {top_k}")

        return doc_scores[:top_k]

    def rerank_with_sources(
        self,
        query: str,
        docs_with_scores: List[Tuple[Document, float, str]],
        top_k: int = None,
    ) -> List[Tuple[Document, float, str]]:
        """
        对带来源信息的检索结果进行重排序。
        返回格式与 vector_store.py 的混合检索结果一致。
        """
        if top_k is None:
            top_k = RERANKER_TOP_K

        if not docs_with_scores:
            return []

        docs = [item[0] for item in docs_with_scores]
        reranked = self.rerank(query, docs, top_k=top_k)

        return [(doc, score * 100, "rerank") for doc, score in reranked]
