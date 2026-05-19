"""
confidence.py - 答案可信度评估模块

功能：
  1. 多维度可信度评分（来源匹配度、权威性、一致性、时效性、完整性）
  2. 答案溯源链记录（检索 → 精排 → 推理 → 生成）
  3. 低置信度答案标记
"""
import json
import logging
import math
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "conversations.db")


class ConfidenceEvaluator:
    """答案可信度评估器"""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS answer_provenance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        sources_json TEXT DEFAULT NULL,
                        provenance_json TEXT DEFAULT NULL,
                        confidence_json TEXT DEFAULT NULL,
                        created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_answer_provenance_session
                    ON answer_provenance(session_id)
                """)
                conn.commit()
            finally:
                conn.close()

    def evaluate(
        self,
        sources: List[Dict],
        answer: str,
        question: str,
        retrieval_details: Dict = None,
    ) -> Dict:
        """
        评估答案可信度

        Args:
            sources: 检索到的来源文档列表
            answer: LLM 生成的答案
            question: 用户问题
            retrieval_details: 检索过程详情（用于溯源树）

        Returns:
            {
                "overall_score": 85,
                "dimensions": {
                    "source_match": {"score": 90, "label": "来源匹配度"},
                    "authority": {"score": 85, "label": "来源权威性"},
                    "consistency": {"score": 88, "label": "信息一致性"},
                    "freshness": {"score": 75, "label": "时效性"},
                    "completeness": {"score": 87, "label": "完整性"},
                },
                "level": "high",
                "provenance_tree": {...}
            }
        """
        dimensions = {}

        # 1. 来源匹配度：基于 Reranker 分数
        dimensions["source_match"] = self._eval_source_match(sources)

        # 2. 来源权威性：基于文档元数据
        dimensions["authority"] = self._eval_authority(sources)

        # 3. 信息一致性：多个来源之间的一致性
        dimensions["consistency"] = self._eval_consistency(sources, answer)

        # 4. 时效性：文档更新时间
        dimensions["freshness"] = self._eval_freshness(sources)

        # 5. 完整性：答案是否覆盖了问题的各个方面
        dimensions["completeness"] = self._eval_completeness(sources, answer, question)

        # 综合评分（加权平均）
        weights = {
            "source_match": 0.30,
            "authority": 0.15,
            "consistency": 0.20,
            "freshness": 0.15,
            "completeness": 0.20,
        }
        overall = sum(
            dimensions[k]["score"] * weights.get(k, 0.2)
            for k in dimensions
        )
        overall = round(overall)

        level = "high" if overall >= 80 else "medium" if overall >= 60 else "low"

        # 构建溯源树
        provenance_tree = self._build_provenance_tree(sources, retrieval_details)

        return {
            "overall_score": overall,
            "level": level,
            "dimensions": dimensions,
            "provenance_tree": provenance_tree,
        }

    def _eval_source_match(self, sources: List[Dict]) -> Dict:
        """评估来源匹配度"""
        if not sources:
            return {"score": 0, "label": "来源匹配度", "detail": "无检索结果"}

        scores = []
        for s in sources:
            sim = s.get("similarity", 0)
            if isinstance(sim, str):
                try:
                    sim = float(sim.replace("%", ""))
                except (ValueError, TypeError):
                    sim = 0
            scores.append(sim)

        if not scores:
            return {"score": 0, "label": "来源匹配度", "detail": "无法计算相似度"}

        avg_score = sum(scores) / len(scores)
        max_score = max(scores)

        # 综合平均分和最高分
        final = round(avg_score * 0.6 + max_score * 0.4)
        final = min(100, max(0, final))

        return {
            "score": final,
            "label": "来源匹配度",
            "detail": f"最高 {max_score:.0f}%，平均 {avg_score:.0f}%，共 {len(sources)} 个来源",
        }

    def _eval_authority(self, sources: List[Dict]) -> Dict:
        """评估来源权威性"""
        if not sources:
            return {"score": 0, "label": "来源权威性", "detail": "无来源"}

        authority_keywords = ["制度", "办法", "规定", "手册", "政策", "法规", "条例", "标准", "规范"]
        official_extensions = [".pdf", ".docx", ".doc"]

        total_score = 0
        for s in sources:
            source_name = s.get("source", "")
            score = 65  # 基础分（有路径即有一定可信度）

            # 文件名包含权威关键词
            for kw in authority_keywords:
                if kw in source_name:
                    score += 12
                    break

            # 正式文档格式
            for ext in official_extensions:
                if source_name.lower().endswith(ext):
                    score += 10
                    break

            # 有明确来源路径
            if "/" in source_name or "\\" in source_name:
                score += 5

            total_score += min(100, score)

        avg = round(total_score / len(sources)) if sources else 0

        return {
            "score": avg,
            "label": "来源权威性",
            "detail": f"共 {len(sources)} 个来源，平均权威度 {avg}%",
        }

    def _eval_consistency(self, sources: List[Dict], answer: str) -> Dict:
        """评估信息一致性"""
        if len(sources) <= 1:
            return {"score": 90, "label": "信息一致性", "detail": "单一来源，无法交叉验证"}

        # 简化版：来源数量越多，一致性基础分越高（因为有交叉验证）
        # 实际生产环境可以用 LLM 做更精确的一致性判断
        base_score = min(95, 70 + len(sources) * 5)

        return {
            "score": base_score,
            "label": "信息一致性",
            "detail": f"{len(sources)} 个来源可交叉验证",
        }

    def _eval_freshness(self, sources: List[Dict]) -> Dict:
        """评估时效性"""
        if not sources:
            return {"score": 0, "label": "时效性", "detail": "无来源"}

        now = datetime.now()
        freshness_scores = []

        for s in sources:
            # 尝试从元数据中获取更新时间
            updated_str = s.get("updated_at") or s.get("created_at") or ""
            if updated_str:
                try:
                    updated = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                    days_old = (now - updated.replace(tzinfo=None)).days
                    if days_old <= 30:
                        freshness_scores.append(95)
                    elif days_old <= 90:
                        freshness_scores.append(85)
                    elif days_old <= 180:
                        freshness_scores.append(70)
                    elif days_old <= 365:
                        freshness_scores.append(55)
                    else:
                        freshness_scores.append(40)
                except (ValueError, TypeError):
                    freshness_scores.append(75)
            else:
                freshness_scores.append(75)  # 无时间信息视为中性，不做惩罚

        avg = round(sum(freshness_scores) / len(freshness_scores)) if freshness_scores else 75

        return {
            "score": avg,
            "label": "时效性",
            "detail": f"平均时效评分 {avg}%",
        }

    def _eval_completeness(self, sources: List[Dict], answer: str, question: str) -> Dict:
        """评估完整性"""
        if not answer:
            return {"score": 0, "label": "完整性", "detail": "无答案内容"}

        score = 80  # 基础分

        # 答案长度检查
        if len(answer) > 200:
            score += 5
        if len(answer) > 500:
            score += 5

        # 包含引用来源
        if "【来源" in answer or "来源：" in answer:
            score += 5

        # 结构化程度
        structure_markers = ["核心结论", "具体规定", "注意事项", "操作建议", "步骤", "总结"]
        found = sum(1 for m in structure_markers if m in answer)
        score += min(10, found * 3)

        return {
            "score": min(100, score),
            "label": "完整性",
            "detail": f"答案 {len(answer)} 字，结构完整度 {min(100, score)}%",
        }

    def _build_provenance_tree(
        self, sources: List[Dict], retrieval_details: Dict = None
    ) -> Dict:
        """构建答案溯源树"""
        tree = {
            "question": "",
            "stages": [],
        }

        # 阶段1：检索
        retrieval_stage = {
            "name": "检索阶段",
            "type": "retrieval",
            "items": [],
        }
        if retrieval_details:
            retrieval_stage["details"] = {
                "method": retrieval_details.get("method", "混合检索"),
                "candidate_count": retrieval_details.get("candidate_count", len(sources)),
            }
        for i, s in enumerate(sources):
            retrieval_stage["items"].append({
                "index": i + 1,
                "source": s.get("source", "未知"),
                "similarity": s.get("similarity", 0),
                "retrieval_type": s.get("retrieval_type", "unknown"),
                "preview": (s.get("full_content") or s.get("preview") or "")[:200],
            })
        tree["stages"].append(retrieval_stage)

        # 阶段2：精排（如果有 Reranker）
        reranked = [s for s in sources if s.get("retrieval_type") == "rerank"]
        if reranked:
            rerank_stage = {
                "name": "Reranker 精排阶段",
                "type": "rerank",
                "items": [],
            }
            for i, s in enumerate(reranked):
                rerank_stage["items"].append({
                    "index": i + 1,
                    "source": s.get("source", "未知"),
                    "similarity": s.get("similarity", 0),
                })
            tree["stages"].append(rerank_stage)

        # 阶段3：推理生成
        tree["stages"].append({
            "name": "LLM 推理生成阶段",
            "type": "generation",
            "items": [{"description": "基于检索结果综合推理，生成最终答案"}],
        })

        return tree

    def save_provenance(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: List[Dict],
        provenance_tree: Dict,
        confidence: Dict,
    ):
        """保存答案溯源记录"""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT INTO answer_provenance
                       (session_id, question, answer, sources_json, provenance_json, confidence_json)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (session_id, question, answer,
                     json.dumps(sources, ensure_ascii=False),
                     json.dumps(provenance_tree, ensure_ascii=False),
                     json.dumps(confidence, ensure_ascii=False))
                )
                conn.commit()
            finally:
                conn.close()

    def get_provenance(self, session_id: str) -> List[Dict]:
        """获取会话的溯源记录"""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM answer_provenance WHERE session_id = ? ORDER BY created_at DESC",
                    (session_id,)
                ).fetchall()
                result = []
                for r in rows:
                    item = dict(r)
                    for field in ["sources_json", "provenance_json", "confidence_json"]:
                        if item.get(field):
                            try:
                                item[field.replace("_json", "")] = json.loads(item[field])
                            except (json.JSONDecodeError, TypeError):
                                pass
                    result.append(item)
                return result
            finally:
                conn.close()


confidence_evaluator = ConfidenceEvaluator()