"""
collaboration.py - 协作式知识生态模块

功能：
  1. 答案评论与讨论
  2. 专家路由（低置信度问题自动推送给专家）
  3. 团队知识共建（文档协作编辑、审批流程）
"""
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "conversations.db")


class AnswerFeedback:
    """答案反馈与评论管理"""

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
                    CREATE TABLE IF NOT EXISTS answer_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        username TEXT DEFAULT 'anonymous',
                        rating TEXT DEFAULT '',
                        comment TEXT DEFAULT '',
                        created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS answer_comments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        feedback_id INTEGER NOT NULL,
                        username TEXT DEFAULT 'anonymous',
                        content TEXT NOT NULL,
                        parent_id INTEGER DEFAULT NULL,
                        created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                        FOREIGN KEY (feedback_id) REFERENCES answer_feedback(id)
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS expert_routing (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question TEXT NOT NULL,
                        confidence_score REAL DEFAULT 0,
                        assigned_expert TEXT DEFAULT '',
                        status TEXT DEFAULT 'pending',
                        expert_answer TEXT DEFAULT '',
                        created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                        resolved_at TEXT DEFAULT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_answer_feedback_session
                    ON answer_feedback(session_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_expert_routing_status
                    ON expert_routing(status)
                """)
                conn.commit()
            finally:
                conn.close()

    def submit_feedback(
        self,
        session_id: str,
        question: str,
        answer: str,
        username: str = "anonymous",
        rating: str = "",
        comment: str = "",
    ) -> int:
        """提交答案反馈"""
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    """INSERT INTO answer_feedback (session_id, question, answer, username, rating, comment)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (session_id, question, answer, username, rating, comment)
                )
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    def add_comment(
        self,
        feedback_id: int,
        username: str,
        content: str,
        parent_id: int = None,
    ) -> int:
        """添加评论"""
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    """INSERT INTO answer_comments (feedback_id, username, content, parent_id)
                       VALUES (?, ?, ?, ?)""",
                    (feedback_id, username, content, parent_id)
                )
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    def get_feedback(self, session_id: str = None, limit: int = 50) -> List[Dict]:
        """获取反馈列表"""
        with self._lock:
            conn = self._get_conn()
            try:
                if session_id:
                    rows = conn.execute(
                        "SELECT * FROM answer_feedback WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                        (session_id, limit)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM answer_feedback ORDER BY created_at DESC LIMIT ?",
                        (limit,)
                    ).fetchall()

                result = []
                for r in rows:
                    item = dict(r)
                    # 获取评论
                    comments = conn.execute(
                        "SELECT * FROM answer_comments WHERE feedback_id = ? ORDER BY created_at ASC",
                        (item["id"],)
                    ).fetchall()
                    item["comments"] = [dict(c) for c in comments]
                    item["comment_count"] = len(item["comments"])
                    result.append(item)
                return result
            finally:
                conn.close()

    def get_feedback_stats(self) -> Dict:
        """获取反馈统计"""
        with self._lock:
            conn = self._get_conn()
            try:
                total = conn.execute("SELECT COUNT(*) as cnt FROM answer_feedback").fetchone()["cnt"]
                positive = conn.execute(
                    "SELECT COUNT(*) as cnt FROM answer_feedback WHERE rating = 'positive'"
                ).fetchone()["cnt"]
                negative = conn.execute(
                    "SELECT COUNT(*) as cnt FROM answer_feedback WHERE rating = 'negative'"
                ).fetchone()["cnt"]
                total_comments = conn.execute(
                    "SELECT COUNT(*) as cnt FROM answer_comments"
                ).fetchone()["cnt"]

                return {
                    "total_feedback": total,
                    "positive": positive,
                    "negative": negative,
                    "neutral": total - positive - negative,
                    "total_comments": total_comments,
                    "satisfaction_rate": round(positive / total * 100, 1) if total > 0 else 0,
                }
            finally:
                conn.close()

    def get_user_feedback_for_qa(
        self,
        question: str,
        answer: str,
        username: str = "anonymous",
    ) -> Optional[Dict]:
        """查询用户对特定问答的反馈"""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    """SELECT * FROM answer_feedback
                       WHERE question = ? AND answer = ? AND username = ?
                       ORDER BY created_at DESC LIMIT 1""",
                    (question, answer, username)
                ).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def get_user_feedback_batch(
        self,
        qa_pairs: List[Dict],
        username: str = "anonymous",
    ) -> Dict[str, Dict]:
        """批量查询用户对多个问答的反馈状态

        Args:
            qa_pairs: [{"question": "...", "answer": "..."}, ...]
            username: 用户名

        Returns:
            {"question|||answer": {"rating": "positive", ...}, ...}
        """
        result = {}
        with self._lock:
            conn = self._get_conn()
            try:
                for pair in qa_pairs:
                    row = conn.execute(
                        """SELECT * FROM answer_feedback
                           WHERE question = ? AND answer = ? AND username = ?
                           ORDER BY created_at DESC LIMIT 1""",
                        (pair["question"], pair["answer"], username)
                    ).fetchone()
                    if row:
                        key = pair["question"] + "|||" + pair["answer"]
                        result[key] = dict(row)
                return result
            finally:
                conn.close()

    def upsert_feedback(
        self,
        session_id: str,
        question: str,
        answer: str,
        username: str = "anonymous",
        rating: str = "",
        comment: str = "",
    ) -> Dict:
        """提交或更新反馈（toggle模式）"""
        with self._lock:
            conn = self._get_conn()
            try:
                existing = conn.execute(
                    """SELECT id, rating FROM answer_feedback
                       WHERE question = ? AND answer = ? AND username = ?
                       ORDER BY created_at DESC LIMIT 1""",
                    (question, answer, username)
                ).fetchone()

                if existing:
                    existing_rating = existing["rating"]
                    if existing_rating == rating:
                        conn.execute(
                            "DELETE FROM answer_feedback WHERE id = ?",
                            (existing["id"],)
                        )
                        conn.commit()
                        return {"action": "deleted", "rating": None, "feedback_id": None}
                    else:
                        conn.execute(
                            """UPDATE answer_feedback SET rating = ?, comment = ?, session_id = ?,
                               created_at = datetime('now', 'localtime') WHERE id = ?""",
                            (rating, comment, session_id, existing["id"])
                        )
                        conn.commit()
                        return {"action": "updated", "rating": rating, "feedback_id": existing["id"]}
                else:
                    cursor = conn.execute(
                        """INSERT INTO answer_feedback (session_id, question, answer, username, rating, comment)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (session_id, question, answer, username, rating, comment)
                    )
                    conn.commit()
                    return {"action": "created", "rating": rating, "feedback_id": cursor.lastrowid}
            finally:
                conn.close()

    def delete_feedback(self, feedback_id: int) -> bool:
        """删除反馈"""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM answer_feedback WHERE id = ?", (feedback_id,))
                conn.commit()
                return True
            finally:
                conn.close()


class ExpertRouter:
    """专家路由器"""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def route_question(
        self,
        question: str,
        confidence_score: float,
        suggested_expert: str = "",
    ) -> int:
        """将低置信度问题路由给专家"""
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    """INSERT INTO expert_routing (question, confidence_score, assigned_expert, status)
                       VALUES (?, ?, ?, 'pending')""",
                    (question, confidence_score, suggested_expert)
                )
                conn.commit()
                logger.info("问题已路由给专家：confidence=%.1f%%, question=%s",
                            confidence_score, question[:50])
                return cursor.lastrowid
            finally:
                conn.close()

    def resolve_question(self, routing_id: int, expert_answer: str, expert_name: str = ""):
        """专家回答问题"""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """UPDATE expert_routing SET status='resolved', expert_answer=?,
                       assigned_expert=?, resolved_at=datetime('now','localtime')
                       WHERE id=?""",
                    (expert_answer, expert_name, routing_id)
                )
                conn.commit()
            finally:
                conn.close()

    def get_pending_questions(self) -> List[Dict]:
        """获取待处理的问题"""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM expert_routing WHERE status = 'pending' ORDER BY created_at DESC"
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_routing_stats(self) -> Dict:
        """获取路由统计"""
        with self._lock:
            conn = self._get_conn()
            try:
                total = conn.execute("SELECT COUNT(*) as cnt FROM expert_routing").fetchone()["cnt"]
                pending = conn.execute(
                    "SELECT COUNT(*) as cnt FROM expert_routing WHERE status='pending'"
                ).fetchone()["cnt"]
                resolved = conn.execute(
                    "SELECT COUNT(*) as cnt FROM expert_routing WHERE status='resolved'"
                ).fetchone()["cnt"]

                return {
                    "total_routed": total,
                    "pending": pending,
                    "resolved": resolved,
                    "resolution_rate": round(resolved / total * 100, 1) if total > 0 else 0,
                }
            finally:
                conn.close()


answer_feedback = AnswerFeedback()
expert_router = ExpertRouter()