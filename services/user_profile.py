"""
user_profile.py - 用户画像与个性化适配引擎

功能：
  1. 用户画像自动构建（基于提问历史静默学习）
  2. 自适应回答引擎（根据画像调整回答深度和风格）
  3. 个人知识空间（收藏答案、标注笔记、导出个人手册）
  4. 回答风格偏好管理
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


class UserProfileManager:
    """用户画像管理器"""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._lock = threading.RLock()
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
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        role TEXT DEFAULT 'viewer',
                        expertise_level TEXT DEFAULT 'intermediate',
                        interests TEXT DEFAULT '[]',
                        style_preference TEXT DEFAULT 'detailed',
                        created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                        updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_bookmarks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        sources TEXT DEFAULT NULL,
                        note TEXT DEFAULT '',
                        created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_question_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        question TEXT NOT NULL,
                        topic TEXT DEFAULT '',
                        created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_profiles_username
                    ON user_profiles(username)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_bookmarks_username
                    ON user_bookmarks(username)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_question_history_username
                    ON user_question_history(username)
                """)
                conn.commit()
            finally:
                conn.close()

    def get_profile(self, username: str) -> Dict:
        """获取用户画像"""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    "SELECT * FROM user_profiles WHERE username = ?",
                    (username,)
                ).fetchone()
                if row:
                    return {
                        "username": row["username"],
                        "role": row["role"],
                        "expertise_level": row["expertise_level"],
                        "interests": json.loads(row["interests"] or "[]"),
                        "style_preference": row["style_preference"],
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
                return self._create_default_profile(username)
            finally:
                conn.close()

    def _create_default_profile(self, username: str) -> Dict:
        """创建默认用户画像"""
        profile = {
            "username": username,
            "role": "viewer",
            "expertise_level": "intermediate",
            "interests": [],
            "style_preference": "detailed",
        }
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO user_profiles
                       (username, role, expertise_level, interests, style_preference)
                       VALUES (?, ?, ?, ?, ?)""",
                    (username, profile["role"], profile["expertise_level"],
                     json.dumps(profile["interests"], ensure_ascii=False),
                     profile["style_preference"])
                )
                conn.commit()
            finally:
                conn.close()
        return profile

    def update_profile(self, username: str, updates: Dict) -> Dict:
        """更新用户画像"""
        profile = self.get_profile(username)
        allowed_fields = ["role", "expertise_level", "interests", "style_preference"]
        for field in allowed_fields:
            if field in updates:
                if field == "interests" and isinstance(updates[field], list):
                    profile[field] = updates[field]
                elif field != "interests":
                    profile[field] = updates[field]

        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    """UPDATE user_profiles SET role=?, expertise_level=?,
                       interests=?, style_preference=?, updated_at=datetime('now','localtime')
                       WHERE username=?""",
                    (profile["role"], profile["expertise_level"],
                     json.dumps(profile["interests"], ensure_ascii=False),
                     profile["style_preference"], username)
                )
                conn.commit()
            finally:
                conn.close()
        return profile

    def update_style_preference(self, username: str, style: str) -> Dict:
        """更新回答风格偏好"""
        valid_styles = ["concise", "detailed", "technical", "plain"]
        if style not in valid_styles:
            style = "detailed"
        return self.update_profile(username, {"style_preference": style})

    def record_question(self, username: str, question: str, topic: str = ""):
        """记录用户提问历史，用于画像推断"""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO user_question_history (username, question, topic) VALUES (?, ?, ?)",
                    (username, question, topic)
                )
                conn.commit()
            finally:
                conn.close()

    def get_question_history(self, username: str, limit: int = 50) -> List[Dict]:
        """获取用户提问历史"""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM user_question_history WHERE username = ? ORDER BY created_at DESC LIMIT ?",
                    (username, limit)
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def add_bookmark(self, username: str, question: str, answer: str,
                     sources: List[Dict] = None, note: str = "") -> int:
        """收藏答案到个人知识空间"""
        sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    """INSERT INTO user_bookmarks (username, question, answer, sources, note)
                       VALUES (?, ?, ?, ?, ?)""",
                    (username, question, answer, sources_json, note)
                )
                conn.commit()
                return cursor.lastrowid
            finally:
                conn.close()

    def update_bookmark_note(self, bookmark_id: int, username: str, note: str):
        """更新收藏笔记"""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "UPDATE user_bookmarks SET note = ? WHERE id = ? AND username = ?",
                    (note, bookmark_id, username)
                )
                conn.commit()
            finally:
                conn.close()

    def delete_bookmark(self, bookmark_id: int, username: str):
        """删除收藏"""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "DELETE FROM user_bookmarks WHERE id = ? AND username = ?",
                    (bookmark_id, username)
                )
                conn.commit()
            finally:
                conn.close()

    def get_bookmarks(self, username: str) -> List[Dict]:
        """获取用户所有收藏"""
        with self._lock:
            conn = self._get_conn()
            try:
                rows = conn.execute(
                    "SELECT * FROM user_bookmarks WHERE username = ? ORDER BY created_at DESC",
                    (username,)
                ).fetchall()
                result = []
                for r in rows:
                    item = dict(r)
                    if item.get("sources"):
                        try:
                            item["sources"] = json.loads(item["sources"])
                        except (json.JSONDecodeError, TypeError):
                            item["sources"] = []
                    result.append(item)
                return result
            finally:
                conn.close()

    def check_bookmark_exists(self, username: str, question: str, answer: str) -> Optional[Dict]:
        """检查用户是否已收藏某个问答"""
        with self._lock:
            conn = self._get_conn()
            try:
                row = conn.execute(
                    """SELECT * FROM user_bookmarks
                       WHERE username = ? AND question = ? AND answer = ?
                       ORDER BY created_at DESC LIMIT 1""",
                    (username, question, answer)
                ).fetchone()
                if row:
                    item = dict(row)
                    if item.get("sources"):
                        try:
                            item["sources"] = json.loads(item["sources"])
                        except (json.JSONDecodeError, TypeError):
                            item["sources"] = []
                    return item
                return None
            finally:
                conn.close()

    def delete_bookmark_by_qa(self, username: str, question: str, answer: str) -> bool:
        """根据问答内容删除收藏"""
        with self._lock:
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    "DELETE FROM user_bookmarks WHERE username = ? AND question = ? AND answer = ?",
                    (username, question, answer)
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def toggle_bookmark(self, username: str, question: str, answer: str,
                        sources: List[Dict] = None, note: str = "") -> Dict:
        """收藏toggle：已收藏则取消，未收藏则添加"""
        existing = self.check_bookmark_exists(username, question, answer)
        if existing:
            self.delete_bookmark_by_qa(username, question, answer)
            return {"action": "deleted", "bookmark_id": None}
        else:
            bookmark_id = self.add_bookmark(username, question, answer, sources, note)
            return {"action": "created", "bookmark_id": bookmark_id}

    def get_bookmarks_batch_check(self, username: str, qa_pairs: List[Dict]) -> Dict[str, bool]:
        """批量检查多个问答是否已收藏

        Args:
            username: 用户名
            qa_pairs: [{"question": "...", "answer": "..."}, ...]

        Returns:
            {"question|||answer": true/false, ...}
        """
        result = {}
        with self._lock:
            conn = self._get_conn()
            try:
                for pair in qa_pairs:
                    row = conn.execute(
                        """SELECT id FROM user_bookmarks
                           WHERE username = ? AND question = ? AND answer = ?
                           LIMIT 1""",
                        (username, pair["question"], pair["answer"])
                    ).fetchone()
                    key = pair["question"] + "|||" + pair["answer"]
                    result[key] = row is not None
                return result
            finally:
                conn.close()

    def get_adaptive_prompt_context(self, username: str, style: str = None) -> str:
        """根据用户画像生成自适应 Prompt 上下文"""
        profile = self.get_profile(username)
        if style is None:
            style = profile.get("style_preference", "detailed")
        expertise = profile.get("expertise_level", "intermediate")
        role = profile.get("role", "viewer")

        style_instructions = {
            "concise": "请用简洁精炼的语言回答，直接给出核心结论和关键要点，省略不必要的背景说明。",
            "detailed": "请给出详细完整的回答，包含背景说明、具体步骤、注意事项和引用来源。",
            "technical": "请使用专业术语和技术性语言回答，深入原理和细节，适合专业人士阅读。",
            "plain": "请用通俗易懂的语言回答，避免专业术语，多用比喻和举例，适合非专业人士理解。",
        }

        expertise_instructions = {
            "beginner": "用户是新手，请从基础概念开始解释，多用举例说明。",
            "intermediate": "用户有一定基础，可以直接给出要点和操作指引。",
            "expert": "用户是专家，可以深入讨论细节、边界情况和最佳实践。",
        }

        parts = []
        parts.append(style_instructions.get(style, style_instructions["detailed"]))
        parts.append(expertise_instructions.get(expertise, expertise_instructions["intermediate"]))

        if role == "admin":
            parts.append("用户是管理员，可以展示系统配置、统计数据等管理相关信息。")

        return "\n".join(parts)


profile_manager = UserProfileManager()