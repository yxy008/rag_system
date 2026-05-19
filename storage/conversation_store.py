"""
conversation_store.py - 对话历史持久化模块

使用 SQLite 将对话记录持久化到本地文件，替代原有的内存 defaultdict 方案。
支持：
  - 按会话查询历史
  - 自动清理过期记录
  - 会话列表查询
  - 会话重命名
"""
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "conversations.db")
MAX_HISTORY_PER_SESSION = 20
RETENTION_DAYS = 30


class ConversationStore:
    """基于 SQLite 的对话历史存储"""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """初始化数据库表"""
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        session_name TEXT DEFAULT '',
                        user_id TEXT DEFAULT '',
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        sources TEXT DEFAULT NULL,
                        created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                    )
                """)
                # 兼容旧数据库：先添加缺失的列，再创建索引（否则索引会因列不存在而失败）
                try:
                    conn.execute("ALTER TABLE conversation_history ADD COLUMN sources TEXT DEFAULT NULL")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE conversation_history ADD COLUMN user_id TEXT DEFAULT ''")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE conversation_history ADD COLUMN username TEXT DEFAULT ''")
                except sqlite3.OperationalError:
                    pass
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_session_id
                    ON conversation_history(session_id)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_created_at
                    ON conversation_history(created_at)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_id
                    ON conversation_history(user_id)
                """)
                conn.commit()
            finally:
                conn.close()

    def save(self, session_id: str, role: str, content: str, sources: Optional[List[Dict]] = None, user_id: str = ""):
        """保存一条对话记录，可选附带来源信息和用户ID"""
        sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
        with self._lock:
            conn = self._get_conn()
            try:
                # 获取当前会话名称（如果已有）
                existing_name = None
                if role != "user":
                    name_row = conn.execute(
                        "SELECT session_name FROM conversation_history WHERE session_id = ? AND session_name != '' LIMIT 1",
                        (session_id,),
                    ).fetchone()
                    if name_row:
                        existing_name = name_row["session_name"]

                # 如果是该会话的第一条用户消息，自动设置会话名称
                if role == "user":
                    existing = conn.execute(
                        "SELECT COUNT(*) as cnt FROM conversation_history WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()
                    if existing and existing["cnt"] == 0:
                        session_name = content[:30] + ("..." if len(content) > 30 else "")
                        conn.execute(
                            "INSERT INTO conversation_history (session_id, session_name, user_id, role, content, sources) VALUES (?, ?, ?, ?, ?, ?)",
                            (session_id, session_name, user_id, role, content, sources_json),
                        )
                        conn.commit()
                        self._trim_history(conn, session_id)
                        return

                # 后续记录也带上 session_name 和 user_id（如果已知）
                if existing_name:
                    conn.execute(
                        "INSERT INTO conversation_history (session_id, session_name, user_id, role, content, sources) VALUES (?, ?, ?, ?, ?, ?)",
                        (session_id, existing_name, user_id, role, content, sources_json),
                    )
                else:
                    conn.execute(
                        "INSERT INTO conversation_history (session_id, user_id, role, content, sources) VALUES (?, ?, ?, ?, ?)",
                        (session_id, user_id, role, content, sources_json),
                    )
                conn.commit()
                self._trim_history(conn, session_id)
            finally:
                conn.close()

    def save_pair(self, session_id: str, question: str, answer: str, sources: Optional[List[Dict]] = None, user_id: str = ""):
        """批量保存一问一答，可选附带来源信息和用户ID"""
        sources_json = json.dumps(sources, ensure_ascii=False) if sources else None
        with self._lock:
            conn = self._get_conn()
            try:
                # 如果是该会话的第一条消息，自动设置会话名称
                existing = conn.execute(
                    "SELECT COUNT(*) as cnt FROM conversation_history WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                is_first = not existing or existing["cnt"] == 0

                if is_first:
                    session_name = question[:30] + ("..." if len(question) > 30 else "")
                    conn.execute(
                        "INSERT INTO conversation_history (session_id, session_name, user_id, role, content) VALUES (?, ?, ?, ?, ?)",
                        (session_id, session_name, user_id, "user", question),
                    )
                else:
                    conn.execute(
                        "INSERT INTO conversation_history (session_id, user_id, role, content) VALUES (?, ?, ?, ?)",
                        (session_id, user_id, "user", question),
                    )
                conn.execute(
                    "INSERT INTO conversation_history (session_id, user_id, role, content, sources) VALUES (?, ?, ?, ?, ?)",
                    (session_id, user_id, "assistant", answer, sources_json),
                )
                conn.commit()
                self._trim_history(conn, session_id)
            finally:
                conn.close()

    def get_history(self, session_id: str, user_id: str = "") -> List[Dict[str, str]]:
        """获取会话的对话历史（用于 LLM 上下文），包含来源信息，可按用户ID过滤"""
        conn = self._get_conn()
        try:
            if user_id:
                rows = conn.execute(
                    "SELECT role, content, sources FROM conversation_history WHERE session_id = ? AND user_id = ? ORDER BY created_at ASC",
                    (session_id, user_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT role, content, sources FROM conversation_history WHERE session_id = ? ORDER BY created_at ASC",
                    (session_id,),
                ).fetchall()
            result = []
            for row in rows:
                item = {"role": row["role"], "content": row["content"]}
                if row["sources"]:
                    try:
                        item["sources"] = json.loads(row["sources"])
                    except (json.JSONDecodeError, TypeError):
                        item["sources"] = None
                result.append(item)
            return result
        finally:
            conn.close()

    def get_history_entities(self, session_id: str, user_id: str = "") -> List[Dict]:
        """获取会话的对话历史（原始实体列表，用于前端展示），可按用户ID过滤"""
        conn = self._get_conn()
        try:
            if user_id:
                rows = conn.execute(
                    "SELECT id, session_id, role, content, created_at FROM conversation_history WHERE session_id = ? AND user_id = ? ORDER BY created_at ASC",
                    (session_id, user_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, session_id, role, content, created_at FROM conversation_history WHERE session_id = ? ORDER BY created_at ASC",
                    (session_id,),
                ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def delete_session(self, session_id: str, user_id: str = ""):
        """删除指定会话的所有历史，可按用户ID过滤"""
        with self._lock:
            conn = self._get_conn()
            try:
                if user_id:
                    cursor = conn.execute(
                        "DELETE FROM conversation_history WHERE session_id = ? AND user_id = ?",
                        (session_id, user_id),
                    )
                else:
                    cursor = conn.execute(
                        "DELETE FROM conversation_history WHERE session_id = ?",
                        (session_id,),
                    )
                deleted = cursor.rowcount
                conn.commit()
                logger.info("已删除会话 %s 的历史记录，共 %d 条", session_id, deleted)
            finally:
                conn.close()

    def get_active_session_count(self, user_id: str = "") -> int:
        """获取活跃会话数，user_id 为空则统计所有"""
        conn = self._get_conn()
        try:
            if user_id:
                row = conn.execute(
                    "SELECT COUNT(DISTINCT session_id) as cnt FROM conversation_history WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(DISTINCT session_id) as cnt FROM conversation_history"
                ).fetchone()
            return row["cnt"] if row else 0
        finally:
            conn.close()

    def get_all_sessions(self, user_id: str = "") -> List[Dict]:
        """获取指定用户的会话列表（用于前端会话管理），user_id 为空则返回所有"""
        conn = self._get_conn()
        try:
            if user_id:
                rows = conn.execute("""
                    SELECT
                        session_id,
                        MAX(session_name) as session_name,
                        MIN(created_at) as created_at,
                        MAX(created_at) as last_active_at,
                        COUNT(*) as message_count
                    FROM conversation_history
                    WHERE user_id = ?
                    GROUP BY session_id
                    ORDER BY last_active_at DESC
                """, (user_id,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT
                        session_id,
                        MAX(session_name) as session_name,
                        MIN(created_at) as created_at,
                        MAX(created_at) as last_active_at,
                        COUNT(*) as message_count
                    FROM conversation_history
                    GROUP BY session_id
                    ORDER BY last_active_at DESC
                """).fetchall()

            sessions = []
            for row in rows:
                name = row["session_name"] or ""
                # 如果 session_name 仍为空，尝试从第一条用户消息中提取
                if not name:
                    first_msg = conn.execute(
                        "SELECT content FROM conversation_history WHERE session_id = ? AND role = 'user' ORDER BY created_at ASC LIMIT 1",
                        (row["session_id"],),
                    ).fetchone()
                    if first_msg:
                        content = first_msg["content"]
                        name = content[:30] + ("..." if len(content) > 30 else "")

                sessions.append({
                    "session_id": row["session_id"],
                    "session_name": name,
                    "created_at": row["created_at"],
                    "last_active_at": row["last_active_at"],
                    "message_count": row["message_count"],
                })
            return sessions
        finally:
            conn.close()

    def rename_session(self, session_id: str, new_name: str, user_id: str = "") -> bool:
        """重命名会话，可按用户ID过滤"""
        with self._lock:
            conn = self._get_conn()
            try:
                if user_id:
                    cursor = conn.execute(
                        "UPDATE conversation_history SET session_name = ? WHERE session_id = ? AND user_id = ?",
                        (new_name, session_id, user_id),
                    )
                else:
                    cursor = conn.execute(
                        "UPDATE conversation_history SET session_name = ? WHERE session_id = ?",
                        (new_name, session_id),
                    )
                updated = cursor.rowcount
                conn.commit()
                if updated > 0:
                    logger.info("会话 %s 已重命名为 %s", session_id, new_name)
                return updated > 0
            finally:
                conn.close()

    def _trim_history(self, conn: sqlite3.Connection, session_id: str):
        """限制每个 session 的历史记录数量（保留最近 N 轮）"""
        max_records = MAX_HISTORY_PER_SESSION * 2
        rows = conn.execute(
            "SELECT id FROM conversation_history WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()

        if len(rows) > max_records:
            to_delete = len(rows) - max_records
            ids_to_delete = [row["id"] for row in rows[:to_delete]]
            conn.execute(
                "DELETE FROM conversation_history WHERE id IN ({})".format(
                    ",".join("?" * len(ids_to_delete))
                ),
                ids_to_delete,
            )
            conn.commit()
            logger.debug("会话 %s 历史记录已裁剪，删除 %d 条旧记录", session_id, to_delete)

    def cleanup_expired(self):
        """清理过期记录"""
        with self._lock:
            conn = self._get_conn()
            try:
                cutoff = (datetime.now() - timedelta(days=RETENTION_DAYS)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                cursor = conn.execute(
                    "DELETE FROM conversation_history WHERE created_at < ?",
                    (cutoff,),
                )
                deleted = cursor.rowcount
                conn.commit()
                if deleted > 0:
                    logger.info("定时清理完成：删除 %d 条过期对话记录（超过 %d 天）", deleted, RETENTION_DAYS)
            finally:
                conn.close()


# 全局单例
conversation_store = ConversationStore()