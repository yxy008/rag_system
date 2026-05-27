"""
api_key_manager.py - API Key 管理模块

提供 API Key 的创建、验证、吊销、查询等功能，
用于 OpenAPI 接口的身份认证和访问控制。
"""

import hashlib
import logging
import os
import secrets
import sqlite3
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# 数据库路径（与 app.py 中一致）
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "conversations.db")


class APIKeyManager:
    """API Key 管理器"""

    def __init__(self):
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """初始化 api_keys 表"""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_key TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    is_active INTEGER DEFAULT 1,
                    rate_limit INTEGER DEFAULT 60,
                    created_by TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    last_used_at TEXT,
                    usage_count INTEGER DEFAULT 0
                )
            """)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _hash_key(key: str) -> str:
        """对 API Key 进行哈希存储（仅存储哈希值，不存储明文）"""
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def create_key(self, name: str, description: str = "",
                   rate_limit: int = 60, created_by: str = "") -> dict:
        """
        创建新的 API Key

        Args:
            name: Key 名称（便于识别用途）
            description: Key 描述
            rate_limit: 每分钟请求限制
            created_by: 创建者

        Returns:
            包含 api_key（明文，仅此一次返回）和元信息的字典
        """
        # 生成 API Key：前缀 "rag-" + 32字节随机hex
        raw_key = f"rag-{secrets.token_hex(32)}"
        key_hash = self._hash_key(raw_key)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = self._get_conn()
        try:
            cursor = conn.execute(
                """INSERT INTO api_keys
                   (api_key, name, description, rate_limit, created_by, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (key_hash, name, description, rate_limit, created_by, now),
            )
            conn.commit()
            key_id = cursor.lastrowid
        finally:
            conn.close()

        logger.info("API Key 创建成功：name=%s, id=%d", name, key_id)

        return {
            "id": key_id,
            "name": name,
            "api_key": raw_key,  # 明文仅此一次返回
            "description": description,
            "rate_limit": rate_limit,
            "created_at": now,
        }

    def validate_key(self, raw_key: str) -> Optional[dict]:
        """
        验证 API Key 是否有效

        Args:
            raw_key: 客户端传入的 API Key 明文

        Returns:
            有效时返回 Key 信息字典，无效时返回 None
        """
        if not raw_key or not raw_key.startswith("rag-"):
            return None

        key_hash = self._hash_key(raw_key)
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM api_keys WHERE api_key = ? AND is_active = 1",
                (key_hash,),
            ).fetchone()

            if not row:
                return None

            # 更新最后使用时间和使用次数
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                """UPDATE api_keys
                   SET last_used_at = ?, usage_count = usage_count + 1
                   WHERE id = ?""",
                (now, row["id"]),
            )
            conn.commit()

            return {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "rate_limit": row["rate_limit"],
                "created_by": row["created_by"],
                "created_at": row["created_at"],
                "usage_count": row["usage_count"] + 1,
            }
        finally:
            conn.close()

    def revoke_key(self, key_id: int) -> bool:
        """
        吊销 API Key（软删除，设为不活跃）

        Args:
            key_id: Key 的数据库 ID

        Returns:
            是否成功
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "UPDATE api_keys SET is_active = 0 WHERE id = ?",
                (key_id,),
            )
            conn.commit()
            success = cursor.rowcount > 0
        finally:
            conn.close()

        if success:
            logger.info("API Key 已吊销：id=%d", key_id)
        return success

    def activate_key(self, key_id: int) -> bool:
        """
        重新激活已吊销的 API Key

        Args:
            key_id: Key 的数据库 ID

        Returns:
            是否成功
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "UPDATE api_keys SET is_active = 1 WHERE id = ?",
                (key_id,),
            )
            conn.commit()
            success = cursor.rowcount > 0
        finally:
            conn.close()

        if success:
            logger.info("API Key 已重新激活：id=%d", key_id)
        return success

    def delete_key(self, key_id: int) -> bool:
        """
        永久删除 API Key

        Args:
            key_id: Key 的数据库 ID

        Returns:
            是否成功
        """
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "DELETE FROM api_keys WHERE id = ?",
                (key_id,),
            )
            conn.commit()
            success = cursor.rowcount > 0
        finally:
            conn.close()

        if success:
            logger.info("API Key 已永久删除：id=%d", key_id)
        return success

    def list_keys(self, include_inactive: bool = False) -> list:
        """
        列出所有 API Key

        Args:
            include_inactive: 是否包含已吊销的 Key

        Returns:
            Key 信息列表（不含 api_key 哈希值）
        """
        conn = self._get_conn()
        try:
            if include_inactive:
                rows = conn.execute(
                    "SELECT id, name, description, is_active, rate_limit, "
                    "created_by, created_at, last_used_at, usage_count "
                    "FROM api_keys ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, name, description, is_active, rate_limit, "
                    "created_by, created_at, last_used_at, usage_count "
                    "FROM api_keys WHERE is_active = 1 ORDER BY created_at DESC"
                ).fetchall()

            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_key(self, key_id: int) -> Optional[dict]:
        """
        获取单个 API Key 的信息

        Args:
            key_id: Key 的数据库 ID

        Returns:
            Key 信息字典或 None
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT id, name, description, is_active, rate_limit, "
                "created_by, created_at, last_used_at, usage_count "
                "FROM api_keys WHERE id = ?",
                (key_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def update_key(self, key_id: int, name: str = None,
                   description: str = None, rate_limit: int = None) -> bool:
        """
        更新 API Key 的信息

        Args:
            key_id: Key 的数据库 ID
            name: 新名称（可选）
            description: 新描述（可选）
            rate_limit: 新速率限制（可选）

        Returns:
            是否成功
        """
        updates = []
        params = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if rate_limit is not None:
            updates.append("rate_limit = ?")
            params.append(rate_limit)

        if not updates:
            return False

        params.append(key_id)
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                f"UPDATE api_keys SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()


# 全局单例
api_key_manager = APIKeyManager()
