"""
evaluation.py - 评估模块

追踪系统运行指标，包括：
  - 请求统计：总请求数、LLM 调用次数、缓存命中/未命中
  - Token 统计：总 Token 消耗、每次请求 Token 数、Token 趋势
  - 延迟统计：平均延迟、检索延迟
  - 时间序列：最近 N 次请求的详细信息（用于图表展示）
  - 缓存效率：精确匹配命中率、语义匹配命中率、总体命中率

持久化方案：使用 SQLite 存储指标数据，服务重启后数据不丢失。
  - eval_counters 表：累计计数器（单行记录）
  - eval_recent_requests 表：最近请求记录（保留最近 N 条）
"""
import json
import logging
import os
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

from core.config import BASE_DIR

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(str(BASE_DIR), "data", "conversations.db")


class EvaluationService:
    """评估服务：追踪和统计系统运行指标（SQLite 持久化）"""

    def __init__(self, max_recent_requests: int = 100):
        self._lock = threading.Lock()
        self._max_recent = max_recent_requests

        # 请求统计
        self._total_requests = 0
        self._llm_requests = 0
        self._exact_cache_hits = 0
        self._semantic_cache_hits = 0
        self._cache_misses = 0

        # Token 统计
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_tokens = 0

        # 延迟统计
        self._total_llm_latency_ms = 0.0
        self._total_retrieval_latency_ms = 0.0

        # 最近请求记录
        self._recent_requests: deque = deque(maxlen=max_recent_requests)

        # 初始化数据库并从数据库恢复数据
        self._init_db()
        self._load_from_db()

    def _get_conn(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """初始化数据库表结构"""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eval_counters (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    total_requests INTEGER DEFAULT 0,
                    llm_requests INTEGER DEFAULT 0,
                    exact_cache_hits INTEGER DEFAULT 0,
                    semantic_cache_hits INTEGER DEFAULT 0,
                    cache_misses INTEGER DEFAULT 0,
                    total_input_tokens INTEGER DEFAULT 0,
                    total_output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_llm_latency_ms REAL DEFAULT 0.0,
                    total_retrieval_latency_ms REAL DEFAULT 0.0,
                    updated_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eval_recent_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    question TEXT DEFAULT '',
                    from_cache INTEGER DEFAULT 0,
                    cache_match_type TEXT,
                    latency_ms REAL DEFAULT 0,
                    retrieval_latency_ms REAL DEFAULT 0,
                    source_count INTEGER DEFAULT 0,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_eval_recent_timestamp
                ON eval_recent_requests(timestamp DESC)
            """)
            conn.commit()
        finally:
            conn.close()

    def _load_from_db(self):
        """从数据库恢复累计计数器和最近请求记录"""
        conn = self._get_conn()
        try:
            row = conn.execute("SELECT * FROM eval_counters WHERE id = 1").fetchone()
            if row:
                self._total_requests = row["total_requests"] or 0
                self._llm_requests = row["llm_requests"] or 0
                self._exact_cache_hits = row["exact_cache_hits"] or 0
                self._semantic_cache_hits = row["semantic_cache_hits"] or 0
                self._cache_misses = row["cache_misses"] or 0
                self._total_input_tokens = row["total_input_tokens"] or 0
                self._total_output_tokens = row["total_output_tokens"] or 0
                self._total_tokens = row["total_tokens"] or 0
                self._total_llm_latency_ms = row["total_llm_latency_ms"] or 0.0
                self._total_retrieval_latency_ms = row["total_retrieval_latency_ms"] or 0.0
                logger.info(f"从数据库恢复指标：总请求 {self._total_requests}，LLM调用 {self._llm_requests}")

            rows = conn.execute(
                "SELECT * FROM eval_recent_requests ORDER BY id DESC LIMIT ?",
                (self._max_recent,)
            ).fetchall()
            for row in reversed(rows):
                record = {
                    "timestamp": row["timestamp"],
                    "question": row["question"] or "",
                    "from_cache": bool(row["from_cache"]),
                    "cache_match_type": row["cache_match_type"],
                    "latency_ms": row["latency_ms"] or 0,
                    "retrieval_latency_ms": row["retrieval_latency_ms"] or 0,
                    "source_count": row["source_count"] or 0,
                    "input_tokens": row["input_tokens"] or 0,
                    "output_tokens": row["output_tokens"] or 0,
                    "total_tokens": row["total_tokens"] or 0,
                }
                self._recent_requests.append(record)
            if rows:
                logger.info(f"从数据库恢复 {len(rows)} 条最近请求记录")
        finally:
            conn.close()

    def _save_counters(self, conn: sqlite3.Connection):
        """保存累计计数器到数据库"""
        conn.execute("""
            INSERT OR REPLACE INTO eval_counters
            (id, total_requests, llm_requests, exact_cache_hits, semantic_cache_hits,
             cache_misses, total_input_tokens, total_output_tokens, total_tokens,
             total_llm_latency_ms, total_retrieval_latency_ms, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self._total_requests, self._llm_requests,
            self._exact_cache_hits, self._semantic_cache_hits, self._cache_misses,
            self._total_input_tokens, self._total_output_tokens, self._total_tokens,
            self._total_llm_latency_ms, self._total_retrieval_latency_ms,
            datetime.now().isoformat()
        ))

    def _save_recent_request(self, conn: sqlite3.Connection, record: Dict):
        """保存一条最近请求记录到数据库"""
        conn.execute("""
            INSERT INTO eval_recent_requests
            (timestamp, question, from_cache, cache_match_type, latency_ms,
             retrieval_latency_ms, source_count, input_tokens, output_tokens, total_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["timestamp"], record["question"],
            1 if record["from_cache"] else 0, record["cache_match_type"],
            record["latency_ms"], record["retrieval_latency_ms"],
            record["source_count"], record["input_tokens"],
            record["output_tokens"], record["total_tokens"]
        ))

    def _prune_old_records(self, conn: sqlite3.Connection):
        """清理超出保留数量的旧记录"""
        conn.execute("""
            DELETE FROM eval_recent_requests WHERE id NOT IN (
                SELECT id FROM eval_recent_requests ORDER BY id DESC LIMIT ?
            )
        """, (self._max_recent,))

    def record_request(
        self,
        from_cache: bool,
        cache_match_type: Optional[str],
        total_latency_ms: float,
        retrieval_latency_ms: float,
        source_count: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        question: str = "",
    ):
        """记录一次请求（内存 + SQLite 持久化）"""
        with self._lock:
            self._total_requests += 1

            if from_cache:
                if cache_match_type == "exact":
                    self._exact_cache_hits += 1
                else:
                    self._semantic_cache_hits += 1
            else:
                self._llm_requests += 1
                self._cache_misses += 1
                self._total_llm_latency_ms += total_latency_ms
                self._total_retrieval_latency_ms += retrieval_latency_ms

                # Token 统计（仅 LLM 调用时记录）
                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens
                self._total_tokens += input_tokens + output_tokens

            # 记录到时间序列
            record = {
                "timestamp": datetime.now().isoformat(),
                "question": question[:80] if question else "",
                "from_cache": from_cache,
                "cache_match_type": cache_match_type,
                "latency_ms": round(total_latency_ms, 1),
                "retrieval_latency_ms": round(retrieval_latency_ms, 1),
                "source_count": source_count,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
            self._recent_requests.append(record)

            # 持久化到 SQLite
            try:
                conn = self._get_conn()
                try:
                    self._save_counters(conn)
                    self._save_recent_request(conn, record)
                    self._prune_old_records(conn)
                    conn.commit()
                finally:
                    conn.close()
            except Exception as e:
                logger.warning(f"指标持久化失败（不影响主流程）：{e}")

    def get_stats(self) -> Dict:
        """获取评估统计信息"""
        with self._lock:
            total = self._total_requests
            llm = self._llm_requests
            exact_hits = self._exact_cache_hits
            semantic_hits = self._semantic_cache_hits
            misses = self._cache_misses
            cache_hits = exact_hits + semantic_hits

            stats = {
                "total_requests": total,
                "llm_requests": llm,
                "cache_hits": cache_hits,
                "cache_misses": misses,
                "exact_cache_hits": exact_hits,
                "semantic_cache_hits": semantic_hits,
            }

            # 命中率
            if total > 0:
                stats["overall_hit_rate"] = f"{cache_hits / total * 100:.1f}%"
                stats["exact_hit_rate"] = f"{exact_hits / total * 100:.1f}%"
                stats["semantic_hit_rate"] = f"{semantic_hits / total * 100:.1f}%"
            else:
                stats["overall_hit_rate"] = "0.0%"
                stats["exact_hit_rate"] = "0.0%"
                stats["semantic_hit_rate"] = "0.0%"

            # 延迟统计
            if llm > 0:
                stats["avg_llm_latency_ms"] = f"{self._total_llm_latency_ms / llm:.0f}"
                stats["avg_retrieval_latency_ms"] = f"{self._total_retrieval_latency_ms / llm:.0f}"
            else:
                stats["avg_llm_latency_ms"] = "0"
                stats["avg_retrieval_latency_ms"] = "0"

            # Token 统计
            stats["total_input_tokens"] = self._total_input_tokens
            stats["total_output_tokens"] = self._total_output_tokens
            stats["total_tokens"] = self._total_tokens
            if llm > 0:
                stats["avg_input_tokens"] = round(self._total_input_tokens / llm)
                stats["avg_output_tokens"] = round(self._total_output_tokens / llm)
            else:
                stats["avg_input_tokens"] = 0
                stats["avg_output_tokens"] = 0

            # 最近请求记录
            stats["recent_requests"] = list(self._recent_requests)

            return stats

    def reset(self):
        """重置所有统计（内存 + 数据库）"""
        with self._lock:
            self._total_requests = 0
            self._llm_requests = 0
            self._exact_cache_hits = 0
            self._semantic_cache_hits = 0
            self._cache_misses = 0
            self._total_input_tokens = 0
            self._total_output_tokens = 0
            self._total_tokens = 0
            self._total_llm_latency_ms = 0.0
            self._total_retrieval_latency_ms = 0.0
            self._recent_requests.clear()

            try:
                conn = self._get_conn()
                try:
                    conn.execute("DELETE FROM eval_counters")
                    conn.execute("DELETE FROM eval_recent_requests")
                    conn.commit()
                finally:
                    conn.close()
            except Exception as e:
                logger.warning(f"重置数据库失败：{e}")

            logger.info("评估统计已重置")


# 全局单例
evaluation_service = EvaluationService()