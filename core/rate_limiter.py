"""
rate_limiter.py - 速率限制模块

基于滑动窗口（Sliding Window）算法的请求速率限制。

方案对比：
  - 固定窗口：简单但存在边界突发问题（窗口切换瞬间可能双倍流量）
  - 滑动日志：精确但内存占用高（需存储每个请求时间戳）
  - 滑动窗口（本实现）：折中方案，兼顾精度和性能

核心机制：
  1. 以用户 IP 或 session_id 为 key 进行限流
  2. 在时间窗口内记录请求时间戳
  3. 每次请求时清理过期时间戳，统计窗口内请求数
  4. 超过限制返回 429 Too Many Requests
"""
import logging
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SlidingWindowRateLimiter:
    """
    滑动窗口速率限制器

    使用示例：
        limiter = SlidingWindowRateLimiter(window_seconds=60, max_requests=100)
        allowed, retry_after = limiter.is_allowed("192.168.1.1")
        if not allowed:
            return 429, retry_after
    """

    def __init__(
        self,
        window_seconds: int = 60,
        max_requests: int = 60,
        cleanup_interval: int = 300,
    ):
        """
        Args:
            window_seconds: 时间窗口大小（秒），默认 60 秒
            max_requests: 窗口内最大请求数，默认 60 次
            cleanup_interval: 清理过期 key 的间隔（秒），默认 300 秒
        """
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.cleanup_interval = cleanup_interval

        # key -> 请求时间戳列表
        self._windows: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def is_allowed(self, key: str) -> Tuple[bool, float]:
        """
        检查指定 key 是否允许通过

        Args:
            key: 限流标识（如 IP 地址、session_id 等）

        Returns:
            (allowed, retry_after_seconds)
            - allowed: True 表示允许通过
            - retry_after_seconds: 如果不允许，建议多少秒后重试
        """
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds

            # 获取或初始化该 key 的时间戳列表
            timestamps = self._windows[key]

            # 清理过期时间戳（窗口外的）
            while timestamps and timestamps[0] < window_start:
                timestamps.pop(0)

            # 检查是否超过限制
            if len(timestamps) >= self.max_requests:
                # 计算需要等待的时间
                oldest_in_window = timestamps[0]
                retry_after = oldest_in_window + self.window_seconds - now
                retry_after = max(0.1, retry_after)
                return False, retry_after

            # 记录本次请求时间戳
            timestamps.append(now)

            # 定期清理过期 key（避免内存泄漏）
            self._maybe_cleanup(now)

            return True, 0.0

    def _maybe_cleanup(self, now: float):
        """定期清理长时间未使用的 key"""
        if now - self._last_cleanup < self.cleanup_interval:
            return

        self._last_cleanup = now
        expired_keys = []
        window_start = now - self.window_seconds

        for key, timestamps in self._windows.items():
            # 清理过期时间戳
            while timestamps and timestamps[0] < window_start:
                timestamps.pop(0)
            # 如果该 key 没有任何有效时间戳，标记删除
            if not timestamps:
                expired_keys.append(key)

        for key in expired_keys:
            del self._windows[key]

        if expired_keys:
            logger.debug("速率限制器清理了 %d 个过期 key", len(expired_keys))

    def get_stats(self) -> Dict:
        """获取速率限制统计信息"""
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds

            active_keys = 0
            total_requests_in_window = 0
            blocked_keys = 0

            for timestamps in self._windows.values():
                # 统计窗口内的请求
                count = sum(1 for t in timestamps if t >= window_start)
                if count > 0:
                    active_keys += 1
                    total_requests_in_window += count
                    if count >= self.max_requests:
                        blocked_keys += 1

            return {
                "window_seconds": self.window_seconds,
                "max_requests": self.max_requests,
                "active_keys": active_keys,
                "total_requests_in_window": total_requests_in_window,
                "blocked_keys": blocked_keys,
                "total_tracked_keys": len(self._windows),
            }

    def reset(self):
        """重置所有限流状态"""
        with self._lock:
            self._windows.clear()
            logger.info("速率限制器已重置")


# 全局单例（默认：每分钟 60 次请求）
rate_limiter = SlidingWindowRateLimiter(
    window_seconds=60,
    max_requests=60,
)