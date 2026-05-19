"""
llm_retry.py - LLM API 调用重试模块

当调用 LLM API 触发 429 (Too Many Requests) 错误时，
使用指数退避（Exponential Backoff）策略进行重试。

重试策略：
  - 退避公式：wait_time = 2^n + random(0, 2) 秒
    其中 n 为当前重试次数（从 0 开始）
  - 最大重试次数：默认 5 次
  - 仅对 429 状态码进行重试
  - 其他错误直接抛出

示例：
  第 0 次重试：等待 2^0 + random(0,2) = 1~3 秒
  第 1 次重试：等待 2^1 + random(0,2) = 2~4 秒
  第 2 次重试：等待 2^2 + random(0,2) = 4~6 秒
  第 3 次重试：等待 2^3 + random(0,2) = 8~10 秒
  第 4 次重试：等待 2^4 + random(0,2) = 16~18 秒
"""
import logging
import random
import time
from functools import wraps
from typing import Callable, TypeVar, Any

logger = logging.getLogger(__name__)

T = TypeVar("T")

# 默认最大重试次数
DEFAULT_MAX_RETRIES = 5

# 退避基数
BACKOFF_BASE = 2

# 随机抖动范围（秒）
JITTER_MIN = 0.0
JITTER_MAX = 2.0


def with_exponential_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base: int = BACKOFF_BASE,
    jitter_min: float = JITTER_MIN,
    jitter_max: float = JITTER_MAX,
):
    """
    装饰器：为函数添加指数退避重试能力

    仅对包含 429 状态码的异常进行重试，其他异常直接抛出。

    Args:
        max_retries: 最大重试次数
        base: 退避基数
        jitter_min: 随机抖动最小值（秒）
        jitter_max: 随机抖动最大值（秒）

    Usage:
        @with_exponential_backoff(max_retries=5)
        def call_llm(prompt):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_str = str(e).lower()

                    # 判断是否为 429 错误
                    is_429 = (
                        "429" in str(e)
                        or "too many requests" in error_str
                        or "rate limit" in error_str
                        or "rate_limit" in error_str
                    )

                    if not is_429 or attempt >= max_retries:
                        if attempt >= max_retries and is_429:
                            logger.error(
                                "LLM API 调用失败：已达到最大重试次数 %d，最后错误：%s",
                                max_retries, e,
                            )
                        raise

                    # 计算等待时间：2^n + random(0, 2)
                    wait_time = pow(base, attempt) + random.uniform(jitter_min, jitter_max)

                    logger.warning(
                        "LLM API 返回 429（Too Many Requests），"
                        "第 %d/%d 次重试，等待 %.1f 秒后重试...",
                        attempt + 1, max_retries, wait_time,
                    )

                    time.sleep(wait_time)

            # 理论上不会执行到这里
            if last_exception:
                raise last_exception
            raise RuntimeError("重试逻辑异常：未捕获到异常但也没有返回结果")

        return wrapper

    return decorator


def retry_with_backoff(
    func: Callable[..., T],
    *args,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base: int = BACKOFF_BASE,
    jitter_min: float = JITTER_MIN,
    jitter_max: float = JITTER_MAX,
    **kwargs,
) -> T:
    """
    函数式调用：对指定函数进行指数退避重试

    Args:
        func: 要重试的函数
        *args: 函数的位置参数
        max_retries: 最大重试次数
        base: 退避基数
        jitter_min: 随机抖动最小值（秒）
        jitter_max: 随机抖动最大值（秒）
        **kwargs: 函数的关键字参数

    Returns:
        函数执行结果

    Usage:
        result = retry_with_backoff(call_llm, prompt, max_retries=5)
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()

            is_429 = (
                "429" in str(e)
                or "too many requests" in error_str
                or "rate limit" in error_str
                or "rate_limit" in error_str
            )

            if not is_429 or attempt >= max_retries:
                if attempt >= max_retries and is_429:
                    logger.error(
                        "LLM API 调用失败：已达到最大重试次数 %d，最后错误：%s",
                        max_retries, e,
                    )
                raise

            wait_time = pow(base, attempt) + random.uniform(jitter_min, jitter_max)

            logger.warning(
                "LLM API 返回 429（Too Many Requests），"
                "第 %d/%d 次重试，等待 %.1f 秒后重试...",
                attempt + 1, max_retries, wait_time,
            )

            time.sleep(wait_time)

    if last_exception:
        raise last_exception
    raise RuntimeError("重试逻辑异常：未捕获到异常但也没有返回结果")