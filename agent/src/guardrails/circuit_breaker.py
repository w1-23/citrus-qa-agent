"""
Circuit Breaker — 断路器模式
防止级联失败，当操作连续失败 N 次后自动断开，冷却后自动半开重试。
"""
import time
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """断路器 — 线程安全，支持逐实例隔离"""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        half_open_max_retries: int = 1,
    ):
        self._name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_retries = half_open_max_retries

        self._lock = threading.Lock()
        self._state: str = "CLOSED"
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._half_open_retries: int = 0

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def _transition(self, new_state: str):
        old = self._state
        self._state = new_state
        logger.info(
            f"[CircuitBreaker:{self._name}] {old} → {new_state} "
            f"(failures={self._failure_count})"
        )

    def call(self, fn, *args, **kwargs):
        """执行受断路器保护的操作"""
        with self._lock:
            if self._state == "OPEN":
                if time.time() - self._last_failure_time >= self._recovery_timeout:
                    self._transition("HALF_OPEN")
                    self._half_open_retries = 0
                else:
                    raise CircuitBreakerOpenError(
                        f"[CircuitBreaker:{self._name}] 断路器已断开，"
                        f"冷却剩余 {self._recovery_timeout - (time.time() - self._last_failure_time):.0f}s"
                    )

            if self._state == "HALF_OPEN" and self._half_open_retries >= self._half_open_max_retries:
                self._transition("OPEN")
                self._last_failure_time = time.time()
                raise CircuitBreakerOpenError(
                    f"[CircuitBreaker:{self._name}] 半开重试已达上限"
                )

        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.time()
                if self._failure_count >= self._failure_threshold:
                    self._transition("OPEN")
                elif self._state == "HALF_OPEN":
                    self._half_open_retries += 1
            raise e

        with self._lock:
            if self._state == "HALF_OPEN":
                logger.info(f"[CircuitBreaker:{self._name}] 半开重试成功，恢复关闭")
            self._state = "CLOSED"
            self._failure_count = 0
            self._half_open_retries = 0

        return result

    async def acall(self, fn, *args, **kwargs):
        """异步执行受断路器保护的操作"""
        with self._lock:
            if self._state == "OPEN":
                if time.time() - self._last_failure_time >= self._recovery_timeout:
                    self._transition("HALF_OPEN")
                    self._half_open_retries = 0
                else:
                    raise CircuitBreakerOpenError(
                        f"[CircuitBreaker:{self._name}] 断路器已断开，"
                        f"冷却剩余 {self._recovery_timeout - (time.time() - self._last_failure_time):.0f}s"
                    )

            if self._state == "HALF_OPEN" and self._half_open_retries >= self._half_open_max_retries:
                self._transition("OPEN")
                self._last_failure_time = time.time()
                raise CircuitBreakerOpenError(
                    f"[CircuitBreaker:{self._name}] 半开重试已达上限"
                )

        try:
            import asyncio
            if asyncio.iscoroutinefunction(fn):
                result = await fn(*args, **kwargs)
            else:
                result = await asyncio.to_thread(fn, *args, **kwargs)
        except Exception as e:
            with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.time()
                if self._failure_count >= self._failure_threshold:
                    self._transition("OPEN")
                elif self._state == "HALF_OPEN":
                    self._half_open_retries += 1
            raise e

        with self._lock:
            if self._state == "HALF_OPEN":
                logger.info(f"[CircuitBreaker:{self._name}] 半开重试成功，恢复关闭")
            self._state = "CLOSED"
            self._failure_count = 0
            self._half_open_retries = 0

        return result

    def reset(self):
        with self._lock:
            self._state = "CLOSED"
            self._failure_count = 0
            self._half_open_retries = 0
            logger.info(f"[CircuitBreaker:{self._name}] 手动重置为 CLOSED")


class CircuitBreakerOpenError(Exception):
    pass
