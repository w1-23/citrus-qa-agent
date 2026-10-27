"""Bounded executors for synchronous work used by async request handlers."""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Any


def _workers(name: str, default: int, lower: int, upper: int) -> int:
    try:
        return min(upper, max(lower, int(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


class ExecutorSaturatedError(RuntimeError):
    """Raised when a bounded executor has no worker or pending slot available."""


class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    """Thread pool whose running plus queued jobs have a fixed upper bound.

    ``submit`` fails fast when full. ``submit_wait`` is for synchronous worker
    threads that fan out work to a separate pool; it applies backpressure there
    without blocking the asyncio event loop. A slot is released only when the
    underlying concurrent future finishes, including after an awaiting request
    times out or is cancelled.
    """

    def __init__(self, max_workers: int, max_pending: int, *, thread_name_prefix: str):
        self.max_pending = max(0, int(max_pending))
        self._submission_slots = threading.BoundedSemaphore(max(1, int(max_workers)) + self.max_pending)
        super().__init__(max_workers=max_workers, thread_name_prefix=thread_name_prefix)

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any):
        if not self._submission_slots.acquire(blocking=False):
            raise ExecutorSaturatedError(
                f"{self._max_workers} 个工作线程和 {self.max_pending} 个排队槽位均已占满")
        try:
            future = super().submit(fn, *args, **kwargs)
        except BaseException:
            self._submission_slots.release()
            raise
        future.add_done_callback(lambda _future: self._submission_slots.release())
        return future

    def submit_wait(self, fn: Callable[..., Any], /, *args: Any,
                    timeout: float | None = None, **kwargs: Any):
        """Submit from a worker thread, waiting for bounded capacity first."""
        if timeout is None:
            timeout = max(1, _workers("CITRUSQA_EXECUTOR_QUEUE_WAIT_SECONDS", 60, 1, 600))
        if not self._submission_slots.acquire(timeout=timeout):
            raise ExecutorSaturatedError(
                f"线程池持续繁忙，等待 {timeout if timeout is not None else '不限时'} 秒后仍无空位")
        try:
            future = super().submit(fn, *args, **kwargs)
        except BaseException:
            self._submission_slots.release()
            raise
        future.add_done_callback(lambda _future: self._submission_slots.release())
        return future


def _pending(name: str, default: int) -> int:
    try:
        return min(2048, max(0, int(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return max(0, int(default))


_tool_workers = _workers("CITRUSQA_TOOL_WORKERS", 12, 2, 64)
_attachment_workers = _workers("CITRUSQA_ATTACHMENT_WORKERS", 2, 1, 8)
_rag_channel_workers = _workers("CITRUSQA_RAG_CHANNEL_WORKERS", 4, 2, 16)
_rag_vector_workers = _workers("CITRUSQA_RAG_VECTOR_WORKERS", 8, 2, 24)
_rag_bm25_workers = _workers("CITRUSQA_RAG_BM25_WORKERS", 4, 2, 16)
_audit_network_workers = _workers("CITRUSQA_AUDIT_NETWORK_WORKERS", 8, 2, 24)
_web_fetch_workers = _workers("CITRUSQA_WEB_FETCH_WORKERS", 12, 2, 32)

TOOL_EXECUTOR = BoundedThreadPoolExecutor(
    _tool_workers, _pending("CITRUSQA_TOOL_PENDING", 32), thread_name_prefix="citrus-tool")
ATTACHMENT_EXECUTOR = BoundedThreadPoolExecutor(
    _attachment_workers, _pending("CITRUSQA_ATTACHMENT_PENDING", 8), thread_name_prefix="citrus-attachment")
RAG_CHANNEL_EXECUTOR = BoundedThreadPoolExecutor(
    _rag_channel_workers, _pending("CITRUSQA_RAG_CHANNEL_PENDING", 16), thread_name_prefix="citrus-rag-channel")
RAG_VECTOR_EXECUTOR = BoundedThreadPoolExecutor(
    _rag_vector_workers, _pending("CITRUSQA_RAG_VECTOR_PENDING", 256), thread_name_prefix="citrus-rag-vector")
RAG_BM25_EXECUTOR = BoundedThreadPoolExecutor(
    _rag_bm25_workers, _pending("CITRUSQA_RAG_BM25_PENDING", 128), thread_name_prefix="citrus-rag-bm25")
RAG_MODEL_SEMAPHORE = threading.BoundedSemaphore(
    _workers("CITRUSQA_RAG_MODEL_CONCURRENCY", 2, 1, 8)
)
AUDIT_NETWORK_EXECUTOR = BoundedThreadPoolExecutor(
    _audit_network_workers, _pending("CITRUSQA_AUDIT_NETWORK_PENDING", 32), thread_name_prefix="citrus-audit-network")
WEB_FETCH_EXECUTOR = BoundedThreadPoolExecutor(
    _web_fetch_workers, _pending("CITRUSQA_WEB_FETCH_PENDING", 48), thread_name_prefix="citrus-web-fetch")
