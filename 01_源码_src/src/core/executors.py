"""Bounded executors for synchronous work used by async request handlers."""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor


def _workers(name: str, default: int, lower: int, upper: int) -> int:
    try:
        return min(upper, max(lower, int(os.getenv(name, str(default)))))
    except (TypeError, ValueError):
        return default


TOOL_EXECUTOR = ThreadPoolExecutor(
    max_workers=_workers("CITRUSQA_TOOL_WORKERS", 12, 2, 64),
    thread_name_prefix="citrus-tool",
)
ATTACHMENT_EXECUTOR = ThreadPoolExecutor(
    max_workers=_workers("CITRUSQA_ATTACHMENT_WORKERS", 2, 1, 8),
    thread_name_prefix="citrus-attachment",
)
RAG_CHANNEL_EXECUTOR = ThreadPoolExecutor(
    max_workers=_workers("CITRUSQA_RAG_CHANNEL_WORKERS", 4, 2, 16),
    thread_name_prefix="citrus-rag-channel",
)
RAG_VECTOR_EXECUTOR = ThreadPoolExecutor(
    max_workers=_workers("CITRUSQA_RAG_VECTOR_WORKERS", 8, 2, 24),
    thread_name_prefix="citrus-rag-vector",
)
RAG_BM25_EXECUTOR = ThreadPoolExecutor(
    max_workers=_workers("CITRUSQA_RAG_BM25_WORKERS", 4, 2, 16),
    thread_name_prefix="citrus-rag-bm25",
)
RAG_MODEL_SEMAPHORE = threading.BoundedSemaphore(
    _workers("CITRUSQA_RAG_MODEL_CONCURRENCY", 2, 1, 8)
)
AUDIT_NETWORK_EXECUTOR = ThreadPoolExecutor(
    max_workers=_workers("CITRUSQA_AUDIT_NETWORK_WORKERS", 8, 2, 24),
    thread_name_prefix="citrus-audit-network",
)
WEB_FETCH_EXECUTOR = ThreadPoolExecutor(
    max_workers=_workers("CITRUSQA_WEB_FETCH_WORKERS", 12, 2, 32),
    thread_name_prefix="citrus-web-fetch",
)
