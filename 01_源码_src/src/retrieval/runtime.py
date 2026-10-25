"""Process-wide retriever instance used by the local hybrid-search tool."""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)
_RAG_INSTANCE = None
_RAG_LOCK = threading.Lock()


def get_rag():
    """Load the shared corpus/index/models once per service process."""
    global _RAG_INSTANCE
    if _RAG_INSTANCE is None:
        with _RAG_LOCK:
            if _RAG_INSTANCE is None:
                from src.retrieval.multi_retriever import MultiBatchRetriever
                _RAG_INSTANCE = MultiBatchRetriever()
                logger.info("Local retriever singleton initialized")
    return _RAG_INSTANCE
