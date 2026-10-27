"""Small bounded in-process cache for completed local RAG query results."""
from __future__ import annotations

import copy
import json
import threading
import time
from collections import OrderedDict


class QueryResultCache:
    def __init__(self, max_entries: int = 300, ttl_seconds: int = 86400,
                 max_bytes: int = 64 * 1024 * 1024, max_entry_bytes: int = 1024 * 1024):
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.max_bytes = max(1024, int(max_bytes))
        self.max_entry_bytes = max(1024, int(max_entry_bytes))
        self._lock = threading.RLock()
        self._items: OrderedDict[str, tuple[float, int, object]] = OrderedDict()
        self._bytes = 0

    def get(self, key: str):
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            created, size, value = item
            if now - created >= self.ttl_seconds:
                self._items.pop(key, None)
                self._bytes = max(0, self._bytes - size)
                return None
            self._items.move_to_end(key)
            return copy.deepcopy(value)

    def set(self, key: str, value) -> None:
        try:
            size = len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
        except Exception:
            return
        if size > self.max_entry_bytes or size > self.max_bytes:
            return
        with self._lock:
            old = self._items.pop(key, None)
            if old:
                self._bytes = max(0, self._bytes - old[1])
            self._items[key] = (time.monotonic(), size, copy.deepcopy(value))
            self._bytes += size
            while len(self._items) > self.max_entries or self._bytes > self.max_bytes:
                _, (_, removed_size, _) = self._items.popitem(last=False)
                self._bytes = max(0, self._bytes - removed_size)


def build_cache_key(query: str, families: list[str], categories: list[str], limit: int,
                    rag_identity: int, settings) -> str:
    payload = {
        "query": " ".join(str(query or "").casefold().split()),
        "families": sorted(set(families)), "categories": sorted(set(categories)),
        "limit": int(limit), "rag": int(rag_identity),
        "rrf_k": int(getattr(settings, "RRF_K", 60) or 60),
        "bm25_weight": float(getattr(settings, "RRF_WEIGHT_BM25", 1.0) or 1.0),
        "dense_weight": float(getattr(settings, "RRF_WEIGHT_ORIG_DENSE", 1.0) or 1.0),
        "decay": float(getattr(settings, "LOCAL_HYBRID_ADJACENT_DECAY_RATIO", 0.6) or 0.6),
        "window": int(getattr(settings, "CANDIDATE_WINDOW", 20) or 20),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def get_query_cache(settings) -> QueryResultCache | None:
    if not bool(getattr(settings, "RAG_CACHE_ENABLED", False)):
        return None
    max_entries = min(max(int(getattr(settings, "RAG_CACHE_SIZE", 300) or 300), 1), 1000)
    ttl = min(max(int(getattr(settings, "RAG_CACHE_TTL_HOURS", 24) or 24), 1), 24 * 30)
    config = (max_entries, ttl * 3600)
    cache = getattr(get_query_cache, "_cache", None)
    if cache is None or getattr(get_query_cache, "_config", None) != config:
        cache = QueryResultCache(max_entries=max_entries, ttl_seconds=ttl * 3600)
        get_query_cache._cache = cache
        get_query_cache._config = config
    return cache
