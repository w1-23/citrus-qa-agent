# -*- coding: utf-8 -*-
"""draft_store — 草稿先行跨任务暂存仓（v8.16.1）。

草稿 worker（expert/light load 节点后台启动的 asyncio task）产出：
  - DRAFT_ZH  → 立即 emit draft 事件（前端 3-5s 上屏，不经此仓）
  - DRAFT_EN / MULTI_QUERY / SUMMARY  → search_multi 多路检索结果
    暂存于此，供 retrieve-agent 的 run_agent 在组装确定性证据回执前
    pop 并入 collected_artifacts["main_results"]（多路检索证据进 [n] 清单）。

与 contextvar 的区别：worker 是 create_task 派生协程，set 的 contextvar 只在该
任务上下文可见，父任务读不到 → 用模块级仓 + (session_id, request_id) 键解决。
键含 request_id 防止同会话并发请求串仓（旧请求被 abort 后其条目自然过期）。
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class DraftStore:
    """线程安全、TTL 过期的请求级草稿仓。

    put 由草稿 worker 调用；pop 由 run_agent（retrieve-agent 回执组装前）调用。
    TTL 过期条目在 put/pop 时惰性清理，防止无 retrieve-agent 场景泄漏。
    """

    def __init__(self, max_entries: int = 64, ttl_sec: int = 180) -> None:
        self._max_entries = max_entries
        self._ttl_sec = ttl_sec
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}  # key -> {"ts": float, "payload": dict}

    @staticmethod
    def _key(session_id: str, request_id: str = "") -> str:
        return f"{session_id or ''}|{request_id or ''}"

    def put(self, session_id: str, request_id: str, payload: dict) -> None:
        if not payload:
            return
        key = self._key(session_id, request_id)
        now = time.time()
        with self._lock:
            self._evict_locked(now)
            self._entries[key] = {"ts": now, "payload": payload}
            # 超限时淘汰最旧条目（防异常路径无限增长）
            if len(self._entries) > self._max_entries:
                oldest = min(self._entries.items(), key=lambda kv: kv[1]["ts"])
                self._entries.pop(oldest[0], None)
        logger.debug(f"[draft_store] put key={key[:40]}...")

    def pop(self, session_id: str, request_id: str = "") -> dict | None:
        key = self._key(session_id, request_id)
        now = time.time()
        with self._lock:
            entry = self._entries.pop(key, None)
        if entry is None:
            return None
        if now - entry["ts"] > self._ttl_sec:
            logger.debug(f"[draft_store] pop expired key={key[:40]}...")
            return None
        return entry["payload"]

    def _evict_locked(self, now: float) -> None:
        expired = [k for k, e in self._entries.items()
                   if now - e["ts"] > self._ttl_sec]
        for k in expired:
            self._entries.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


# 进程级单例（与 MultiBatchRetriever 同为进程级共享对象）
draft_store = DraftStore()