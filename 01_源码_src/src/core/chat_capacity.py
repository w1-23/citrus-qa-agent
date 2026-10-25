"""Single-worker FIFO capacity gate for chat requests."""
from __future__ import annotations

import asyncio
import os


class ChatCapacity:
    def __init__(self) -> None:
        self.max_active = max(1, int(os.getenv("CITRUSQA_MAX_ACTIVE_CHATS", "10")))
        self.max_pending = max(0, int(os.getenv("CITRUSQA_MAX_PENDING_CHATS", "100")))
        self.wait_timeout = max(1, int(os.getenv("CITRUSQA_QUEUE_WAIT_SECONDS", "900")))
        self._semaphore = asyncio.Semaphore(self.max_active)
        self._lock = asyncio.Lock()
        self._queued = 0
        self._active = 0

    async def acquire(self, on_wait=None) -> None:
        counted = False
        async with self._lock:
            if self._semaphore.locked():
                if self._queued >= self.max_pending:
                    raise OverflowError("当前排队人数已满，请稍后重试。")
                self._queued += 1
                counted = True
                position = self._queued
            else:
                position = 0
        try:
            if counted and on_wait:
                await on_wait(position, self.max_active, self.max_pending)
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self.wait_timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError("排队等待超过 15 分钟，请稍后重试。") from exc
        finally:
            if counted:
                async with self._lock:
                    self._queued = max(0, self._queued - 1)
        async with self._lock:
            self._active += 1

    async def release(self) -> None:
        async with self._lock:
            self._active = max(0, self._active - 1)
            self._semaphore.release()

    async def snapshot(self) -> dict[str, int]:
        async with self._lock:
            return {"active": self._active, "queued": self._queued,
                    "max_active": self.max_active, "max_pending": self.max_pending,
                    "wait_timeout_seconds": self.wait_timeout}


chat_capacity = ChatCapacity()
