"""Single-worker FIFO capacity gate for chat requests."""
from __future__ import annotations

import asyncio
import os
from collections import deque


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return min(maximum, max(minimum, value))


class ChatCapacity:
    def __init__(self) -> None:
        self.max_active = _env_int("CITRUSQA_MAX_ACTIVE_CHATS", 10, 1, 128)
        self.max_pending = _env_int("CITRUSQA_MAX_PENDING_CHATS", 100, 0, 1000)
        self.max_queued_payload_bytes = _env_int(
            "CITRUSQA_MAX_QUEUED_ATTACHMENT_BYTES", 64 * 1024 * 1024, 0, 1024 * 1024 * 1024)
        self.wait_timeout = _env_int("CITRUSQA_QUEUE_WAIT_SECONDS", 900, 1, 24 * 60 * 60)
        self._condition = asyncio.Condition()
        self._queue: deque[object] = deque()
        self._queued_payload_bytes = 0
        self._active = 0

    async def acquire(self, on_wait=None, *, queued_payload_bytes: int = 0) -> None:
        ticket = object()
        queued = False
        promoted = False
        payload_bytes = max(0, int(queued_payload_bytes or 0))
        async with self._condition:
            # Reserve an active slot atomically. New work cannot jump ahead of
            # an existing waiter, and simultaneous arrivals cannot bypass the
            # bounded queue while the semaphore's first waiter is being set up.
            if not self._queue and self._active < self.max_active:
                self._active += 1
                return
            if len(self._queue) >= self.max_pending:
                raise OverflowError("当前排队人数已满，请稍后重试。")
            if payload_bytes and self._queued_payload_bytes + payload_bytes > self.max_queued_payload_bytes:
                raise OverflowError("等待队列的附件容量已满，请稍后重试；本轮附件尚未解码或处理。")
            self._queue.append(ticket)
            self._queued_payload_bytes += payload_bytes
            queued = True
            position = len(self._queue)

        try:
            if on_wait:
                await on_wait(position, self.max_active, self.max_pending)
            async with asyncio.timeout(self.wait_timeout):
                async with self._condition:
                    await self._condition.wait_for(
                        lambda: bool(self._queue)
                        and self._queue[0] is ticket
                        and self._active < self.max_active
                    )
                    self._queue.popleft()
                    self._queued_payload_bytes = max(0, self._queued_payload_bytes - payload_bytes)
                    self._active += 1
                    promoted = True
                    self._condition.notify_all()
        except TimeoutError as exc:
            seconds = self.wait_timeout
            duration = f"{seconds // 60} 分钟" if seconds % 60 == 0 else f"{seconds} 秒"
            raise TimeoutError(f"排队等待超过 {duration}，请稍后重试。") from exc
        finally:
            if queued and not promoted:
                cleanup = asyncio.create_task(self._remove_waiter(ticket, payload_bytes))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise

    async def _remove_waiter(self, ticket: object, payload_bytes: int) -> None:
        async with self._condition:
            try:
                self._queue.remove(ticket)
            except ValueError:
                return
            self._queued_payload_bytes = max(0, self._queued_payload_bytes - payload_bytes)
            self._condition.notify_all()

    async def release(self) -> None:
        async with self._condition:
            self._active = max(0, self._active - 1)
            self._condition.notify_all()

    async def snapshot(self) -> dict[str, int]:
        async with self._condition:
            return {"active": self._active, "queued": len(self._queue),
                    "queued_payload_bytes": self._queued_payload_bytes,
                    "max_active": self.max_active, "max_pending": self.max_pending,
                    "max_queued_payload_bytes": self.max_queued_payload_bytes,
                    "wait_timeout_seconds": self.wait_timeout}


chat_capacity = ChatCapacity()
