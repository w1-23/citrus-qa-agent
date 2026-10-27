"""Local concurrency checks for the shared chat capacity gate."""
from __future__ import annotations

import asyncio
import sys

from src.core.chat_capacity import ChatCapacity


def _run(coro):
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(coro)


def test_ten_active_two_waiting_and_cancellation_recover_capacity():
    async def scenario():
        capacity = ChatCapacity()
        capacity.max_active = 10
        capacity.max_pending = 2
        capacity.wait_timeout = 5
        release_active = asyncio.Event()
        entered = 0
        current = 0
        peak = 0

        async def worker(wait_for_release: bool):
            nonlocal entered, current, peak
            await capacity.acquire()
            entered += 1
            current += 1
            peak = max(peak, current)
            try:
                if wait_for_release:
                    await release_active.wait()
            finally:
                current -= 1
                await capacity.release()

        active = [asyncio.create_task(worker(True)) for _ in range(10)]
        for _ in range(1000):
            if (await capacity.snapshot())["active"] == 10:
                break
            await asyncio.sleep(0.001)
        assert (await capacity.snapshot())["active"] == 10

        queued_a = asyncio.create_task(worker(False))
        queued_b = asyncio.create_task(worker(False))
        for _ in range(1000):
            if (await capacity.snapshot())["queued"] == 2:
                break
            await asyncio.sleep(0.001)
        assert (await capacity.snapshot())["queued"] == 2

        try:
            await capacity.acquire()
        except OverflowError:
            pass
        else:
            raise AssertionError("third waiting task should be rejected at queue capacity")

        queued_b.cancel()
        try:
            await queued_b
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled queue waiter should propagate cancellation")
        assert (await capacity.snapshot())["queued"] == 1

        release_active.set()
        await asyncio.gather(*active, queued_a)
        snapshot = await capacity.snapshot()
        assert snapshot["active"] == 0
        assert snapshot["queued"] == 0
        assert entered == 11
        assert peak <= 10

    _run(scenario())


def test_queued_payload_limit_and_cancel_cleanup():
    async def scenario():
        capacity = ChatCapacity()
        capacity.max_active = 1
        capacity.max_pending = 3
        capacity.max_queued_payload_bytes = 10
        capacity.wait_timeout = 5
        await capacity.acquire()
        queued = asyncio.create_task(capacity.acquire(queued_payload_bytes=8))
        for _ in range(1000):
            if (await capacity.snapshot())["queued"] == 1:
                break
            await asyncio.sleep(0.001)
        try:
            await capacity.acquire(queued_payload_bytes=3)
        except OverflowError as exc:
            assert "附件容量" in str(exc)
        else:
            raise AssertionError("queued payload over budget must be rejected")
        queued.cancel()
        try:
            await queued
        except asyncio.CancelledError:
            pass
        await capacity.release()
        snapshot = await capacity.snapshot()
        assert snapshot["active"] == snapshot["queued"] == snapshot["queued_payload_bytes"] == 0

    _run(scenario())


def test_queue_timeout_removes_ticket_and_payload_reservation():
    async def scenario():
        capacity = ChatCapacity()
        capacity.max_active = 1
        capacity.max_pending = 2
        capacity.max_queued_payload_bytes = 100
        capacity.wait_timeout = 0.02
        await capacity.acquire()

        try:
            await capacity.acquire(queued_payload_bytes=37)
        except TimeoutError as exc:
            assert "排队等待" in str(exc)
            assert "0.02 秒" in str(exc)
        else:
            raise AssertionError("a waiter that exceeds its deadline must time out")

        snapshot = await capacity.snapshot()
        assert snapshot["active"] == 1
        assert snapshot["queued"] == 0
        assert snapshot["queued_payload_bytes"] == 0
        await capacity.release()
        assert (await capacity.snapshot())["active"] == 0

    _run(scenario())


def test_capacity_configuration_is_bounded_and_invalid_values_fall_back(monkeypatch):
    monkeypatch.setenv("CITRUSQA_MAX_ACTIVE_CHATS", "999999")
    monkeypatch.setenv("CITRUSQA_MAX_PENDING_CHATS", "not-an-integer")
    monkeypatch.setenv("CITRUSQA_MAX_QUEUED_ATTACHMENT_BYTES", "-1")
    monkeypatch.setenv("CITRUSQA_QUEUE_WAIT_SECONDS", "999999999")
    capacity = ChatCapacity()
    assert capacity.max_active == 128
    assert capacity.max_pending == 100
    assert capacity.max_queued_payload_bytes == 0
    assert capacity.wait_timeout == 24 * 60 * 60


def test_simultaneous_arrivals_cannot_exceed_active_or_queue_limit():
    async def scenario():
        capacity = ChatCapacity()
        capacity.max_active = 2
        capacity.max_pending = 3
        capacity.wait_timeout = 5
        await asyncio.gather(*(capacity.acquire() for _ in range(2)))
        waiters = [asyncio.create_task(capacity.acquire()) for _ in range(8)]
        for _ in range(1000):
            if (await capacity.snapshot())["queued"] == 3:
                break
            await asyncio.sleep(0.001)
        assert (await capacity.snapshot())["queued"] == 3
        for task in waiters:
            if not task.done():
                task.cancel()
        results = await asyncio.gather(*waiters, return_exceptions=True)
        assert sum(isinstance(result, OverflowError) for result in results) == 5
        await capacity.release()
        await capacity.release()
        snapshot = await capacity.snapshot()
        assert snapshot["active"] == snapshot["queued"] == 0

    _run(scenario())
