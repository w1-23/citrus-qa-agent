"""Bounds for synchronous tool and attachment worker pools."""
from __future__ import annotations

import asyncio
import threading

import pytest

from src.core.executors import BoundedThreadPoolExecutor, ExecutorSaturatedError


def test_executor_bounds_running_and_queued_work():
    executor = BoundedThreadPoolExecutor(1, 1, thread_name_prefix="test-bound")
    entered = threading.Event()
    release = threading.Event()
    try:
        first = executor.submit(lambda: (entered.set(), release.wait(5)))
        assert entered.wait(2)
        second = executor.submit(lambda: "queued")
        with pytest.raises(ExecutorSaturatedError):
            executor.submit(lambda: "overflow")
        release.set()
        first.result(timeout=2)
        assert second.result(timeout=2) == "queued"
        assert executor.submit(lambda: "recovered").result(timeout=2) == "recovered"
    finally:
        release.set()
        executor.shutdown(wait=True, cancel_futures=True)


def test_cancelled_await_does_not_free_running_thread_slot_early():
    async def scenario():
        executor = BoundedThreadPoolExecutor(1, 0, thread_name_prefix="test-cancel")
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def blocking_work():
            entered.set()
            release.wait(5)
            finished.set()

        try:
            future = asyncio.get_running_loop().run_in_executor(executor, blocking_work)
            assert await asyncio.to_thread(entered.wait, 2)
            future.cancel()
            await asyncio.sleep(0)
            with pytest.raises(ExecutorSaturatedError):
                executor.submit(lambda: None)
            release.set()
            assert await asyncio.to_thread(finished.wait, 2)
            assert executor.submit(lambda: "available").result(timeout=2) == "available"
        finally:
            release.set()
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)

    asyncio.run(scenario())
