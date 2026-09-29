"""Background task governance — fire-and-forget 任务治理（v8.3.7 M1）。

禁止裸 create_task：所有后台协程经 spawn() 启动，
- 持有强引用（防 GC 静默丢任务）
- done 回调记录异常（防静默失败）
- drain() 供 shutdown 时等待关键写入落库
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_tasks: set = set()


def _on_done(t: asyncio.Task) -> None:
    _tasks.discard(t)
    if not t.cancelled():
        try:
            exc = t.exception()
            if exc is not None:
                logger.warning(f"[Background] task failed: {exc}")
        except (asyncio.CancelledError, Exception):
            pass


def spawn(coro) -> None:
    """启动后台任务：持有引用 + 异常日志。"""
    t = asyncio.create_task(coro)
    _tasks.add(t)
    t.add_done_callback(_on_done)


def adopt(t: asyncio.Task) -> None:
    """接管已有 Task（SSE 断连保活：graph_task 转交后台持有引用，防 GC 取消）。"""
    _tasks.add(t)
    t.add_done_callback(_on_done)


def pending_count() -> int:
    return len(_tasks)


async def drain(timeout: float = 5.0) -> None:
    """等待在途后台任务完成（shutdown 时调用）。"""
    if not _tasks:
        return
    try:
        await asyncio.wait_for(
            asyncio.gather(*list(_tasks), return_exceptions=True),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"[Background] drain timeout, {len(_tasks)} tasks still pending")
