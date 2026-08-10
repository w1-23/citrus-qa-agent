"""API Gateway — v8.3.0 with Fast Guard + ContextManager + structured SSE.

FastGuard: rule-based short-circuit for trivial greetings (<1ms, no graph).
SSE event protocol: all events use canonical {event, data} format with
JSON-string data. Five structured event types: thinking, tool_call_start,
tool_executing, tool_result, text. Heartbeat + Tool progress keepalive.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from src.config import PROJECT_ROOT, settings, FeatureFlags
from src.guardrails.memory import memory_store
from src.session.manager import session_manager
from src.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


FAST_GUARD_PATTERNS: set[str] = {
    "你好", "您好", "hello", "hi", "hey",
    "在吗", "谢谢", "thanks", "thank you",
    "再见", "bye", "拜拜", "晚安", "早安",
    "你是谁", "你能做什么", "你会什么", "你有什么功能",
    "what can you do", "who are you",
}


def _is_fast_guard_hit(query: str) -> bool:
    cleaned = query.strip().lower()
    if len(cleaned) > 12:
        return False
    return cleaned in FAST_GUARD_PATTERNS


def _fast_guard_reply(query: str) -> str:
    return (
        "你好！我是柑橘科研助手。\n\n"
        "我可以帮助你:\n"
        "- 检索柑橘领域文献（基因、病害、代谢、栽培等）\n"
        "- 机制解析与对比分析\n"
        "- 实验设计与统计分析\n"
        "- 学术综述撰写与保存\n\n"
        "请问有什么柑橘科研相关的问题？"
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Lifespan] starting...")
    try:
        from src.retrieval.init import eager_load_rag
        eager_load_rag()
        logger.info("[Lifespan] RAG engine warmed up")
    except Exception as e:
        logger.error(f"[Lifespan] RAG warm-up failed: {e}")
    yield
    try:
        from src.retrieval.multi_retriever import MultiBatchRetriever
        MultiBatchRetriever().close()
    except Exception as e:
        logger.debug(f"[Lifespan] Qdrant close: {e}")
    logger.info("[Lifespan] shutdown complete")


app = FastAPI(title="Citrus QA Agent v8.3", version="8.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    query: str
    light_mode: Optional[bool] = None


@app.post("/api/v2/chat")
async def chat_v2(req: ChatRequest):
    query = req.query
    sid = req.session_id or str(uuid.uuid4())
    mode = "light" if req.light_mode else "expert"

    # Fast Guard: pure greeting -> direct reply, no graph, no RAG
    if _is_fast_guard_hit(query):
        reply = _fast_guard_reply(query)
        t0 = time.perf_counter()

        async def guard_events():
            from src.core.progress_bus import _encode_event, log_sse_frame
            elapsed = int((time.perf_counter() - t0) * 1000)
            evt = _encode_event("done", {
                "session_id": sid,
                "answer": reply,
                "gen_time_ms": elapsed,
            })
            log_sse_frame(evt)
            yield evt
            try:
                from langchain_core.messages import HumanMessage, AIMessage
                asyncio.create_task(
                    session_manager.save_messages(sid, [
                        HumanMessage(content=query),
                        AIMessage(content=reply),
                    ])
                )
            except Exception:
                pass

        return EventSourceResponse(guard_events())

    logger.info(f"[API v2] mode={mode} session={sid[:8]}... query={query[:60]}")

    from src.graph.graph import build_graph
    from src.graph.state import AgentState

    graph = build_graph(mode)
    initial_state: AgentState = {
        "query": query,
        "session_id": sid,
        "mode": mode,
        "messages": [],
        "answer": "",
    }
    t0 = time.perf_counter()

    async def event_generator():
        from src.core.progress_bus import (
            get_progress_queue, get_log_queue,
            reset_progress_queue, SSELogHandler,
            _encode_event, log_sse_frame,
            reset_tool_call_accumulator, get_tool_call_accumulator,
            get_running_tools, get_tool_elapsed, clear_tool_timers,
        )

        reset_progress_queue()
        reset_tool_call_accumulator()
        clear_tool_timers()
        progress_queue = get_progress_queue()
        log_queue = get_log_queue()
        event_queue: asyncio.Queue = asyncio.Queue()

        async def bridge_progress():
            while True:
                try:
                    evt = await asyncio.wait_for(progress_queue.get(), timeout=0.3)
                    await event_queue.put(evt)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception:
                    break

        async def bridge_logs():
            while True:
                try:
                    evt = await asyncio.wait_for(log_queue.get(), timeout=0.3)
                    await event_queue.put(evt)
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception:
                    break

        async def tool_heartbeat():
            """Send progress events for tools that have been running > 2s."""
            try:
                while True:
                    try:
                        await asyncio.sleep(2.0)
                    except asyncio.CancelledError:
                        break
                    running = get_running_tools()
                    for call_id, tool_name, elapsed in running:
                        if elapsed > 2.0:
                            evt = _encode_event("tool_executing", {
                                "message": f"{tool_name or '工具'} 执行中... ({elapsed:.0f}s)",
                                "tool_call_id": call_id,
                                "tool_name": tool_name,
                            })
                            await event_queue.put(evt)
            except asyncio.CancelledError:
                pass

        progress_task = asyncio.create_task(bridge_progress())
        log_task = asyncio.create_task(bridge_logs())
        heartbeat_task = asyncio.create_task(tool_heartbeat())

        async def process_graph():
            try:
                async for node_output in graph.astream(initial_state, stream_mode="updates"):
                    for node_name, output in node_output.items():
                        if "_trace" in output:
                            trace = output["_trace"]
                            if trace.get("node") in ("load_context", "expert_load"):
                                await event_queue.put(_encode_event("status", {
                                    "stage": "loading",
                                    "summary": trace.get("summary", ""),
                                }))

                        if node_name in ("load_context", "expert_load", "save_context", "expert_save"):
                            continue

                        if node_name in ("light_retrieve",):
                            await event_queue.put(_encode_event("status", {
                                "stage": "retrieval_done",
                                "main_count": len(output.get("main_results", [])),
                            }))

                        if node_name in ("supervisor", "light_synthesize", "light_react"):
                            ans = output.get("answer", "")
                            if ans:
                                await asyncio.sleep(0.5)
                                await event_queue.put(_encode_event("done", {
                                    "session_id": sid,
                                    "answer": ans,
                                    "gen_time_ms": int((time.perf_counter() - t0) * 1000),
                                }))
                        elif node_name not in ("light_retrieve",):
                            ans = output.get("answer", "")
                            if ans:
                                await event_queue.put(_encode_event("done", {
                                    "session_id": sid,
                                    "answer": ans,
                                    "gen_time_ms": int((time.perf_counter() - t0) * 1000),
                                }))

                        ref_data = output.get("references_data")
                        if ref_data:
                            await event_queue.put(_encode_event("citations", ref_data))
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"[SSE v2] graph processing error: {e}")
                try:
                    await event_queue.put(_encode_event("error", {"message": str(e)}))
                except Exception:
                    pass
            finally:
                # Drain pending bridge/log events before sending sentinel
                for _ in range(10):
                    if progress_queue.empty() and log_queue.empty():
                        break
                    await asyncio.sleep(0.1)
                heartbeat_task.cancel()
                await event_queue.put(None)

        graph_task = asyncio.create_task(process_graph())

        try:
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    hb = _encode_event("heartbeat", {})
                    log_sse_frame(hb)
                    yield hb
                    continue

                if event is None:
                    break
                log_sse_frame(event)
                yield event
        except asyncio.CancelledError:
            logger.warning(f"[SSE v2] Client disconnected (session={sid[:8]})")
            graph_task.cancel()
            try:
                await graph_task
            except Exception:
                pass
            raise
        finally:
            if not graph_task.done():
                graph_task.cancel()
            if not progress_task.done():
                progress_task.cancel()
            if not log_task.done():
                log_task.cancel()
            if not heartbeat_task.done():
                heartbeat_task.cancel()

    return EventSourceResponse(event_generator())


@app.get("/api/v1/flags")
async def get_flags():
    return {"flags": FeatureFlags.all_flags()}


@app.post("/api/v1/flags/toggle")
async def toggle_flag(flag_name: str, value: bool):
    if flag_name not in FeatureFlags.all_flags():
        raise HTTPException(status_code=404, detail=f"Flag '{flag_name}' not found")
    FeatureFlags.set_flag(flag_name, value)
    return {"status": "ok", "flag": flag_name, "value": value}


@app.get("/api/v1/memory/{session_id}")
async def get_memory(session_id: str):
    return {"status": "ok", "session_id": session_id[:8]}


@app.get("/api/v1/hitl/pending")
async def hitl_pending():
    """Stub: HITL disabled in v8.1.1. Always returns empty."""
    return {"pending": []}


@app.post("/api/v1/hitl/resolve")
async def hitl_resolve():
    """Stub: HITL disabled in v8.1.1."""
    return {"status": "ok"}


@app.get("/")
async def serve_frontend():
    html_path = PROJECT_ROOT / "index.html"
    if not html_path.exists():
        return {"error": f"index.html not found at {html_path}"}
    return FileResponse(html_path)


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "8.1.1"}


@app.post("/api/v1/session/{session_id}/clear")
async def clear_session_endpoint(session_id: str):
    try:
        await session_manager.clear_session(session_id)
        memory_store.clear_session(session_id)
        logger.info(f"[API] session cleared: {session_id[:8]}...")
        return {"status": "ok", "message": f"Session {session_id[:8]} cleared"}
    except Exception as e:
        logger.error(f"[API] clear session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/session/new")
async def new_session_endpoint():
    new_id = str(uuid.uuid4())
    await session_manager.get_or_create_session(new_id)
    logger.info(f"[API] new session: {new_id}")
    return {"status": "ok", "session_id": new_id}
