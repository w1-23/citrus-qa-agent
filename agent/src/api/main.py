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
from pydantic import BaseModel, Field

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
    """v8.4.4: 去掉 12 字符长度门槛（"what can you do" 15 字符曾永不命中）——
    问候语归一化后精确/前缀匹配；带问候+科研问句的复合输入仍走完整链路。"""
    cleaned = " ".join(query.strip().lower().split())
    if cleaned in FAST_GUARD_PATTERNS:
        return True
    _CHITCHAT_TAIL = {"", "啊", "呀", "哈", "呢", "嘛", "!", "！", "~", "～", "！", "，", ","}
    for pat in ("你好", "您好", "hello", "hi", "hey", "在吗", "谢谢",
                "thanks", "再见", "拜拜", "晚安", "早安"):
        if cleaned == pat:
            return True
        if cleaned.startswith(pat):
            rest = cleaned[len(pat):].lstrip(" ，,。.")
            if rest in _CHITCHAT_TAIL:
                return True
    return False


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
    # v8.4.4: 清理工具大结果 offload 临时文件（workspace/tmp/，避免累积）
    try:
        from src.tools.registry import cleanup_offload_files
        n = cleanup_offload_files()
        if n:
            logger.info(f"[Lifespan] 清理 offload 临时文件 {n} 个")
    except Exception as e:
        logger.debug(f"[Lifespan] offload cleanup: {e}")
    # v8.3.7: 等待在途后台历史写入落库（防关服务丢历史）
    try:
        from src.core.background import drain
        await drain(timeout=5.0)
    except Exception as e:
        logger.debug(f"[Lifespan] background drain: {e}")
    logger.info("[Lifespan] shutdown complete")


app = FastAPI(title="Citrus QA Agent v8.3", version="8.3.3", lifespan=lifespan)
# v8.3.3: 无 Cookie 鉴权场景下不允许 "*" + credentials 组合（浏览器规范拒绝），仅开 Origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    # v8.3.7: 长度上限（防单条超长文本打爆预算/日志）+ 客户端幂等 ID（重试复用）
    query: str = Field(..., min_length=1, max_length=20000)
    client_request_id: Optional[str] = Field(default=None, max_length=64)
    light_mode: Optional[bool] = None


@app.post("/api/v2/chat")
async def chat_v2(req: ChatRequest):
    query = req.query
    sid = await session_manager.get_or_create_session(req.session_id)
    # v8.3.3: 请求级追踪 ID（日志串线）
    from src.core.tracing import new_request_id
    rid = new_request_id()
    # v8.4.3: 会话 ID 入 contextvar（工具执行/权限判定需要）
    from src.core.tracing import set_session_id
    set_session_id(sid)
    # v8.3.7 M1: 幂等键（客户端稳定 ID 优先，服务端 30s 桶兜底）——重发/重试不重复写历史
    from src.session.manager import compute_idempotency_key
    idem_key = compute_idempotency_key(sid, query, req.client_request_id or "")
    # 模式完全由客户端 light_mode 决定（用户手动切换，无服务端自动升级）
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
                from src.core.background import spawn
                spawn(session_manager.save_messages(
                    sid,
                    [HumanMessage(content=query), AIMessage(content=reply)],
                    idem_key,
                ))
            except Exception:
                pass

        return EventSourceResponse(guard_events())

    logger.info(f"[API v2] mode={mode} session={sid[:8]}... query={query[:60]} req={rid}")

    # v8.4.1: 业务日志（logs/business.log）——按 req= 可串出整条请求链路
    try:
        from src.core.business_logger import blog
        blog("request_start", session=sid[:8], mode=mode,
             query=query[:120], idem=idem_key[:16])
    except Exception:
        pass

    from src.graph.graph import build_graph
    from src.graph.state import AgentState

    graph = build_graph(mode)
    # v8.3.7 M2: 任务 job（write 类长任务断连保活，状态可查）
    from src.core import jobs as jobs_mod
    job_id = jobs_mod.create_job(sid, rid, "chat")
    from src.core.tracing import set_job_id
    set_job_id(job_id)
    initial_state: AgentState = {
        "query": query,
        "session_id": sid,
        "mode": mode,
        "messages": [],
        "answer": "",
        "idempotency_key": idem_key,
    }
    t0 = time.perf_counter()

    async def event_generator():
        from src.core.progress_bus import (
            get_progress_queue, get_log_queue,
            set_request_queue, clear_request_queue,
            _encode_event, log_sse_frame,
            get_running_tools, get_tool_elapsed, clear_tool_timers,
        )

        # v8.3.3: 每个请求独立队列（contextvars 绑定），并发会话互不串扰
        request_queue: asyncio.Queue = asyncio.Queue()
        set_request_queue(request_queue)
        clear_tool_timers()
        log_queue = get_log_queue()
        event_queue: asyncio.Queue = asyncio.Queue()

        async def bridge_progress():
            while True:
                try:
                    evt = await asyncio.wait_for(request_queue.get(), timeout=0.3)
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

        # v8.3.7 M2: 观察者标志——write 类任务断连保活后丢弃后续事件（防队列堆积）
        observer_connected = True
        keep_alive = False

        async def emit_to_client(evt) -> None:
            """仅在客户端在线时投递事件（保活后丢弃）。"""
            if observer_connected:
                await event_queue.put(evt)

        async def process_graph():
            try:
                async for node_output in graph.astream(initial_state, stream_mode="updates",
                                                       recursion_limit=settings.RECURSION_LIMIT):
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
                            await emit_to_client(_encode_event("status", {
                                "stage": "retrieval_done",
                                "main_count": len(output.get("main_results", [])),
                            }))

                        if node_name in ("supervisor", "light_synthesize", "light_react"):
                            ans = output.get("answer", "")
                            if ans:
                                await asyncio.sleep(0.5)
                                done_payload = {
                                    "session_id": sid,
                                    "answer": ans,
                                    "job_id": job_id,
                                    "gen_time_ms": int((time.perf_counter() - t0) * 1000),
                                }
                                # v8.3.7 M3: 假完成检测元数据（前端可提示"引用未检索支撑"）
                                cit = output.get("citation_info")
                                if cit:
                                    done_payload.update(cit)
                                if output.get("tools_called"):
                                    done_payload["tools_called"] = output["tools_called"]
                                await emit_to_client(_encode_event("done", done_payload))
                                # v8.3.7 M2: 任务完成落状态
                                jobs_mod.update_job(job_id, status="completed",
                                                    progress_summary=ans[:200])
                                # v8.4.1: 业务日志
                                try:
                                    from src.core.business_logger import blog
                                    blog("request_done", answer_chars=len(ans),
                                         tools=output.get("tools_called", 0),
                                         ms=int((time.perf_counter() - t0) * 1000),
                                         job=job_id[:12])
                                except Exception:
                                    pass
                        elif node_name not in ("light_retrieve",):
                            ans = output.get("answer", "")
                            if ans:
                                await emit_to_client(_encode_event("done", {
                                    "session_id": sid,
                                    "answer": ans,
                                    "job_id": job_id,
                                    "gen_time_ms": int((time.perf_counter() - t0) * 1000),
                                }))

                        ref_data = output.get("references_data")
                        if ref_data:
                            await emit_to_client(_encode_event("citations", ref_data))
            except asyncio.CancelledError:
                # 仅普通问答被 cancel（write 保活任务不会到这里）
                jobs_mod.update_job(job_id, status="cancelled")
                pass
            except Exception as e:
                logger.error(f"[SSE v2] graph processing error: {e}")
                jobs_mod.update_job(job_id, status="failed", error=str(e)[:500])
                try:
                    from src.core.business_logger import blog
                    blog("request_error", err=str(e)[:200])
                except Exception:
                    pass
                try:
                    # v8.3.3: 错误脱敏——详细异常只进日志，SSE 只发用户可读信息
                    await emit_to_client(_encode_event("error", {
                        "message": "处理请求时发生内部错误，请稍后重试",
                    }))
                except Exception:
                    pass
            finally:
                # Drain pending bridge/log events before sending sentinel
                for _ in range(10):
                    if request_queue.empty() and log_queue.empty():
                        break
                    await asyncio.sleep(0.1)
                heartbeat_task.cancel()
                if observer_connected:
                    await event_queue.put(None)
                else:
                    # 保活完成：标记任务结束（不推 sentinel，无人消费）
                    jobs_mod.update_job(job_id, status="completed",
                                        progress_summary="(断连保活完成)")
                    logger.info(f"[SSE v2] write job {job_id} 断连保活执行完成")

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
            # v8.3.7 M2: write 类任务断连保活——转交后台继续执行；普通问答仍取消
            if jobs_mod.is_write_job(job_id):
                from src.core.background import adopt
                observer_connected = False
                keep_alive = True
                adopt(graph_task)
                logger.info(f"[SSE v2] write job {job_id} 转入后台保活（断连不杀任务）")
            else:
                graph_task.cancel()
                try:
                    await graph_task
                except Exception:
                    pass
            raise
        finally:
            if not keep_alive:
                if not graph_task.done():
                    graph_task.cancel()
            if not progress_task.done():
                progress_task.cancel()
            if not log_task.done():
                log_task.cancel()
            if not heartbeat_task.done():
                heartbeat_task.cancel()
            clear_request_queue()

    return EventSourceResponse(event_generator())


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """v8.3.7 M2: 任务状态查询（断连保活后可查 running/completed/failed）。"""
    from src.core import jobs as jobs_mod
    job = jobs_mod.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.get("/sessions/{session_id}/jobs")
async def list_session_jobs(session_id: str):
    """v8.3.7 M2: 会话最近任务列表。"""
    from src.core import jobs as jobs_mod
    return {"jobs": jobs_mod.list_for_session(session_id)}


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


# v8.4.4: HITL 旧 stub 端点已删（/api/v1/hitl/pending|resolve 恒空）——
# 权限确认由 /api/v2/permission/grant + 前端审批卡片承担


# ── v8.4.3 结构化权限确认（前端审批卡片闭环）──

class GrantRequest(BaseModel):
    session_id: str = ""
    tool_name: str
    scope: str = "once"   # once | session | workspace


@app.post("/api/v2/permission/grant")
async def permission_grant(req: GrantRequest):
    """授权工具调用（once/session/workspace 范围）。授权不写入对话历史。"""
    if req.scope not in ("once", "session", "workspace"):
        raise HTTPException(status_code=400, detail=f"scope 必须为 once/session/workspace，got {req.scope}")
    ok = await asyncio.to_thread(
        session_manager.grant_permission,
        req.session_id, req.tool_name, req.scope)
    if not ok:
        raise HTTPException(status_code=500, detail="授权记录失败")
    return {"status": "ok", "tool_name": req.tool_name, "scope": req.scope}


# ── v8.4.3 运行时配置（前端"上下文概览"面板单一来源，删本地硬编码）──

@app.get("/api/v2/config")
async def runtime_config():
    return {
        "context": {
            "max_tokens": settings.CONTEXT_BUDGET_MAX_TOKENS,
            "soft_threshold": settings.CONTEXT_BUDGET_SOFT_THRESHOLD,
            "hard_threshold": settings.CONTEXT_BUDGET_HARD_THRESHOLD,
        },
        "permission_mode": settings.PERMISSION_MODE,
        "model": {
            "main": settings.MAIN_MODEL,
            "fast": settings.FAST_MODEL,
        },
    }


@app.get("/")
async def serve_frontend():
    html_path = PROJECT_ROOT / "index.html"
    if not html_path.exists():
        return {"error": f"index.html not found at {html_path}"}
    return FileResponse(html_path)


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "8.3.3"}


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
