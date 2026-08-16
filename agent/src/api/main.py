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
import re
import time
import uuid
from typing import Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT, settings
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
        from src.retrieval import eager_load_rag
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


# v8.4.5: 版本单源（settings.VERSION）——UI/健康检查/API 元数据共用
app = FastAPI(title=f"Citrus QA Agent v{settings.VERSION}", version=settings.VERSION, lifespan=lifespan)
# v8.3.3: 无 Cookie 鉴权场景下不允许 "*" + credentials 组合（浏览器规范拒绝），仅开 Origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# v8.4.11 用户中断（停止功能）：运行中 graph 任务注册表 job_id -> asyncio.Task。
# cancel 端点按会话找到 running job 后在此安全点取消（书中 §4.7.6 取消式处理：
# 不在任意时刻强行掐断，而是向任务发取消信号，在 LLM/tool await 处抛 CancelledError）。
_running_graph_tasks: dict = {}


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
            # v8.4.11: 注册运行任务（cancel 端点按 job_id 定位取消）
            _running_graph_tasks[job_id] = asyncio.current_task()
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
                # v8.4.11: 注销运行任务（正常/取消/异常路径统一清理）
                _running_graph_tasks.pop(job_id, None)
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


# ── v8.4.11 用户中断：停止当前任务（书中 §4.7.6 取消式处理）──
# 用户输入有误/改变主意时点"停止"：前端 abort SSE + 本端点取消该会话
# 所有 running job（普通问答与 write 断连保活任务都覆盖）。取消在安全点
# 生效（task.cancel() → LLM/tool await 处抛 CancelledError，不强行掐断
# 文件写入等临界操作）；停止后可直接修改问题重新发送（新请求 = 新 job）。

class CancelRequest(BaseModel):
    session_id: str


@app.post("/api/v2/chat/cancel")
async def cancel_chat(req: CancelRequest):
    from src.core import jobs as jobs_mod
    cancelled: list[str] = []
    try:
        jobs = jobs_mod.list_for_session(req.session_id, limit=20)
    except Exception as e:
        logger.warning(f"[API] cancel list jobs failed: {e}")
        jobs = []
    for j in jobs:
        if j.get("status") != "running":
            continue
        jid = j.get("job_id") or ""
        if not jid:
            continue
        t = _running_graph_tasks.get(jid)
        if t is not None and not t.done():
            t.cancel()
            cancelled.append(jid)
            # 状态即时置 cancelled（任务内部 CancelledError 分支幂等重复置）
            jobs_mod.update_job(jid, status="cancelled")
            logger.info(f"[API] cancel signal sent: job={jid[:12]} session={req.session_id[:8]}")
        else:
            # 任务不在本进程（如服务重启后残留）——仅修正状态一致性
            jobs_mod.update_job(jid, status="cancelled")
            cancelled.append(jid)
            logger.info(f"[API] stale running job marked cancelled: {jid[:12]}")
    if not cancelled:
        logger.info(f"[API] cancel: no running job for session={req.session_id[:8]}")
    return {"status": "ok", "cancelled": cancelled, "count": len(cancelled)}


# v8.4.14: 死端点已删——/api/v1/flags + /api/v1/flags/toggle（FeatureFlags 0 个开关）、
# /api/v1/memory/{id}（恒返回固定值 stub）、/api/v1/hitl/*（v8.4.4 已删 stub）。
# 权限确认由 /api/v2/permission/grant + 前端审批卡片承担。


# ── v8.6 用户反馈（书 §4.6 反馈循环 / O7 经验学习第一步）──
# 纯记录端点：👍/👎 落库 feedback 表（sessions.db），不进入对话历史、
# 不影响任何问答/检索/写作流程；数据供未来离线"经验沉淀"分析使用。

class FeedbackRequest(BaseModel):
    session_id: str = ""
    message_id: str = ""
    rating: int
    comment: str = ""


@app.post("/api/v2/feedback")
async def submit_feedback(req: FeedbackRequest):
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400,
                            detail="rating 必须为 1(👍) 或 -1(👎)")
    ok = await asyncio.to_thread(
        session_manager.record_feedback,
        req.session_id, req.message_id, req.rating, req.comment)
    if not ok:
        raise HTTPException(status_code=500, detail="反馈记录失败")
    logger.info(f"[API] feedback recorded: session={req.session_id[:8] or '-'} "
                f"rating={req.rating}")
    return {"status": "ok", "rating": req.rating}


# ── v8.4.3 结构化权限确认（前端审批卡片闭环）──

class GrantRequest(BaseModel):
    session_id: str = ""
    tool_name: str
    scope: str = "once"   # once | session | workspace


@app.post("/api/v2/permission/grant")
async def permission_grant(req: GrantRequest):
    """授权工具调用（once/session/workspace 范围）。授权不写入对话历史。

    v8.4.5: 授权后唤醒 ask 模式下挂起的工具执行（同一执行内继续，无需整轮重跑）。
    """
    if req.scope not in ("once", "session", "workspace"):
        raise HTTPException(status_code=400, detail=f"scope 必须为 once/session/workspace，got {req.scope}")
    ok = await asyncio.to_thread(
        session_manager.grant_permission,
        req.session_id, req.tool_name, req.scope)
    if not ok:
        raise HTTPException(status_code=500, detail="授权记录失败")
    try:
        from src.tools.registry import signal_permission_granted
        signal_permission_granted(req.session_id or "", req.tool_name)
    except Exception:
        pass
    return {"status": "ok", "tool_name": req.tool_name, "scope": req.scope}


# ── v8.4.8 HITL 服务端化：权限模式运行时切换（WebUI 徽标菜单）──

class ModeRequest(BaseModel):
    mode: str


@app.post("/api/v2/permission/mode")
async def permission_mode_switch(req: ModeRequest):
    """运行时切换权限模式（ask/auto_workspace/deny）。

    生效即时（settings 运行时更新）；重启后回落到 config.yaml 配置值。
    单用户部署下的 HITL 管理通道：WebUI 徽标菜单切换 + ask 模式审批卡片闭环。
    """
    if req.mode not in ("ask", "auto_workspace", "deny"):
        raise HTTPException(status_code=400,
                            detail=f"mode 必须为 ask/auto_workspace/deny，got {req.mode}")
    settings.PERMISSION_MODE = req.mode
    logger.info(f"[API] permission mode switched to {req.mode}")
    return {"status": "ok", "permission_mode": settings.PERMISSION_MODE}


# ── v8.4.3 运行时配置（前端"上下文概览"面板单一来源，删本地硬编码）──

@app.get("/api/v2/config")
async def runtime_config():
    return {
        "version": settings.VERSION,
        "context": {
            "max_tokens": settings.CONTEXT_BUDGET_MAX_TOKENS,
            "soft_threshold": settings.CONTEXT_BUDGET_SOFT_THRESHOLD,
            "hard_threshold": settings.CONTEXT_BUDGET_HARD_THRESHOLD,
        },
        "permission_mode": settings.PERMISSION_MODE,
        # v8.4.12: 审批卡片超时提示单一来源（config.yaml permission.wait_sec）
        "permission_wait_sec": settings.PERMISSION_WAIT_SEC,
        # v8.5.0 开源版: 只暴露"是否已配置"，永不回传 key 本身
        "has_api_key": bool(settings.RESOLVED_MAIN_API_KEY),
        "model": {
            "main": settings.MAIN_MODEL,
            "fast": settings.FAST_MODEL,
        },
    }


# ── v8.5.0 开源版：WebUI 引导填写 API Key ──
# 用户启动后在前端填写 DeepSeek API Key（唯一指定 deepseek-v4-flash）：
#  - 不写 .env（用户自己的 key 不混入项目配置）
#  - 持久化到 state/api_key（gitignore 内，不入仓库），跨重启保留
#  - 填写即校验（GET /models 连通性），失败提示更换

class ApiKeyRequest(BaseModel):
    api_key: str


@app.post("/api/v2/config/apikey")
async def set_api_key(req: ApiKeyRequest):
    key = (req.api_key or "").strip()
    if len(key) < 16 or not key.startswith("sk-"):
        raise HTTPException(status_code=400,
                            detail="Key 格式不正确（应为 DeepSeek 的 sk- 开头密钥）")
    # 连通性校验：GET {base}/models
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{settings.MAIN_BASE_URL.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {key}"},
            )
        if resp.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=f"Key 校验失败（HTTP {resp.status_code}）："
                       f"请检查密钥是否有效或已过期")
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"[API] apikey connectivity check failed: {e}")
        raise HTTPException(status_code=400, detail=f"无法连接 DeepSeek API: {e}")
    if not settings.save_runtime_api_key(key):
        raise HTTPException(status_code=500, detail="Key 保存失败")
    logger.info("[API] runtime api key configured via WebUI")
    return {"status": "ok", "has_api_key": True}


@app.get("/")
async def serve_frontend():
    html_path = PROJECT_ROOT / "index.html"
    if not html_path.exists():
        return {"error": f"index.html not found at {html_path}"}
    return FileResponse(html_path)


# v8.9 工作区静态服务（会话侧栏"工作区"文件可点击打开；只读）
_workspace_static = (PROJECT_ROOT / "workspace")
if _workspace_static.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/workspace", StaticFiles(directory=str(_workspace_static)), name="workspace")


@app.get("/health")
async def health_check():
    return {"status": "ok", "version": settings.VERSION}


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


# ── v8.9 会话管理（列表 / 重命名 / 软删除 / 工作区文件）──

@app.get("/api/v2/sessions")
async def list_sessions(limit: int = 100):
    """会话列表（排除软删除，按最近更新倒序；标题为空显示"新会话"）。"""
    sessions = await asyncio.to_thread(session_manager.list_sessions, limit)
    return {"sessions": sessions, "count": len(sessions)}


class RenameRequest(BaseModel):
    title: str = ""


@app.post("/api/v2/session/{session_id}/rename")
async def rename_session(session_id: str, req: RenameRequest):
    title = (req.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title 不能为空")
    ok = await asyncio.to_thread(session_manager.rename_session, session_id, title)
    if not ok:
        raise HTTPException(status_code=500, detail="重命名失败")
    return {"status": "ok", "session_id": session_id, "title": title[:60]}


@app.delete("/api/v2/session/{session_id}")
async def delete_session(session_id: str):
    """软删除：deleted_at 标记，从列表隐藏；消息/证据/记忆全部保留（可追溯）。"""
    ok = await asyncio.to_thread(session_manager.soft_delete_session, session_id)
    if not ok:
        raise HTTPException(status_code=500, detail="删除失败")
    logger.info(f"[API] session soft-deleted: {session_id[:8]}")
    return {"status": "ok", "session_id": session_id}


@app.get("/api/v2/session/{session_id}/workspace-files")
async def session_workspace_files(session_id: str):
    """该会话的写作成果文件（workspace/output/）。

    来源两层（v8.10i）：
    1) pipeline_tasks 完成记录（plan_execute 路径落库）；
    2) 兜底扫描 workspace/output/ 最近修改的 md 文件——直接写作/ReAct 等
       非 plan_execute 路径只写文件不落 pipeline_tasks，此前工作区查不到；
       现按修改时间展示最近 20 个（跨会话，部署到用户本地同样适用）。
    """
    import sqlite3
    from pathlib import Path
    files: list[dict] = []
    try:
        with sqlite3.connect(str(PROJECT_ROOT / "state" / "sessions.db")) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT output_path, MAX(updated_at) AS updated_at "
                "FROM pipeline_tasks WHERE session_id=? AND status IN ('done','partial') "
                "GROUP BY output_path ORDER BY updated_at DESC LIMIT 50",
                (session_id,)).fetchall()
        out_root = (PROJECT_ROOT / "workspace" / "output").resolve()
        seen = set()
        for r in rows:
            rel = (r["output_path"] or "").strip()
            if not rel or rel in seen:
                continue
            seen.add(rel)
            p = (out_root / rel).resolve()
            # 路径安全：必须落在 workspace/output 内
            if not str(p).startswith(str(out_root)):
                continue
            if not p.exists() or not p.is_file():
                continue
            st = p.stat()
            files.append({
                "path": rel,
                "name": p.name,
                "size": st.st_size,
                "modified": st.st_mtime,
                "modified_at": r["updated_at"] or "",
                "source": "task",
            })
        # v8.10i 兜底：最近修改的 md（直接写作路径的文件也有记录可点开；排除草稿）
        try:
            recent = sorted(
                (p for p in out_root.glob("*.md")
                 if p.is_file() and not p.name.endswith(".draft.md")),
                key=lambda p: p.stat().st_mtime, reverse=True)[:20]
            for p in recent:
                rel = str(p.relative_to(out_root))
                if rel in seen:
                    continue
                seen.add(rel)
                st = p.stat()
                files.append({
                    "path": rel,
                    "name": p.name,
                    "size": st.st_size,
                    "modified": st.st_mtime,
                    "modified_at": "",
                    "source": "recent",
                })
        except Exception as e:
            logger.debug(f"[API] workspace-files recent scan skipped: {e}")
    except Exception as e:
        logger.warning(f"[API] workspace-files failed: {e}")
    return {"files": files, "count": len(files)}


@app.get("/api/v2/session/{session_id}/citations")
async def session_citations(session_id: str):
    """v8.10k: 会话历史文献引用恢复（来自 session_evidence 每轮落库）。

    前端内存 roundHistory 在刷新/切换会话后丢失——文献引用栏为空；
    本端点按轮次返回引用，前端恢复渲染（同一会话内跨轮累加）。
    """
    import sqlite3
    rounds: list[dict] = []
    try:
        with sqlite3.connect(str(PROJECT_ROOT / "state" / "sessions.db")) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT turn_seq, query, evidence_json FROM session_evidence "
                "WHERE session_id=? ORDER BY turn_seq ASC", (session_id,)).fetchall()
        for r in rows:
            items: list[dict] = []
            try:
                import json as _json
                raw = _json.loads(r["evidence_json"] or "[]")
            except Exception:
                raw = []
            for i, e in enumerate(raw):
                if not isinstance(e, dict):
                    continue
                doi = str(e.get("doi") or "N/A")
                items.append({
                    "ref_id": f"R{r['turn_seq']}-{i + 1}",
                    "type": "main",
                    "doi": doi,
                    "title": str(e.get("title") or "")[:200],
                    "year": str(e.get("year") or ""),
                    "score": float(e.get("score") or 0),
                    "chunk_id": str(e.get("chunk_id") or ""),
                    "text_preview": str(e.get("snippet") or "")[:250],
                })
            if items:
                rounds.append({
                    "round_id": f"hist-{r['turn_seq']}",
                    "query": str(r["query"] or ""),
                    "cited": items,
                    "uncited": [],
                    "historical": [],
                })
    except Exception as e:
        logger.warning(f"[API] session citations failed: {e}")
    return {"session_id": session_id, "rounds": rounds, "count": len(rounds)}


# ── v8.4.9 会话持久化：历史对话读取（前端刷新/关闭重开后恢复渲染）──
# 数据本就持久化在 sessions.db（save_messages 全量入库），此前只缺读取通道；
# 本端点只回用户可见的 Human/AI 轮次，工具/系统消息留在库内不进聊天区。
# v8.7: user 消息为完整上下文 HumanMessage（含记忆召回/格式指南/策略卡片等
# 内部块）——显示层裁剪为 <user_query> 原文，内部上下文不显示到前端
# （存储与模型注入不变，仍走 INV-10 存储全量）。

def _user_display_text(content: str) -> str:
    """用户可见文本：提取 <user_query> 块；无标签（旧数据/直通消息）回退原文。"""
    m = re.search(r"<user_query>\s*([\s\S]*?)\s*</user_query>", content)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return content

@app.get("/api/v2/session/{session_id}/messages")
async def session_messages(session_id: str, limit: int = 200):
    """读取会话历史对话轮 + 上下文快照，供前端刷新/重开恢复渲染。

    返回顺序与库内一致（id 升序）；limit 为对话消息条数上限（默认 200 → 100 轮）。
    v8.4.10: 响应附加 context 快照（与请求期 context_status 事件同构）——
    刷新后上下文概览面板无需等下一次提问即可恢复显示。
    v8.7: user 消息只返回 <user_query> 原文（内部上下文块不显示）。
    """
    try:
        msgs, _ = await session_manager.get_messages_with_ids(session_id)
    except Exception as e:
        logger.warning(f"[API] session_messages read failed: {e}")
        return {"session_id": session_id, "messages": [], "count": 0,
                "context": _build_context_snapshot([])}
    out: list[dict] = []
    from langchain_core.messages import HumanMessage, AIMessage
    for m in msgs:
        if isinstance(m, HumanMessage):
            content = str(getattr(m, "content", "") or "").strip()
            if content:
                out.append({"role": "user", "content": _user_display_text(content)})
        elif isinstance(m, AIMessage):
            content = str(getattr(m, "content", "") or "").strip()
            # 纯工具轮（只有 tool_calls 无文本）不渲染；合成收尾指令已由读路径过滤
            if content:
                out.append({"role": "assistant", "content": content})
        if len(out) >= max(int(limit), 1):
            break
    return {"session_id": session_id, "messages": out, "count": len(out),
            "context": _build_context_snapshot(msgs, session_id)}


def _build_context_snapshot(msgs: list, session_id: str = "") -> dict:
    """上下文快照（v8.4.10）：复用请求期 context_status 的估算口径，字段同构。

    刷新后前端直接 renderContextPanel(snapshot)，无需等下一次提问的
    context_status 事件；压缩状态取会话 checkpoint（msg_id>0 表示已压缩）。
    """
    try:
        from src.core.context_budget import ContextBudget
        est = ContextBudget().estimate_tokens(msgs)
    except Exception:
        est = 0
    try:
        hist_chars = sum(len(getattr(m, "content", "") or "") for m in msgs)
    except Exception:
        hist_chars = 0
    compressed, summary = False, ""
    if session_id:
        try:
            ck = session_manager.get_checkpoint(session_id)
            if ck:
                compressed = bool(ck.get("msg_id"))
                summary = str(ck.get("summary") or "")[:200]
        except Exception:
            compressed = False
    return {
        "history_msgs": len(msgs),
        "history_chars": hist_chars,
        "estimated_tokens": est,
        "max_tokens": settings.CONTEXT_BUDGET_MAX_TOKENS,
        "soft_threshold": settings.CONTEXT_BUDGET_SOFT_THRESHOLD,
        "hard_threshold": settings.CONTEXT_BUDGET_HARD_THRESHOLD,
        "compressed": compressed,
        "compression_len": 0,
        "ltm_recalled": False, "ltm_chars": 0,
        "resident_cards": False, "suggestions": [], "format_hint": "",
    }


# ── v8.10 上下文细分（管理面板逐段查看用）──

@app.get("/api/v2/session/{session_id}/context-detail")
async def context_detail_endpoint(session_id: str, mode: str = "expert"):
    """上下文细分：system 提示词 / 历史消息（区分 user·ai·tool·system）/ 内部记忆块，
    均含原文，供前端"上下文管理"面板逐段展开查看。

    user 消息展示裁剪版（<user_query> 原文，内部块不混入对话文本）；
    内部块（LTM 记忆/用户偏好/格式指南等）单独提取为 memory 段展示。
    """
    try:
        from src.prompts.loader import assemble_system_prompt
        from langchain_core.messages import (
            HumanMessage, AIMessage, ToolMessage, SystemMessage)

        segments: list[dict] = []

        # 1) system 段：静态前缀全文
        sys_prompt = assemble_system_prompt(mode=mode, format_hint=None, query=None)
        segments.append({
            "role": "system", "source": "系统提示词（静态前缀）",
            "chars": len(sys_prompt), "content": sys_prompt,
        })

        # 2) 历史消息段（区分 user/ai/tool/system）
        msgs, _ = await session_manager.get_messages_with_ids(session_id)
        internal: dict[str, list[str]] = {}
        seq = 0
        for m in msgs:
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "ai"
            elif isinstance(m, ToolMessage):
                role = "tool"
            elif isinstance(m, SystemMessage):
                role = "system"
            else:
                role = "other"
            content = str(getattr(m, "content", "") or "")
            if role == "user":
                # 提取内部块 → memory 段；剩余显示 <user_query> 原文
                for tag in ("long_term_memory", "user_preferences", "format_guide",
                            "output_guide", "skill_cards", "strategy_cards",
                            "search_suggestions", "evidence", "task_plan",
                            "resident_cards"):
                    pat = re.compile(rf"<{tag}>([\s\S]*?)</{tag}>", re.IGNORECASE)
                    for mm in pat.finditer(content):
                        v = mm.group(1).strip()
                        if v:
                            internal.setdefault(tag, []).append(v)
                content = _user_display_text(content)
            if not content.strip():
                continue
            seq += 1
            segments.append({
                "role": role,
                "source": f"对话历史 #{seq}",
                "chars": len(content),
                "content": content,
            })

        # 3) 内部记忆块段（同类合并）
        for tag, blocks in internal.items():
            text = "\n\n".join(blocks)
            if not text:
                continue
            segments.append({
                "role": "memory",
                "source": f"记忆/提示块 · {tag}",
                "chars": len(text),
                "content": text,
            })

        return {"session_id": session_id, "segments": segments, "count": len(segments)}
    except Exception as e:
        logger.error(f"[API] context-detail failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
