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
    # v9.0: 启动时一次性拼接固定 system prompt（Supervisor/Retrieve/Write/Lite/…）
    # 之后每次请求复用同一字符串，不再拼接、不再动态选择格式模板（KV cache 最大化复用）
    try:
        from src.prompts.loader import ensure_fixed_prompts
        _prompts = ensure_fixed_prompts()
        logger.info("[Lifespan] fixed prompts warmed: "
                    + ", ".join(f"{k}={len(v)}chars" for k, v in _prompts.items()))
    except Exception as e:
        logger.error(f"[Lifespan] fixed prompts build failed: {e}")
    # v9.1.2（用户真机日志排查: Ruby 旧工具调用 vs 同 session 新路径矛盾）:
    # 启动时打印 supervisor 工具 schema 快照——进程内 schema 单例在 import 时求值、
    # 恒定不变（supervisor_tools → expert_graph._AGENT_TOOLS → bind_tools，无按会话
    # 重建路径）。真机重启后第一眼即可从本行日志判定进程加载的代码版本：
    #   含 call_search_both = 新代码；含 [LEGACY ... PRESENT] 标记 = 旧构建。
    try:
        from src.tools.supervisor_tools import get_supervisor_tool_names
        _sup_names = get_supervisor_tool_names()
        logger.info("[Lifespan] supervisor tools snapshot: "
                    + ", ".join(_sup_names)
                    + (" [LEGACY call_retrieve_agent PRESENT]" 
                       if "call_retrieve_agent" in _sup_names else ""))
    except Exception as e:
        logger.warning(f"[Lifespan] supervisor tools snapshot failed: {e}")
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
    # v8.15: 联网搜索开关（前端「联网搜索」按钮——仅后端主开关 web_search.enabled 开启后生效）
    web_search_enabled: Optional[bool] = None


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
    # v8.13: 结构化诊断事件（JSONL）
    try:
        from src.core.diag import diag
        diag("request_start", mode=mode, query_chars=len(query))
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
        # v8.15: 联网搜索由前端开关逐请求决定（config web_search.enabled 仅作部署默认值，
        # 不设启用门槛）——前端开则本次可联网，关则工具执行层短路为 [DISABLED]
        # v9.2: 开关只经 tracing contextvar（下行 set_web_search_enabled）传递，
        # state 键为写而不读死字段已于重构中删除
    }
    from src.core.tracing import set_web_search_enabled, set_original_query, reset_web_budget
    set_web_search_enabled(bool(req.web_search_enabled))
    # v9.1（用户决策：每个用户请求最多一次联网）: 请求级联网预算重置——
    # deepseek_web_search 入口消费，超预算返回 [WEB_BUDGET_EXHAUSTED]。
    reset_web_budget(1)
    # v8.15.3d: 用户原始问题直传联网工具——deepseek_web_search 把原始问题原样交给
    # DeepSeek 原生联网（output_text 围绕原始问题作答），检索词仅作"搜索参考关键词"
    set_original_query(query)
    t0 = time.perf_counter()

    async def event_generator():
        from src.core.progress_bus import (
            set_request_queue, clear_request_queue,
            _encode_event, log_sse_frame,
            get_running_tools, clear_tool_timers,
        )

        # v8.3.3: 每个请求独立队列（contextvars 绑定），并发会话互不串扰
        request_queue: asyncio.Queue = asyncio.Queue()
        set_request_queue(request_queue)
        clear_tool_timers()
        event_queue: asyncio.Queue = asyncio.Queue()

        async def bridge_progress():
            # v9.2 P4 根治: 事件驱动桥——原 0.3s 轮询（wait_for(timeout=0.3)）带来
            # 至多 300ms 转发延迟，且是后续 sleep(0.5)/轮询排空时序补丁的根因之一。
            # 改阻塞 get()：事件即刻转发；任务取消经 CancelledError 传播（finally
            # 统一 cancel，无泄漏）。
            while True:
                try:
                    evt = await request_queue.get()
                    await event_queue.put(evt)
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
        heartbeat_task = asyncio.create_task(tool_heartbeat())

        # v8.3.7 M2: 观察者标志——write 类任务断连保活后丢弃后续事件（防队列堆积）
        observer_connected = True
        keep_alive = False

        async def emit_to_client(evt) -> None:
            """仅在客户端在线时投递事件（保活后丢弃）。"""
            if observer_connected:
                await event_queue.put(evt)

        async def flush_bridge() -> None:
            # v9.2 P4 根治: 确定性排空替代时序补丁（原 done 前 sleep(0.5) 与
            # finally 10×0.1s 轮询排空）。事件驱动桥下，工具线程返回前已入队的
            # 事件即刻转发；此处显式把 request_queue 中仍在等待转发的事件一次
            # 性搬入 event_queue，并让出数轮事件循环吸收并发入队，保证它们
            # 先于 done / sentinel 到达（done 契约"事件顺序"不变量保持）。
            for _ in range(3):
                while not request_queue.empty():
                    await event_queue.put(request_queue.get_nowait())
                await asyncio.sleep(0)

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

                        # v9.2: 节点名改为真实值（原 "light_synthesize"/"light_react" 是
                        # v8.3.0 前 light 老节点名→ light 恒落入兜底分支：丢 citation_info/
                        # tools_called、job 永不 completed、request_done 日志缺失）
                        if node_name in ("supervisor", "light_supervisor"):
                            ans = output.get("answer", "")
                            if ans:
                                # v9.2 P4 根治: 排空桥替代原 sleep(0.5) 时序猜
                                # 测——工具线程已入队的进度事件先于 done 到达
                                await flush_bridge()
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
                                # v9.1.3: tools 口径与 supervisor_done 一致——tools_called
                                # 为列表（AgentState 已声明，不再被 langgraph 丢弃），取长度
                                _tools_n = len(output.get("tools_called") or [])
                                try:
                                    from src.core.business_logger import blog
                                    blog("request_done", answer_chars=len(ans),
                                         tools=_tools_n,
                                         ms=int((time.perf_counter() - t0) * 1000),
                                         job=job_id[:12])
                                except Exception:
                                    pass
                                # v8.13: 结构化诊断事件（请求成功快照）
                                try:
                                    from src.core.diag import diag
                                    diag("request_done", node=node_name,
                                         answer_chars=len(ans),
                                         tools=_tools_n,
                                         ms=int((time.perf_counter() - t0) * 1000))
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
                # v8.13: 结构化诊断事件（请求失败快照）
                try:
                    from src.core.diag import diag
                    diag("request_error", err=type(e).__name__, msg=str(e)[:200])
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
                # v8.17.15: 草稿任务注册表已删除（草稿全链移除，联网回归
                # retrieve-agent 工具链），此处不再等待草稿——直接关闭连接。
                # v9.2 P4 根治: 确定性排空替代原 10×0.1s 轮询排空——桥中遗留
                # 事件全部先搬到 event_queue，再发 sentinel（事件顺序不变量）
                await flush_bridge()
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
        # v8.15: 数据源开关状态（前端据此动态显示欢迎语/联网按钮可用性）
        "academic_enabled": bool(settings.ACADEMIC_ENABLED),
        "web_search": {
            "enabled": bool(settings.WEB_SEARCH_ENABLED),
            # v8.15: provider 纯信息展示；启用时一律走 DeepSeek Responses 原生 web_search。
            # 注意：旧 .env 残留 WEB_SEARCH_PROVIDER=serper 是历史 Tavily 自建源配置，
            # 已不参与执行路径（仅作展示兜底）。
            "provider": ("deepseek_responses" if settings.WEB_SEARCH_ENABLED
                         else settings.WEB_SEARCH_PROVIDER),
        },
        "model": {
            "main": settings.RESOLVED_MAIN_MODEL,
            "fast": settings.RESOLVED_FAST_MODEL,
            "main_base_url": settings.RESOLVED_MAIN_BASE_URL,
            "fast_base_url": settings.RESOLVED_FAST_BASE_URL,
            "defaults": {
                "main": settings.MAIN_MODEL,
                "fast": settings.FAST_MODEL,
                "main_base_url": settings.MAIN_BASE_URL,
                "fast_base_url": settings.FAST_BASE_URL or settings.MAIN_BASE_URL,
            },
            "runtime_overrides": settings.has_model_overrides,
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
                            detail="Key 格式不正确（应为 sk- 开头的 API Key）")
    # 连通性校验：GET {base}/models
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{settings.RESOLVED_MAIN_BASE_URL.rstrip('/')}/models",
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
        raise HTTPException(status_code=400, detail=f"无法连接 API 服务: {e}")
    if not settings.save_runtime_api_key(key):
        raise HTTPException(status_code=500, detail="Key 保存失败")
    logger.info("[API] runtime api key configured via WebUI")
    return {"status": "ok", "has_api_key": True}


# ── v8.14.1 运行时底座模型切换（前端设置面板）──
# 主/快模型 ID + API 地址，留空 = 恢复默认（yaml/env）；
# 持久化到 state/model_config.json（gitignore 内），跨重启保留。
class ModelConfigRequest(BaseModel):
    main_model: str = ""
    fast_model: str = ""
    main_base_url: str = ""
    fast_base_url: str = ""


@app.post("/api/v2/config/model")
async def set_model_config(req: ModelConfigRequest):
    main_base_url = (req.main_base_url or "").strip().rstrip("/")
    if main_base_url and not (main_base_url.startswith("http://")
                              or main_base_url.startswith("https://")):
        raise HTTPException(status_code=400,
                            detail="API 地址需以 http:// 或 https:// 开头")
    # 连通性校验：仅当已配 Key 且给了主地址时校验（保证地址未拼错）。
    # 401/403 视为「地址可达但 Key 属于其他厂商」——放行并附警告（换地址后需再换 Key）。
    auth_warning = ""
    key = settings.RESOLVED_MAIN_API_KEY
    if key and main_base_url:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{main_base_url}/models",
                    headers={"Authorization": f"Bearer {key}"},
                )
            if resp.status_code not in (200, 401, 403):
                raise HTTPException(
                    status_code=400,
                    detail=f"模型列表校验失败（HTTP {resp.status_code}）：请检查 API 地址")
            if resp.status_code in (401, 403):
                auth_warning = (
                    f"地址可达但当前 Key 校验未通过（HTTP {resp.status_code}）；"
                    f"若该 Key 属于其他厂商，保存模型配置后请再在下方保存新 Key")
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"[API] model config connectivity check failed: {e}")
            raise HTTPException(status_code=400,
                                detail=f"无法连接 {main_base_url}（请检查地址是否拼写正确）: {e}")
    if not settings.save_runtime_model_config(
            main=req.main_model, fast=req.fast_model,
            main_base_url=req.main_base_url, fast_base_url=req.fast_base_url):
        raise HTTPException(status_code=500, detail="模型配置保存失败")
    logger.info("[API] runtime model config saved: %s", settings._runtime_model_cfg)
    return {
        "status": "ok",
        "warning": auth_warning,
        "model": {
            "main": settings.RESOLVED_MAIN_MODEL,
            "fast": settings.RESOLVED_FAST_MODEL,
            "main_base_url": settings.RESOLVED_MAIN_BASE_URL,
            "fast_base_url": settings.RESOLVED_FAST_BASE_URL,
            "runtime_overrides": settings.has_model_overrides,
        },
    }


@app.get("/")
async def serve_frontend():
    html_path = PROJECT_ROOT / "index.html"
    if not html_path.exists():
        return {"error": f"index.html not found at {html_path}"}
    return FileResponse(html_path)


# v8.9 工作区静态服务（会话侧栏"工作区"文件可点击打开；只读）
# v8.13: 暴露面从整个 workspace/ 收窄到 workspace/output/——前端仅引用
# /workspace/output/（写作成果预览），其余目录（input/临时文件等）不再 HTTP 可达
_workspace_static = (PROJECT_ROOT / "workspace" / "output")
if _workspace_static.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/workspace/output", StaticFiles(directory=str(_workspace_static)),
              name="workspace")


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
            # 路径安全：必须落在 workspace/output 内（v8.13: startswith 有同前缀
            # 边界漏洞——output_evil 兄弟目录可绕过 → is_relative_to 严格判定）
            if not p.is_relative_to(out_root):
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
    """v9.2: 会话历史文献引用恢复——按来源分组（RAG/UCR/Web/历史 + 动态批次组）。

    替代 v8.10k 每轮列表结构（ref_id=R{turn}-{i} 与正文 [n]/[Wn]/[Hn] 体系
    不同源、点击不可追踪）：返回与 live 侧栏同构的 groups——
      数字组（rag ∪ ucr，与 live [n] 共用编号池）连续 1..k；
      web 组 W1..Wm；historical 组 H1..Hn（get_evidence_refs 同款跨轮去重）；
      v9.4: 非内置来源（paper1/Citrus varieties1 等）按规范化分组键归组
      （paper1→paper），与 live 侧栏 srcKey 同口径。
    每组条目携带 round_seq（来源轮次）元数据。前端刷新/切换会话后按组
    手风琴渲染，正文历史引用编号可对应组内条目。
    """
    import sqlite3
    import json as _json
    from src.core.evidence import normalize_source_key
    groups: dict[str, list] = {"rag": [], "ucr": [], "web": [], "historical": []}
    seen: dict[str, set] = {"rag": set(), "ucr": set(), "web": set()}
    num_i = 0
    web_i = 0
    try:
        with sqlite3.connect(str(PROJECT_ROOT / "state" / "sessions.db")) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT turn_seq, evidence_json FROM session_evidence "
                "WHERE session_id=? ORDER BY id DESC LIMIT 10", (session_id,)).fetchall()
    except Exception as e:
        logger.warning(f"[API] session citations failed: {e}")
        return {"session_id": session_id, "groups": groups, "count": 0}
    for r in reversed(rows):   # 旧 → 新：编号顺序与回答内首次出现一致
        turn_seq = r["turn_seq"]
        try:
            raw = _json.loads(r["evidence_json"] or "[]")
        except Exception:
            raw = []
        if not isinstance(raw, list):
            continue
        for e in raw:
            if not isinstance(e, dict):
                continue
            src = str(e.get("source") or "").strip() or "rag"
            # v9.4: 非内置来源（paper1/Citrus varieties1 等批次）按规范化分组键
            # 归组（paper1→paper），与 live 侧栏 srcKey 同口径；旧历史条目 ucr/rag/web 原样
            group = src if src in ("rag", "ucr", "web") else normalize_source_key(src)
            doi = str(e.get("doi") or "").strip()
            title = str(e.get("title") or "").strip()[:150]
            key = (doi or title or str(e.get("chunk_id") or "")).strip()
            if not key or key in seen.setdefault(group, set()):
                continue
            seen[group].add(key)
            if group == "web":
                web_i += 1
                ref_id = f"W{web_i}"
            else:
                num_i += 1
                ref_id = str(num_i)
            groups.setdefault(group, []).append({
                "ref_id": ref_id,
                "type": "main" if group != "web" else "web",
                "source": group,
                "doi": doi or "N/A",
                "title": title,
                "year": str(e.get("year") or ""),
                "score": float(e.get("score") or 0),
                "chunk_id": str(e.get("chunk_id") or ""),
                "text_preview": str(e.get("snippet") or "")[:250],
                "round_seq": turn_seq,   # 来源轮次元数据（前端可显示"第 N 轮"）
            })
    # 历史组 = 最近 10 轮证据去重（H1..Hn，与 live historical 组同口径）
    try:
        _hist = session_manager.get_evidence_refs(session_id, limit=10)
        for i, _h in enumerate(_hist or [], 1):
            groups["historical"].append({
                "ref_id": f"H{i}",
                "type": "historical",
                "source": str(_h.get("source") or "rag"),
                "doi": str(_h.get("doi") or "N/A"),
                "url": str(_h.get("url") or ""),
                "title": str(_h.get("title") or "")[:150],
                "year": str(_h.get("year") or ""),
                "text_preview": "",
                "chunk_id": str(_h.get("chunk_id") or ""),
            })
    except Exception as e:
        logger.debug(f"[API] session citations historical failed: {e}")
    return {"session_id": session_id, "groups": groups,
            "count": sum(len(v) for v in groups.values())}


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
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    for m in msgs:
        if isinstance(m, HumanMessage):
            content = str(getattr(m, "content", "") or "").strip()
            if content:
                out.append({"role": "user", "content": _user_display_text(content)})
        elif isinstance(m, AIMessage):
            content = str(getattr(m, "content", "") or "").strip()
            # 纯工具轮（只有 tool_calls 无文本）不渲染；合成收尾指令已由读路径过滤
            if content:
                item: dict = {"role": "assistant", "content": content}
                # v8.10l: 深度思考随历史返回（退出会话再回来可恢复思考过程）
                rc = (getattr(m, "additional_kwargs", None) or {}).get("reasoning_content")
                if rc:
                    item["reasoning"] = rc
                out.append(item)
        elif isinstance(m, ToolMessage):
            # v8.10l: 工具调用结果随历史返回（前端恢复工具链折叠块）
            tcontent = str(getattr(m, "content", "") or "")
            out.append({
                "role": "tool",
                "name": getattr(m, "name", "") or "tool",
                "content": tcontent[:500],
                "truncated": len(tcontent) > 500,
            })
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
        # v8.16.4: 细分端点不再 500——内部异常降级为空信封 + error 标记，
        # 前端据此显示温和失败提示（此前 500 直接落到前端 .catch"细分加载失败"，
        # 且无法与"暂无上下文"区分）
        logger.error(f"[API] context-detail failed: {e}")
        return {"session_id": session_id, "segments": [], "count": 0,
                "error": "context_detail_error"}
