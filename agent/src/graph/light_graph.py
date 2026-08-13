"""Light Graph — LLM 自主路由 Supervisor.

v8.3.0: 移除 hardcoded pre-retrieve/ReAct fallback.
LLM bind_tools 自主决定是否调用搜索工具, max 3 轮.
"""
import asyncio
import logging
import time

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from src.graph.state import AgentState
from src.config import settings, get_deepseek_model
from src.prompts.loader import assemble_system_prompt

logger = logging.getLogger(__name__)

from src.core.progress_bus import (
emit_thinking, emit_text,
    emit_tool_call_start, emit_tool_executing, emit_tool_result,
    emit_status,
)

# v8.3.3: 轮次上限接线 config.yaml light.max_turns（此前硬编码 2）
LIGHT_MAX_TURNS = settings.LIGHT_MAX_TURNS
LIGHT_TOOL_NAMES = ("citrus_rag_search", "read_local_file")


def _build_light_llm(bind_tools=None):
    # v8.4: 客户端进程级复用（llm_pool），避免每请求新建 ChatOpenAI/连接池
    from src.core.llm_pool import get_llm as _pool_get_llm
    client = _pool_get_llm(
        model=get_deepseek_model(),
        api_key=settings.RESOLVED_MAIN_API_KEY,
        base_url=settings.MAIN_BASE_URL,
        temperature=settings.TEMPERATURE_MAIN,
        max_tokens=16384,
        timeout=60,
    )
    if bind_tools:
        client = client.bind_tools(bind_tools)
    return client


async def load_context_node(state: AgentState) -> dict:
    query = state.get("query", "")
    session_id = state.get("session_id", "default")
    mode = state.get("mode", "light")
    logger.info(f"[LightGraph:load] session={session_id[:8]}...")

    try:
        emit_status("step_active", step_id="load", message="加载会话上下文...")
    except Exception:
        pass

    result: dict = {}
    try:
        from src.session.manager import session_manager
        from src.core.context_budget import ContextBudget, ContextBudgetConfig
        from src.core.context_manager import ContextManager, finalize_load_result
        from src.guardrails.memory import memory_store

        budget_config = ContextBudgetConfig(
            max_tokens=settings.CONTEXT_BUDGET_MAX_TOKENS,
            soft_threshold=settings.CONTEXT_BUDGET_SOFT_THRESHOLD,
            hard_threshold=settings.CONTEXT_BUDGET_HARD_THRESHOLD,
        )
        budget = ContextBudget(budget_config)

        ctx_mgr = ContextManager(
            session_manager=session_manager,
            memory_store=memory_store,
            budget=budget,
        )

        ctx = await ctx_mgr.load(session_id, query, mode)
        # v8.4: 收尾装配收敛至 context_manager.finalize_load_result
        # （与 expert 图共用，消除双实现漂移）
        return finalize_load_result(
            ctx,
            session_manager=session_manager,
            session_id=session_id,
            budget=budget,
            node_label="load_context",
            log_prefix="LightGraph",
        )

    except Exception as e:
        logger.warning(f"[LightGraph:load] failed: {e}")
        result["_trace"] = {"node": "load_context", "elapsed_ms": 0, "summary": "unavailable"}
        return result


async def light_supervisor_node(state: AgentState) -> dict:
    """LLM autonomous routing supervisor.

    Binds citrus_rag_search.
    LLM receives light-mode system prompt and decides whether to call tools.
    Max 3 turns, timeout 60s per LLM call.
    """
    query = state.get("query", "")
    format_hint = state.get("format_hint")
    logger.info(f"[LightGraph:supervisor] query={query[:60]}...")

    system_prompt = assemble_system_prompt(
        mode="light", format_hint=format_hint, query=query,
    )

    from src.core.context_manager import LoadedContext, build_human_message
    ctx = LoadedContext(
        session_id=state.get("session_id", ""),
        mode="light",
        query=query,
        history_summary=state.get("history_summary"),
        long_term_memory=state.get("long_term_memory"),
        resident_cards=state.get("resident_cards"),
        search_suggestions=state.get("search_suggestions", []),
        format_hint=format_hint,
    )
    human_msg = build_human_message(ctx)

    messages: list = [SystemMessage(content=system_prompt)]
    history_msgs = list(state.get("messages", []))
    if history_msgs:
        messages.extend(history_msgs)
    # v8.3.8: 历史检索证据块
    if state.get("history_evidence_block"):
        messages.append(HumanMessage(content=state["history_evidence_block"]))
    messages.append(human_msg)
    # v8.3.8: 本轮轨迹起点
    trace_start_index = len(messages) - 1

    from src.tools import _TOOL_REGISTRY_BY_NAME
    tools = [t for t_name in LIGHT_TOOL_NAMES
             if t_name in _TOOL_REGISTRY_BY_NAME
             for t in [_TOOL_REGISTRY_BY_NAME[t_name]]]

    llm = _build_light_llm(bind_tools=tools) if tools else _build_light_llm()

    answer = ""
    all_main_results = []
    all_web_results = []
    tool_call_count = 0
    t0 = time.perf_counter()

    try:
        for turn in range(LIGHT_MAX_TURNS):
            t_llm = time.perf_counter()
            for attempt in range(3):
                try:
                    response = await llm.ainvoke(messages)
                    break
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"[LightGraph:supervisor] LLM retry {attempt+1}/3: {e}")
                        await asyncio.sleep(3)
                    else:
                        raise
            dt_llm = (time.perf_counter() - t_llm) * 1000
            messages.append(response)
            # v8.3.3: 推送真实 token 增量（前端上下文面板实时刷新，避免累计值重复计数）
            # 阶段0: 统一经 cache_metrics 提取（含 prompt_cache 命中字段）
            try:
                from src.core.cache_metrics import emit_usage_from_response
                emit_usage_from_response(state.get("session_id", ""),
                                         "light_supervisor", response)
            except Exception:
                pass

            if not getattr(response, "tool_calls", None):
                answer = response.content or ""
                logger.info(f"[LightGraph:supervisor] turn{turn} done: {len(answer)}c / {dt_llm:.0f}ms")
                break

            if response.content:
                try:
                    emit_thinking(response.content[:800])
                except Exception:
                    pass

            # Emit tool_call_start events for all tools
            for tc in response.tool_calls:
                # tool_calls 元素为 dict（OpenAI 兼容格式），兼容对象形式取值
                if isinstance(tc, dict):
                    tc_id = tc.get("id", "")
                    tc_name = tc.get("name", "?")
                    tc_args = dict(tc.get("args") or {})
                else:
                    tc_id = getattr(tc, "id", "")
                    tc_name = getattr(tc, "name", "?")
                    tc_args = dict(getattr(tc, "args", None) or {})
                logger.info(f"[LightGraph:supervisor] turn{turn}: call {tc_name}({str(tc_args)[:80]})")
                try:
                    emit_tool_call_start(tc_name[:30], tc_args, tc_id)
                    emit_tool_executing(f"调用 {tc_name}...", tc_name[:30], tc_id)
                except Exception:
                    pass

            # Batch execute all tool calls (preserves original API call IDs)
            t_tool = time.perf_counter()
            from src.tools.registry import PartitionedToolNode
            tn = PartitionedToolNode(tools)
            tool_results = await tn.execute_tools(list(response.tool_calls))
            dt_tool = (time.perf_counter() - t_tool) * 1000
            tool_call_count += len(response.tool_calls)

            for idx, tr in enumerate(tool_results):
                messages.append(tr)
                art = getattr(tr, "artifact", {}) or {}
                if isinstance(art, dict):
                    all_main_results.extend(art.get("main_results", []))
                    all_web_results.extend(art.get("web_results", []))
                tc = response.tool_calls[idx] if idx < len(response.tool_calls) else None
                tc_id = getattr(tc, "id", "") if tc else ""
                tc_name = getattr(tc, "name", "") if tc else "tool"
                try:
                    result_text = str(getattr(tr, "content", ""))[:2000]
                    emit_tool_result(tc_name[:30], result_text, tc_id,
                                     summary=f"完成 ({dt_tool:.0f}ms)")
                except Exception:
                    pass

            logger.info(f"[LightGraph:supervisor] turn{turn}: LLM {dt_llm:.0f}ms (total {time.perf_counter()-t0:.1f}s)")

        else:
            logger.info("[LightGraph:supervisor] max turns, forcing final")
            # v8.4.3 工单7: 与 expert 统一——详尽中文措辞 + 临时列表（不入 turn_trace/历史）
            # + 未绑工具客户端（杜绝收尾再发 tool_calls 导致空答）
            final_prompt = (
                "请立即给出最终回答：基于已检索的全部证据，完整、详尽、结构化地作答；"
                "不要精简、不要省略、不要提及工具或轮次限制；信息不足请逐条说明缺口。"
            )
            final_resp = await _build_light_llm().ainvoke(
                messages + [HumanMessage(content=final_prompt)])
            answer = final_resp.content or ""
            if not answer:
                for m in reversed(messages):
                    if getattr(m, "content", None):
                        answer = m.content
                        break

    except Exception as e:
        logger.error(f"[LightGraph:supervisor] error: {e}")
        answer = f"An error occurred: {e}"

    elapsed = (time.perf_counter() - t0) * 1000

    seen_doi = set()
    deduped_main = []
    for r in all_main_results:
        doi = (r.get("doi") or "").strip()
        if doi and doi in seen_doi:
            continue
        seen_doi.add(doi)
        deduped_main.append(r)

    cited_refs = []
    for i, r in enumerate(deduped_main[:20]):
        cited_refs.append({
            "ref_id": i + 1,
            "type": "main",
            "doi": r.get("doi", "N/A"),
            "title": r.get("title", r.get("name", "Untitled")),
            "section_name": r.get("section_name", ""),
            "text_preview": (r.get("abstract") or r.get("snippet") or "")[:300],
            "score": r.get("score", r.get("rerank_score", 0)) or 0,
            "year": r.get("year", ""),
            "authors": r.get("authors", ""),
        })
    for i, wr in enumerate(all_web_results[:5]):
        cited_refs.append({
            "ref_id": f"W{i+1}",
            "type": "web",
            "url": wr.get("url", wr.get("link", "")),
            "title": wr.get("title", wr.get("name", "Untitled")),
            "text_preview": (wr.get("snippet") or wr.get("content") or "")[:300],
            "score": 0,
        })

    references_data = {"cited": cited_refs, "uncited": [], "total": len(cited_refs)}

    if answer:
        chunk_size = 8
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i:i + chunk_size]
            try:
                emit_text(chunk)
            except Exception:
                pass
            await asyncio.sleep(0.012)

    logger.info(
        f"[LightGraph:supervisor] done: {len(answer)}c, "
        f"{tool_call_count} tool calls, "
        f"{len(deduped_main)} main + {len(all_web_results)} web results, "
        f"{elapsed:.0f}ms"
    )

    return {
        "answer": answer,
        "gen_time_ms": elapsed,
        "main_results": deduped_main[:20],
        "web_results": all_web_results[:5],
        "references_data": references_data,
        # v8.3.8: 本轮完整轨迹（save 节点持久化）
        "turn_trace": messages[trace_start_index:],
        "_trace": {"node": "light_supervisor", "elapsed_ms": elapsed,
                   "summary": f"{len(answer)} chars, {tool_call_count} tools"},
    }


async def save_context_node(state: AgentState) -> dict:
    query = state.get("query", "")
    answer = state.get("answer", "")
    session_id = state.get("session_id", "default")
    if not answer:
        return {"_trace": {"node": "save", "elapsed_ms": 0, "summary": "no answer"}}

    try:
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
        from src.session.manager import session_manager, _validate_trace
        from src.core.background import spawn
        # v8.3.8: 完整轨迹持久化（含 tool_calls/ToolMessage 配对）
        trace = list(state.get("turn_trace") or [])
        trace = [m for m in trace if getattr(m, "type", "") != "system"]
        trace = [m for m in trace
                 if not str(getattr(m, "content", "")).startswith("[历史检索证据]")]
        trace = _validate_trace(trace)
        if not (trace and isinstance(trace[-1], AIMessage)
                and not getattr(trace[-1], "tool_calls", None)
                and trace[-1].content == answer):
            trace.append(AIMessage(content=answer))
        spawn(session_manager.save_messages(
            session_id, trace, state.get("idempotency_key", "")))
        # v8.3.8: 证据账本（v8.3.9: 合并全部报告 + chunk_id 可回查）
        report_parts = []
        for m in trace:
            if isinstance(m, ToolMessage) and getattr(m, "name", "") == "call_retrieve_agent":
                report_parts.append(str(m.content))
        report_text = "\n\n---\n\n".join(report_parts)
        main_results = state.get("main_results") or []
        if report_text or main_results:
            evidence = [
                {"doi": r.get("doi", ""),
                 "chunk_id": f"{r.get('paper_id', '')}:{r.get('chunk_index', '')}",
                 "title": str(r.get("title", ""))[:150],
                 "score": r.get("score", r.get("rerank_score", 0)) or 0,
                 "snippet": str(r.get("text", "") or r.get("abstract", ""))[:500]}
                for r in main_results[:30]
            ]
            spawn(session_manager.save_evidence(
                session_id, query, evidence, report_text))
    except Exception as e:
        logger.warning(f"[LightGraph:save] failed: {e}")

    try:
        # v8.4: LTM 提取转后台 + ADD-only 写入（与 expert 图一致）
        def _extract_and_save_ltm(q: str, a: str, sid: str):
            try:
                from src.guardrails.memory import memory_store
                facts = memory_store.extract_key_facts(q, a)
                for f in facts:
                    memory_store.save_long_term_fact(
                        f.get("key", ""),
                        f.get("value", ""),
                        f.get("confidence", 0.5),
                        owner_session=sid,
                        source_query=q,
                    )
            except Exception as e:
                logger.debug(f"[LightGraph:save] LTM background extract failed: {e}")

        if len(answer) > 500:
            spawn(asyncio.to_thread(_extract_and_save_ltm, query, answer, session_id))
    except Exception as e:
        logger.debug(f"[LightGraph:save] LTM skip: {e}")

    return {"_trace": {"node": "save", "elapsed_ms": 0, "summary": "saved"}}


def build_light_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("load_context", load_context_node)
    graph.add_node("light_supervisor", light_supervisor_node)
    graph.add_node("save_context", save_context_node)

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "light_supervisor")
    graph.add_edge("light_supervisor", "save_context")
    graph.add_edge("save_context", END)

    return graph.compile()
