"""Light Graph — LLM 自主路由 Supervisor.

v8.3.0: 移除 hardcoded pre-retrieve/ReAct fallback.
LLM bind_tools 自主决定是否调用搜索工具, max 3 轮.
"""
import asyncio
import json
import logging
import time
import uuid
from typing import AsyncIterator

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from src.graph.state import AgentState
from src.config import settings, get_deepseek_model
from src.prompts.loader import assemble_system_prompt

logger = logging.getLogger(__name__)

from src.core.progress_bus import (
    emit_encoded, emit_thinking, emit_text,
    emit_tool_call_start, emit_tool_executing, emit_tool_result,
    emit_status,
)

LIGHT_MAX_TURNS = 2
LIGHT_TOOL_NAMES = ("citrus_rag_search",)


def _build_light_llm(bind_tools=None):
    kwargs = {
        "model": get_deepseek_model(),
        "api_key": settings.RESOLVED_MAIN_API_KEY,
        "base_url": settings.MAIN_BASE_URL,
        "temperature": settings.TEMPERATURE_MAIN,
        "max_tokens": 16384,
        "timeout": 60,
    }
    client = ChatOpenAI(**kwargs)
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
        from src.core.context_manager import ContextManager, LoadedContext
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
        result["session_id"] = ctx.session_id
        result["history_summary"] = ctx.history_summary
        result["long_term_memory"] = ctx.long_term_memory
        result["search_suggestions"] = ctx.search_suggestions
        result["format_hint"] = ctx.format_hint

        history_msgs = list(ctx.history_messages) if ctx.history_messages else []
        result["messages"] = history_msgs

        load_summary = f"history={len(history_msgs)}msgs"
        if ctx.format_hint:
            load_summary += f", fmt={ctx.format_hint}"
        result["_trace"] = {"node": "load_context", "elapsed_ms": 0, "summary": load_summary}

        hist_chars = sum(len(m.content or "") for m in history_msgs if hasattr(m, 'content'))
        est_tokens = budget.estimate_tokens(history_msgs)
        try:
            emit_encoded("context_status", {
                "history_msgs": len(history_msgs),
                "history_chars": hist_chars,
                "ltm_recalled": bool(ctx.long_term_memory),
                "ltm_chars": len(ctx.long_term_memory or ""),
                "suggestions": ctx.search_suggestions[:3] if ctx.search_suggestions else [],
                "format_hint": ctx.format_hint or "",
                "estimated_tokens": est_tokens,
                "max_tokens": budget_config.max_tokens,
                "soft_threshold": budget_config.soft_threshold,
                "hard_threshold": budget_config.hard_threshold,
                "compressed": bool(ctx.history_summary),
                "compression_len": len(ctx.history_summary or ""),
            })
        except Exception:
            pass
        return result

    except Exception as e:
        logger.warning(f"[LightGraph:load] failed: {e}")
        result["_trace"] = {"node": "load_context", "elapsed_ms": 0, "summary": "unavailable"}
        return result


async def light_supervisor_node(state: AgentState) -> dict:
    """LLM autonomous routing supervisor.

    Binds citrus_rag_search / academic_search / encyclopedia_search.
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
        search_suggestions=state.get("search_suggestions", []),
        format_hint=format_hint,
    )
    human_msg = build_human_message(ctx)

    messages: list = [SystemMessage(content=system_prompt)]
    history_msgs = list(state.get("messages", []))
    if history_msgs:
        messages.extend(history_msgs)
    messages.append(human_msg)

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
                tc_id = getattr(tc, "id", "")
                tc_name = getattr(tc, "name", "?")
                tc_args = {}
                if hasattr(tc, "args") and tc.args:
                    tc_args = dict(tc.args)
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
            messages.append(HumanMessage(content=(
                "You have reached the maximum number of turns. "
                "Do NOT call any more tools. Provide your final answer now."
            )))
            final_resp = await _build_light_llm().ainvoke(messages)
            answer = final_resp.content or ""

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
        from langchain_core.messages import HumanMessage, AIMessage
        from src.session.manager import session_manager
        msgs_to_save = [
            HumanMessage(content=query),
            AIMessage(content=answer),
        ]
        asyncio.create_task(session_manager.save_messages(session_id, msgs_to_save))
    except Exception as e:
        logger.warning(f"[LightGraph:save] failed: {e}")

    try:
        from src.guardrails.memory import memory_store
        if len(answer) > 500:
            facts = await asyncio.to_thread(memory_store.extract_key_facts, query, answer)
            for f in facts:
                memory_store.save_long_term_fact(
                    f.get("key", ""),
                    f.get("value", ""),
                    f.get("confidence", 0.5),
                    owner_session=session_id,
                    source_query=query,
                )
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
