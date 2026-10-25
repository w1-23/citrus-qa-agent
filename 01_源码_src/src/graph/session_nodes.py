"""Shared conversation context loading and persistence nodes."""
from __future__ import annotations

import asyncio
import logging
import time

from src.config import settings
from src.graph.state import AgentState

logger = logging.getLogger(__name__)


async def load_session_context(state: AgentState) -> dict:
    """Load history and memory once for the unified Agent graph."""
    query = state.get("query", "")
    session_id = state.get("session_id", "default")
    # The application now has one Agent graph. Ignore the legacy UI mode so
    # history hints and prompts cannot silently switch to the retired Lite path.
    mode = "expert"
    result: dict = {}
    try:
        from src.session.manager import session_manager
        from src.core.context_budget import ContextBudget, ContextBudgetConfig
        from src.core.context_manager import ContextManager, finalize_load_result
        from src.guardrails.memory import memory_store
        from src.core.progress_bus import emit_status, emit_progress

        budget = ContextBudget(ContextBudgetConfig(
            max_tokens=settings.RESOLVED_CONTEXT_BUDGET_MAX_TOKENS,
            soft_threshold=settings.CONTEXT_BUDGET_SOFT_THRESHOLD,
            hard_threshold=settings.CONTEXT_BUDGET_HARD_THRESHOLD,
        ))
        ctx_mgr = ContextManager(
            session_manager=session_manager,
            memory_store=memory_store,
            budget=budget,
        )
        started = time.perf_counter()
        emit_status("step_active", step_id="load")
        emit_progress("tool_progress", {
            "message": "正在加载会话历史与记忆…", "tool_call_id": ""})
        load_coro = ctx_mgr.load(
            session_id, query, mode,
            generate_hints=not bool(state.get("agent_no_prefetch_hints")))
        if state.get("agent_no_prefetch_hints") and state.get("request_deadline"):
            remaining = max(0.1, float(state["request_deadline"]) - time.monotonic() - 25.0)
            ctx = await asyncio.wait_for(load_coro, timeout=remaining)
        else:
            ctx = await load_coro
        result = finalize_load_result(
            ctx,
            session_manager=session_manager,
            session_id=session_id,
            budget=budget,
            node_label="session_load",
            log_prefix="GeneralAgent",
        )
        emit_status("step_done", step_id="load")
        emit_progress("tool_progress", {
            "message": f"已加载历史 {len(ctx.history_messages)} 条，正在理解问题并规划检索…",
            "tool_call_id": ""})
        try:
            from src.core.diag import diag
            diag("load", ms=round((time.perf_counter() - started) * 1000, 1),
                 raw_msgs=len(ctx.history_messages), compacted=bool(ctx.compacted))
        except Exception:
            pass
        try:
            from src.core.business_logger import blog
            blog("load_done", raw_msgs=len(ctx.history_messages),
                 compacted=ctx.compacted, ltm=bool(ctx.long_term_memory),
                 resident=bool(ctx.resident_cards),
                 evidence=bool(result.get("history_evidence_block")))
        except Exception:
            pass
        return result
    except Exception as exc:  # noqa: BLE001 - context failure must not break a chat
        logger.warning("[SessionNodes:load] failed: %s", exc)
        try:
            from src.core.business_logger import blog
            blog("load_failed", err=str(exc)[:200])
        except Exception:
            pass
        result["_trace"] = {"node": "session_load", "elapsed_ms": 0,
                            "summary": "unavailable"}
        return result


async def save_session_context(state: AgentState) -> dict:
    """Persist the answer, sanitized trace, and evidence ledger."""
    from src.core.agent_loop import default_ltm_gate, run_save_node
    return await run_save_node(
        state,
        log_tag="GeneralAgent",
        include_web=True,
        ltm_gate=default_ltm_gate,
    )
