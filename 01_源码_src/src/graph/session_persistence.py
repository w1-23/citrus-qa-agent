"""Persistence for one General Agent turn."""
from __future__ import annotations

import asyncio
import logging
import time

from langchain_core.messages import AIMessage

from src.core.evidence import EVIDENCE_SNIPPET_MAX_CHARS, clean_doi, render_evidence, src_of

logger = logging.getLogger(__name__)
_LTM_SUBSTANTIAL_MARKERS = ("###", "结论", "摘要", "引言", "核心结论", "局限与边界")


def _worth_extracting_memory(answer: str) -> bool:
    return len(answer) > 500 and any(marker in answer for marker in _LTM_SUBSTANTIAL_MARKERS)


async def persist_agent_turn(state: dict) -> dict:
    """Save the answer, refresh-safe elapsed time, evidence and account memory."""
    query = str(state.get("query") or "")
    answer = str(state.get("answer") or "")
    session_id = str(state.get("session_id") or "default")
    if not answer:
        return {"_trace": {"node": "session_save", "elapsed_ms": 0,
                           "summary": "no answer", "history_saved": False,
                           "generation_ms": 0}}

    saved = False
    generation_ms = 0
    try:
        from src.session.manager import session_manager, _validate_trace

        trace = [
            message for message in (state.get("turn_trace") or [])
            if message is not None and getattr(message, "type", "") != "system"
            and not str(getattr(message, "content", "")).startswith("[历史检索证据]")
        ]
        trace = _validate_trace(trace)
        if not (trace and isinstance(trace[-1], AIMessage)
                and not getattr(trace[-1], "tool_calls", None)
                and trace[-1].content == answer):
            trace.append(AIMessage(content=answer))

        try:
            started_at = float(state.get("request_started_at") or 0)
            generation_ms = int(max(0.0, time.perf_counter() - started_at) * 1000)
        except (TypeError, ValueError):
            generation_ms = int(max(0, state.get("gen_time_ms") or 0))
        if generation_ms > 0:
            metadata = dict(getattr(trace[-1], "response_metadata", {}) or {})
            metadata["citrus_generation_ms"] = generation_ms
            trace[-1].response_metadata = metadata

        saved = await session_manager.save_messages(
            session_id, trace, state.get("idempotency_key", ""))
        if not saved:
            logger.warning("[GeneralAgent:save] conversation history was not written")

        main_results = state.get("main_results") or []
        web_results = state.get("web_results") or []
        if main_results or web_results:
            evidence = [
                {
                    "doi": clean_doi(result.get("doi")),
                    "chunk_id": f"{result.get('paper_id', '')}:{result.get('chunk_index', '')}",
                    "title": str(result.get("title", ""))[:150],
                    "score": result.get("score", result.get("rerank_score", 0)) or 0,
                    "year": str(result.get("year", "")),
                    "source": src_of(result),
                    "snippet": render_evidence(result, max_chars=EVIDENCE_SNIPPET_MAX_CHARS),
                }
                for result in main_results[:30]
            ]
            evidence.extend(
                {
                    "doi": "",
                    "chunk_id": str(result.get("url", ""))[:200],
                    "title": str(result.get("title", ""))[:150],
                    "score": float(result.get("score", 0) or 0),
                    "year": "",
                    "source": "web",
                    "snippet": str(result.get("abstract", result.get("snippet", "")))[:500],
                    "url": str(result.get("url", "")),
                }
                for result in web_results[:30]
            )
            refs_data = state.get("references_data") or {}
            cited_refs = refs_data.get("cited") if isinstance(refs_data, dict) else None
            from src.core.background import spawn

            spawn(session_manager.save_evidence(session_id, query, evidence, "", cited_refs))
    except Exception as exc:  # saving must not discard a completed answer
        logger.warning("[GeneralAgent:save] failed: %s", exc)

    if _worth_extracting_memory(answer):
        try:
            from src.core.background import spawn

            def extract_and_save_memory(q: str, text: str, sid: str, user_id: int) -> None:
                try:
                    from src.guardrails.memory import memory_store
                    facts = memory_store.extract_key_facts(q, text, session_id=sid)
                    for fact in facts:
                        if fact.get("type") == "preference":
                            memory_store.set_preference(
                                user_id, fact.get("key", ""), fact.get("value", ""))
                        else:
                            memory_store.save_long_term_fact(
                                fact.get("key", ""), fact.get("value", ""),
                                fact.get("confidence", 0.5), owner_session=sid,
                                source_query=q, owner_user_id=user_id)
                except Exception as exc:
                    logger.debug("[GeneralAgent:save] memory extraction failed: %s", exc)

            spawn(asyncio.to_thread(
                extract_and_save_memory, query, answer, session_id,
                int(state.get("user_id") or 0)))
        except Exception as exc:
            logger.debug("[GeneralAgent:save] memory extraction skipped: %s", exc)

    return {"_trace": {
        "node": "session_save", "elapsed_ms": 0,
        "summary": "saved" if saved else "already saved or skipped",
        "history_saved": bool(saved), "generation_ms": generation_ms,
    }}
