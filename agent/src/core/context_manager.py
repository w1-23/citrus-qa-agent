"""Context Manager — 统一上下文加载与组装.

v8.1.1: 所有 LLM 调用前通过 ContextManager 获取上下文.
  - 加载完整 messages (不自动压缩)
  - 召回长期记忆
  - 预生成 search_suggestions + format_hint
  - 组装标准 HumanMessage

v8.4 (存储全量·发送裁剪):
  - SQLite 原始轨迹 append-only，压缩绝不写回历史
  - 用户轮边界（load 时）基于 checkpoint 构建发送视图:
      [首消息] + <conversation_summary> + [checkpoint 之后的原始消息]
  - 视图超软阈值 → 批量压缩至 ~50%，checkpoint 持久化到 sessions 表
    （原始消息永不删除，防摘要套摘要——旧摘要作为 prior_summary 增量整合）
  - 循环内绝不压缩（前缀中途变化=缓存必不命中）
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from src.config import settings
from src.core.context_budget import ContextBudget, ContextBudgetLevel

logger = logging.getLogger(__name__)


@dataclass
class LoadedContext:
    session_id: str
    mode: str
    query: str
    history_messages: list[BaseMessage] = field(default_factory=list)
    history_summary: str | None = None
    long_term_memory: str | None = None
    resident_cards: str | None = None    # v8.4: 常驻卡片（双层记忆"概览"层，≤500 字符）
    search_suggestions: list[str] = field(default_factory=list)
    format_hint: str | None = None
    compacted: bool = False    # v8.4: 本轮是否触发过压缩（UI 展示用，摘要已在视图内）

    @property
    def has_history(self) -> bool:
        return bool(self.history_messages or self.history_summary)


class ContextManager:
    def __init__(
        self,
        session_manager=None,
        memory_store=None,
        budget: ContextBudget | None = None,
    ):
        self._session = session_manager
        self._memory = memory_store
        self._budget = budget
        self._fast_llm = None
        self._compact_llm = None
        if budget:
            from src.guardrails.history_compactor import compact_messages
            async def _compact(msgs, query="", prior_summary=""):
                return await compact_messages(
                    msgs,
                    fast_llm=self._get_compact_llm(),
                    max_tokens=self._budget.config.compact_max_tokens,
                    query=query,
                    prior_summary=prior_summary,
                )
            budget.set_compact_fn(_compact)

    def _get_fast_llm(self):
        if self._fast_llm is None:
            from src.core.llm_pool import get_llm as _pool_get_llm
            self._fast_llm = _pool_get_llm(
                model=settings.FAST_MODEL,
                api_key=settings.RESOLVED_FAST_API_KEY,
                base_url=settings.RESOLVED_FAST_BASE_URL,
                temperature=0,
                timeout=12,
            )
        return self._fast_llm

    def _get_compact_llm(self):
        """压缩用 fast 模型 (v8.4): 压缩是高频低价值操作，摘要质量由保留优先级提示保证，
        无需 main 模型；触发频率低时 main 模型成本过高。"""
        if self._compact_llm is None:
            from src.core.llm_pool import get_llm as _pool_get_llm
            self._compact_llm = _pool_get_llm(
                model=settings.FAST_MODEL,
                api_key=settings.RESOLVED_FAST_API_KEY,
                base_url=settings.RESOLVED_FAST_BASE_URL,
                temperature=0,
                timeout=30,
            )
        return self._compact_llm

    async def load(
        self,
        session_id: str,
        query: str,
        mode: str,
    ) -> LoadedContext:
        ctx = LoadedContext(
            session_id=session_id,
            mode=mode,
            query=query,
        )

        if self._session:
            try:
                raw_messages, row_ids = await self._session.get_messages_with_ids(session_id)
                view_msgs, compacted = await self._build_send_view(
                    session_id, raw_messages, row_ids, query)
                ctx.history_messages = view_msgs
                ctx.compacted = compacted
                logger.info(
                    f"[ContextManager] loaded {len(raw_messages)} raw msgs -> "
                    f"view {len(view_msgs)} msgs for session {session_id[:8]}"
                )
            except Exception as e:
                logger.warning(f"[ContextManager] load history failed: {e}")
                # AG-14: 历史加载失败不再静默 — 推送 context_degraded 供前端警示
                try:
                    from src.core.progress_bus import emit_encoded
                    emit_encoded("context_degraded", {"reason": "history_unavailable"})
                except Exception:
                    pass

        if self._memory:
            try:
                # v8.4.5: LTM 语义召回含 embed 推理（≤500 条事实），走线程池防阻塞事件循环
                ltm = await asyncio.to_thread(self._memory.recall_long_term_memory, query)
                if ltm:
                    ctx.long_term_memory = ltm
                    logger.info(f"[ContextManager] LTM recalled: {len(ltm)} chars")
            except Exception as e:
                logger.debug(f"[ContextManager] LTM recall skipped: {e}")
            try:
                cards = await asyncio.to_thread(self._memory.get_resident_cards)
                if cards:
                    ctx.resident_cards = cards
            except Exception as e:
                logger.debug(f"[ContextManager] resident cards skipped: {e}")

        suggestions, format_hint = await self._generate_hints(query)

        if suggestions:
            ctx.search_suggestions = suggestions
        if format_hint:
            ctx.format_hint = format_hint

        return ctx

    async def _build_send_view(
        self,
        session_id: str,
        raw_messages: list,
        row_ids: list,
        query: str,
    ) -> tuple[list, bool]:
        """v8.4: 基于 checkpoint 构建发送视图（用户轮边界，压缩只在此时发生）。

        返回 (视图消息, 是否触发压缩)。视图不写回 SQLite；压缩发生时
        通过 session.set_checkpoint 持久化摘要位置（原始轨迹永不改写）。
        """
        if not raw_messages:
            return [], False
        if self._budget is None:
            return list(raw_messages), False

        checkpoint = {}
        if self._session:
            try:
                checkpoint = self._session.get_checkpoint(session_id)
            except Exception as e:
                logger.debug(f"[ContextManager] get_checkpoint failed: {e}")

        cp_id = int(checkpoint.get("msg_id", 0) or 0)
        cp_summary = checkpoint.get("summary", "") or ""

        view_msgs: list = [raw_messages[0]]
        view_ids: list = [row_ids[0]]
        if cp_summary and cp_id:
            from langchain_core.messages import HumanMessage
            view_msgs.append(HumanMessage(
                content=f"<conversation_summary>\n{cp_summary}\n</conversation_summary>"))
            view_ids.append(0)   # 合成摘要无原始 row id
        for m, mid in zip(raw_messages[1:], row_ids[1:]):
            if mid > cp_id:
                view_msgs.append(m)
                view_ids.append(mid)

        if self._budget and view_msgs:
            try:
                result = await self._budget.check(
                    view_msgs, query=query, ids=view_ids,
                    session_id=session_id, prior_summary=cp_summary)
                if result.level != ContextBudgetLevel.NORMAL:
                    logger.info(
                        f"[ContextManager] budget level={result.level.value}, "
                        f"summary={len(result.summary or '')} chars, "
                        f"cutoff_id={result.cutoff_id}"
                    )
                    # v8.4: 持久化 checkpoint（原始轨迹永不改写，只记录摘要位置）
                    if (self._session and result.cutoff_id is not None
                            and result.summary):
                        try:
                            self._session.set_checkpoint(
                                session_id, result.cutoff_id, result.summary)
                        except Exception as e:
                            logger.warning(
                                f"[ContextManager] persist checkpoint failed: {e}")
                    # 摘要已嵌入视图（<conversation_summary> 消息），
                    # 不再经 build_human_message 二次注入
                    return result.messages, True
            except Exception as e:
                logger.warning(f"[ContextManager] budget check failed: {e}")

        return view_msgs, False

    async def _generate_hints(
        self, query: str
    ) -> tuple[list[str], str | None]:
        async def gen_suggestions() -> list[str]:
            try:
                llm = self._get_fast_llm()
                resp = await llm.ainvoke([
                    SystemMessage(content=(
                        "Extract 2-3 English search angles (5-15 keywords each) "
                        "from the user's query. Output a JSON array. "
                        "If the input is a greeting or chitchat, output []. "
                        'Format: ["keywords...", "keywords..."]'
                    )),
                    HumanMessage(content=query),
                ])
                import json, re
                content = resp.content.strip()
                content = re.sub(r"```\w*\n?", "", content)
                content = re.sub(r"\n```", "", content)
                data = json.loads(content)
                if isinstance(data, list):
                    return [s for s in data[:3] if s and isinstance(s, str)]
                return []
            except Exception as e:
                logger.debug(f"[ContextManager] suggestions gen failed: {e}")
                return []

        async def gen_format_hint() -> str | None:
            try:
                llm = self._get_fast_llm()
                resp = await llm.ainvoke([
                    SystemMessage(content=(
                        "Classify output format. Output exactly ONE word:\n"
                        "fact / compare / method / design / review / task / fallback\n"
                        "fact: factual query | compare: comparison | method: mechanism/pathway\n"
                        "design: experiment design | review: paper/review writing | task: file ops\n"
                        "fallback: uncertain or chitchat"
                    )),
                    HumanMessage(content=query),
                ])
                hint = resp.content.strip().lower()
                valid = {"fact", "compare", "method", "design", "review", "task", "fallback"}
                return hint if hint in valid else "fallback"
            except Exception as e:
                logger.debug(f"[ContextManager] format_hint gen failed: {e}")
                return None

        s_task = asyncio.create_task(gen_suggestions())
        f_task = asyncio.create_task(gen_format_hint())
        suggestions = await s_task
        format_hint = await f_task
        return suggestions, format_hint

    def build_human_message(
        self,
        ctx: LoadedContext,
        *,
        retrieval_context: str | None = None,
    ) -> HumanMessage:
        return build_human_message(ctx, retrieval_context=retrieval_context)


def build_human_message(
    ctx: LoadedContext,
    *,
    retrieval_context: str | None = None,
) -> HumanMessage:
    """Standalone function to build a standard HumanMessage from a LoadedContext."""
    blocks: list[str] = []

    if ctx.long_term_memory:
        blocks.append(
            f"<long_term_memory>\n{ctx.long_term_memory}\n"
            f"</long_term_memory>"
        )

    if ctx.resident_cards:
        blocks.append(
            f"<resident_cards>\n{ctx.resident_cards}\n"
            f"</resident_cards>"
        )

    if ctx.search_suggestions:
        items = "\n".join(f"- {s}" for s in ctx.search_suggestions[:3])
        blocks.append(
            f"<search_suggestions>\n"
            f"System-generated search angle suggestions. "
            f"You may adopt, modify, or ignore them:\n"
            f"{items}\n"
            f"</search_suggestions>"
        )

    if ctx.format_hint:
        blocks.append(
            f"<format_hint>\n"
            f"Suggested output format: {ctx.format_hint}. "
            f"You may override this.\n"
            f"</format_hint>"
        )

    if retrieval_context:
        blocks.append(
            f"<retrieval_context>\n{retrieval_context}\n"
            f"</retrieval_context>"
        )

    blocks.append(f"<user_query>\n{ctx.query}\n</user_query>")

    # 阶段1 静态前缀: format 指南/策略卡片等动态内容追加到末尾
    # (不在 SystemMessage 中 → 前缀字节级稳定，缓存跨请求命中)
    try:
        from src.prompts.loader import build_dynamic_blocks
        dynamic = build_dynamic_blocks(
            format_hint=ctx.format_hint,
            query=ctx.query,
        )
        if dynamic:
            blocks.append(dynamic)
    except Exception:
        pass

    return HumanMessage(content="\n\n".join(blocks))


def finalize_load_result(
    ctx: LoadedContext,
    *,
    session_manager,
    session_id: str,
    budget,
    node_label: str,
    log_prefix: str,
) -> dict:
    """v8.4: expert/light 两图共用的 load 节点收尾装配（消除重复实现）。

    组装 state result dict（消息/摘要/记忆/证据块/_trace）+ 推送 context_status。
    """
    result: dict = {
        "session_id": ctx.session_id,
        "history_summary": ctx.history_summary,
        "compacted": ctx.compacted,
        "long_term_memory": ctx.long_term_memory,
        "resident_cards": ctx.resident_cards,
        "search_suggestions": ctx.search_suggestions,
        "format_hint": ctx.format_hint,
    }

    history_msgs = list(ctx.history_messages) if ctx.history_messages else []
    if history_msgs:
        result["messages"] = history_msgs

    # 历史检索证据块（跨轮复用）——supervisor 装配时注入
    try:
        block = session_manager.build_evidence_block(session_id, limit=2)
        if block:
            result["history_evidence_block"] = block
    except Exception as e:
        logger.warning(f"[{log_prefix}:load] evidence block failed: {e}")

    load_summary = f"history={len(history_msgs)}msgs"
    if result.get("history_evidence_block"):
        load_summary += ", evidence_block=yes"
    if ctx.format_hint:
        load_summary += f", fmt={ctx.format_hint}"
    if ctx.compacted:
        load_summary += ", compacted=yes"
    result["_trace"] = {"node": node_label, "elapsed_ms": 0, "summary": load_summary}

    hist_chars = sum(len(m.content or "") for m in history_msgs if hasattr(m, 'content'))
    est_tokens = budget.estimate_tokens(history_msgs) if budget is not None else 0
    try:
        from src.core.progress_bus import emit_encoded
        emit_encoded("context_status", {
            "history_msgs": len(history_msgs),
            "history_chars": hist_chars,
            "ltm_recalled": bool(ctx.long_term_memory),
            "ltm_chars": len(ctx.long_term_memory or ""),
            "resident_cards": bool(ctx.resident_cards),
            "suggestions": ctx.search_suggestions[:3] if ctx.search_suggestions else [],
            "format_hint": ctx.format_hint or "",
            "estimated_tokens": est_tokens,
            "max_tokens": budget.config.max_tokens if budget is not None else 0,
            "soft_threshold": budget.config.soft_threshold if budget is not None else 0,
            "hard_threshold": budget.config.hard_threshold if budget is not None else 0,
            "compressed": bool(ctx.compacted),
            "compression_len": 0,
        })
    except Exception:
        pass
    return result
