"""Context Manager — 统一上下文加载与组装.

v8.1.1: 所有 LLM 调用前通过 ContextManager 获取上下文.
  - 加载完整 messages (不自动压缩)
  - 召回长期记忆
  - 预生成 search_suggestions + format_hint
  - 组装标准 HumanMessage
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
    search_suggestions: list[str] = field(default_factory=list)
    format_hint: str | None = None

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
            async def _compact(msgs):
                return await compact_messages(msgs, fast_llm=self._get_compact_llm())
            budget.set_compact_fn(_compact)

    def _get_fast_llm(self):
        if self._fast_llm is None:
            from langchain_openai import ChatOpenAI
            self._fast_llm = ChatOpenAI(
                model=settings.FAST_MODEL,
                api_key=settings.RESOLVED_FAST_API_KEY,
                base_url=settings.RESOLVED_FAST_BASE_URL,
                temperature=0,
                timeout=12,
            )
        return self._fast_llm

    def _get_compact_llm(self):
        """压缩用 main 模型：触发频率低，摘要质量优先于成本 (v8.3.1)."""
        if self._compact_llm is None:
            from langchain_openai import ChatOpenAI
            self._compact_llm = ChatOpenAI(
                model=settings.MAIN_MODEL,
                api_key=settings.RESOLVED_MAIN_API_KEY,
                base_url=settings.MAIN_BASE_URL,
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
                raw_messages = await self._session.get_messages(session_id)
                ctx.history_messages = raw_messages
                logger.info(
                    f"[ContextManager] loaded {len(raw_messages)} messages "
                    f"for session {session_id[:8]}"
                )
            except Exception as e:
                logger.warning(f"[ContextManager] load history failed: {e}")
                # AG-14: 历史加载失败不再静默 — 推送 context_degraded 供前端警示
                try:
                    from src.core.progress_bus import emit_encoded
                    emit_encoded("context_degraded", {"reason": "history_unavailable"})
                except Exception:
                    pass

        if self._budget and ctx.history_messages:
            try:
                result = await self._budget.check(ctx.history_messages)
                ctx.history_messages = result.messages
                if result.level != ContextBudgetLevel.NORMAL:
                    ctx.history_summary = result.summary
                    logger.info(
                        f"[ContextManager] budget level={result.level.value}, "
                        f"summary={len(result.summary or '')} chars"
                    )
                    # 压缩结果持久化 (AG-6): 事务内替换历史，避免每轮重复压缩
                    if self._session:
                        try:
                            await self._session.replace_history(session_id, result.messages)
                        except Exception as e:
                            logger.warning(f"[ContextManager] persist compaction failed: {e}")
            except Exception as e:
                logger.warning(f"[ContextManager] budget check failed: {e}")

        if self._memory:
            try:
                ltm = self._memory.recall_long_term_memory(query)
                if ltm:
                    ctx.long_term_memory = ltm
                    logger.info(f"[ContextManager] LTM recalled: {len(ltm)} chars")
            except Exception as e:
                logger.debug(f"[ContextManager] LTM recall skipped: {e}")

        suggestions, format_hint = await self._generate_hints(query)

        if suggestions:
            ctx.search_suggestions = suggestions
        if format_hint:
            ctx.format_hint = format_hint

        return ctx

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

    if ctx.history_summary:
        blocks.append(
            f"<conversation_summary>\n{ctx.history_summary}\n"
            f"</conversation_summary>"
        )

    if ctx.long_term_memory:
        blocks.append(
            f"<long_term_memory>\n{ctx.long_term_memory}\n"
            f"</long_term_memory>"
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

    return HumanMessage(content="\n\n".join(blocks))
