"""Context Budget — 发送视图构建与用户轮边界批量压缩 (v8.4).

架构（存储全量·发送裁剪，书 2.7 压缩与 KV Cache 互补）:
  - SQLite 原始轨迹 append-only，永不改写（压缩结果绝不写回历史）
  - check() 只在"用户轮边界"（新请求 load 时）调用——循环内绝不压缩，
    否则前缀中途变化，缓存必不命中（KV Cache 铁律）
  - 视图 ≥ soft 阈值 → 一次性批量压缩至 ~target_ratio（不逐步挤牙膏：
    每次压缩=一次缓存破坏，批量=只破坏一次）
  - 保护名单: 最近 protect_recent_turns 轮 Q/A 原样保留；被引用证据全文、
    DOI/evidence_id/artifact_id 标识符由压缩提示词保证保留（压缩只针对
    旧工具过程、错误日志、低相关证据）
  - checkpoint: 会话表记录"已摘要至哪条消息 + 摘要文本"，跨请求增量复用，
    防摘要套摘要
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ContextBudgetLevel(Enum):
    NORMAL = "normal"
    SUMMARIZE = "summarize"
    TRUNCATE = "truncate"


@dataclass
class ContextBudgetConfig:
    max_tokens: int = 512000          # 发送视图预算
    soft_threshold: float = 0.75      # ≥75% 批量压缩（一次压到 ~50%）
    hard_threshold: float = 0.93      # ≥93% 规则式保护截断
    target_ratio: float = 0.50        # 批量压缩目标：压到预算 ~50%
    protect_recent_turns: int = 3     # 保护名单：最近 N 轮 Q/A 不压缩
    keep_recent_turns: int = 2        # TRUNCATE 硬截断保留最近轮数
    summarize_ratio: float = 0.70
    compact_max_tokens: int = 800
    enabled: bool = True


@dataclass
class BudgetResult:
    level: ContextBudgetLevel
    messages: list
    summary: str | None = None
    cutoff_id: Optional[int] = None   # v8.4: 被摘要覆盖的最后一条原始消息 row id（checkpoint 定位）


def _estimate_chars_tokens(text: str) -> int:
    """混合语言 token 估算 (AG-15)。

    对齐 opencode 的字符估算思路（len/4，面向英文代码），本项目适配中英混合：
      - 中文/日文/韩文 CJK: ~1.2 token/字（DeepSeek BPE 实测区间 1.0-1.5）
      - 其他（英文/数字/符号）: 4 字符/token（与 opencode 一致）
    无外部 tokenizer 依赖，偏差目标 <15%。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(int(cjk * 1.2 + other / 4), 1)


# 压缩失败熔断器（书 2.7.4 L3 全量压缩熔断）: 连续失败 ≥3 次后降级为规则式，
# 避免在压缩失败的会话上持续烧钱。key=session_id（跨请求存活）
_compaction_failures: dict = {}


def _record_compaction_failure(session_id: str) -> bool:
    """返回 True 表示熔断（已连续失败 ≥3 次）。"""
    n = _compaction_failures.get(session_id, 0) + 1
    _compaction_failures[session_id] = n
    if n >= 3:
        logger.warning(
            f"[ContextBudget] 压缩熔断 (session={session_id[:8]}): "
            f"连续失败 {n} 次，降级为规则式截断")
        return True
    return False


def _reset_compaction_failures(session_id: str) -> None:
    _compaction_failures.pop(session_id, None)


def _keep_identifiers(content: str, max_chars: int = 2500) -> str:
    """规则式截断: 保留标识符行（DOI/evidence_id/artifact_id/引用编号），
    丢弃过程性文本。透明标记（书 2.7: 截断必须显式标注）。"""
    import re
    if not content:
        return content
    ident_lines: list[str] = []
    for line in content.splitlines():
        if re.search(
            r"\b(doi|evidence_id|artifact_id|chunk_id|source_id|pmid|url)\b",
            line, re.IGNORECASE) or re.match(r"^\s*\[?\d+\]?\s", line):
            ident_lines.append(line[:200])
    head = content[:max_chars]
    body = (f"{head}\n[TRUNCATED: 原始 {len(content)} 字符，仅保留前 {max_chars} 字符；"
            f"完整内容已被发送视图压缩规则截断]") if len(content) > max_chars else content
    if ident_lines:
        body += "\n[保留标识符]\n" + "\n".join(ident_lines[:20])
    return body


class ContextBudget:
    def __init__(self, config: ContextBudgetConfig | None = None) -> None:
        self.config = config or ContextBudgetConfig()
        self._compact_fn: Callable | None = None

    def set_compact_fn(self, fn: Callable) -> None:
        """fn(messages, query="") -> str 摘要文本。"""
        self._compact_fn = fn

    def estimate_tokens(self, messages: list) -> int:
        total = 0
        for msg in messages:
            content = ""
            if hasattr(msg, "content"):
                c = msg.content
                if isinstance(c, str):
                    content = c
                elif isinstance(c, list):
                    for block in c:
                        if isinstance(block, dict):
                            content += str(block.get("text", ""))
                else:
                    content = str(c) if c else ""
            total += _estimate_chars_tokens(content)

            tcs = getattr(msg, "tool_calls", None)
            if tcs:
                for tc in tcs:
                    args = getattr(tc, "args", {}) or {}
                    args_str = str(args) if args else ""
                    total += 30 + max(len(args_str) // 2, 1)
        return int(total)

    def _split_turns(self, items: list) -> list[list]:
        """按 user/human 消息边界切轮。items 为消息或 (msg, id) 元组，保留原序。"""
        turns = []
        current: list = []
        for item in items:
            msg = item[0] if isinstance(item, tuple) else item
            current.append(item)
            role = getattr(msg, "type", None) or getattr(msg, "role", "")
            if role in ("human", "user") and len(current) > 1:
                turns.append(current[:-1])
                current = [item]
        if current:
            turns.append(current)
        return turns

    def _trim_noise(self, turns: list) -> list:
        """噪声优先删除（占位/错误消息），证据与结论保护。

        删除 ToolMessage 时必须同步修剪配对 AIMessage.tool_calls（INV-01），
        否则历史重放触发 OpenAI 400。
        """
        NOISE_NAMES = {"budget_skip", "circuit_breaker"}
        NOISE_PREFIXES = ("[ERR_", "[circuit_breaker]", "[budget]",
                          "[ERR_TIMEOUT]", "[ERR_NETWORK]", "[ERR_HITL_REJECT]")

        def _msg(item):
            return item[0] if isinstance(item, tuple) else item

        def _is_noise_tool(m) -> bool:
            from langchain_core.messages import ToolMessage
            if not isinstance(m, ToolMessage):
                return False
            if getattr(m, "name", "") in NOISE_NAMES:
                return True
            content = str(getattr(m, "content", ""))
            return content.startswith(NOISE_PREFIXES)

        cleaned = []
        for turn in turns:
            drop_ids = {getattr(_msg(m), "tool_call_id", "")
                        for m in turn if _is_noise_tool(_msg(m))
                        and getattr(_msg(m), "tool_call_id", "")}
            if not drop_ids:
                cleaned.append(turn)
                continue
            kept = []
            for item in turn:
                m = _msg(item)
                if getattr(m, "tool_call_id", "") in drop_ids:
                    continue
                tcs = getattr(m, "tool_calls", None)
                if tcs:
                    filtered = [
                        tc for tc in tcs
                        if (tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", ""))
                        not in drop_ids
                    ]
                    if len(filtered) != len(tcs):
                        try:
                            m = m.model_copy(update={"tool_calls": filtered or None})
                            item = (m, item[1]) if isinstance(item, tuple) else m
                        except Exception:
                            pass
                kept.append(item)
            if kept:
                cleaned.append(kept)
        return cleaned

    async def check(
        self,
        messages: list,
        query: str = "",
        ids: Optional[list] = None,
        session_id: str = "",
        prior_summary: str = "",
    ) -> BudgetResult:
        """用户轮边界（新请求 load）调用：估算视图，超软阈值批量压缩。

        messages: 发送视图候选（可能已含 checkpoint 摘要）
        ids: 与 messages 一一对应的原始 row id（checkpoint 定位用；缺省用序号）
        prior_summary: checkpoint 已有摘要（增量压缩：作为 prior_summary 传给
            压缩 LLM，防摘要套摘要时丢旧信息）
        """
        if not self.config.enabled:
            return BudgetResult(level=ContextBudgetLevel.NORMAL, messages=list(messages))

        if ids is None or len(ids) != len(messages):
            ids = list(range(len(messages)))

        tokens = self.estimate_tokens(messages)
        max_t = self.config.max_tokens
        ratio = tokens / max_t if max_t > 0 else 0
        logger.debug(f"[ContextBudget] tokens={tokens}/{max_t} ({ratio:.1%})")

        if ratio < self.config.soft_threshold:
            return BudgetResult(level=ContextBudgetLevel.NORMAL, messages=list(messages))

        return await self._compress(messages, query, ids, session_id, prior_summary)

    async def _compress(
        self,
        messages: list,
        query: str,
        ids: list,
        session_id: str,
        prior_summary: str,
    ) -> BudgetResult:
        cfg = self.config
        paired = list(zip(messages, ids))
        turns = self._split_turns(paired)
        turns = self._trim_noise(turns)

        if len(turns) <= cfg.protect_recent_turns + 1:
            return BudgetResult(level=ContextBudgetLevel.NORMAL, messages=list(messages))

        protect_n = cfg.protect_recent_turns
        compressible = [item for turn in turns[:-protect_n] for item in turn]
        protected = [item for turn in turns[-protect_n:] for item in turn]

        anchor = paired[0] if paired else None
        # 压缩范围 = 除锚点首消息外的全部可压缩轮次
        compressible = [item for item in compressible if item is not anchor]

        if not compressible:
            return BudgetResult(level=ContextBudgetLevel.NORMAL, messages=list(messages))

        comp_msgs = [m for m, _ in compressible]
        cutoff_id = max(mid for _, mid in compressible)
        summary = await self._safe_compact(comp_msgs, query, session_id, prior_summary)

        from langchain_core.messages import HumanMessage
        new_messages = []
        if anchor:
            new_messages.append(anchor[0])
        if summary:
            new_messages.append(HumanMessage(
                content=f"<conversation_summary>\n{summary}\n</conversation_summary>"))
        new_messages.extend(m for m, _ in protected)

        est = self.estimate_tokens(new_messages)
        ratio = est / cfg.max_tokens if cfg.max_tokens > 0 else 0
        logger.info(
            f"[ContextBudget] 批量压缩: {len(comp_msgs) if compressible else 0} msgs -> "
            f"summary {len(summary)} chars; 视图 {len(new_messages)} msgs / {est} tok ({ratio:.1%})"
        )

        if ratio >= cfg.hard_threshold:
            new_messages = self._hard_trim(new_messages, cfg)
            return BudgetResult(
                level=ContextBudgetLevel.TRUNCATE,
                messages=new_messages,
                summary=summary,
                cutoff_id=cutoff_id,
            )
        return BudgetResult(
            level=ContextBudgetLevel.SUMMARIZE,
            messages=new_messages,
            summary=summary,
            cutoff_id=cutoff_id,
        )

    async def _safe_compact(self, messages: list, query: str, session_id: str,
                            prior_summary: str = "") -> str:
        """LLM 压缩 + 熔断器（连续失败 ≥3 → 规则式截断）。"""
        if not self._compact_fn:
            return self._rules_summary(messages)
        if session_id and _compaction_failures.get(session_id, 0) >= 3:
            return self._rules_summary(messages)
        try:
            summary = await self._compact_fn(
                messages, query=query, prior_summary=prior_summary)
            if session_id:
                _reset_compaction_failures(session_id)
            return summary or self._rules_summary(messages)
        except Exception as e:
            logger.warning(f"[ContextBudget] LLM 压缩失败: {e}")
            if session_id:
                _record_compaction_failure(session_id)
            return self._rules_summary(messages)

    def _rules_summary(self, messages: list) -> str:
        """规则式摘要（压缩 LLM 不可用/熔断时）: 保留标识符 + 最近几条要点。"""
        parts: list[str] = []
        for m in messages[-6:]:
            role = getattr(m, "type", "") or getattr(m, "role", "")
            content = str(getattr(m, "content", ""))[:300]
            if content:
                parts.append(f"[{role}]: {content}")
        return "（规则式摘要）\n" + "\n".join(parts)[:2000]

    def _hard_trim(self, messages: list, cfg: ContextBudgetConfig) -> list:
        """硬阈值兜底: 大工具输出只留标识符行，仍超限则缩到最近 keep_recent_turns 轮。"""
        from langchain_core.messages import ToolMessage
        trimmed = []
        for m in messages:
            if isinstance(m, ToolMessage) and len(str(m.content or "")) > 3000:
                try:
                    trimmed.append(m.model_copy(
                        update={"content": _keep_identifiers(str(m.content))}))
                    continue
                except Exception:
                    pass
            trimmed.append(m)
        if self.estimate_tokens(trimmed) / max(cfg.max_tokens, 1) < cfg.hard_threshold:
            return trimmed
        turns = self._split_turns(trimmed)
        keep = max(1, cfg.keep_recent_turns)
        if len(turns) <= keep + 1:
            return trimmed
        head = [turns[0][0]]
        tail = [item for turn in turns[-keep:] for item in turn]
        return head + tail
