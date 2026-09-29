# -*- coding: utf-8 -*-
"""Web-Agent 自主检索循环 —— v10.7 多智能体联网架构（最多 2 轮）。

分工（计划 §2.4「语义决策由模型，约束由代码」）：

| 决策 | 谁决定 |
|---|---|
| 搜什么 | Supervisor（模型，转成明确指令） |
| 相关性 / 充分性 | Web Agent（模型，见 `web_agent_judge`） |
| 两轮之间换不换策略 | Web Agent（模型，judge 给 `next_strategy`） |
| **轮次上限** | **代码**（`web_agent.max_rounds`，默认 2） |
| **预算** | **代码**（每轮一次工具调用；工具内部另有请求级预算） |
| **输出分离** | **代码**（`web_output_separator`） |

停止条件（计划 §2.3）：
  第 1 轮相关且充分 → 立即停止；相关但不充分 / 不相关 → 进第 2 轮（可换策略）；
  第 2 轮后仍不充分 → 停止并标注「未充分检索」；仍不相关 → 停止并标注「未找到相关信息」。

**只用自己的上下文**：每轮 judge 只看「指令 + 本轮结果 + 上一轮查询」，
不读会话全文（计划 §1 决策 6）。

fail-open：judge 异常 → 视为充分并停止（退化为单轮，不烧多余预算）；
工具异常 → 记为 error 并停止（不重试）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from src.config import settings
from src.core.web_agent_judge import judge_round

logger = logging.getLogger(__name__)

#: 停止原因（写进审计记录，便于逐轮复核）
STOP_SUFFICIENT = "sufficient"
STOP_MAX_ROUNDS_INSUFFICIENT = "max_rounds_insufficient"
STOP_MAX_ROUNDS_IRRELEVANT = "max_rounds_irrelevant"
STOP_DISABLED = "disabled"
STOP_BUDGET = "budget"
STOP_ERROR = "error"
STOP_EMPTY = "empty"
STOP_JUDGE_FAIL_OPEN = "judge_fail_open"


def _dedupe(items: list) -> list:
    """按 URL（无 URL 时按标题）跨轮去重，保持首次出现顺序。"""
    out, seen = [], set()
    for it in items or []:
        if not isinstance(it, dict):
            continue
        key = str(it.get("url") or it.get("link") or it.get("title")
                  or it.get("name") or "").strip().lower().rstrip("/")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(it)
    return out


def _status_of(text: str, items: list, summary: str) -> str:
    t = str(text or "")
    if "[DISABLED]" in t:
        return "disabled"
    if "[WEB_BUDGET_EXHAUSTED]" in t:
        return "budget"
    if t.startswith("[ERR_") or "[AgentError]" in t or t.startswith("[Error"):
        return "error"
    if not items and not str(summary or "").strip():
        return "empty"
    return "ok"


def _round2_query(instruction: str, prev_query: str, next_strategy: str) -> str:
    """构造第 2 轮查询：优先用 judge 给的新策略，否则退化为"换个角度"的显式改写。

    刻意**不是**把同一句再问一遍（计划 §2.3 要求两轮有实质差异）。
    """
    ns = str(next_strategy or "").strip()
    if ns:
        # judge 给的是"该怎么搜"的描述；把它与原始指令拼成一条可执行的查询
        return f"{instruction}｜换个角度：{ns}"
    return (f"{instruction}｜上一轮结果不足。请改用与「{prev_query[:60]}」不同的检索角度："
            f"换关键词（同义词/学名/标识符）或换粒度（更具体的指标、年份、地区）。")


def _tag_round(items: list, rno: int) -> list:
    """给本轮联网条目打上 `round_no`（v10.13 · Task E）。

    用途：侧栏要"**一个 round 一个 web 侧栏**"分组显示，而条目本身原先不带轮次信息
    （`prepare_display_items` 只能默认全部算第 1 轮）。
    就地打标会污染 `artifact` 里别处引用同一 dict 的地方，故**浅拷贝**后再打。
    打标失败不影响检索（本函数绝不抛）。
    """
    out = []
    for it in (items or []):
        if isinstance(it, dict):
            try:
                d = dict(it)
                d.setdefault("round_no", rno)
                out.append(d)
            except Exception:                                  # noqa: BLE001
                out.append(it)
        else:
            out.append(it)
    return out


async def run_web_agent_loop(
    instruction: str,
    *,
    max_rounds: Optional[int] = None,
    self_judge: Optional[bool] = None,
    allow_strategy_switch: Optional[bool] = None,
    stop_when_sufficient: Optional[bool] = None,
    tool_call: Optional[Callable[[str], Awaitable[tuple]]] = None,
    judge_fn: Optional[Callable[..., Awaitable[dict]]] = None,
) -> dict:
    """跑 Web Agent 的自主检索循环，返回聚合结果 + 逐轮审计。

    ``tool_call(query) -> (text, artifact)``：默认走 `deepseek_web_search`；
    测试可注入桩。``judge_fn`` 默认 `judge_round`。
    """
    cfg = settings
    max_rounds = int(max_rounds if max_rounds is not None
                     else getattr(cfg, "WEB_AGENT_MAX_ROUNDS", 2) or 2)
    self_judge = bool(self_judge if self_judge is not None
                      else getattr(cfg, "WEB_AGENT_SELF_JUDGE", True))
    allow_switch = bool(allow_strategy_switch if allow_strategy_switch is not None
                        else getattr(cfg, "WEB_AGENT_ALLOW_STRATEGY_SWITCH", True))
    stop_suff = bool(stop_when_sufficient if stop_when_sufficient is not None
                     else getattr(cfg, "WEB_AGENT_STOP_WHEN_SUFFICIENT", True))
    max_rounds = max(1, min(max_rounds, 2))          # 代码硬上限：最多 2 轮
    _judge = judge_fn or judge_round

    if tool_call is None:
        from src.tools.deepseek_web import deepseek_web_search

        async def tool_call(q: str):                              # noqa: E306
            return await asyncio.to_thread(deepseek_web_search.func, q)

    rounds: list[dict] = []
    all_items: list = []
    all_summaries: list[str] = []
    #: 各轮工具**原始文本**。必须带出去：`search_both.run_web_agent` 的 `result_text`
    #: = 原始文本 + 联网侧回执，这是工具契约的一部分（回滚档下模型靠它拿到联网正文）。
    #: v10.8 修：上一版把 `text` 写成空串，导致 result_text 只剩回执——
    #: 回滚档模型拿不到任何联网正文（由 `_scratch/verify_tool_result_emit.py` 抓出）。
    raw_texts: list[str] = []
    query = str(instruction or "")
    prev_queries: list[str] = []
    stop_reason = STOP_EMPTY
    status = "ok"

    for rno in range(1, max_rounds + 1):
        try:
            text, artifact = await tool_call(query)
        except Exception as e:                                     # noqa: BLE001
            logger.error(f"[WebLoop] round{rno} 工具异常: {type(e).__name__}: {e}")
            rounds.append({"round": rno, "query": query, "n_items": 0,
                           "status": "error", "error": f"{type(e).__name__}: {e}",
                           "stop_reason": STOP_ERROR})
            status, stop_reason = "error", STOP_ERROR
            break

        artifact = artifact if isinstance(artifact, dict) else {}
        items = list(artifact.get("web_results") or [])
        summary = str(artifact.get("web_summary") or "").strip()
        _raw = str(text or "").strip()
        if _raw:
            raw_texts.append(_raw)
        status = _status_of(text, items, summary)

        rec = {"round": rno, "query": query, "n_items": len(items),
               "summary_chars": len(summary), "status": status,
               "relevant": None, "sufficient": None, "reason": "",
               "strategy_switch": None, "stop_reason": None, "judge_mode": None}

        # 工具层短路：开关关闭 / 预算用尽 / 网络错误 → 不再judge、不再下一轮
        if status in ("disabled", "budget", "error"):
            rec["stop_reason"] = {"disabled": STOP_DISABLED, "budget": STOP_BUDGET,
                                  "error": STOP_ERROR}[status]
            rounds.append(rec)
            stop_reason = rec["stop_reason"]
            all_items.extend(_tag_round(items, rno))
            if summary:
                all_summaries.append(summary)
            break

        all_items.extend(_tag_round(items, rno))
        if summary:
            all_summaries.append(summary)

        # 自评（可关：self_judge=false → 退化为"只搜一轮"）
        if not self_judge:
            rec.update({"judge_mode": "off", "stop_reason": STOP_SUFFICIENT,
                        "relevant": True, "sufficient": True})
            rounds.append(rec)
            stop_reason = STOP_SUFFICIENT
            break

        j = await _judge(str(instruction or ""), items, summary,
                         round_no=rno)
        rec.update({"relevant": j.get("relevant"), "sufficient": j.get("sufficient"),
                    "reason": j.get("reason") or "", "judge_mode": j.get("mode")})

        if j.get("mode") == "fail-open":
            rec["stop_reason"] = STOP_JUDGE_FAIL_OPEN
            rounds.append(rec)
            stop_reason = STOP_JUDGE_FAIL_OPEN
            break

        # 第 1 轮充分 → 立即停止（计划 §2.3 第一行）
        if stop_suff and j.get("relevant") and j.get("sufficient"):
            rec["stop_reason"] = STOP_SUFFICIENT
            rounds.append(rec)
            stop_reason = STOP_SUFFICIENT
            break

        # 还有轮次 → 决定第 2 轮查询（可换策略）
        if rno < max_rounds:
            nxt = _round2_query(str(instruction or ""), query,
                                j.get("next_strategy") if allow_switch else "")
            rec["strategy_switch"] = (nxt != query)
            prev_queries.append(query)
            query = nxt
            rounds.append(rec)
            continue

        # 用尽轮次 → 停止并标注（计划 §2.3 最后两行）
        if j.get("relevant"):
            rec["stop_reason"] = STOP_MAX_ROUNDS_INSUFFICIENT
            stop_reason = STOP_MAX_ROUNDS_INSUFFICIENT
        else:
            rec["stop_reason"] = STOP_MAX_ROUNDS_IRRELEVANT
            stop_reason = STOP_MAX_ROUNDS_IRRELEVANT
        rounds.append(rec)

    # 停止标注（给回执/审计用；不写进给用户的联网小节——那由分离器决定）
    note = ""
    if stop_reason == STOP_MAX_ROUNDS_INSUFFICIENT:
        note = "未充分检索"
    elif stop_reason == STOP_MAX_ROUNDS_IRRELEVANT:
        note = "未找到相关信息"
    elif stop_reason == STOP_EMPTY:
        note = "联网未返回结果"

    _deduped = _dedupe(all_items)
    # ── v10.13（阶段 0 审计发现）：`web_results` 必须保持**工具原始返回的语义** ──
    #   上一轮 `_tag_round` 就地给条目加了 `round_no`，而 `_dedupe(all_items)` 把这些
    #   打过标的对象直接当成 `web_results` 透出 ⇒ 破坏了 `test_v91_web_agent.py` 的
    #   "artifacts 透传"断言（按内容比较），回归从 248/0 掉到 **246/2**（审计实测）。
    #   正解：**两个视图分开** ——
    #     · `web_results`        = 原始条目（剥掉展示层字段）→ 保持工具契约不变
    #     · `web_results_round`  = 带 `round_no` 的副本 → 仅供侧栏分组（计划新增项 7）
    _ROUND_KEY = "round_no"
    web_results_clean = []
    web_results_round = []
    for _it in _deduped:
        if isinstance(_it, dict) and _ROUND_KEY in _it:
            _c = {k: v for k, v in _it.items() if k != _ROUND_KEY}
            web_results_clean.append(_c)
            web_results_round.append(dict(_it))
        else:
            web_results_clean.append(_it)
            if isinstance(_it, dict):
                _d = dict(_it)
                _d.setdefault(_ROUND_KEY, 1)
                web_results_round.append(_d)
            else:
                web_results_round.append(_it)

    return {
        "instruction": str(instruction or ""),
        "rounds": rounds,
        "rounds_used": len(rounds),
        "web_results": web_results_clean,          # 工具原始语义（不含 round_no）
        "web_results_round": web_results_round,    # 侧栏分组用（含 round_no）
        "web_summaries": all_summaries,
        # 各轮工具原始文本（拼接）；`run_web_agent` 用它组装 result_text（工具契约）
        "raw_text": "\n\n".join(raw_texts),
        "status": status,
        "stop_reason": stop_reason,
        "note": note,
        "strategy_switched": any(r.get("strategy_switch") for r in rounds),
        "queries": [r.get("query") for r in rounds],
    }
