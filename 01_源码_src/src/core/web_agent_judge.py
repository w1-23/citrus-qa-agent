# -*- coding: utf-8 -*-
"""Web-Agent 自评（相关性 + 充分性）—— v10.7 多智能体联网架构。

**为什么单独成模块**：新架构把"搜几轮、要不要换策略"从代码硬编码改成
**模型语义判断 + 代码约束**（计划 §2.4）。本模块只负责**判断**，
轮次/预算/上限由 `web_agent_loop` 强制——语义归模型、约束归代码。

判定口径（与 `prompts/source/21_web_agent_search.md` 一致）：

* ``relevant``   ：结果是否与指令要查的问题相关（主题一致，不是答非所问/广告导购）；
* ``sufficient`` ：结果是否**足以回答**指令里的问题（覆盖关键子问题、含可用具体信息）；
* ``next_strategy``：不足时给出**与上一轮有实质差异**的检索角度（换关键词/粒度/来源/语言）。

**fail-open 取向**：判断器异常时返回 ``relevant=True, sufficient=True``（＝停止），
即退化为"只搜一轮"的旧行为——宁可少搜一轮，也不因为判断器坏了而多烧一次联网预算。

**看不到会话全文**：本模块的输入只有"指令 + 本轮结果"（Web Agent 自己的上下文），
符合计划 §1 决策 6。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Awaitable, Callable, Optional

from src.config import settings
from src.core.llm_pool import get_llm

logger = logging.getLogger(__name__)

_CACHE: dict[str, dict] = {}
_CACHE_MAX = 256

JUDGE_SYSTEM = """你是联网检索结果的裁决器。给定【检索指令】与【本轮检索结果】，只输出 JSON：
{"relevant": true, "sufficient": false, "reason": "一句话理由", "next_strategy": "下一轮该怎么搜"}

判定标准：
- relevant：结果是否与指令要查的问题相关（主题一致；答非所问、纯广告导购、企业宣传 = false）。
- sufficient：结果是否**足以回答**指令里的问题——覆盖了关键子问题、且含可用的具体信息
  （数字、年份、口径、结论）。只是"提到过这个主题"不算充分。
- 不确定时：relevant 取 true（宁可再搜一轮也别漏），sufficient 取 false。
- next_strategy：若 sufficient 为 false，给出**与上一轮有实质差异**的检索角度
  （换关键词/同义词/学名、换粒度、换来源取向、换语言或改写成数据查询的说法）；
  若已充分，给空串 ""。
只输出 JSON，不要解释、不要复述结果。"""


def _view(items: list, limit: int = 160, max_items: int = 12) -> str:
    """把本轮结果压成可判读的短视图（标题 + 摘要片段 + URL 域名）。"""
    out = []
    for i, it in enumerate((items or [])[:max_items], 1):
        title = str(it.get("title") or it.get("name") or "")
        body = str(it.get("snippet") or it.get("abstract") or it.get("content")
                   or it.get("text") or "")
        body = re.sub(r"\s+", " ", body).strip()
        url = str(it.get("url") or "")
        host = url.split("//", 1)[-1].split("/", 1)[0] if url else ""
        out.append(f"[{i}] {title[:90]} :: {body[:limit]}"
                   + (f" ({host})" if host else ""))
    return "\n".join(out)


def _cache_key(instruction: str, views: str, round_no: int) -> str:
    raw = f"{round_no}||{(instruction or '')[:200]}||{views[:400]}"
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()


def _extract_json(text: str) -> dict:
    """从模型输出里取 JSON；解析失败返回 {}（调用方 fail-open）。"""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {}
    raw = m.group(0)
    for candidate in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):
        try:
            d = json.loads(candidate)
            return d if isinstance(d, dict) else {}
        except Exception:                                      # noqa: BLE001
            continue
    return {}


def _as_bool(v, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1", "是", "相关", "充分"):
            return True
        if s in ("false", "no", "0", "否", "不相关", "不充分"):
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return default


async def judge_round(instruction: str, items: list, summary: str = "", *,
                      round_no: int = 1,
                      llm_call: Optional[Callable[[list], Awaitable[str]]] = None,
                      timeout: float = 20.0) -> dict:
    """判断本轮结果的相关性与充分性。**fail-open：异常 → 停止（充分）。**

    返回 ``{relevant, sufficient, reason, next_strategy, mode, n_items}``。
    """
    items = items or []
    views = _view(items)
    base = {"round": round_no, "n_items": len(items),
            "relevant": True, "sufficient": True,
            "reason": "", "next_strategy": "", "mode": "noop"}

    # 无结果 = 不相关且不充分（交给轮次控制决定要不要再来一轮）
    if not items and not str(summary or "").strip():
        return {**base, "relevant": False, "sufficient": False,
                "reason": "本轮无检索结果", "mode": "empty"}

    ck = _cache_key(instruction, views, round_no)
    if ck in _CACHE:
        return {**_CACHE[ck], "mode": "cached"}

    user = (f"【检索指令】{instruction or '(未提供)'}\n\n"
            f"【第 {round_no} 轮结果】（{len(items)} 条）\n{views or '(无条目)'}")
    if str(summary or "").strip():
        user += f"\n\n【原生综述正文（截断）】\n{str(summary)[:1200]}"

    try:
        if llm_call is not None:
            raw = await llm_call([
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ])
            text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
        else:
            llm = get_llm(
                settings.RESOLVED_FAST_MODEL, settings.RESOLVED_FAST_API_KEY,
                settings.RESOLVED_FAST_BASE_URL or settings.RESOLVED_MAIN_BASE_URL,
                temperature=0.0, timeout=timeout, max_tokens=300, thinking_off=True)
            resp = await llm.ainvoke([
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user},
            ])
            text = getattr(resp, "content", None) or str(resp)

        d = _extract_json(text)
        if not d:
            logger.warning("[WebJudge] 输出无法解析为 JSON（fail-open → 停止）")
            return {**base, "mode": "parse-fail"}
        out = {
            "round": round_no, "n_items": len(items),
            "relevant": _as_bool(d.get("relevant"), True),
            "sufficient": _as_bool(d.get("sufficient"), True),
            "reason": str(d.get("reason") or "")[:200],
            "next_strategy": str(d.get("next_strategy") or "")[:300],
            "mode": "llm",
        }
        if len(_CACHE) < _CACHE_MAX:
            _CACHE[ck] = {k: v for k, v in out.items() if k != "mode"}
        logger.info(f"[WebJudge] round{round_no}: relevant={out['relevant']} "
                    f"sufficient={out['sufficient']} reason={out['reason'][:60]}")
        return out
    except Exception as e:                                     # noqa: BLE001
        logger.warning(f"[WebJudge] fail-open（{type(e).__name__}: {e}）→ 停止")
        return {**base, "mode": "fail-open"}
