"""证据裁决（卡点②）—— 生成前按用户确认意图丢弃无关证据 + 联网来源分级。

接入点：expert_graph._execute_supervisor_tools —— 每轮检索产物在写回
all_main_results / all_web_results（进而影响引用池与 write-agent 上下文）之前过一遍：

  1) 程序化（always，零成本）：web 条目经 source_tiers 域名分级，写入
     source_tier 字段（T3 = 低可信，下游标注/交叉要求；不在此处直接丢）；
  2) LLM（fast，每批一次）：按 intent_confirmed 判断相关性，丢弃答非所问/
     广告导购/无关背景；**上限 = min(3, n//2)**，不确定性一律保留；
  3) fail-open：模型/解析/超时任何异常 → 不丢任何东西（宁留噪声，不丢证据）。

产物 adjudication_log 进 state，随 audit_report 一起落日志/前端事件，可追溯。
用户决策对应：不做证据融合，按用户最初意图取舍；无用的不进入生成上下文。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Awaitable, Callable, Optional

from src.config import settings
from src.core.llm_pool import get_llm
from src.core.source_tiers import grade_web_items

logger = logging.getLogger(__name__)

_CACHE: dict[str, set[str]] = {}
_CACHE_MAX = 256

ADJ_SYSTEM = """你是检索结果相关性裁决器。给定【用户意图】与候选证据列表，只输出 JSON：
{"drop": ["w2", "m5"]}

裁决规则：
- 丢弃：主题与用户意图不同、答非所问、纯广告/导购/企业宣传、与问题无关的背景铺垫；
- 保留：与意图直接相关，或能提供必要背景/机制对照/数据佐证；
- 不确定的一律保留；最多丢弃 {max_drop} 条；
- 只输出 JSON，不要解释、不要复述证据。"""


def _view(key: str, item: dict, limit: int = 140) -> str:
    title = str(item.get("title") or item.get("name") or "")
    body = str(item.get("text") or item.get("abstract") or item.get("snippet")
               or item.get("text_preview") or "")
    body = re.sub(r"\s+", " ", body).strip()
    return f"[{key}] {title[:80]} :: {body[:limit]}"


def _cache_key(intent: str, views: list) -> str:
    raw = (intent or "")[:200] + "||" + "||".join(v[:120] for v in views)
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()


def _extract_drops(text: str) -> set:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return set()
    try:
        data = json.loads(m.group(0))
    except Exception:
        try:
            data = json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except Exception:
            return set()
    drops = data.get("drop") if isinstance(data, dict) else None
    if not isinstance(drops, list):
        return set()
    return {str(d).strip().lower() for d in drops if str(d).strip()}


def _enforce_cap(drops: set, n_main: int, n_web: int) -> set:
    """上限 min(3, n//2)；且不把某一类清空（至少保留 1 条）。"""
    total = n_main + n_web
    cap = min(3, total // 2)
    out = set()
    dropped_main = dropped_web = 0
    for d in sorted(drops):
        if len(out) >= cap:
            break
        if d.startswith("m"):
            if n_main - dropped_main <= 1:
                continue
            dropped_main += 1
        elif d.startswith("w"):
            if n_web - dropped_web <= 1:
                continue
            dropped_web += 1
        else:
            continue
        out.add(d)
    return out


async def adjudicate_batch(intent: str, main_items: list, web_items: list, *,
                           turn: int = 0,
                           llm_call: Optional[Callable[[list], Awaitable[str]]] = None,
                           timeout: float = 20.0) -> dict:
    """单批裁决。返回 {dropped_keys, tier_counts, mode, ...}；fail-open。"""
    main_items = main_items or []
    # web 条目自动域名分级（幂等；T3 = 低可信标记，供下游引用/口径提示使用）
    web_items = grade_web_items(web_items or []) if web_items else []
    n_main, n_web = len(main_items), len(web_items)
    tier_counts: dict = {}
    if web_items:
        for it in web_items:
            t = str(it.get("source_tier") or "T3")
            tier_counts[t] = tier_counts.get(t, 0) + 1

    base = {"turn": turn, "n_main": n_main, "n_web": n_web,
            "tier_counts": tier_counts, "dropped_keys": [], "mode": "noop"}
    if n_main + n_web == 0:
        return base

    views = ([_view(f"m{i}", it) for i, it in enumerate(main_items)]
             + [_view(f"w{i}", it) for i, it in enumerate(web_items)])
    max_drop = min(3, (n_main + n_web) // 2)
    if max_drop <= 0:
        return base

    ck = _cache_key(intent or "", views)
    if ck in _CACHE:
        return {**base, "dropped_keys": sorted(_CACHE[ck]), "mode": "cached"}

    try:
        drops: set = set()
        if llm_call is not None:
            raw = await llm_call([
                {"role": "system",
                 "content": ADJ_SYSTEM.replace("{max_drop}", str(max_drop))},
                {"role": "user",
                 "content": f"【用户意图】{intent or '(未提供)'}\n\n【候选证据】\n"
                            + "\n".join(views)},
            ])
            drops = _extract_drops(raw if isinstance(raw, str)
                                   else getattr(raw, "content", str(raw)))
        else:
            llm = get_llm(
                settings.RESOLVED_FAST_MODEL, settings.RESOLVED_FAST_API_KEY,
                settings.RESOLVED_FAST_BASE_URL or settings.RESOLVED_MAIN_BASE_URL,
                temperature=0.0, timeout=timeout, max_tokens=200, thinking_off=True)
            resp = await llm.ainvoke([
                {"role": "system",
                 "content": ADJ_SYSTEM.replace("{max_drop}", str(max_drop))},
                {"role": "user",
                 "content": f"【用户意图】{intent or '(未提供)'}\n\n【候选证据】\n"
                            + "\n".join(views)},
            ])
            drops = _extract_drops(getattr(resp, "content", None) or str(resp))
        drops = _enforce_cap(drops, n_main, n_web)
        if len(_CACHE) < _CACHE_MAX:
            _CACHE[ck] = drops
        return {**base, "dropped_keys": sorted(drops),
                "mode": "llm" if drops else "llm-keep-all"}
    except Exception as e:
        logger.warning(f"[Adjudicate] fail-open ({type(e).__name__}: {e})")
        return {**base, "mode": "fail-open"}


def split_by_drops(main_items: list, web_items: list, drops: set):
    """按 m{i}/w{i} 键拆分保留项（保持原顺序）。"""
    kept_main = [it for i, it in enumerate(main_items or []) if f"m{i}" not in drops]
    kept_web = [it for i, it in enumerate(web_items or []) if f"w{i}" not in drops]
    return kept_main, kept_web
