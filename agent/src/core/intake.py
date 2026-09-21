"""Intake（卡点①）—— 用户中文提问后的「一次」意图判定 + 条件 HITL 澄清卡。

流程（对应 v10 设计讨论）：
    用户中文提问
      → judge_intent()            fast 模型判一次，输出结构化 IntentProfile
      → decide_card()             有歧义且答错代价高 → 弹卡；否则 interpretation[0] 直接生效
      → [前端卡片]                2~3 个中文候选理解（统计题带口径区分）+ 自填，超时按推荐项放行
      → resolve_choice()          产出 intent_confirmed（中文）
      → 其后英文改写/检索/回答全部以 intent_confirmed 为锚点

设计约束：
  - 只调一次 fast 模型（temperature=0，max_tokens 受限），失败返回 None 由调用方放行原问题；
  - 不阻塞其它请求：挂起表在 API 层，per-request 独立；
  - 开关：环境变量 CITRUS_INTENT_CONFIRM=0 关闭；CITRUS_INTENT_WAIT_SEC 调整等待秒数。
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from src.config import settings
from src.core.llm_pool import get_llm

logger = logging.getLogger(__name__)

INTENT_ENABLED = os.getenv("CITRUS_INTENT_CONFIRM", "1").strip().lower() not in ("0", "false", "no")
INTENT_WAIT_SEC = int(os.getenv("CITRUS_INTENT_WAIT_SEC", "30"))
# 弹卡阈值：歧义分达到该值且存在多个实质不同理解时才弹（宁少勿多，避免拖慢首响）
AMBIGUITY_THRESHOLD = float(os.getenv("CITRUS_INTENT_AMBIGUITY", "0.55"))

INTENT_SYSTEM = """你是柑橘科研问答系统的「意图确认」模块。用户在中文提问后，你先做一次判断，只输出 JSON：
{
 "intent_summary": "一句话复述你理解的用户意图（中文）",
 "recency_required": true/false,               // 是否明确要求最新/近年/实时/行情数据
 "deliverable_format": "text|table|checklist|diagram",
 "ambiguity_score": 0.0,                        // 0~1，不同理解会导致不同答案的程度
 "clarify_reason": "",                          // 为什么有歧义；无歧义留空
 "needs_clarification": false,
 "interpretations": [                           // 1~3 个候选理解；无歧义时只给 1 个
   {"id": 1, "text": "候选理解（中文，可直接作为检索目标）",
    "caliber": "",                              // 统计口径说明（仅统计/比例类问题填写）
    "evidence_plan": "该理解对应的取证方向（一句话）"}
 ]
}
判断规则：
- 只有「不同理解会导致不同证据、不同答案」才算歧义；措辞模糊但答案一致的，不弹卡。
- 统计/比例/数量对比类问题：候选理解必须按统计口径区分（如「比例」要区分 实生选种 / 杂交育种 / 芽变选种 口径），caliber 字段写清口径。
- 上下文或前文已明确的对象，不算歧义。
- 宁少勿多：不确定时倾向 needs_clarification=false。"""


def _extract_json(text: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        try:  # 容错：去掉尾随逗号
            return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except Exception:
            return None


def _normalize(profile: dict, query: str) -> dict:
    """结构规范化：缺字段补默认值，interpretations 保证 ≥1 且 id 连续。"""
    out = {
        "intent_summary": str(profile.get("intent_summary") or query)[:300],
        "recency_required": bool(profile.get("recency_required")),
        "deliverable_format": str(profile.get("deliverable_format") or "text"),
        "ambiguity_score": 0.0,
        "clarify_reason": str(profile.get("clarify_reason") or "")[:300],
        "needs_clarification": bool(profile.get("needs_clarification")),
        "interpretations": [],
    }
    try:
        out["ambiguity_score"] = max(0.0, min(1.0, float(profile.get("ambiguity_score") or 0.0)))
    except (TypeError, ValueError):
        out["ambiguity_score"] = 0.0
    for i, it in enumerate((profile.get("interpretations") or [])[:3]):
        if not isinstance(it, dict):
            continue
        out["interpretations"].append({
            "id": i + 1,
            "text": str(it.get("text") or "").strip()[:400],
            "caliber": str(it.get("caliber") or "").strip()[:120],
            "evidence_plan": str(it.get("evidence_plan") or "").strip()[:200],
        })
    if not out["interpretations"]:
        out["interpretations"] = [{"id": 1, "text": query[:400],
                                   "caliber": "", "evidence_plan": ""}]
    return out


async def judge_intent(query: str, context: str = "") -> Optional[dict]:
    """fast 模型一次判定；任何异常返回 None（调用方放行原问题，不阻断主流程）。"""
    if not INTENT_ENABLED or not (query or "").strip():
        return None
    try:
        llm = get_llm(
            settings.RESOLVED_FAST_MODEL,
            settings.RESOLVED_FAST_API_KEY,
            settings.RESOLVED_FAST_BASE_URL or settings.RESOLVED_MAIN_BASE_URL,
            temperature=0.0, timeout=30, max_tokens=700, thinking_off=True,
        )
        user = ((f"【最近对话】\n{context}\n\n" if context else "")
                + f"【用户提问】{query}")
        resp = await llm.ainvoke(
            [{"role": "system", "content": INTENT_SYSTEM},
             {"role": "user", "content": user}])
        text = getattr(resp, "content", None) or str(resp)
        profile = _extract_json(text)
        if not profile:
            logger.warning("[Intake] judge_intent 无法解析 JSON，跳过")
            return None
        return _normalize(profile, query)
    except Exception as e:
        logger.warning(f"[Intake] judge_intent failed: {e}")
        return None


def decide_card(profile: Optional[dict], *, mode: str = "expert") -> bool:
    """弹卡决策：歧义有代价才弹。无 profile / light 模式 / 仅 1 个理解 → 不弹。"""
    if not profile or not INTENT_ENABLED or mode != "expert":
        return False
    if not profile.get("needs_clarification"):
        return False
    if len(profile.get("interpretations") or []) < 2:
        return False
    return float(profile.get("ambiguity_score") or 0.0) >= AMBIGUITY_THRESHOLD


def resolve_choice(profile: dict, choice: Optional[dict]) -> str:
    """用户选择 → intent_confirmed 文本。未选择（超时/异常）→ 推荐项 interpretation[0]。

    自填文本优先；否则按 choice_id 匹配候选；再否则回退 intent_summary。
    """
    opts = profile.get("interpretations") or []
    if choice:
        custom = str(choice.get("custom_text") or "").strip()
        if custom:
            return custom[:2000]
        cid = choice.get("choice_id")
        if cid is not None:
            for o in opts:
                try:
                    if int(o.get("id")) == int(cid):
                        return str(o.get("text") or plan_fallback(profile))[:2000]
                except (TypeError, ValueError):
                    continue
    if opts:
        return str(opts[0].get("text") or plan_fallback(profile))[:2000]
    return plan_fallback(profile)


def plan_fallback(profile: dict) -> str:
    return str((profile or {}).get("intent_summary") or "")
