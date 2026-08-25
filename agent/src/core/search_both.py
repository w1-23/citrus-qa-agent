# -*- coding: utf-8 -*-
"""v9.1: call_search_both —— Supervisor 统一检索入口（本地 + 联网并行执行）。

架构（用户决策）：
  Supervisor 唯一决策者（构造 local_goal/web_goal，不拆解、不预判跳过）；
  Retrieve-Agent 只做本地 RAG/UCR（工具列表无联网工具）；Web-Agent 无 LLM 决策
  （直接调用 deepseek_web_search 一次）；两者 asyncio.gather 并行、互不阻塞，
  消除"本地 10s 被 web 78s 阻塞"的木桶效应（总等待 = max(本地, 联网)）。

拆解层级（防乘法膨胀）：
  Supervisor：完整意图，不拆解 → Retrieve-Agent：2-4 个聚焦角度（唯一拆解层）
  → citrus_rag_search 内部 HyDE 多路（工具层行为）→ 合计 2-4 × 9 = 18-36 路。

返回形态与 call_retrieve_agent 时代一致：{"agent","result","artifacts",
"tools_called","status"}——expert_graph 循既有装配路径收集 main_results /
web_results / web_summaries 进 cited_refs 与 evidence 账本，零协议改动。
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_EMPTY_ARTS = {"main_results": [], "web_results": [], "web_summaries": []}


async def run_web_agent(web_goal: str) -> dict:
    """Web-Agent 极简执行器：无 LLM 决策，原样使用 web_goal 调一次联网。

    任何情况只调用一次（预算/开关短路由工具内部判定并返回控制信号）；
    输出 = 工具回执 + 一句「联网检索员判定」；证据经 artifact 透传。
    """
    from src.tools.deepseek_web import deepseek_web_search

    _goal = str(web_goal or "").strip()
    try:
        content, artifact = await asyncio.to_thread(deepseek_web_search.func, _goal)
    except Exception as e:  # 工具本体抛错兜底（正常路径工具内捕获并返回 [ERR_*]）
        logger.error(f"[WebAgent] deepseek_web_search 调用异常: {e}")
        content, artifact = (f"[ERR_NETWORK] 联网搜索调用异常: {e}", dict(_EMPTY_ARTS))
    artifact = artifact or {}
    if not isinstance(artifact, dict):
        artifact = dict(_EMPTY_ARTS)

    text = str(content or "")
    web_items = list(artifact.get("web_results") or [])
    summary = str(artifact.get("web_summary") or "").strip()

    if "[DISABLED]" in text:
        judge = "联网检索员判定：前端联网开关关闭，本轮无联网证据。"
        status = "disabled"
    elif "[WEB_BUDGET_EXHAUSTED]" in text:
        judge = "联网检索员判定：本次请求联网预算已用尽（仅允许 1 次）。"
        status = "budget"
    elif text.startswith("[ERR_") or "[AgentError]" in text or text.startswith("[Error"):
        judge = "联网检索员判定：联网检索失败，本轮无联网证据（详见回执）。"
        status = "error"
    elif not web_items and not summary:
        judge = "联网检索员判定：联网未返回有效信息（0 引用 0 摘要），不重试。"
        status = "empty"
    elif not summary:
        judge = (f"联网检索员判定：获得 {len(web_items)} 条引用但无正文摘要"
                 "（模型未生成综述；可依据标题/URL，不重试）。")
        status = "ok"
    else:
        judge = (f"联网检索员判定：获得 {len(web_items)} 条引用、正文摘要 "
                 f"{len(summary)} 字，可支撑时效与实时信息裁决。")
        status = "ok"

    result_text = f"{text.strip()}\n\n{judge}"
    try:
        from src.core.progress_bus import emit_tool_result
        emit_tool_result("deepseek_web_search", result_text[:100000], "web-agent",
                         summary=judge[:60])
    except Exception:
        pass
    return {
        "agent": "web-agent",
        "result": result_text,
        "artifacts": {
            "main_results": [],
            "web_results": web_items,
            "web_summaries": [summary] if summary else [],
        },
        "tools_called": 1,
        "status": status,
    }


async def call_search_both(local_goal: str, web_goal: str,
                           fallback_query: str = "",
                           *, session_id: str = "",
                           seen_queries: list | None = None) -> dict:
    """Supervisor 工具执行体：本地 + 联网并行检索，合并回执与证据。

    核心原则（用户决策）：不预判、不跳过——任一 goal 为空时用另一个或
    fallback_query 填充兜底；本地与联网互不阻塞，证据取回后由 Supervisor 仲裁。
    """
    from src.core.agent_runner import run_agent

    local_goal = str(local_goal or "").strip()
    web_goal = str(web_goal or "").strip()
    fallback = (str(fallback_query or "").strip() or local_goal or web_goal)
    if not fallback:
        return {"agent": "call_search_both",
                "result": "[ERR_PARSE] call_search_both 需要至少一个非空目标"
                          "（local_goal/web_goal）。",
                "artifacts": dict(_EMPTY_ARTS), "tools_called": 0, "status": "error"}
    local_goal = local_goal or fallback
    web_goal = web_goal or fallback

    async def _run_local():
        return await run_agent(
            "retrieve-agent",
            {"goal": local_goal, "query": local_goal[:200]},
            context="", timeout_sec=120,
            session_id=session_id, seen_queries=seen_queries,
        )

    local_res, web_res = await asyncio.gather(
        _run_local(), run_web_agent(web_goal))

    merged_arts: dict = {
        "main_results": [],
        "web_results": [],
        "web_summaries": [],
    }
    for r in (local_res, web_res):
        arts = (r or {}).get("artifacts") or {}
        if isinstance(arts, dict):
            for k in merged_arts:
                merged_arts[k].extend(arts.get(k) or [])

    parts: list[str] = []
    lr = str((local_res or {}).get("result") or "").strip()
    if lr:
        parts.append(f"## 本地检索结果（retrieve-agent）\n{lr}")
    wr = str((web_res or {}).get("result") or "").strip()
    if wr:
        parts.append(f"## 联网检索结果（web-agent）\n{wr}")
    merged_text = "\n\n".join(p for p in parts if p) or "(无检索结果)"

    _weberr = (web_res or {}).get("status") in ("error", "empty")
    _localerr = str((local_res or {}).get("status")) == "error"
    return {
        "agent": "call_search_both",
        "result": merged_text,
        "artifacts": merged_arts,
        "tools_called": ((local_res or {}).get("tools_called") or 0)
                        + ((web_res or {}).get("tools_called") or 0),
        "status": "error" if (_localerr and _weberr) else "ok",
    }