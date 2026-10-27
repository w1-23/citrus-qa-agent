# -*- coding: utf-8 -*-
"""v10.7: call_search_both —— Supervisor 统一检索入口（本地 + 联网**并行**执行）。

架构（v10.7 用户决策：多智能体 + 并行 + 最多两轮 + 本地与联网分离输出）：
  · Supervisor：**看会话全文缓存**，理解上下文，把意图转成**明确检索指令**
    （local_goal / web_goal），一次 `call_search_both` 唤起两个子 Agent；
  · Retrieve-Agent：只做本地 RAG/UCR（工具列表无联网工具）；
  · **Web-Agent：自主联网检索**——看**自己的上下文**（指令 + 自己的检索结果），
    每轮判断**相关性/充分性**，不充分可**换策略**再搜，**最多 2 轮**，
    第 1 轮充分即停止（`core/web_agent_loop.py` + `core/web_agent_judge.py`）；
  · 两者 `asyncio.gather` **并行**、互不阻塞（总等待 = max(本地, 联网)）；
  · **不做跨源仲裁**：联网内容不进引用池、不占编号，由 `web_output_separator`
    净化后**单独输出**给用户（URL + 来源分级，用户自行判断）。

拆解层级（防乘法膨胀）：
  Supervisor：完整意图，不拆解 → Retrieve-Agent：2-4 个聚焦角度（唯一拆解层）
  → citrus_rag_search 内部 HyDE 多路（工具层行为）→ 合计 2-4 × 9 = 18-36 路。

返回形态与 call_retrieve_agent 时代一致：{"agent","result","artifacts",
"tools_called","status"}——expert_graph 循既有装配路径收集 main_results /
web_results / web_summaries 进 cited_refs 与 evidence 账本，零协议改动。
`artifacts["web_agent_rounds"]` 另带**逐轮审计**（轮次/查询/相关性/充分性/换策略/停止原因）。
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_EMPTY_ARTS = {"main_results": [], "web_results": [], "web_summaries": []}


async def run_web_agent(web_goal: str) -> dict:
    """Web-Agent：**自主**联网检索（v10.7，最多 2 轮 + 自评相关性/充分性）。

    与旧版的区别（旧版是"无 LLM 决策、原样调一次工具"）：

      · 轮次：**最多 2 轮**（代码硬上限）；第 1 轮相关且充分 → 立即停止；
        不充分/不相关 → 进第 2 轮并**换策略**；第 2 轮后仍不满足 → 停止并标注。
      · 判断：相关性/充分性由 Web Agent 自己的 LLM 判定（`web_agent_judge`），
        **只看自己的上下文**（指令 + 自己的检索结果），不看会话全文。
      · 预算：每轮一次工具调用；工具内部另有请求级预算（`[WEB_BUDGET_EXHAUSTED]`）。

    输出：聚合后的 `web_results` / `web_summaries` + 一句「联网侧回执」（代码生成）
    + `artifacts["web_agent_rounds"]` 逐轮审计（计划 §3.4 的七个字段）。
    内容仍由 `web_output_separator` 净化后**单独输出**，不进引用池。
    """
    from src.core.web_agent_loop import run_web_agent_loop

    _goal = str(web_goal or "").strip()
    # ⚠️ 兜底：`call_search_both` 用 `asyncio.gather` 并行跑本地与联网，
    #   而 gather 默认会把**任一分支的异常**抛穿整个调用——联网侧一旦出意外，
    #   **本地检索结果也会一起丢**（旧版只在工具调用处 try/except，覆盖不全）。
    #   故这里把联网分支整体兜住：异常 → 降级为 error 结果，本地分支不受影响。
    try:
        loop = await run_web_agent_loop(_goal)
    except Exception as e:                                     # noqa: BLE001
        logger.error(f"[WebAgent] 自主循环异常（降级为 error，保住本地分支）: "
                     f"{type(e).__name__}: {e}")
        loop = {"instruction": _goal, "rounds": [], "rounds_used": 0,
                "web_results": [], "web_summaries": [], "status": "error",
                "stop_reason": "error", "note": "", "strategy_switched": False,
                "queries": []}

    # v10.8 修：这里原写成 `text = ""`，导致 result_text 只剩回执——
    #   回滚档下模型拿不到任何联网正文（由 `_scratch/verify_tool_result_emit.py` 抓出）。
    #   现从循环带出的 `raw_text`（各轮工具原始文本）恢复，保住工具契约。
    text = str(loop.get("raw_text") or "")
    web_items = list(loop.get("web_results") or [])
    # ── v10.13（阶段 0 审计修正）：带 `round_no` 的副本**由 loop 单独提供** ──────
    #   `web_agent_loop` 现在同时给出两个视图：
    #     · `web_results`       = 工具原始语义（**不含** `round_no`）→ artifacts 透传不变
    #     · `web_results_round` = 含 `round_no` 的副本 → 仅供侧栏分组
    #   早先在 search_both 里做浅拷贝补 `round_no`，但 loop 那边已经把标打进了原始
    #   条目 ⇒ 回归掉到 246/2。现改为直接消费 loop 提供的干净视图。
    web_items_round = [dict(_it) if isinstance(_it, dict) else _it
                       for _it in (loop.get("web_results_round") or [])]
    if len(web_items_round) != len(web_items):          # 兜底：loop 未提供时自行构造
        web_items_round = []
        for _it in web_items:
            if isinstance(_it, dict):
                _d = dict(_it)
                _d.setdefault("round_no", 1)
                web_items_round.append(_d)
            else:
                web_items_round.append(_it)
    summaries = [s for s in (loop.get("web_summaries") or []) if str(s).strip()]
    summary = max(summaries, key=len) if summaries else ""
    status = str(loop.get("status") or "ok")
    rounds = list(loop.get("rounds") or [])
    n_rounds = int(loop.get("rounds_used") or len(rounds) or 0)

    if status == "disabled":
        judge = "联网侧回执：本轮未联网（开关关闭），无联网结果。"
    elif status == "budget":
        judge = "联网侧回执：本轮联网预算已用尽，无联网结果。"
    elif status == "error":
        judge = "联网侧回执：联网检索失败，无联网结果。"
    elif not web_items and not summary:
        judge = "联网侧回执：联网未返回结果（0 引用、0 正文）。"
    else:
        _why = {"sufficient": "第 1 轮即充分" if n_rounds == 1 else "达到充分",
                "max_rounds_insufficient": "未充分检索",
                "max_rounds_irrelevant": "未找到相关信息",
                "judge_fail_open": "自评不可用（按充分处理）"}.get(
                    str(loop.get("stop_reason")), str(loop.get("stop_reason")))
        _sw = "，两轮间换过策略" if loop.get("strategy_switched") else ""
        judge = (f"联网侧回执：{n_rounds} 轮检索，共获得 {len(web_items)} 条引用（URL 去重）"
                 + (f"、原生回答正文 {len(summary)} 字" if summary else "、无原生回答正文")
                 + f"；停止原因：{_why}{_sw}。")

    result_text = f"{text.strip()}\n\n{judge}" if text.strip() else judge
    # ── v10.6（分离输出）：推给前端的 `output` 也要净化 ──────────────────────
    #   `emit_tool_result` 的 `output` 字段：前端**不显示**它（index.html:2676 只用
    #   `data.output.length` 估算上下文占用），所以它既不是展示面、却又是"模型上下文"
    #   的计量口径。分离档下模型只收到计数占位说明 ⇒ 若这里仍推联网全文，会
    #   ① 把带 `[Wn]` 编号的联网正文送到浏览器（与"净化：无 [W1] 等标记"的意图相悖）；
    #   ② 把模型**从未收到**的正文计入上下文占用（实测单例 3,000+ 字 ≈ 2,100 token 虚高）。
    #   故分离档下推同一份计数说明；回滚档保持原样。
    _emit_text = result_text
    try:
        from src.core.web_output_separator import SEPARATION as _SEPS
        from src.core.web_output_separator import model_facing_web_notice as _mfn
        if _SEPS.enabled and getattr(_SEPS, "hide_from_model", True):
            _emit_text = _mfn(len(web_items), len(summary))
    except Exception:                                          # noqa: BLE001
        pass
    try:
        from src.core.progress_bus import emit_tool_result
        emit_tool_result("deepseek_web_search", _emit_text[:100000], "web-agent",
                         summary=judge[:60])
    except Exception:
        pass
    return {
        "agent": "web-agent",
        "result": result_text,
        "artifacts": {
            "main_results": [],
            # ⚠️ v10.13（Task E）：这里必须是**原始对象**，不能带 `round_no` 等展示层字段。
            #   实测踩坑：早先把 `_tag_round` 的结果放进 `web_results`，导致既有测试
            #   `test_v91_web_agent.py` 的"artifacts 透传"断言（`== items`，按内容比较）
            #   失败——那是对的：`web_results` 的语义是"工具原始返回"。
            #   带轮次的副本另放 `web_results_round`，只给展示层（侧栏分组）用。
            "web_results": web_items,
            "web_results_round": web_items_round,
            "web_summaries": summaries,
            # 计划 §3.4：逐轮审计（轮次/指令/结果量/相关性/充分性/换策略/停止原因）
            "web_agent_rounds": rounds,
            "web_agent_stop_reason": loop.get("stop_reason"),
            "web_agent_note": loop.get("note") or "",
        },
        "tools_called": max(1, n_rounds),
        "status": status,
    }


async def call_search_both(local_goal: str, web_goal: str,
                           fallback_query: str = "",
                           *, session_id: str = "",
                           seen_queries: list | None = None) -> dict:
    """Supervisor 工具执行体：本地 + 联网**并行**检索，合并回执与证据。

    核心原则（用户决策）：不预判、不跳过——任一 goal 为空时用另一个或
    fallback_query 填充兜底；本地与联网 `asyncio.gather` 互不阻塞。
    **不做跨源仲裁**：联网证据不进引用池，由 `web_output_separator` 单独输出。
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

    # ── v10.12（修复 P2-1）：本地/联网**分段完成提示** ─────────────────────────
    #   现象（用户观感）：15:30:04 本地开始 → 15:30:29 联网回执到达，中间 25 秒
    #   界面没有任何"本地已完成"的信号，用户以为整体结束、实际联网还在跑。
    #   设计约束（任务书明确）：
    #     · **不改并行结构**（仍是 asyncio.gather）；
    #     · **不引入轮询**（事件驱动是既定设计）；
    #     · 用现有 `emit_status` 通道发分割事件（不新增事件类型，前端契约不变）。
    #   做法：给两条分支各包一层——**谁先完成就先发自己的事件**，并带上
    #   "另一条还在跑"的语义；两条都完成后由 `gather` 之后的收尾统一发 done。
    #   全部包在 try/except 里：**进度提示失败绝不影响检索本身**。
    import time as _time
    _t_start = _time.perf_counter()
    _timing: dict[str, float] = {}

    async def _run_local_wrapped():
        _n = 0
        try:
            r = await _run_local()
            _n = len(((r or {}).get("artifacts") or {}).get("main_results") or [])
        finally:
            _timing["local"] = _time.perf_counter() - _t_start
            try:
                if not _web_done:
                    from src.core.progress_bus import emit_status as _es
                    _es("stage", status="local_done_web_running",
                        message=f"本地检索完成（{_n} 篇文献）· 联网检索中…",
                        local_result_count=_n)
            except Exception:                                  # noqa: BLE001
                pass
        return r

    async def _run_web_wrapped():
        nonlocal _web_done
        _n = 0
        try:
            r = await run_web_agent(web_goal)
            _n = len(((r or {}).get("artifacts") or {}).get("web_results") or [])
        finally:
            _timing["web"] = _time.perf_counter() - _t_start
            _web_done = True
            try:
                from src.core.progress_bus import emit_status as _es
                _es("stage", status="web_done",
                    message=f"联网检索完成（{_n} 条），正在生成答案…",
                    web_result_count=_n)
            except Exception:                                  # noqa: BLE001
                pass
        return r

    _web_done = False
    local_res, web_res = await asyncio.gather(
        _run_local_wrapped(), _run_web_wrapped())

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
    _local_section = f"## 本地检索结果（retrieve-agent）\n{lr}" if lr else ""
    if _local_section:
        parts.append(_local_section)
    wr = str((web_res or {}).get("result") or "").strip()
    _web_section = f"## 联网检索结果（web-agent）\n{wr}" if wr else ""
    if _web_section:
        parts.append(_web_section)
    merged_text = "\n\n".join(p for p in parts if p) or "(无检索结果)"

    # ── v10.12（修复 P0-1）：把**分段文本**也带出去 ────────────────────────────
    #   缺陷：`expert_graph` 的"对模型隐藏联网正文"分支此前把**整个** tool 内容
    #   替换成联网占位说明，而 `merged_text` 是「本地段 + 联网段」拼接的字符串
    #   ⇒ 本地 `[n]` 主文献清单被联网说明一起吞掉。实测（2026-09-25 15:32 运行）：
    #   `[WebSep] 工具回执对模型隐藏联网正文：20603c → 占位说明（15 条）`
    #   ⇒ Supervisor reasoning 里出现 "it doesn't show local evidence. Hmm,
    #     this is a problem... But I don't see them."，最终答案退化为
    #   "本轮本地检索未返回任何可引用证据。"
    #   修法：这里只**额外**提供两段原文，`result` 保持逐字不变（零行为回归）；
    #   由 `expert_graph` 按段重组 = 保留本地段、只替换联网段。
    merged_arts["local_section_text"] = _local_section
    merged_arts["web_section_text"] = _web_section

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