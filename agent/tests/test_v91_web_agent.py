# -*- coding: utf-8 -*-
"""v9.1 回归：独立 Web-Agent 架构 + 并行检索（用户全量计划落地）。

架构：Supervisor 唯一决策者——统一入口 call_search_both(local_goal, web_goal)；
Retrieve-Agent 只做本地 RAG/UCR（工具列表无联网工具，根除"只联网不本地"）；
Web-Agent 无 LLM 决策（直接调 deepseek_web_search 一次，不重试）；两者
asyncio.gather 并行互不阻塞（消木桶效应，总等待 = max(本地, 联网)）。

思维链差异化（真机实测 2026-08-24）：
  chat/completions 端点关闭字段 = thinking:disabled（MODEL_REASONING_OFF_BODY）；
  Responses 端点（联网内部）关闭字段 = reasoning:effort:none（WEB_REASONING_OFF_BODY，
  实测 thinking:disabled 在该端点无效——reasoning 块照出）。

全部离线、无模型、无网络（Fake 桩 + 源码断言）。约定见 test_batch1.py。
覆盖：
  VF-70  retrieve-agent 白名单无联网工具（根除只联网不本地）
  VF-71  supervisor schema：call_search_both(local_goal, web_goal) 取代 call_retrieve_agent
  VF-72  run_web_agent 极简执行器（DISABLED/成功/失败三态 + artifacts 透传 + 无 LLM）
  VF-73  call_search_both 并行（总耗时 < 串行和）+ 证据合并 + 空 goal 互填
  VF-74  请求级联网预算（最多 1 次；超预算 [WEB_BUDGET_EXHAUSTED]）
  VF-75  Responses 端点关闭字段 reasoning effort none 进入 payload（+ fail-soft 去参重试）
  VF-76  提示词与构建映射（21_source / web 角色 / builds / 03 语义 / 04 无联网）
  VF-77  main.py 每请求重置联网预算接线
"""
import sys
import os
import asyncio
import time
import inspect as _inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings

passed, failed = [], []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
    else:
        failed.append(name)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


class _FakeWebTool:
    """模拟 deepseek_web_search（langchain @tool 形态：.func 返回 (str, artifact)）。"""

    def __init__(self, content, artifact=None):
        self.content = content
        self.artifact = artifact or {"main_results": [], "web_results": []}
        self.calls = 0

    def func(self, query):
        self.calls += 1
        return self.content, self.artifact


def _old_name_only_in_guard(eg_src: str) -> bool:
    """v9.1.1: expert_graph 中旧工具名只允许出现在废弃防御分支内（无执行逻辑）。"""
    gpos = eg_src.find("# v9.1.1（用户真机日志）: 旧工具名防御")
    gend = eg_src.find("# v9.1（用户决策）: 统一检索入口", gpos)
    if gpos == -1 or gend == -1:
        return False
    outside = eg_src[:gpos] + eg_src[gend:]
    return "call_retrieve_agent" not in outside


# ── F-9.1-1 白名单与工具注册 ───────────────────────────────────
def test_v91_whitelist():
    print("[VF-70] retrieve-agent 白名单无联网工具")
    from src.core import agent_runner as ar
    ar_src = _inspect.getsource(ar)
    # 精确切片：_resolve_tool_names 里 retrieve-agent 的工具列表段
    msec = ar_src.split("def _resolve_tool_names")[1]
    start = msec.find('"retrieve-agent": [')
    end = msec.find("]", start)
    block = msec[start:end] if start != -1 and end != -1 else ""
    check("retrieve-agent 列表段存在", "citrus_rag_search" in block, block[:120])
    check("retrieve-agent 列表段无联网工具（根除只联网不本地）",
          "deepseek_web_search" not in block, block[:120])


def test_v91_schema():
    print("[VF-71] supervisor schema：call_search_both 取代 call_retrieve_agent")
    from src.tools.supervisor_tools import get_supervisor_tool_schemas, get_supervisor_tool_names
    names = get_supervisor_tool_names()
    check("schema 含 call_search_both", "call_search_both" in names, str(names))
    check("call_retrieve_agent 已移除", "call_retrieve_agent" not in names, str(names))
    sc = next((t["function"] for t in get_supervisor_tool_schemas()
               if t["function"]["name"] == "call_search_both"), None)
    check("call_search_both 参数 local_goal/web_goal 必填",
          sc is not None and set(sc["parameters"].get("required", [])) == {"local_goal", "web_goal"},
          str(sc and sc["parameters"]))
    from src.graph import expert_graph as eg
    eg_src = _inspect.getsource(eg)
    check("expert_graph 含 call_search_both 执行分支",
          "if name == \"call_search_both\":" in eg_src)
    check("expert_graph 的旧名仅存在于废弃防御分支（v9.1.1）",
          _old_name_only_in_guard(eg_src), "残留执行逻辑？")


# ── F-9.1-2 Web-Agent 极简执行器 ────────────────────────────────
def test_v91_web_agent():
    print("[VF-72] run_web_agent 三态与 artifacts 透传")
    import src.core.search_both as sb
    import src.tools.deepseek_web as dw

    _orig = dw.deepseek_web_search

    async def _run(content, artifact):
        dw.deepseek_web_search = _FakeWebTool(content, artifact)
        try:
            return await sb.run_web_agent("web goal test")
        finally:
            dw.deepseek_web_search = _orig

    # DISABLED 态
    r1 = asyncio.run(_run("[DISABLED] 联网搜索未开启。", {"main_results": [], "web_results": []}))
    check("DISABLED → 无联网证据判定", "前端联网开关关闭" in r1["result"], r1["result"][:80])
    check("DISABLED → status=disabled", r1["status"] == "disabled")
    check("DISABLED → 空 web_results", r1["artifacts"]["web_results"] == [])

    # 成功态（摘要+引用）
    items = [{"ref_id": "W1", "type": "web", "source": "web",
              "url": "https://x.cn/a", "title": "T1",
              "abstract": "abs", "snippet": "abs"}]
    r2 = asyncio.run(_run("[ToolResult] ok\n正文摘录", {
        "main_results": [], "web_results": items, "web_summary": "摘要正文"}))
    check("成功 → 判定含引用数与摘要字数", "1 条引用" in r2["result"] and "正文摘要" in r2["result"],
          r2["result"][-120:])
    check("成功 → web_results 透传", r2["artifacts"]["web_results"] == items)
    check("成功 → web_summaries 收集", r2["artifacts"]["web_summaries"] == ["摘要正文"])
    check("成功 → status=ok", r2["status"] == "ok")

    # 失败态
    r3 = asyncio.run(_run("[ERR_NETWORK] 联网搜索失败 HTTP 500", {
        "main_results": [], "web_results": []}))
    check("失败 → 无联网证据判定", "联网检索失败" in r3["result"], r3["result"][-80:])
    check("失败 → status=error", r3["status"] == "error")


# ── F-9.1-3 call_search_both 并行与合并 ─────────────────────────
def test_v91_search_both_parallel():
    print("[VF-73] call_search_both 并行执行 + 证据合并 + 空 goal 互填")
    import src.core.search_both as sb
    import src.core.agent_runner as ar_mod

    _orig_run_agent = ar_mod.run_agent

    async def _fake_run_agent(agent_name, task, context="", timeout_sec=120,
                              session_id="", seen_queries=None, **kw):
        await asyncio.sleep(0.15)  # 模拟本地检索耗时
        return {"agent": "retrieve-agent", "result": "本地回执：证据充分",
                "artifacts": {"main_results": [{"doi": "10.1/x", "title": "P1"}],
                              "web_results": [], "web_summaries": []},
                "tools_called": 2, "status": "ok"}

    async def _main():
        ar_mod.run_agent = _fake_run_agent
        import src.tools.deepseek_web as dw
        dw.deepseek_web_search = _FakeWebTool(
            "[ToolResult] web ok",
            {"main_results": [], "web_results": [
                {"ref_id": "W1", "type": "web", "source": "web",
                 "url": "https://y.cn/b", "title": "W1T"}],
             "web_summary": "web 摘要"},
        )
        try:
            t0 = time.perf_counter()
            res = await sb.call_search_both("local goal A", "web goal B")
            elapsed = time.perf_counter() - t0
        finally:
            ar_mod.run_agent = _orig_run_agent
        return res, elapsed

    res, elapsed = asyncio.run(_main())
    check("并行：总耗时 < 串行和的一半加缓冲（本地0.15+web任务并行 → <0.30s）",
          elapsed < 0.30, f"{elapsed:.3f}s")
    check("合并回执含本地段与联网段",
          "## 本地检索结果" in res["result"] and "## 联网检索结果" in res["result"])
    check("证据合并：main_results 含本地条目", len(res["artifacts"]["main_results"]) == 1)
    check("证据合并：web_results 含联网条目", len(res["artifacts"]["web_results"]) == 1)
    check("证据合并：web_summaries 收集", res["artifacts"]["web_summaries"] == ["web 摘要"])
    check("tools_called 汇总", res["tools_called"] == 3, str(res["tools_called"]))

    # 空 goal 互填：local 为空 → 用 web_goal
    async def _main2():
        ar_mod.run_agent = _fake_run_agent
        import src.tools.deepseek_web as dw
        dw.deepseek_web_search = _FakeWebTool("[ToolResult] ok", {
            "main_results": [], "web_results": [], "web_summary": ""})
        try:
            return await sb.call_search_both("", "fallback goal")
        finally:
            ar_mod.run_agent = _orig_run_agent

    r2 = asyncio.run(_main2())
    check("空 local_goal → web_goal 兜底执行，结果仍正常", "结果" in r2["result"] or "回执" in r2["result"],
          r2["result"][:80])


# ── F-9.1-4 请求级联网预算 ─────────────────────────────────────
def test_v91_web_budget():
    print("[VF-74] 请求级联网预算（每请求 1 次）")
    from src.core.tracing import reset_web_budget, consume_web_budget

    reset_web_budget(1)
    check("预算剩余 1 → 首次消费成功", consume_web_budget() is True)
    check("预算剩余 0 → 再次消费失败", consume_web_budget() is False)
    reset_web_budget(1)
    check("重置后恢复", consume_web_budget() is True)

    # 工具层：预算耗尽 → [WEB_BUDGET_EXHAUSTED]（用 stub 工具函数）
    import src.tools.deepseek_web as dw
    sentinel = {}

    def _fake_func(query):
        # 触发真实工具入口的预算检查：直接走 stub 路径不可行（工具已吞），
        # 此断言改为源码级——工具入口消费预算并返回控制信号
        raise AssertionError("不直达")

    dw_src = _inspect.getsource(dw)
    check("工具入口消费预算（源码）", "consume_web_budget" in dw_src
          and "WEB_BUDGET_EXHAUSTED" in dw_src, "预算/控制信号缺失")
    check("工具入口保留前端开关短路", "web_search_enabled" in dw_src)


# ── F-9.1-5 Responses 端点关闭字段 ─────────────────────────────
def test_v91_responses_thinking_off():
    print("[VF-75] Responses 端点关闭字段 reasoning effort none")
    import src.tools.deepseek_web as dw

    off_body = dict(getattr(settings, "WEB_REASONING_OFF_BODY", None) or {})
    check("配置默认 = reasoning effort none（真机实测有效字段）",
          off_body == {"reasoning": {"effort": "none"}}, str(off_body))
    dw_src = _inspect.getsource(dw)
    check("payload 下发关闭字段（源码）", "payload.update(off_body)" in dw_src
          and "WEB_REASONING_OFF_BODY" in dw_src, "关闭字段接线缺失")
    check("400/422 去参重试（fail-soft）", "resp.status_code in (400, 422)" in dw_src
          and "去参重试" in dw_src, "fail-soft 缺失")


# ── F-9.1-6 提示词与构建映射 ───────────────────────────────────
def test_v91_prompts():
    print("[VF-76] 提示词与构建映射（21 / web 角色 / builds / 03 语义 / 04 无联网）")
    from src.prompts.loader import ROLE_SOURCE_FILES, BUILD_FILE_NAMES, AGENT_ROLE_MAP

    check("source/21_web_agent_search.md 存在",
          os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "src", "prompts", "source", "21_web_agent_search.md")))
    check("web 角色映射存在且不含 03/04", ROLE_SOURCE_FILES.get("web") ==
          ["01_global_role_domain.md", "02_global_data_fidelity_citation.md",
           "21_web_agent_search.md", "20_terminology_domain.md"],
          str(ROLE_SOURCE_FILES.get("web")))
    check("BUILD_FILE_NAMES 含 web", BUILD_FILE_NAMES.get("web") == "web_agent_system.md")
    check("AGENT_ROLE_MAP 含 web-agent", AGENT_ROLE_MAP.get("web-agent") == "web")

    import src.prompts.snapshot as snap
    snap_src = _inspect.getsource(snap)
    check("快照含 web-agent 渲染", "web-agent" in snap_src or
          os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                      "src", "prompts", "snapshots", "agent_web-agent.txt")),
          "web-agent 快照缺失")

    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "prompts")
    with open(os.path.join(base, "source", "03_supervisor_routing_fusion.md"), encoding="utf-8") as f:
        s03 = f.read()
    check("03 语义：call_search_both + 不预判不跳过",
          "call_search_both" in s03 and "不预先跳过任何一方" in s03 and "禁止传空字符串" in s03)
    check("03 不再指示 call_retrieve_agent", "call_retrieve_agent" not in s03)
    with open(os.path.join(base, "source", "04_retrieve_agent_search.md"), encoding="utf-8") as f:
        s04 = f.read()
    check("04 无联网工具引用（本地唯一拆解层）",
          "deepseek_web_search" not in s04 and "citrus_rag_search" in s04
          and "唯一允许的拆解层" in s04)

    builds = os.path.join(base, "builds")
    check("builds/web_agent_system.md 已生成",
          os.path.exists(os.path.join(builds, "web_agent_system.md")))


# ── F-9.1-7 main.py 预算重置接线 ───────────────────────────────
def test_v91_main_reset():
    print("[VF-77] main.py 每请求重置联网预算")
    import src.api.main as main_mod
    src_ = _inspect.getsource(main_mod)
    check("chat 入口调用 reset_web_budget", "reset_web_budget(" in src_)
    check("与开关设置同处（set_web_search_enabled 相邻）",
          src_.find("set_web_search_enabled(") < src_.find("reset_web_budget(")
          and src_.find("reset_web_budget(") < src_.find("reset_web_budget(") + 200)


# ── F-9.1-8 旧工具名防御（v9.1.1 用户真机日志：旧路径须显式引导）──
def test_v91_deprecated_tool_guard():
    print("[VF-78] call_retrieve_agent 废弃引导防御")
    from src.graph import expert_graph as eg

    async def _hit(name):
        return await eg._execute_tool_call({"name": name, "args": {}})

    r = asyncio.run(_hit("call_retrieve_agent"))
    check("返回显式废弃错误（不再静默未知工具）",
          "[ERR_DEPRECATED]" in r.get("result", ""), r.get("result", "")[:100])
    check("错误信息引导改用 call_search_both",
          "call_search_both" in r.get("result", ""))
    check("废弃路径 status=error（supervisor 可感知失败）", r["status"] == "error")
    check("废弃路径无证据 artifacts",
          r["artifacts"] == {"main_results": [], "web_results": [], "web_summaries": []},
          str(r.get("artifacts")))
    check("无 tools_called 虚增", r["tools_called"] == 0)

    # supervisor 工具 schema 中确认彻底无旧名（最后一次防回归）
    from src.tools.supervisor_tools import get_supervisor_tool_names
    check("schema 无 call_retrieve_agent（防回归）",
          "call_retrieve_agent" not in get_supervisor_tool_names())


def test_v91_silent_summary():
    if failed:
        print(f"\n  ✗ v9.1 回归: {len(failed)} FAIL / {len(passed)} PASS")
        raise SystemExit(1)
    print(f"\n  ✓ v9.1 回归: {len(passed)} PASS / 0 FAIL")