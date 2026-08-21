# -*- coding: utf-8 -*-
"""v8.15.3 特性回归：联网失败熔断 / 检索统计回传 / 数据源覆盖提示 / 推理控制。

全部离线、无模型、无网络（仅验证逻辑与门控纯函数）。约定见 test_batch1.py。
覆盖：
  VF-9   rag_stats_note 检索统计回执（早停阈值 + 并发防串号）
  VF-10  build_evidence_report web_unavailable 熔断提示（确定性回执）
  VF-11  model.reasoning_mode 配置默认 + llm_pool thinking_off 接线
  VF-12  deepseek_web_search HTTP 失败详情（状态码落回执，超时单独分支）
"""
import sys
import os

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


# ── F-15.3-1 检索统计回执（决策器每轮检索后自审依据）────────────────
def test_v8153_rag_stats_note():
    print("[VF-9] rag_stats_note 检索统计回执")
    from src.tools.search import rag_stats_note

    check("无统计 → 空串", rag_stats_note(None) == "")
    check("非 dict → 空串", rag_stats_note("x") == "")
    check("候选 0 → 空串", rag_stats_note({"candidates": 0, "passed": 0, "filtered": 0}) == "")
    ok_note = rag_stats_note({"candidates": 20, "passed": 12, "filtered": 3, "query": "citrus hlb"})
    check("通过率高 → 仅统计不提示", "[检索统计]" in ok_note and "相关性低" not in ok_note, ok_note)
    low_note = rag_stats_note({"candidates": 20, "passed": 4, "filtered": 6, "query": "citrus hlb"})
    check("过滤占比 60% → 提示相关性低", "相关性低" in low_note, low_note)
    thin_note = rag_stats_note({"candidates": 20, "passed": 2, "filtered": 1, "query": "citrus hlb"})
    check("通过 ≤2 → 提示相关性低", "相关性低" in thin_note, thin_note)
    check("统计数字字面量正确", "候选 20 条" in ok_note and "通过 12 条" in ok_note, ok_note)
    # 并发防串号：last_stats 归属查询与本次查询不一致 → 不输出
    mismatch = rag_stats_note({"candidates": 20, "passed": 1, "filtered": 9, "query": "other query"},
                              expect_query="citrus hlb")
    check("串号统计被丢弃", mismatch == "", mismatch)
    match = rag_stats_note({"candidates": 20, "passed": 1, "filtered": 9, "query": "citrus hlb"},
                           expect_query="citrus hlb")
    check("同查询统计正常输出", match != "" and "相关性低" in match, match)


# ── F-15.3-2 确定性回执的联网熔断提示 ──────────────────────────────
def test_v8153_evidence_report_web_unavailable():
    print("[VF-10] build_evidence_report 联网熔断提示")
    from src.core.agent_runner import build_evidence_report

    arts = {"main_results": [{"title": "P1", "doi": "10.1/x", "year": 2023, "text": "body"}],
            "web_results": []}
    base = build_evidence_report(arts, "citrus hlb", 2)
    check("未熔断默认无提示", "已标记不可用" not in base and "⚠" not in base)
    broke = build_evidence_report(arts, "citrus hlb", 2, web_unavailable=True)
    check("熔断后回执明示联网不可用", "⚠ 联网搜索本次请求已标记不可用" in broke, broke)
    check("熔断提示含不再重试语义", "未再重试" in broke and "如实声明缺口" in broke)
    # 熔断提示不应影响正常证据条目
    check("熔断提示下证据完整", "[1]" in broke and "[RAG]" in broke, broke)


# ── F-15.3-3 推理控制（配置默认不发送参数，off 时接线 thinking:disabled）──
def test_v8153_reasoning_mode():
    print("[VF-11] model.reasoning_mode 接线")
    check("默认值 = default（不发送任何参数）",
          getattr(settings, "MODEL_REASONING_MODE", "default") == "default")
    from src.core.llm_pool import get_llm

    llm_off = get_llm("t-model", "k-1", "https://x", max_tokens=64, thinking_off=True)
    check("thinking_off → model_kwargs.thinking.type=disabled",
          llm_off.model_kwargs.get("thinking", {}).get("type") == "disabled",
          str(llm_off.model_kwargs))
    llm_on = get_llm("t-model", "k-1", "https://x", max_tokens=64, thinking_off=False)
    check("默认 → 不发送 thinking 参数", llm_on.model_kwargs == {}, str(llm_on.model_kwargs))
    check("缓存键区分 thinking_off", llm_off is not llm_on)
    # 同参数复用同一实例（进程级复用不回退）
    llm_off2 = get_llm("t-model", "k-1", "https://x", max_tokens=64, thinking_off=True)
    check("同参数复用实例", llm_off2 is llm_off)


# ── F-15.3-4 联网失败详情（HTTP 状态码 / 超时分支）─────────────────
def test_v8153_web_failure_details():
    print("[VF-12] deepseek_web_search 失败详情")
    import src.tools.deepseek_web as dw
    from src.core.tracing import set_web_search_enabled

    set_web_search_enabled(True)

    class _FakeResp:
        status_code = 503
        text = "<html>Service Unavailable</html>"

    class _FakeRequestsHTTP:
        def post(self, *a, **k):
            return _FakeResp()

    class _FakeRequestsTimeout(_FakeRequestsHTTP):
        class exceptions:
            class Timeout(Exception):
                pass

        def post(self, *a, **k):
            raise self.exceptions.Timeout("timed out after 30s")

    old = dw.requests
    try:
        dw.requests = _FakeRequestsHTTP()
        c, a = dw.deepseek_web_search.func("最新柑橘行情")
        check("HTTP 失败 → [ERR_NETWORK] 且带状态码",
              c.startswith("[ERR_NETWORK]") and "HTTP 503" in c, c[:80])
        check("HTTP 失败 artifact 双键空", a == {"main_results": [], "web_results": []}, str(a))
        dw.requests = _FakeRequestsTimeout()
        c2, _a2 = dw.deepseek_web_search.func("最新柑橘行情")
        check("超时 → 单独分支含超时提示", c2.startswith("[ERR_NETWORK]") and "超时" in c2, c2[:80])
    finally:
        dw.requests = old
        set_web_search_enabled(False)


# ── F-15.3-5 联网失败熔断状态机（_web_streak_step 纯函数）──────────
def test_v8153_web_streak_state_machine():
    print("[VF-13] 联网失败熔断状态机")
    from src.core.agent_runner import _web_streak_step

    check("[ERR_NETWORK] 失败 +1", _web_streak_step(0, "[ERR_NETWORK] 调用失败") == 1)
    check("连续失败累计", _web_streak_step(1, "[ERR_EMPTY] 无内容") == 2)
    check("熔断占位 [ERR] 不计不重置", _web_streak_step(2, "[ERR] 联网搜索已连续失败…") == 2)
    check("[DISABLED] 清零", _web_streak_step(2, "[DISABLED] 联网搜索未开启") == 0)
    check("成功结果清零", _web_streak_step(2, "[ToolResult] deepseek_web_search…") == 0)
    check("空内容清零", _web_streak_step(1, "") == 0)


# ── F-15.3-6 提示词机制回归（防未来清理误删 v8.15.3 机制标记）──────
def test_v8153_prompt_mechanisms():
    print("[VF-14] 提示词机制标记存在性")
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]  # agent/
    ra = (root / "src/prompts/agents/retrieve-agent.md").read_text(encoding="utf-8")
    dg = (root / "src/prompts/system/decision_guide.md").read_text(encoding="utf-8")

    check("retrieve-agent 含数据源覆盖边界", "数据源覆盖边界" in ra)
    check("retrieve-agent 覆盖表(政策/新闻=本地不覆盖)",
          "政府工作报告" in ra and "最新新闻" in ra)
    check("retrieve-agent 早停阈值规则(通过≤2/过滤≥50%)",
          "通过 ≤2 条" in ra and "过滤占比 ≥50%" in ra)
    check("retrieve-agent 联网失败禁再调", "[ERR_NETWORK]" in ra and "禁止再次调用" in ra)
    # v8.15.3d: 原始问题直传 + 仲裁规则标记
    check("retrieve-agent 联网原始问题直传说明",
          "自动直传" in ra and "搜索参考关键词" in ra)
    check("decision_guide 含覆盖表+自审", "数据源覆盖边界与检索前自审" in dg)
    check("decision_guide 含回答前自审(引用对齐)", "回答前自审" in dg and "引用对齐" in dg)
    check("decision_guide 含证据来源仲裁规则", "证据来源仲裁规则" in dg)
    check("decision_guide 仲裁: 时效优先联网", "时效信息优先联网" in dg)
    check("decision_guide 仲裁: 冲突并列禁止折中",
          "并列展示" in dg and "禁止捏造折中值" in dg)


# ── F-15.3-7 单工具超时覆盖（联网工具 120s，防执行层 60s 硬上限误杀）──
def test_v8153_tool_timeout_override():
    print("[VF-15] 单工具超时覆盖")
    from src.tools.registry import _tool_exec_timeout

    check("联网工具放行 120s", _tool_exec_timeout("deepseek_web_search") == 120,
          str(_tool_exec_timeout("deepseek_web_search")))
    check("其他工具默认 60s", _tool_exec_timeout("citrus_rag_search") == 60,
          str(_tool_exec_timeout("citrus_rag_search")))
    # 误配/钳制（覆盖非法值 → 回退默认；负数钳到下限 1s）
    old = settings.TOOL_TIMEOUTS
    try:
        settings.TOOL_TIMEOUTS = {"deepseek_web_search": "abc"}
        check("非法覆盖回退默认", _tool_exec_timeout("deepseek_web_search") == 60,
              str(_tool_exec_timeout("deepseek_web_search")))
        settings.TOOL_TIMEOUTS = {"deepseek_web_search": -3}
        check("超时下限钳制 1s", _tool_exec_timeout("deepseek_web_search") == 1,
              str(_tool_exec_timeout("deepseek_web_search")))
    finally:
        settings.TOOL_TIMEOUTS = old


# ── F-15.3-8 原始问题直传联网工具（input 以原始问题开头，检索词作参考）──
def test_v8153d_original_query_direct():
    print("[VF-16] 原始问题直传联网工具")
    import src.tools.deepseek_web as dw
    from src.core.tracing import set_original_query, original_query
    from src.core.tracing import set_web_search_enabled

    set_original_query("2026年柑橘产业政府工作报告有哪些政策？")
    check("contextvar 写入/读取",
          original_query() == "2026年柑橘产业政府工作报告有哪些政策？")

    captured: dict = {}

    class _FakeResp:
        status_code = 200

        def json(self):
            return {"output": [
                {"type": "message", "content": [{"type": "output_text", "text":
                    "根据检索，[政府工作报告](https://gov.example.com/report)指出……"}]}]}

    class _FakeRequests:
        def post(self, url, **kw):
            captured["url"] = url
            captured["payload"] = kw.get("json", {})
            return _FakeResp()

    old = dw.requests
    set_web_search_enabled(True)
    try:
        dw.requests = _FakeRequests()
        c, a = dw.deepseek_web_search.func("citrus policy report 2026")
    finally:
        dw.requests = old
        set_original_query("")
        set_web_search_enabled(False)

    inp = str(captured.get("payload", {}).get("input", ""))
    check("input 以原始问题开头",
          inp.startswith("2026年柑橘产业政府工作报告有哪些政策？"), inp[:80])
    check("检索词作为搜索参考关键词",
          "搜索参考关键词" in inp and "citrus policy report 2026" in inp, inp)
    check("仍要求标注真实网址", "真实网址" in inp)
    check("返回内容含 [W1] 引用", "[W1]" in c, c[:120])
    ok_url = (a.get("web_results") or [{}])[0].get("url") == "https://gov.example.com/report"
    check("artifact 带 URL", bool(a.get("web_results")) and ok_url, str(a))
    # v8.15.3f: 正文 summary 必须进 artifact（此前只进 content，回执拿不到 → 只有 URL 无正文）
    check("artifact 带 web_summary 正文", bool(a.get("web_summary")), str(a.get("web_summary")))


# ── F-15.3-10 网络综述正文进确定性回执（根治"只有 URL 无正文"）──
def test_v8153f_web_summary_in_report():
    print("[VF-18] 网络综述正文进回执")
    from src.core.agent_runner import build_evidence_report

    arts = {
        "main_results": [],
        "web_results": [{"ref_id": "W1", "title": "2026柑橘产业报告",
                         "url": "https://gov.example.com/report", "source": "web"}],
        "web_summaries": ["2026年柑橘产业政府工作报告指出，全年产量同比增长……"],
    }
    rep = build_evidence_report(arts, "2026柑橘政府报告", 0)
    check("回执含网络综述段", "## 网络综述（DeepSeek 原生联网回答正文）" in rep, rep[:200])
    check("回执含综述正文", "2026年柑橘产业政府工作报告指出" in rep, rep[:300])
    check("回执仍含 [W1] 来源清单",
          "[W1] 2026柑橘产业报告" in rep and "https://gov.example.com/report" in rep, rep)
    check("回执引导 [n]/[Wn] 引用", "联网事实必须挂 [Wn]" in rep, rep[-200:])

    rep2 = build_evidence_report(
        {"main_results": [], "web_results": arts["web_results"]}, "q", 0)
    check("无摘要时不渲染综述段", "网络综述" not in rep2)


# ── F-15.3-9 加载阶段状态推送（消除静默期，前端零改动复用现有事件）──
def test_v8153e_load_progress_events():
    print("[VF-17] load 阶段进度事件")
    import asyncio
    import json
    from pathlib import Path
    from src.core.progress_bus import (
        set_request_queue, clear_request_queue, emit_status, emit_progress)

    q = asyncio.Queue()
    set_request_queue(q)
    try:
        emit_status("step_active", step_id="load")
        emit_progress("tool_progress", {
            "message": "已加载历史 12 条，正在理解问题并规划检索…", "tool_call_id": ""})
        emit_status("step_done", step_id="load")
        items = [q.get_nowait() for _ in range(3)]
    finally:
        clear_request_queue()

    check("事件入队 3 条", len(items) == 3, str(items))
    first = json.loads(items[0]["data"]) if isinstance(items[0].get("data"), str) else {}
    check("step_active 载荷(event=status, stage=step_active)",
          items[0]["event"] == "status" and first.get("stage") == "step_active"
          and first.get("step_id") == "load", str(items[0]))
    second = json.loads(items[1]["data"]) if isinstance(items[1].get("data"), str) else {}
    check("tool_progress 带自定义文案（前端 updateStatus 直显）",
          items[1]["event"] == "tool_progress"
          and "正在理解问题并规划检索" in second.get("message", ""), str(items[1]))
    third = json.loads(items[2]["data"]) if isinstance(items[2].get("data"), str) else {}
    check("step_done 载荷", items[2]["event"] == "status" and third.get("stage") == "step_done",
          str(items[2]))

    # 源码断言：expert/light 两个 load 节点都已接上推送（防静默期回归）
    root = Path(__file__).resolve().parents[1]  # agent/
    ex = (root / "src/graph/expert_graph.py").read_text(encoding="utf-8")
    lg = (root / "src/graph/light_graph.py").read_text(encoding="utf-8")
    check("expert load 节点含 step_active+tool_progress 推送",
          'emit_status("step_active", step_id="load")' in ex
          and 'emit_status("step_done", step_id="load")' in ex
          and "正在理解问题并规划检索" in ex)
    check("light load 节点含完成推送", 'emit_status("step_done", step_id="load")' in lg
          and "已加载历史" in lg)


# ── 汇总 ──────────────────────────────────────────────────────────
def _summary():
    print(f"\n[VF-15.3] PASS {len(passed)} / FAIL {len(failed)}"
          f"  ({len(passed) + len(failed)} total)")
    for f in failed:
        print(f"  [FAIL] {f}")
    return len(failed) == 0


if __name__ == "__main__":
    import types as _types
    _funcs = [v for v in list(globals().values())
              if isinstance(v, _types.FunctionType)
              and v.__name__.startswith("test_")]
    for _f in sorted(_funcs, key=lambda x: x.__name__):
        try:
            _f()
        except Exception as _e:
            if os.environ.get("PYTEST_CURRENT_TEST"):
                raise
            print(f"  {_e}")
    ok = _summary()
    sys.exit(0 if ok else 1)