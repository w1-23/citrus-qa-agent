# -*- coding: utf-8 -*-
"""v8.16.2 回答质量修复回归：回执重排（截尾防护）/ cap 扩容 / 数字专名保真规则 /
草稿业务日志 / load 分段计时插桩。

全部离线、无模型、无网络（回执为纯函数，容量以实测数据编码为回归护栏）。
覆盖：
  VF-25  回执重排 + 容量护栏（rich 场景 47.6K 必须 ≤ cap 且 [Wn]/导引前置）
  VF-26  cap 60000 + 源码接线（draft_done/draft_skipped 业务日志、load_stages 插桩）
  VF-27  decision_guide 数字与专名保真规则落地
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


def _arts(n_docs, chunk_len, n_web_sum=0, n_web=0):
    """构造符合生产形态的回执输入：chunk ≤2000（v8.4.6 语料约束），综述 ≤4000，
    联网条目 500 字摘要——与 v8.16.2 实测脚本同一数据形态。"""
    main = [{"title": f"Paper {i}", "doi": f"10.{i}/x", "year": 2024,
             "text": "x" * chunk_len} for i in range(n_docs)]
    ws = ["w" * 4000 for _ in range(n_web_sum)]
    web = [{"title": f"W{i}", "url": f"https://x/{i}", "abstract": "a" * 500}
           for i in range(n_web)]
    return {"main_results": main, "web_results": web, "web_summaries": ws}


# ── VF-25 回执重排 + 容量护栏（v8.16.2 Phase 1 核心）──────────────
def test_v8162_report_reorder():
    print("[VF-25] build_evidence_report 重排 + 容量护栏")
    from src.core.agent_runner import build_evidence_report

    rich = _arts(15, 2000, 3, 8)
    rep = build_evidence_report(rich, "2026 citrus policy", 8)

    # v8.16.2 实测：rich 场景 47,587 chars，旧 cap 40000 截掉 7.6K ——
    # 且旧顺序下截断的正是文末 [W1..W8] 清单与引用导引（问题1 实证根因）。
    i_guide = rep.find("引用编号请使用下列")
    i_sum = rep.find("## 网络综述")
    i_w1 = rep.find("[W1]")
    i_first = rep.find("[1][RAG]")
    check("引用导引前置（截断线上方）", 0 < i_guide < i_sum < i_w1 < i_first,
          f"{i_guide}/{i_sum}/{i_w1}/{i_first}")
    check("rich 场景零截断（≤ cap）", len(rep) <= settings.TOOL_RESULT_CAPS["retrieve-agent"],
          f"{len(rep)} > {settings.TOOL_RESULT_CAPS['retrieve-agent']}")
    check("旧 cap 40000 本会截断（护栏意义）", len(rep) > 40000, len(rep))
    # 重排前后总量不变（内容完整、仅顺序变化）——抽验关键段存在
    check("综述含 DeepSeek 原生联网标题", "## 网络综述（DeepSeek 原生联网回答正文）" in rep)
    check("联网条目 8 条齐全", "[W8]" in rep and "[W9]" not in rep)

    # 空结果分支不受重排影响
    empty = build_evidence_report({"main_results": [], "web_results": [],
                                   "web_summaries": []}, "q", 0)
    check("空结果分支保留", "未检索到相关文献。" in empty and "网络综述" not in empty)

    # 最坏（chunk 3000 渲染阀）残余截断记录在案：不回归到"导引被切"
    worst = build_evidence_report(_arts(15, 3000, 3, 8), "q", 8)
    check("worst 残余截断不吞导引", "引用编号请使用下列" in worst)


# ── VF-26 cap + 源码接线 ──────────────────────────────────────────
def test_v8162_cap_and_wiring():
    print("[VF-26] cap 60000 + 接线断言")
    cap = settings.TOOL_RESULT_CAPS.get("retrieve-agent")
    check("retrieve-agent cap == 60000", cap == 60000, str(cap))

    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(src, "src", "tools", "deepseek_web.py"),
              encoding="utf-8") as f:
        dw = f.read()
    check("draft_worker 有 draft_done/draft_skipped 业务日志",
          'blog("draft_done"' in dw and 'blog("draft_skipped"' in dw)
    check("_draft_blog fail-soft 封装存在", "def _draft_blog" in dw)

    with open(os.path.join(src, "src", "core", "context_manager.py"),
              encoding="utf-8") as f:
        cm = f.read()
    check("load 有 load_stages 分段插桩", 'diag("load_stages"' in cm)
    check("load 计时变量完备", "_t_load" in cm and "_t_hints0" in cm)


# ── VF-27 数字与专名保真规则 ─────────────────────────────────────
def test_v8162_prompt_rule():
    print("[VF-27] decision_guide 数字与专名保真")
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(src, "src", "prompts", "system", "decision_guide.md"),
              encoding="utf-8") as f:
        txt = f.read()
    check("含数字与专名保真规则（v8.16.2）",
          "数字与专名保真（v8.16.2）" in txt
          and "禁止把具体数字概括为" in txt
          and "保留原始数值" in txt)


# ── 汇总 ──────────────────────────────────────────────────────────
def _summary():
    print(f"\n[VF-16.2] PASS {len(passed)} / FAIL {len(failed)}"
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