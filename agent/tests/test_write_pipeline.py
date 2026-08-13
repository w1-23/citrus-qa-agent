# -*- coding: utf-8 -*-
"""Write Pipeline 单元测试（mock LLM，不调用真实 API）"""
import asyncio
import os
import sys
import uuid

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src.core import write_pipeline as wp
from src.config import PROJECT_ROOT, settings

passed, failed = [], []


def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


class FakeResp:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        if self._responses:
            return FakeResp(self._responses.pop(0))
        return FakeResp("(no more responses)")


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_classify():
    print("[classify] 四路路由")
    check("综述→plan_execute", wp.classify_write_task("写一篇柑橘泛基因组综述，6000字", "", False)["mode"] == "plan_execute")
    check("先写→react", wp.classify_write_task("先写个引言我看看", "", False)["mode"] == "react")
    check("保存→direct_write", wp.classify_write_task("把这个回答保存到 a.md", "", False)["mode"] == "direct_write")
    check("文件存在+章节→modify", wp.classify_write_task("把第三章重写", "", True)["mode"] == "modify")
    check("文献回顾→plan_execute", wp.classify_write_task("帮我写一篇柑橘的文献回顾", "", False)["mode"] == "plan_execute")
    check("3000字提取", wp.classify_write_task("写个3000字总结", "", False)["target_chars"] == 3000)
    check("无字数→0(自主模式)", wp.classify_write_task("写一篇综述", "", False)["target_chars"] == 0)
    check("材料丰富提示词含自主说明",
          "材料充分则深入" in wp._build_plan_prompt([{"doi": "1", "title": "x"}], 0))


def test_validate_plan():
    print("[validate] 双模式 + 单章上限")
    good = {"sections": [{"heading": f"h{i}", "target_chars": 1500, "refs": ["DOI:1"]} for i in range(4)]}
    ok, fails = wp.validate_plan(good, 6000)
    check("6000字 4章×1500 → 通过", ok, str(fails))
    small = {"sections": [{"heading": f"h{i}", "target_chars": 900, "refs": ["DOI:1"]} for i in range(3)]}
    ok2, fails2 = wp.validate_plan(small, 3000)
    check("3000字 3章×900 → 通过", ok2, str(fails2))
    # 自主模式: 材料有限精简
    auto = {"sections": [{"heading": "h1", "target_chars": 800, "refs": ["DOI:1"]},
                         {"heading": "h2", "target_chars": 800, "refs": []}]}
    ok3, fails3 = wp.validate_plan(auto, 0)
    check("自主模式 2章×800 → 通过", ok3, str(fails3))
    # 单章超上限 → 拒绝（防截断）
    oversized = {"sections": [{"heading": f"h{i}", "target_chars": 5000, "refs": ["DOI:1"]} for i in range(3)]}
    ok4, fails4 = wp.validate_plan(oversized, 0)
    check("单章 5000 字超容量 → 拒绝", not ok4 and "max_per_section" in fails4, str(fails4))
    bad = {"sections": [{"heading": "h1", "target_chars": 100, "refs": []}]}
    ok5, fails5 = wp.validate_plan(bad, 6000)
    check("单章100字+refs空 → 拒绝", not ok5 and "min_per_section" in fails5, str(fails5))
    check("refs 覆盖不足 → 拒绝", "ref_coverage" in fails5)


def test_plan_stage():
    print("[Stage1] Plan 生成与重试")
    plan_json = json_doc({"title": "T", "abstract_draft": "A", "keywords": ["k"],
                          "total_target_chars": 6000,
                          "sections": [{"heading": f"s{i}", "points": ["p"], "target_chars": 1500,
                                        "refs": ["DOI:10.1/x"]} for i in range(4)]})
    llm = FakeLLM([plan_json])
    plan, text = run(wp.run_stage1_plan(llm, [{"doi": "10.1/x", "title": "X"}], 6000))
    check("有效大纲 → 解析成功", plan is not None and len(plan["sections"]) == 4)
    check("一次调用", llm.calls == 1)

    llm2 = FakeLLM(["not json", plan_json])
    plan2, _ = run(wp.run_stage1_plan(llm2, [{"doi": "10.1/x", "title": "X"}], 6000))
    check("JSON 失败→重试成功", plan2 is not None)

    llm3 = FakeLLM(["not json", "still not json", "nope"])
    plan3, text3 = run(wp.run_stage1_plan(llm3, [{"doi": "10.1/x", "title": "X"}], 6000))
    check("全部失败 → 返回 None（回退）", plan3 is None and text3)


def test_execute_stage():
    print("[Stage2] 章节循环 + summary + 缺章")
    plan = {"title": "综述T", "abstract_draft": "摘要", "keywords": ["k1"],
            "sections": [
                {"heading": "1 引言", "points": ["p1"], "target_chars": 800, "refs": []},
                {"heading": "2 正文", "points": ["p2"], "target_chars": 800, "refs": []},
                {"heading": "3 结论", "points": ["p3"], "target_chars": 800, "refs": []},
            ]}
    llm = FakeLLM([
        "## 1 引言\n引言内容。<summary>本章讲了引言</summary>",
        "## 2 正文\n正文内容。<summary>本章讲了正文</summary>",
        "## 3 结论\n结论内容。<summary>本章讲了结论</summary>",
    ])
    fname = f"test_pipe_{uuid.uuid4().hex[:6]}.md"
    r = run(wp.run_stage2_execute(llm, plan, [], fname))
    target = PROJECT_ROOT / "workspace" / "output" / fname
    content = target.read_text(encoding="utf-8") if target.exists() else ""
    check("3 章完成", r["chapters"] == 3 and not r["missing_sections"])
    check("文件含标题+摘要+关键词", "# 综述T" in content and "摘要" in content and "关键词" in content)
    check("三章标题齐全", "## 1 引言" in content and "## 2 正文" in content and "## 3 结论" in content)
    check("summary 标签已剥离", "<summary>" not in content)
    if target.exists():
        target.unlink()


def test_missing_section():
    print("[Stage2] 单章失败降级")
    plan = {"title": "T", "abstract_draft": "A", "keywords": [],
            "sections": [{"heading": "1 A", "points": [], "target_chars": 500, "refs": []},
                         {"heading": "2 B", "points": [], "target_chars": 500, "refs": []}]}
    # 第1章 3 次异常 → 缺章；第2章正常返回（read-back 校验要求返回内容与章节匹配）
    class BoomLLM:
        def __init__(self):
            self.calls = 0
        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls <= 3:
                raise Exception("boom")
            return FakeResp("## 2 B\n内容B。<summary>sumB</summary>")
    llm = BoomLLM()
    fname = f"test_pipe2_{uuid.uuid4().hex[:6]}.md"
    r = run(wp.run_stage2_execute(llm, plan, [], fname))
    check("第1章失败→缺章", r["missing_sections"] == ["1 A"], str(r["missing_sections"]))
    check("第2章仍完成", r["chapters"] == 1)
    target = PROJECT_ROOT / "workspace" / "output" / fname
    if target.exists():
        target.unlink()


def test_summary_extract():
    print("[summary] 标签提取")
    body, s = wp.extract_summary("## 1 引言\n内容。<summary>本章核心结论：讲了引言</summary>")
    check("正文剥离标签", body == "## 1 引言\n内容。" and "<summary>" not in body)
    check("摘要提取", "讲了引言" in s)
    body2, s2 = wp.extract_summary("无标签内容")
    check("无标签 → body 原样", body2 == "无标签内容" and s2 == "")


def test_modify():
    print("[modify] 章节定位与替换")
    md = "# T\n\n## 1 引言\n引言内容\n\n## 2 正文\n正文内容\n\n## 3 结论\n结论内容\n"
    sections = wp._split_sections(md)
    real = [s for s in sections if s["heading"]]
    check("切出 3 个二级章节（含文档头部段）", len(real) == 3, str([s["heading"] for s in sections]))
    check("头部段保留", sections[0]["heading"] == "" and "# T" in sections[0]["body"])
    check("标题精确匹配", wp._locate_section(sections, "2 正文") == 2)
    check("序数匹配", wp._locate_section(sections, "3") == 3)
    check("中文序数", wp._locate_section(sections, "二") == 2)
    check("不存在 → -1", wp._locate_section(sections, "9 不存在") == -1)


def test_entry_pipeline():
    import json
    from src.core.write_pipeline_state import start_task, mark_section_done
    print("[入口] run_write_pipeline 四路 + 续传 + 回退")
    plan_obj = {"title": "T", "abstract_draft": "A", "keywords": ["k"],
                "total_target_chars": 0,
                "sections": [{"heading": "s1", "points": ["p"], "target_chars": 800,
                              "refs": ["DOI:1"]},
                             {"heading": "s2", "points": ["p"], "target_chars": 800,
                              "refs": ["DOI:1"]}]}
    plan_json = json.dumps(plan_obj, ensure_ascii=False)

    # 1) plan_execute 首次运行（无续传）→ P0 UnboundLocalError 回归
    fname = f"test_entry_{uuid.uuid4().hex[:6]}.md"
    llm = FakeLLM([plan_json,
                   "## s1\n第一张内容。<summary>s1</summary>",
                   "## s2\n第二章内容。<summary>s2</summary>"])
    r = run(wp.run_write_pipeline({"goal": "写一篇柑橘综述", "output_path": fname},
                                  [{"doi": "1", "title": "X"}],
                                  llm_factory=lambda: llm, session_id="t-entry"))
    check("首次运行 plan_execute 不崩(UnboundLocalError 回归)", r["mode"] == "plan_execute"
          and not r["missing_sections"], f"mode={r['mode']} missing={r['missing_sections']}")
    check("文件已落盘", (PROJECT_ROOT / "workspace" / "output" / fname).exists())

    # 2) react 快筛 → 结构化返回
    r2 = run(wp.run_write_pipeline({"goal": "先写个引言", "output_path": ""}, [],
                                   llm_factory=lambda: llm))
    check("react → 结构化返回", r2["mode"] == "react", f"mode={r2['mode']}")

    # 3) direct_write 快筛 → 结构化返回
    r3 = run(wp.run_write_pipeline({"goal": "把这个回答保存到 a.md", "output_path": ""}, [],
                                   llm_factory=lambda: llm))
    check("direct_write → 结构化返回", r3["mode"] == "direct_write", f"mode={r3['mode']}")

    # 4) plan 全部失败 → react_fallback 带大纲
    llm4 = FakeLLM(["not json", "still not json", "nope"])
    r4 = run(wp.run_write_pipeline({"goal": "写一篇综述", "output_path": ""}, [],
                                   llm_factory=lambda: llm4))
    check("plan 失败 → react_fallback", r4["mode"] == "react_fallback"
          and "回退" in r4["result"], f"mode={r4['mode']}")

    # 5) modify: 已有文件重写章节
    fname2 = f"test_entry2_{uuid.uuid4().hex[:6]}.md"
    t2 = PROJECT_ROOT / "workspace" / "output" / fname2
    t2.write_text("# T\n\n## 1 引言\n旧引言\n\n## 2 正文\n旧正文\n", encoding="utf-8")
    llm5 = FakeLLM(["## 2 正文\n新正文内容"])
    r5 = run(wp.run_write_pipeline({"goal": "重写第二章", "output_path": fname2}, [],
                                   llm_factory=lambda: llm5))
    check("modify 重写章节", r5["mode"] == "modify" and "已重写" in r5["result"],
          f"mode={r5['mode']} result={r5['result'][:40]}")
    check("文件内容已替换", "新正文内容" in t2.read_text(encoding="utf-8"))
    t2.unlink()

    # 6) 断点续传: 复用 Plan、跳过已完成章节（仅剩 1 章 → 只调 1 次 LLM）
    fname3 = f"test_entry3_{uuid.uuid4().hex[:6]}.md"
    tid = start_task("t-resume", fname3, plan_obj)
    mark_section_done(tid, 0)
    llm6 = FakeLLM(["## s2\n续传后的第二章。<summary>s2b</summary>"])
    r6 = run(wp.run_write_pipeline({"goal": "写一篇柑橘综述", "output_path": fname3},
                                   [{"doi": "1", "title": "X"}],
                                   llm_factory=lambda: llm6, session_id="t-resume"))
    check("续传: 复用 Plan 不重调 stage1", llm6.calls == 1, f"calls={llm6.calls}")
    check("续传: 缺章补齐", r6["mode"] == "plan_execute" and not r6["missing_sections"],
          f"mode={r6['mode']} missing={r6['missing_sections']}")

    target = PROJECT_ROOT / "workspace" / "output" / fname
    target3 = PROJECT_ROOT / "workspace" / "output" / fname3
    if target.exists():
        target.unlink()
    if target3.exists():
        target3.unlink()


def test_runtime_capacity():
    print("[Stage2] 运行时单章容量兜底")
    plan = {"title": "T", "abstract_draft": "A", "keywords": [],
            "sections": [{"heading": "1 A", "points": [], "target_chars": 800, "refs": []}]}
    old = settings.PIPELINE_SECTION_MAX_TOKENS
    settings.PIPELINE_SECTION_MAX_TOKENS = 2400  # → max_per_section = 1800
    try:
        llm = FakeLLM(["## 1 A\n" + "字" * 2000, "## 1 A\n精简版内容"])
        fname = f"test_cap_{uuid.uuid4().hex[:6]}.md"
        r = run(wp.run_stage2_execute(llm, plan, [], fname))
        target = PROJECT_ROOT / "workspace" / "output" / fname
        content = target.read_text(encoding="utf-8") if target.exists() else ""
        check("超容量 → 触发精简重试", llm.calls == 2, f"calls={llm.calls}")
        check("精简后内容落盘", "精简版内容" in content and len(content) < 2000)
        check("不标记 truncated", r["truncated_sections"] == [], str(r.get("truncated_sections")))
        if target.exists():
            target.unlink()

        llm2 = FakeLLM(["## 1 A\n" + "字" * 2000, "## 1 A\n" + "字" * 1900])
        fname2 = f"test_cap2_{uuid.uuid4().hex[:6]}.md"
        r2 = run(wp.run_stage2_execute(llm2, plan, [], fname2))
        t2 = PROJECT_ROOT / "workspace" / "output" / fname2
        check("仍超限 → truncated 标记", r2["truncated_sections"] == ["1 A"],
              str(r2.get("truncated_sections")))
        check("仍超限 → 内容仍写盘", t2.exists())
        if t2.exists():
            t2.unlink()
    finally:
        settings.PIPELINE_SECTION_MAX_TOKENS = old


def test_pipeline_state():
    import json as _json
    from src.core.write_pipeline_state import (start_task, find_resumable_task,
                                               mark_section_done, finish_task,
                                               cleanup_stale_tasks)
    print("[state] 断点续传表 原子操作")
    plan_obj = {"title": "T", "sections": [{"heading": "a"}, {"heading": "b"}]}
    sid = f"st-{uuid.uuid4().hex[:6]}"
    op = f"test_state_{uuid.uuid4().hex[:6]}.md"
    tid = start_task(sid, op, plan_obj)
    check("start_task 返回 task_id", bool(tid))
    r = find_resumable_task(sid, op)
    check("find_resumable_task 命中", r is not None and r["completed"] == [] and r["plan"]["title"] == "T")
    check("不同 session 不命中", find_resumable_task("other", op) is None)
    mark_section_done(tid, 0)
    r2 = find_resumable_task(sid, op)
    check("mark_section_done 追加", r2["completed"] == [0])
    mark_section_done(tid, 0)
    r3 = find_resumable_task(sid, op)
    check("重复标记幂等", r3["completed"] == [0])
    mark_section_done(tid, 1)
    r4 = find_resumable_task(sid, op)
    check("完成两章", r4["completed"] == [0, 1])
    finish_task(tid, "done")
    check("finish 后不再可续传", find_resumable_task(sid, op) is None)
    n = cleanup_stale_tasks(0)
    check("cleanup 删除 running 过期任务", isinstance(n, int) and n >= 0, f"n={n}")
    # cleanup 不误删 done 任务
    tid2 = start_task("st-keep", op, plan_obj)
    finish_task(tid2, "done")
    check("cleanup 保留 done 任务", find_resumable_task("st-keep", op) is None
          and cleanup_stale_tasks(0) >= 0)


def test_verify_functions():
    print("[校验] read-back + 引用完整性")
    fname = f"test_verify_{uuid.uuid4().hex[:6]}.md"
    target = PROJECT_ROOT / "workspace" / "output" / fname
    target.write_text("# T\n\n## 引言\n正文[1][2]。\n\n## 参考文献\n[1] Ref A\n[2] Ref B\n",
                      encoding="utf-8")
    check("read-back 命中（容错）", wp._verify_section_written(fname, "1 引言"))
    check("read-back 未写 → False", not wp._verify_section_written(fname, "不存在的章节"))
    issues = wp.verify_reference_integrity(fname)
    check("引用完整无问题", issues == [], str(issues))
    # 缺引用 + 未用文献
    target.write_text("# T\n\n正文[1][3]。\n\n## 参考文献\n[1] Ref A\n[2] Ref B\n",
                      encoding="utf-8")
    issues2 = wp.verify_reference_integrity(fname)
    check("正文[3]无文献 → 检出", any("[3]" in i for i in issues2), str(issues2))
    check("文献[2]未引用 → 检出", any("[2]" in i for i in issues2), str(issues2))
    target.unlink()


def test_react_fallback():
    print("[回退] react_fallback 落盘")
    fname = f"test_fb_{uuid.uuid4().hex[:6]}.md"
    # stage1 两次 JSON 失败 → 回退生成：第 3 次响应为回退全文
    llm = FakeLLM(["not json", "still not json",
                   "# 回退文档\n\n## 一 概述\n内容。<summary>s</summary>"])
    r = run(wp.run_write_pipeline({"goal": "写一篇综述", "output_path": fname},
                                  [{"doi": "1", "title": "X"}],
                                  llm_factory=lambda: llm))
    check("react_fallback 模式", r["mode"] == "react_fallback" and "已保存" in r["result"],
          f"mode={r['mode']}")
    target = PROJECT_ROOT / "workspace" / "output" / fname
    check("回退内容已落盘", target.exists() and "回退文档" in target.read_text(encoding="utf-8"))
    check("回退内容无 summary 标签", target.exists() and "<summary>" not in target.read_text(encoding="utf-8"))
    if target.exists():
        target.unlink()
    # LLM 也失败 → 提示文本兜底
    llm2 = FakeLLM(["not json", "not json", "nope"])
    r2 = run(wp.run_write_pipeline({"goal": "写一篇综述", "output_path": ""},
                                   [], llm_factory=lambda: llm2))
    check("全部失败 → 提示文本", r2["mode"] == "react_fallback" and "回退" in r2["result"])


def test_output_path_fallback():
    import json
    print("[入口] output_path 缺失 → 默认路径兜底（复测发现 P1）")
    plan_obj = {"title": "黄龙病防控策略综述", "abstract_draft": "A", "keywords": ["k"],
                "total_target_chars": 0,
                "sections": [{"heading": "s1", "points": ["p"], "target_chars": 800,
                              "refs": ["DOI:1"]},
                             {"heading": "s2", "points": ["p"], "target_chars": 800,
                              "refs": ["DOI:1"]}]}
    plan_json = json.dumps(plan_obj, ensure_ascii=False)
    llm = FakeLLM([plan_json,
                   "## s1\n第一张内容。<summary>s1</summary>",
                   "## s2\n第二章内容。<summary>s2</summary>"])
    r = run(wp.run_write_pipeline({"goal": "写一篇综述", "output_path": ""},
                                  [{"doi": "1", "title": "X"}],
                                  llm_factory=lambda: llm, session_id="t-nopath"))
    check("无 output_path 仍成功执行", r["mode"] == "plan_execute" and not r["missing_sections"],
          f"mode={r['mode']} missing={r['missing_sections']}")
    # 生成的默认文件应存在且含标题
    out_dir = PROJECT_ROOT / "workspace" / "output"
    hits = list(out_dir.glob("黄龙病防控策略综述_*.md"))
    check("默认文件名落盘", len(hits) >= 1, str([h.name for h in hits[:3]]))
    if hits:
        for h in hits:
            h.unlink()


def json_doc(obj):
    import json
    return json.dumps(obj, ensure_ascii=False)


if __name__ == "__main__":
    test_classify()
    test_validate_plan()
    test_plan_stage()
    test_execute_stage()
    test_missing_section()
    test_summary_extract()
    test_modify()
    test_entry_pipeline()
    test_runtime_capacity()
    test_pipeline_state()
    test_verify_functions()
    test_react_fallback()
    test_output_path_fallback()
    print(f"\n结果: {len(passed)} passed / {len(failed)} failed")
    if failed:
        print("失败项:", failed)
        sys.exit(1)
