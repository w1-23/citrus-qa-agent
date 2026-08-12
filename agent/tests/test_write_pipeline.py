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
    llm = FakeLLM([None, "## 1 A\n内容A。<summary>sumA</summary>"])  # 第1章失败3次
    # FakeLLM 返回 None 会 AttributeError → 走重试 → 最终 missing
    class BoomLLM:
        def __init__(self):
            self.calls = 0
        async def ainvoke(self, messages):
            self.calls += 1
            if self.calls <= 3:
                raise Exception("boom")
            return FakeResp("## 1 A\n内容A。<summary>sumA</summary>")
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
    print(f"\n结果: {len(passed)} passed / {len(failed)} failed")
    if failed:
        print("失败项:", failed)
        sys.exit(1)
