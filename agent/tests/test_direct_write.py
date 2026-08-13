# -*- coding: utf-8 -*-
"""direct-write 双层判定回归测试（v8.3.1）:
  ① _has_retrieval_markers 否定排除（启发式，免费）
  ② _classify_document LLM 语义确认（失败保守回退 False）
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.expert_graph import _has_retrieval_markers, _classify_document

passed, failed = [], []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
    else:
        failed.append(name)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def test_retrieval_markers():
    print("[①] 否定排除 _has_retrieval_markers")

    # 用户踩坑场景: 用户指令(含标题/摘要/引言/关键词) + 检索结果(构建函数输出带标记)
    instruction = (
        "基于以下检索到的柑橘文献撰写综述，需包含：标题、摘要、关键词、引言、正文、结论、参考文献。"
        "1) Butelli et al. 2017 DOI: 10.1104/pp.16.01701; 2) DOI: 10.1111/tpj.16866; "
        "3) DOI: 10.1093/hr/uhad190; 4) DOI: 10.3390/genes11070807; 5) DOI: 10.1093/hr/uhac175; "
        "6) DOI: 10.1016/j.xplc.2020.100138; 7) DOI: 10.1111/jpi.70078; 8) DOI: 10.1111/pbi.70371; "
        "9) DOI: 10.1007/s11032-024-01490-9; 10) DOI: 10.3390/plants12233965"
    )
    retrieval = "检索结果:\n[1] Paper A\n    Authors: N/A\n    Year: 2020  DOI: 10.1/x\n    Abstract: 花青素"
    check("指令+多条DOI → 排除", _has_retrieval_markers(instruction) is True)
    check("检索结果标记 → 排除", _has_retrieval_markers(retrieval) is True)
    check("编号+字段行 → 排除", _has_retrieval_markers("[1] Title\n    Authors: N/A\n    Year: 2023") is True)

    # 真文档不应被排除（否定排除不能误伤）
    doc = (
        "# 柑橘花青素调控机制综述\n## 摘要\n花青素积累受 Ruby/MYB 调控。\n"
        "## 引言\n重要品质性状。\n## 结论\n综上所述。\n## 参考文献\n[1] Butelli 2017. [2] 综述内容"
    )
    check("真文档 → 不排除（交由 LLM 确认）", _has_retrieval_markers(doc) is False)


async def _classify_mock(resp_content, fail=False):
    """mock _classify_document 的 OpenAI 调用路径无法直接注入，改为验证真实函数的
    失败回退：传入会抛错的环境（无 API key 时 OpenAI 调用会失败 → 返回 False 保守回退）。"""
    return resp_content


def test_classify_fallback():
    print("[②] LLM 分类 _classify_document 保守回退")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'graph', 'expert_graph.py'),
               encoding='utf-8').read()
    check("存在 _classify_document", "_classify_document" in src)
    check("输出限制 max_tokens=5（单 token 分类）", "max_tokens=5" in src)
    check("失败回退 return False", "conservative fallback" in src and "return False" in src)
    check("温度 0", "temperature=0" in src)

    # 真实调用失败路径: 无 API key 时应抛错并返回 False（保守回退，不直写）
    loop = asyncio.new_event_loop()
    try:
        result = loop.run_until_complete(_classify_document("测试文本" * 100))
        check("无 key 环境 → 回退 False", result is False)
    finally:
        loop.close()


def test_branch_wiring():
    print("[分支接线] direct-write 双层条件")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'graph', 'expert_graph.py'),
               encoding='utf-8').read()
    check("分支含否定排除", "_has_retrieval_markers(context)" in src)
    check("分支含 LLM 确认", "await _classify_document(context)" in src)
    check("旧启发式定义已移除", "def _looks_like_document" not in src)


if __name__ == "__main__":
    test_retrieval_markers()
    test_classify_fallback()
    test_branch_wiring()
    print(f"\n结果: {len(passed)} passed / {len(failed)} failed")
    if failed:
        print("失败项:", failed)
        sys.exit(1)
