# -*- coding: utf-8 -*-
"""direct-write 误判回归测试：复现"用户指令+检索列表被误判为文档"场景 (v8.3.1)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.graph.expert_graph import _looks_like_document, _build_full_retrieval_context

passed, failed = [], []


def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def test_direct_write_misjudge_fixed():
    print("[direct-write] 误判场景回归")

    # 场景 1（用户实际踩坑）: 用户指令(含 标题/摘要/引言/关键词) + 检索结果列表
    user_instruction = (
        "基于以下检索到的柑橘文献撰写综述：1) Butelli et al. 2017, DOI: 10.1104/pp.16.01701; "
        "2) The Plant Journal, DOI: 10.1111/tpj.16866; 3) DOI: 10.1093/hr/uhad190; "
        "4) DOI: 10.3390/genes11070807; 5) DOI: 10.1093/hr/uhac175; 6) DOI: 10.1016/j.xplc.2020.100138; "
        "7) RAG来源; 8) DOI: 10.1111/jpi.70078; 9) DOI: 10.1186/s12864-017-4366-2; "
        "10) DOI: 10.1111/pbi.70371; 11) DOI: 10.32473/edis-HS1472-2023; "
        "12) DOI: 10.1007/s11032-024-01490-9; 13) DOI: 10.3390/plants12233965; "
        "14) DOI: 10.3390/biology11071078; 15) DOI: 10.3389/fpls.2022.945738。"
        "综述需包含：标题、摘要、关键词、引言、正文、结论、局限与边界、参考文献。"
    ) * 1
    retrieval = _build_full_retrieval_context(
        [{"title": f"Paper {i}", "authors": "N/A", "year": 2020, "doi": f"10.1{i}/x.y",
          "abstract": "花青素调控"} for i in range(10)],
        [],
    )
    mixed = user_instruction + "\n\n" + retrieval
    check("含'摘要/引言/关键词'的用户指令+检索列表 → 不判为文档",
          _looks_like_document(mixed) is False, f"len={len(mixed)}")
    check("纯检索结果(构建函数输出) → 不判为文档",
          _looks_like_document(retrieval) is False)

    # 场景 2: 真文档 → 仍应判为文档（不能误伤；direct write 要求 >2000 字符，测试用 >200 即可触发）
    real_doc = (
        "# 柑橘花青素调控机制综述\n"
        "## 摘要\n柑橘花青素积累受 Ruby/MYB 转录因子核心调控，涉及结构基因 CHS、DFR、ANS、UFGT 的协同表达。\n"
        "## 引言\n花青素是柑橘果实重要的品质性状，其在血橙等品种中的组织特异性积累受到广泛关注。\n"
        "## 正文\nRuby1 启动子中的逆转座子插入变异是花青素驯化选择的关键事件。低温通过冷响应 ERF 转录因子激活 Ruby1 表达。\n"
        "## 结论\n综上所述，柑橘花青素调控网络以 Ruby/MYB 为核心，受环境与表观遗传多重调控。\n"
        "## 参考文献\n[1] Butelli et al. 2017, Plant Physiology. [2] 相关综述内容。"
    )
    check("真文档(#标题+摘要+引言+正文+结论+参考文献) → 判为文档",
          _looks_like_document(real_doc) is True, f"len={len(real_doc)}")

    # 场景 3: 短文本 → 不判（direct write 需 >2000 字符，这里防御 len<200）
    check("短文本 → 不判为文档", _looks_like_document("摘要：短") is False)

    # 场景 4: 构建函数输出带"检索结果:"标记
    check("检索结果带标记", retrieval.startswith("检索结果:"), retrieval[:20])


if __name__ == "__main__":
    test_direct_write_misjudge_fixed()
    print(f"\n结果: {len(passed)} passed / {len(failed)} failed")
    if failed:
        print("失败项:", failed)
        sys.exit(1)
