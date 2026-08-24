# -*- coding: utf-8 -*-
"""v9.0/v9.1 固定 system prompt 架构回归测试。

验证 "source/ 21 份源文件 + 启动时固定拼接 + 进程级缓存" 的核心不变量：
  1. 角色 → 源文件映射完整且章节顺序符合设计（Supervisor 14 份 / Retrieve 5 份 /
     Write 6 份 / Lite 4 份 / Analyze 4 份 / Web 4 份）。
  2. 每次请求复用同一字符串：format_hint / query 不影响内容，进程内缓存稳定。
  3. builds/ 落盘文件与运行时 getter 输出一致（KV cache 前缀字节级复用前提）。
  4. 全部 21 份源文件存在且非空。

全部离线，无模型、无网络。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompts import loader

passed, failed = [], []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
    else:
        failed.append(name)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def test_pb01_source_files_complete():
    print("[PB-01] source/ 21 份源文件齐全且非空")
    from pathlib import Path
    source_dir = Path(loader.PROMPT_DIR) / "source"
    files = {p.name for p in source_dir.glob("*.md")}
    expected = {
        "01_global_role_domain.md", "02_global_data_fidelity_citation.md",
        "03_supervisor_routing_fusion.md", "04_retrieve_agent_search.md",
        "05_academic_writing_common.md", "06_review_planner.md",
        "07_review_chapter_writer.md", "08_data_analysis_experiment.md",
        "09_format_fact.md", "10_format_mechanism.md", "11_format_compare.md",
        "12_format_review.md", "13_format_experiment.md", "14_format_task.md",
        "15_format_default.md", "16_tool_usage_file_rules.md",
        "17_data_source_boundaries.md", "18_evidence_arbitration_citation.md",
        "19_lite_mode.md", "20_terminology_domain.md",
        "21_web_agent_search.md",
    }
    check("21 份源文件齐全", files == expected, f"missing={sorted(expected - files)}")
    for name in sorted(expected):
        check(f"源文件非空: {name}", (source_dir / name).stat().st_size > 50)


def test_pb02_role_mapping_orders():
    print("[PB-02] 角色→源文件映射与章节顺序")
    prompts = loader.ensure_fixed_prompts(reload=True)

    supervisor = prompts["supervisor"]
    sv_heads = [ln for ln in supervisor.splitlines() if ln.startswith("#") and not ln.startswith("##")]
    check("Supervisor 章节数=14",
          len(sv_heads) == 14, str(sv_heads))
    expected_sv = [
        "# 角色与领域范围（全局）",
        "# 数据保真与来源标注（全局）",
        "# Supervisor 决策与证据融合",
        "# 输出格式：事实查询",
        "# 输出格式：机制查询",
        "# 输出格式：对比查询",
        "# 输出格式：综述",
        "# 输出格式：实验设计",
        "# 输出格式：任务执行",
        "# 输出格式：默认",
        "# 工具使用边界与文件保存规则",
        "# 数据源覆盖边界与检索前自审",
        "# 证据来源仲裁与引用规则（[n] / [Wn]）",
        "# 术语规范与领域限定细则",
    ]
    h1 = [ln for ln in supervisor.splitlines() if ln.startswith("# 输出格式：对比查询")]
    check("Supervisor 章节顺序正确", sv_heads == expected_sv and len(h1) == 1)

    retrieve = prompts["retrieve"]
    rv_heads = [ln for ln in retrieve.splitlines()
                if ln.startswith("# ") and not ln.startswith("## ")]
    check("Retrieve 章节=5（01/02/04/17/20）", len(rv_heads) == 5, str(rv_heads))
    check("Retrieve 含检索执行 Agent 章节",
          "# 检索执行 Agent（Retrieve-Agent）" in rv_heads)

    write = prompts["write"]
    wr_heads = [ln for ln in write.splitlines()
                if ln.startswith("# ") and not ln.startswith("## ")]
    check("Write 章节=6（01/02/05/06/07/20）", len(wr_heads) == 6, str(wr_heads))
    check("Write 含 Plan/Execute 章节",
          "# 综述写作规划器（Plan 阶段）" in wr_heads
          and "# 综述单章撰写（Execute 阶段）" in wr_heads)


def test_pb03_fixed_and_cached():
    print("[PB-03] 固定字符串 + 进程级缓存（KV cache 复用前提）")
    from src.prompts.loader import assemble_system_prompt, build_dynamic_blocks

    s1 = assemble_system_prompt(mode="expert", format_hint="fact", query="Q1")
    s2 = assemble_system_prompt(mode="expert", format_hint="review", query="Q2")
    check("不同 format_hint/query → 同一字符串", s1 == s2)
    check("Supervisor 提示词含全部格式模板",
          all(t in s1 for t in ("输出格式：事实查询", "输出格式：机制查询",
                                "输出格式：对比查询", "输出格式：综述",
                                "输出格式：实验设计", "输出格式：任务执行",
                                "输出格式：默认")))
    check("无动态格式块（恒空串）", build_dynamic_blocks(format_hint="fact", query="Q") == "")

    # 进程级缓存：同一角色两次取引同一缓存对象
    a = loader.get_system_prompt("supervisor")
    b = loader.get_system_prompt("supervisor")
    check("supervisor 缓存对象稳定", a is b)

    # reload 重建后字节级一致（确定性拼接）
    rebuilt = loader.ensure_fixed_prompts(reload=True)
    check("reload 后字节级一致", rebuilt["supervisor"] == s1
          and rebuilt["retrieve"] == loader.get_system_prompt("retrieve")
          and rebuilt["write"] == loader.get_system_prompt("write"))


def test_pb04_builds_on_disk():
    print("[PB-04] builds/ 落盘文件与 getter 一致")
    from pathlib import Path
    builds_dir = Path(loader.PROMPT_DIR) / "builds"
    prompts = loader.ensure_fixed_prompts()
    for role, fname in loader.BUILD_FILE_NAMES.items():
        target = builds_dir / fname
        check(f"builds/{fname} 存在", target.exists())
        on_disk = target.read_text(encoding="utf-8").strip()
        check(f"builds/{fname} 与缓存一致", on_disk == prompts[role].strip(),
              f"disk={len(on_disk)} vs cache={len(prompts[role])}")


def test_pb05_agent_role_fixed_prompts():
    print("[PB-05] 子 Agent 固定提示词内容正确")
    from src.prompts.loader import assemble_agent_prompt

    ra = assemble_agent_prompt("retrieve-agent")
    wa = assemble_agent_prompt("write-agent")
    aa = assemble_agent_prompt("analyze-agent")
    check("retrieve 含检索执行 Agent 章节", "# 检索执行 Agent（Retrieve-Agent）" in ra)
    check("retrieve 不重复 Supervisor 格式模板", "输出格式：事实查询" not in ra)
    check("write 含写作规范章节", "# 学术写作通用规范（Write-Agent）" in wa)
    check("retrieve/write 均含全局角色", "你是柑橘科研助手" in ra and "你是柑橘科研助手" in wa)
    check("write/retrieve 固定不随 skills 变化",
          assemble_agent_prompt("write-agent", skills=["x"]) == wa)
    check("analyze 保留位含结果归因", "结果归因" in aa)


def test_pb06_lite_prompt():
    print("[PB-06] Lite 固定提示词（01+02+19+20）")
    lite = loader.get_system_prompt("lite")
    check("lite 含轻量模式规则", "轻量模式" in lite)
    check("lite 不含 Supervisor 路由章节", "证据融合" not in lite)
    check("lite 固定", lite == loader.get_system_prompt("lite"))


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
    print(f"\nprompt-builds tests: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)