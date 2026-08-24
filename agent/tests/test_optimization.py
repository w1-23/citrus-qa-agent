# -*- coding: utf-8 -*-
"""性能优化回归（v8.3.1）:
  ① academic_search min_keep=3（治因：打断"全丢→换词→再丢"空转）
  ② supervisor ≤2 retrieve + retrieve 单轮 ≤2 工具（止血，BUDGET_LIMIT 告知）
  ③ write-agent max_tokens 12000 + prompt 分块（治截断）
  ④ write 结果回显预览 + 假引用移除
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

passed, failed = [], []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
    else:
        failed.append(name)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def test_min_keep():
    print("[①] academic_search min_keep")
    src = open(os.path.join(BASE, 'src', 'tools', 'search.py'), encoding='utf-8').read()
    check("含 min_keep=3", "min_keep=3" in src)
    check("全丢时保留 top3", "deduped[:3]" in src)
    check("非柑橘标注", "非柑橘相关，仅供参考" in src)

    # 行为模拟: citrus filter 全丢场景
    import sys as _s
    _s.path.insert(0, BASE)
    from src.tools.search import _is_citrus_related
    check("非柑橘条目被识别", _is_citrus_related("Genome sequencing of tomato") is False)
    check("柑橘条目被识别", _is_citrus_related("Citrus sinensis genome assembly") is True)


def test_budget_removed():
    print("[②] BUDGET_LIMIT 已删除（回滚硬阈值）")
    expert = open(os.path.join(BASE, 'src', 'graph', 'expert_graph.py'), encoding='utf-8').read()
    runner = open(os.path.join(BASE, 'src', 'core', 'agent_runner.py'), encoding='utf-8').read()
    cfg = open(os.path.join(BASE, 'config.yaml'), encoding='utf-8').read()
    cfgpy = open(os.path.join(BASE, 'src', 'config.py'), encoding='utf-8').read()
    check("expert_graph 无 BUDGET_LIMIT", "BUDGET_LIMIT" not in expert)
    check("agent_runner 无 BUDGET_LIMIT", "BUDGET_LIMIT" not in runner)
    check("config.yaml 无预算键", "search_budget_per_" not in cfg)
    check("config.py 无预算字段", "SEARCH_BUDGET" not in cfgpy)
    # v8.4.6: 直接执行（仅新增检索角度去重的代码级预过滤，无预算类过滤）
    check("agent_runner 恢复直接执行", "execute_tools([tc for _, tc in exec_calls])" in runner)


def test_reason_feedback():
    print("[统一协议] 原因回传（状态+原因+建议）")
    ret = open(os.path.join(BASE, 'src', 'retrieval', 'multi_retriever.py'), encoding='utf-8').read()
    search = open(os.path.join(BASE, 'src', 'tools', 'search.py'), encoding='utf-8').read()
    reg = open(os.path.join(BASE, 'src', 'tools', 'registry.py'), encoding='utf-8').read()
    check("multi_retriever 含 last_empty_reason", "last_empty_reason" in ret)
    check("阈值拦截归因 threshold_blocked", "threshold_blocked" in ret and "threshold_blocked" in search)
    check("无匹配归因 no_match", '"no_match"' in ret or "'no_match'" in ret)
    check("RAG 空结果附建议(换特异词)", "换更特异的柑橘术语" in search)
    check("RAG 空结果附建议(换源)", "academic_search 学术源补充" in search)
    check("academic 网络失败回传", "[ERR_NETWORK] 学术源请求失败" in search and "source_errors" in search)
    check("academic 空结果附建议", "换更特异的英文关键词" in search)
    check("_classify_error 补建议策略", "建议: " in reg and "pip install" in reg)
    check("write-agent 归因要求", "结果归因" in open(os.path.join(BASE, 'src', 'prompts', 'source', '05_academic_writing_common.md'), encoding='utf-8').read())
    check("analyze-agent 归因要求", "结果归因" in open(os.path.join(BASE, 'src', 'prompts', 'source', '08_data_analysis_experiment.md'), encoding='utf-8').read())

    # 行为模拟: 归因文本生成
    import sys as _s
    _s.path.insert(0, BASE)
    from src.tools.registry import _classify_error
    import asyncio
    msg = _classify_error(FileNotFoundError("x.md"))
    check("FILE_NOT_FOUND 含建议", "检查路径" in msg)
    msg2 = _classify_error(asyncio.TimeoutError())
    check("TIMEOUT 含建议", "本地源" in msg2)


def test_write_token():
    print("[③] write-agent 截断")
    runner = open(os.path.join(BASE, 'src', 'core', 'agent_runner.py'), encoding='utf-8').read()
    prom = open(os.path.join(BASE, 'src', 'prompts', 'source', '05_academic_writing_common.md'), encoding='utf-8').read()
    check("max_tokens 12000", "max_t = 12000" in runner)
    check("旧值 32768 已移除", "32768 if" not in runner)
    check("prompt 分块指令", "每轮只写 1-2 个章节" in prom)
    check("prompt 禁止单轮全文", "禁止单轮生成全文" in prom)


def test_write_preview():
    print("[④] 写入回显 + 假引用")
    fops = open(os.path.join(BASE, 'src', 'tools', 'file_ops.py'), encoding='utf-8').read()
    prom = open(os.path.join(BASE, 'src', 'prompts', 'source', '05_academic_writing_common.md'), encoding='utf-8').read()
    check("write 返回含内容预览", "内容预览" in fops)
    check("预览截断 200 字符", "preview_src[:200]" in fops or "content[:200]" in fops)
    check("append 预览显示新增块", "new_content if actual_mode" in fops)
    check("假引用已删除", "read_local_file" not in prom)

    # 行为模拟: write 返回格式
    import sys as _s
    _s.path.insert(0, BASE)
    from src.tools.file_ops import write_local_file
    from src.config import PROJECT_ROOT
    import uuid
    fname = f"test_opt_{uuid.uuid4().hex[:6]}.md"
    r = write_local_file.func(fname, "测试内容" * 50, "write")
    check("返回以 Success 开头", r.startswith("Success:"))
    check("返回含预览", "内容预览: 测试内容测试内容" in r)
    p = PROJECT_ROOT / "workspace" / "output" / fname
    if p.exists():
        p.unlink()


if __name__ == "__main__":
    test_min_keep()
    test_budget_removed()
    test_reason_feedback()
    test_write_token()
    test_write_preview()
    print(f"\n结果: {len(passed)} passed / {len(failed)} failed")
    if failed:
        print("失败项:", failed)
        sys.exit(1)
