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
    (passed if cond else failed).append(name)
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


def test_budget_limit():
    print("[②] 检索预算（配置驱动）")
    expert = open(os.path.join(BASE, 'src', 'graph', 'expert_graph.py'), encoding='utf-8').read()
    runner = open(os.path.join(BASE, 'src', 'core', 'agent_runner.py'), encoding='utf-8').read()
    cfg = open(os.path.join(BASE, 'config.yaml'), encoding='utf-8').read()
    check("supervisor 含 BUDGET_LIMIT", "BUDGET_LIMIT" in expert)
    check("supervisor 用配置预算", "SEARCH_BUDGET_PER_SUPERVISOR_TURN" in expert)
    check("agent_runner 用配置预算", "SEARCH_BUDGET_PER_RETRIEVE_TURN" in runner)
    check("config.yaml 含两个预算键", "search_budget_per_supervisor_turn" in cfg
          and "search_budget_per_retrieve_turn" in cfg)
    check("无硬编码 2（用 budget 变量）", "retrieve_calls[budget:]" in expert
          and "tool_calls_to_run[budget:]" in runner)
    check("被拒调用回传 ToolMessage", 'tool_call_id=excess_id' in runner)

    # 行为模拟: 预算从 settings 读取
    import sys as _s
    _s.path.insert(0, BASE)
    from src.config import settings
    check("settings 读取预算=2", settings.SEARCH_BUDGET_PER_SUPERVISOR_TURN == 2
          and settings.SEARCH_BUDGET_PER_RETRIEVE_TURN == 2)


def test_write_token():
    print("[③] write-agent 截断")
    runner = open(os.path.join(BASE, 'src', 'core', 'agent_runner.py'), encoding='utf-8').read()
    prom = open(os.path.join(BASE, 'src', 'prompts', 'agents', 'write-agent.md'), encoding='utf-8').read()
    check("max_tokens 12000", "max_t = 12000" in runner)
    check("旧值 32768 已移除", "32768 if" not in runner)
    check("prompt 分块指令", "每轮只写 1-2 个章节" in prom)
    check("prompt 禁止单轮全文", "禁止尝试在单轮内生成全文" in prom)


def test_write_preview():
    print("[④] 写入回显 + 假引用")
    fops = open(os.path.join(BASE, 'src', 'tools', 'file_ops.py'), encoding='utf-8').read()
    prom = open(os.path.join(BASE, 'src', 'prompts', 'agents', 'write-agent.md'), encoding='utf-8').read()
    check("write 返回含内容预览", "内容预览" in fops)
    check("预览截断 200 字符", "content[:200]" in fops)
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
    test_budget_limit()
    test_write_token()
    test_write_preview()
    print(f"\n结果: {len(passed)} passed / {len(failed)} failed")
    if failed:
        print("失败项:", failed)
        sys.exit(1)
