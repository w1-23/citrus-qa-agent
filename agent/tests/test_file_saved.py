# -*- coding: utf-8 -*-
"""职责矩阵重构回归（v8.3.1）:
  ① forced save / file_saved 信号已彻底删除（无第三方补偿写入）
  ② supervisor 持有 write_local_file 直写工具（保存现成内容）
  ③ retrieve-agent 3 轮 / light 绑定 read_local_file
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

passed, failed = [], []


def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def test_no_fallback():
    print("[职责] 无兜底写入 / 无布尔信号")
    expert = open(os.path.join(BASE, 'src', 'graph', 'expert_graph.py'), encoding='utf-8').read()
    runner = open(os.path.join(BASE, 'src', 'core', 'agent_runner.py'), encoding='utf-8').read()
    check("expert_graph 无 forced save", "forced save" not in expert and "forced_save" not in expert)
    check("expert_graph 无 file_saved", "file_saved" not in expert)
    check("agent_runner 无 file_saved", "file_saved" not in runner)


def test_supervisor_direct_write():
    print("[职责] supervisor 直写工具")
    expert = open(os.path.join(BASE, 'src', 'graph', 'expert_graph.py'), encoding='utf-8').read()
    check("_AGENT_TOOLS 含 write_local_file", '"name": "write_local_file"' in expert)
    check("处理分支存在", 'tc_dict["name"] == "write_local_file"' in expert)
    check("描述含'原样保存/verbatim'", "verbatim" in expert and "call_write_agent instead" in expert)
    check("write_local_file 分支执行写盘", "write_local_file.func, path, content, mode" in expert)


def test_retrieve_turns():
    print("[职责] retrieve-agent 3 轮")
    runner = open(os.path.join(BASE, 'src', 'core', 'agent_runner.py'), encoding='utf-8').read()
    check("_get_max_turns retrieve=3", '"retrieve-agent": 3' in runner)
    prom = open(os.path.join(BASE, 'src', 'prompts', 'agents', 'retrieve-agent.md'), encoding='utf-8').read()
    check("prompt 含轮次语义", "最多 3 轮" in prom and "同一轮内" in prom)


def test_light_read():
    print("[职责] light 绑定 read_local_file")
    light = open(os.path.join(BASE, 'src', 'graph', 'light_graph.py'), encoding='utf-8').read()
    check("LIGHT_TOOL_NAMES 含 read_local_file",
          'LIGHT_TOOL_NAMES = ("citrus_rag_search", "read_local_file")' in light)
    rules = open(os.path.join(BASE, 'src', 'prompts', 'system', 'light_rules.md'), encoding='utf-8').read()
    check("light_rules 边界更新", "读取单个本地文件" in rules)


def test_dead_code_removed():
    print("[职责] 死代码已删")
    init_src = open(os.path.join(BASE, 'src', 'tools', '__init__.py'), encoding='utf-8').read()
    reg = open(os.path.join(BASE, 'src', 'core', 'registries.py'), encoding='utf-8').read()
    check("tools/__init__ 无 get_tools_for_mode", "get_tools_for_mode" not in init_src)
    check("tools/__init__ 无 get_all_tools", "get_all_tools" not in init_src)
    check("registries 无 MODE_ALLOWED", "MODE_ALLOWED" not in reg)


def test_prompt_boundaries():
    print("[职责] prompt 双向边界")
    guide = open(os.path.join(BASE, 'src', 'prompts', 'system', 'decision_guide.md'), encoding='utf-8').read()
    writer = open(os.path.join(BASE, 'src', 'prompts', 'agents', 'write-agent.md'), encoding='utf-8').read()
    check("decision_guide 含 write_local_file 直写指引", "直接调 write_local_file" in guide)
    check("decision_guide 含 call_write_agent 撰写指引", "调 **call_write_agent**" in guide)
    check("write-agent 含原样保存指令", "原样写入" in writer and "不要改写" in writer)
    check("write-agent 自检含补写", "mode=\"append\"" in writer)


if __name__ == "__main__":
    test_no_fallback()
    test_supervisor_direct_write()
    test_retrieve_turns()
    test_light_read()
    test_dead_code_removed()
    test_prompt_boundaries()
    print(f"\n结果: {len(passed)} passed / {len(failed)} failed")
    if failed:
        print("失败项:", failed)
        sys.exit(1)
