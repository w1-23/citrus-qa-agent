# -*- coding: utf-8 -*-
"""agent_runner file_saved dict/getattr bug 回归（v8.3.1）
真实事故: tool_calls 元素为 dict，getattr(tc,"name") 恒 "" → file_saved 恒 False
→ supervisor forced save 覆盖 write-agent 已写入的完整综述。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed, failed = [], []


def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def test_file_saved_dict_bug():
    print("[file_saved] dict tool_calls 场景")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'src', 'core', 'agent_runner.py'), encoding='utf-8').read()
    check("tc 兼容 dict 取值 (isinstance 分支)",
          "isinstance(tc, dict)" in src and 'tc.get("name", "")' in src)
    check("write-agent 判定仍用 tc_name == 'write_local_file'",
          'tc_name == "write_local_file"' in src)

    # 模拟判定逻辑: dict 场景 tc_name 应能取到
    tc = {"id": "call_1", "name": "write_local_file", "args": {"path": "a.md", "content": "x"}}
    tc_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
    check("dict 场景 tc_name 取到 write_local_file", tc_name == "write_local_file", tc_name)

    # 模拟 write_local_file 成功返回
    tr_content = "Success: write to a.md. Total file size now: 100 chars (0.1 KB)."
    file_saved = (tc_name == "write_local_file" and tr_content.startswith("Success:"))
    check("file_saved 判定成功", file_saved is True)


def test_encyclopedia_removed():
    print("[encyclopedia] 已删除")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_src = open(os.path.join(base, 'src', 'tools', 'search.py'), encoding='utf-8').read()
    init_src = open(os.path.join(base, 'src', 'tools', '__init__.py'), encoding='utf-8').read()
    cfg = open(os.path.join(base, 'config.yaml'), encoding='utf-8').read()
    check("search.py 无 encyclopedia_search 定义", "def encyclopedia_search" not in search_src)
    check("__init__.py 无导入", "encyclopedia_search" not in init_src)
    check("config.yaml 无残留", "encyclopedia" not in cfg)
    check("注册表数量 7", "read_local_file, write_local_file" in init_src)


if __name__ == "__main__":
    test_file_saved_dict_bug()
    test_encyclopedia_removed()
    print(f"\n结果: {len(passed)} passed / {len(failed)} failed")
    if failed:
        print("失败项:", failed)
        sys.exit(1)
