# -*- coding: utf-8 -*-
"""v8.17.18 回归：retrieve-agent 全挂双根因（真机日志实证）。

真机日志两处致命错误：
  1. AsyncCompletions.create() got an unexpected keyword argument 'thinking'
     —— llm_pool.py 把 {"thinking": {"type": "disabled"}} 塞进 model_kwargs，
     langchain-openai 将其展开为 create() 关键字参数，api.deepseek.com 不接受 →
     retrieve-agent 每轮 LLM 调用失败（3 次重试全败）→ 工具调用 0 次、检索没执行。
     修复：不再发送任何 thinking 参数（V3.2 类模型思维链默认关闭，不发送即等效关闭）。
     search.py HyDE 的 extra_body 同字段一并移除。
  2. cannot access local variable 'placeholder_results'
     —— agent_runner.py 原先在 turn 循环体内初始化该变量，而 LLM 全失败（response
     is None → break 提前出循环）后，收尾段 budget/dedup 统计仍引用它 →
     UnboundLocalError。修复：初始化提前到 turn 循环外。

全部离线、无模型、无网络（源码/实例级断言）。约定见 test_batch1.py。
覆盖：
  VF-55  llm_pool 不再发送 thinking 参数（源码级 + 实例级）
  VF-56  search.py HyDE 不再发送 thinking（extra_body 同移除）
  VF-57  agent_runner placeholder_results 初始化先于 turn 循环（防 UnboundLocalError）
  VF-58  retrieve-agent thinking_off 接线保留（reasoning_mode 配置仍生效路径）
  VF-59  收尾段 budget/dedup 统计仍引用 placeholder_results（与 VF-57 配对成立）
"""
import sys
import os
import inspect as _inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed, failed = [], []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
    else:
        failed.append(name)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


# ── F-17.18-1 thinking 参数移除（bug 1）────────────────────────────
def test_v81718_thinking_param_removed():
    print("[VF-55] llm_pool 不再发送 thinking 参数")
    from src.core.llm_pool import get_llm
    import src.core.llm_pool as lp
    lp_src = _inspect.getsource(lp)

    check("源码无 kw['thinking'] 发送",
          'kw["thinking"]' not in lp_src and "type\": \"disabled" not in lp_src,
          "仍然残留 thinking 发送代码")
    check("源码无 model_kwargs 传参（docstring 说明文字除外）",
          "model_kwargs=" not in lp_src, lp_src[:200])
    # 实例级：thinking_off=True 与 False 均不携带任何模型参数
    for off in (True, False):
        llm = get_llm("t-model", "k-1", "https://x", max_tokens=64, thinking_off=off)
        check(f"thinking_off={off} → model_kwargs 为空",
              getattr(llm, "model_kwargs", None) == {},
              str(getattr(llm, "model_kwargs", None)))


def test_v81718_hyde_thinking_removed():
    print("[VF-56] search.py HyDE 不再发送 thinking（extra_body 移除）")
    import src.tools.search as search_mod
    s_src = _inspect.getsource(search_mod)

    check("HyDE 无 thinking disabled 残留",
          '"thinking": {"type": "disabled"}' not in s_src, "extra_body thinking 仍残留")
    check("HyDE 无 extra_body 组装残留", "extra[\"extra_body\"]" not in s_src, "extra_body 仍残留")


# ── F-17.18-2 placeholder_results 初始化提前（bug 2）───────────────
def test_v81718_placeholder_init_before_loop():
    print("[VF-57] agent_runner placeholder_results 初始化先于 turn 循环")
    from src.core import agent_runner as ar
    ar_src = _inspect.getsource(ar)

    init_pos = ar_src.find("placeholder_results: dict = {}")
    loop_pos = ar_src.find("for turn in range(max_turns):")
    check("初始化存在且先于 turn 循环",
          init_pos != -1 and loop_pos != -1 and init_pos < loop_pos,
          f"init@{init_pos} loop@{loop_pos}")
    # 循环体内不应再重复初始化（避免混淆）
    body = ar_src[loop_pos:loop_pos + 2000]
    check("循环体内无重复初始化", "placeholder_results: dict = {}" not in body)


def test_v81718_placeholder_wiring_kept():
    print("[VF-58/59] retrieve-agent 接线与收尾统计保留")
    from src.core import agent_runner as ar
    ar_src = _inspect.getsource(ar)

    check("retrieve-agent thinking_off 接线保留",
          'thinking_off=(agent_name in ("retrieve-agent",)' in ar_src)
    check("收尾段 budget/dedup 统计仍引用 placeholder_results",
          "for m in placeholder_results.values()" in ar_src)
    check("[WEB_BUDGET_EXHAUSTED] 占位写入仍在",
          "placeholder_results[idx] = ToolMessage(" in ar_src
          and "[WEB_BUDGET_EXHAUSTED]" in ar_src)


def test_v81718_silent_summary():
    if failed:
        print(f"\n  ✗ v8.17.18 回归: {len(failed)} FAIL / {len(passed)} PASS")
        raise SystemExit(1)
    print(f"\n  ✓ v8.17.18 回归: {len(passed)} PASS / 0 FAIL")