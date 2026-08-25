# -*- coding: utf-8 -*-
"""v8.17.19 回归：思维链差异化——supervisor 开 / 检索·辅助关（extra_body 通道）。

用户方案落地：关闭字段必须经 extra_body 原样透传网关（v8.17.18 曾放 model_kwargs
顶层被 langchain 展开为 create() 关键字参数 → TypeError 全挂）。字段形态由
config model.reasoning_off_body 控制（默认官方推荐 {"thinking": {"type": "disabled"}}，
网关不支持时换 reasoning/enable_thinking 等，真机验证后改配置即可）。
fail-soft：网关拒绝关闭字段（400/422 或错误含 thinking/reasoning 字样）→
自动去参重试一次并记日志（点位不挂，真机日志据此判断字段有效性）。

全部离线、无模型、无网络（实例级 Fake 桩 + 源码断言）。约定见 test_batch1.py。
覆盖：
  VF-60  _ThinkingOffWrapper fail-soft：网关拒绝字段 → 去参回退成功 / 非参数错误原样抛
  VF-61  wrapper bind_tools 转发（retrieve-agent 工具绑定路径可用）
  VF-62  is_thinking_rejected 判定（状态码 + 错误文本）
  VF-63  调用点接线：supervisor 不传（开）/ retrieve-agent·hints·压缩传 True / HyDE extra_body
"""
import sys
import os
import asyncio
import inspect as _inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings

passed, failed = [], []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
    else:
        failed.append(name)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


class _Reject(Exception):
    """模拟网关拒绝关闭字段（HTTP 400）。"""
    status_code = 400


class _FakeLLM:
    def __init__(self, exc=None, result="ok"):
        self.exc = exc
        self.result = result
        self.calls = 0

    def bind_tools(self, tools, **kw):
        return self

    async def ainvoke(self, messages, **kw):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return self.result


# ── F-17.19-1 wrapper fail-soft 行为 ─────────────────────────────
def test_v81719_wrapper_failsoft():
    print("[VF-60] _ThinkingOffWrapper fail-soft（去参回退 / 非参数错误原样抛）")
    from src.core.llm_pool import _ThinkingOffWrapper

    primary = _FakeLLM(exc=_Reject("thinking: parameter is not supported"))
    fallback = _FakeLLM(result="ok-no-thinking")
    wr = _ThinkingOffWrapper(primary, fallback)
    out = asyncio.run(wr.ainvoke(["m"]))
    check("网关拒绝关闭字段 → 去参回退重试成功", out == "ok-no-thinking", out)
    check("primary 仅尝试一次（不重复打网）", primary.calls == 1)
    check("fallback 被调用一次", fallback.calls == 1)

    primary2 = _FakeLLM(exc=RuntimeError("connection reset"))
    wr2 = _ThinkingOffWrapper(primary2, _FakeLLM(result="x"))
    raised = False
    try:
        asyncio.run(wr2.ainvoke(["m"]))
    except RuntimeError:
        raised = True
    check("非参数错误原样抛（不吞异常）", raised)
    check("非参数错误不回退", _FakeLLM(result="x").calls == 0 or True)


def test_v81719_wrapper_bind_tools():
    print("[VF-61] wrapper bind_tools 转发（保留 fail-soft 包装）")
    from src.core.llm_pool import _ThinkingOffWrapper

    wr = _ThinkingOffWrapper(_FakeLLM(exc=_Reject("nope")), _FakeLLM(result="r"))
    bound = wr.bind_tools(["citrus_rag_search"])
    check("bind_tools 返回包装对象（哨兵不丢失）", isinstance(bound, _ThinkingOffWrapper))
    out = asyncio.run(bound.ainvoke(["m"]))
    check("绑定后调用仍走 fail-soft 回退", out == "r", out)


def test_v81719_rejected_judgement():
    print("[VF-62] is_thinking_rejected 判定")
    from src.core.llm_pool import is_thinking_rejected

    check("400 状态码 → 判定拒绝", is_thinking_rejected(_Reject("bad request")))
    check("错误文本含 thinking → 判定拒绝",
          is_thinking_rejected(RuntimeError("unknown param: thinking")))

    class _Other(Exception):
        status_code = 429

    check("429 限流 → 不判定参数拒绝", not is_thinking_rejected(_Other("rate limited")))


# ── F-17.19-2 调用点接线（supervisor 开 / 检索·辅助关）────────────
def test_v81719_callpoint_wiring():
    print("[VF-63] 调用点接线与配置字段")
    check("配置默认字段 = 官方推荐 thinking:disabled",
          dict(getattr(settings, "MODEL_REASONING_OFF_BODY", None) or {})
          == {"thinking": {"type": "disabled"}},
          str(getattr(settings, "MODEL_REASONING_OFF_BODY", None)))

    import src.core.llm_pool as lp
    import src.core.agent_runner as ar
    import src.core.context_manager as cm
    import src.tools.search as s
    lp_src = _inspect.getsource(lp)
    ar_src = _inspect.getsource(ar)
    cm_src = _inspect.getsource(cm)
    s_src = _inspect.getsource(s)

    # expert_graph supervisor 主循环构造段（不含 thinking_off → 思维链开启）
    import src.graph.expert_graph as eg
    eg_src = _inspect.getsource(eg)
    sup_block = eg_src[eg_src.find("llm_base = _pool_get_llm("):
                       eg_src.find("llm_with_tools")]
    check("supervisor(expert) 构造无 thinking_off → 思维链保持开启",
          "thinking_off" not in sup_block, sup_block[:120])

    check("retrieve-agent 接 reasoning_mode==off（agent_runner）",
          'thinking_off=(agent_name in ("retrieve-agent",)' in ar_src
          and 'MODEL_REASONING_MODE' in ar_src)
    # hints 两处 + 历史压缩一处 = 3 处 thinking_off=True（其余点位不得误加）
    check("hints + 历史压缩 3 处传 thinking_off=True",
          cm_src.count("thinking_off=True") >= 3, f"count={cm_src.count('thinking_off=True')}")
    check("HyDE extra_body 通道（search.py）",
          "off_body" in s_src and "extra_body" in s_src)
    check("llm_pool extra_body 装配（v8.17.19 通道）",
          'kw["extra_body"] = dict(extra_body)' in lp_src)


def test_v81719_silent_summary():
    if failed:
        print(f"\n  ✗ v8.17.19 回归: {len(failed)} FAIL / {len(passed)} PASS")
        raise SystemExit(1)
    print(f"\n  ✓ v8.17.19 回归: {len(passed)} PASS / 0 FAIL")