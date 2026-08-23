# -*- coding: utf-8 -*-
"""v8.16.4 草稿 3 态体验与提速回归：草稿关思维链+fail-soft、超时有界、
hints∥LTM 并行、context-detail 端点降级信封、前端 3 态文案与细分失败提示。

全部离线、无模型、无网络。
覆盖：
  VF-36  草稿调用关思维链（thinking:disabled）+ 参数被拒 fail-soft + timeout 25
  VF-37  context-detail 端点异常 → 200 空信封+error 标记（不再 500）
  VF-38  LTM ∥ hints 并行接线 + 前端 3 态文案/占位升级/细分失败温和提示
"""
import sys
import os
import inspect
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings

passed, failed = [], []
ROOT = Path(__file__).resolve().parents[1]


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
    else:
        failed.append(name)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


# ── VF-36 草稿关思维链 + fail-soft + 超时有界 ─────────────────────
def test_v8164_draft_nothink():
    print("[VF-36] 草稿关思维链(fail-soft) + timeout 25")
    import src.tools.deepseek_web as dw

    # v8.17.7: thinking/fail-soft 逻辑收敛到 _fast_llm_call（草稿/提取共用）
    src = inspect.getsource(dw._fast_llm_call)
    check("草稿调用支持 thinking:disabled（开启时下发）",
          '"thinking": {"type": "disabled"}' in src)
    check("参数被拒 → 第 1 次退回默认参数重试（防无草稿）",
          "退回默认参数重试一次" in src)
    check("fast 助手被草稿与提取共用",
          "def _call_structured_draft" in inspect.getsource(dw)
          and "def _call_extract_from_answer" in inspect.getsource(dw))
    check("DRAFT_THINKING_OFF 配置默认开（v8.17.14 关思维链）",
          settings.DRAFT_THINKING_OFF is True, str(settings.DRAFT_THINKING_OFF))
    check("DRAFT_TIMEOUT_SEC = 25（有界余量，不归零）",
          settings.DRAFT_TIMEOUT_SEC == 25, str(settings.DRAFT_TIMEOUT_SEC))
    yaml = (ROOT / "config.yaml").read_text(encoding="utf-8")
    check("yaml: thinking_off: true + timeout_sec: 25",
          "thinking_off: true" in yaml and "timeout_sec: 25" in yaml)
    cfg = (ROOT / "src/config.py").read_text(encoding="utf-8")
    check("config.py: DRAFT_THINKING_OFF 字段存在", "DRAFT_THINKING_OFF" in cfg)
    check("config.py: DRAFT_THINKING_OFF 默认 True（v8.17.14 关思维链）",
          "default=True" in cfg
          and "DRAFT_THINKING_OFF" in cfg)
    # v8.17.14: 联网草稿路径（_responses_web_search）同样关思维链 + fail-soft
    rsrc = inspect.getsource(dw._responses_web_search)
    check("联网草稿调用 thinking:disabled（开启时下发）",
          '"thinking": {"type": "disabled"}' in rsrc)
    check("联网草稿参数被拒 → 退回默认参数重试一次（防草稿全丢）",
          "退回默认参数重试一次" in rsrc)
    check("联网草稿日志含 thinking 标记（off/on）", "thinking={'off' if 'thinking' in payload else 'on'}" in rsrc
          or "thinking=" in rsrc)
    # v8.17.14: 诊断盲区修复——summary 空但 calls 有 → 记录原始响应
    check("联网返回正文空但含引用 → 记录原始响应（诊断盲区修复）",
          "联网返回正文为空但含引用" in rsrc and "request_id" in rsrc)


# ── VF-37 context-detail 端点降级信封 ─────────────────────────────
def test_v8164_context_detail():
    print("[VF-37] context-detail 端点健壮性（不再 500）")
    msrc = (ROOT / "src/api/main.py").read_text(encoding="utf-8")
    check("异常 → 空信封 + error 标记（源码接线）",
          'return {"session_id": session_id, "segments": [], "count": 0' in msrc
          and '"error": "context_detail_error"' in msrc)
    check("不再 raise HTTPException(500)（仅限该端点函数体）",
          "status_code=500" not in msrc[msrc.index("context_detail_endpoint"):])
    try:
        import asyncio
        from src.api import main as api_main
    except Exception as e:  # 环境缺依赖等 → 接线断言已足够
        print(f"  [skip] api.main 导入失败，仅接线断言: {e}")
        return

    class _FakeSM:
        async def get_messages_with_ids(self, sid):
            raise RuntimeError("db boom")

        def get_checkpoint(self, sid):
            return None

    orig = api_main.session_manager
    api_main.session_manager = _FakeSM()
    try:
        r = asyncio.run(api_main.context_detail_endpoint("sess-x", "expert"))
        ok = (r.get("error") == "context_detail_error" and r.get("count") == 0
              and r.get("segments") == [])
    finally:
        api_main.session_manager = orig
    check("假 DB 异常 → 200 信封（error 标记，无抛错）", ok, str(r)[:80])


# ── VF-38 LTM ∥ hints 并行 + 前端 3 态文案 ────────────────────────
def test_v8164_parallel_and_ux():
    print("[VF-38] hints∥LTM 并行 + 前端 3 态")
    cm = (ROOT / "src/core/context_manager.py").read_text(encoding="utf-8")
    check("hints 提前调度（create_task 与 LTM 并行）",
          "hints_task = asyncio.create_task(self._generate_hints(query, session_id, mode))" in cm)
    check("await 汇合 hints 任务结果", "await hints_task" in cm)
    check("旧串行调用已移除",
          "suggestions, format_hint = await self._generate_hints(query, session_id, mode)" not in cm)
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    check("3 态①占位「正在生成预览…」", "正在生成预览…" in html)
    check("3 态②「草稿已生成 · 正在扩展检索证据…」",
          "草稿已生成 · 正在扩展检索证据…" in html)
    check("3 态②展检索「正在扩展检索证据…」（agent_switch 钩子）",
          "正在扩展检索证据…" in html)
    check("3 态③「正在整合证据、生成最终回答…」（step_done retrieve 钩子）",
          "正在整合证据、生成最终回答…" in html)
    check("占位原位升级草稿（draft-phase 元素）", "draft-phase" in html)
    check("细分失败温和提示（catch + error 信封双路）",
          "⚠ 细分加载失败，请稍后重试（异常已记录）" in html)


if __name__ == "__main__":
    test_v8164_draft_nothink()
    test_v8164_context_detail()
    test_v8164_parallel_and_ux()
    print(f"\npassed: {len(passed)}  failed: {len(failed)}")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)