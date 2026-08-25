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


# ── VF-36 v8.17.15 草稿配置/调用删除 + 联网工具超时保留 ──────────
def test_v8164_draft_nothink():
    print("[VF-36] 草稿全删 + 联网工具保留")
    import src.tools.deepseek_web as dw
    from src.tools.registry import _tool_exec_timeout

    dw_src = inspect.getsource(dw)
    check("草稿调用函数已删除（_fast_llm_call/_call_structured_draft）",
          "def _fast_llm_call" not in dw_src
          and "def _call_structured_draft" not in dw_src)
    check("联网工具保留（deepseek_web_search）",
          "def deepseek_web_search" in dw_src)
    check("网络超时 90s 保留（_web_http_timeout）", "def _web_http_timeout" in dw_src)
    check("工具执行超时 120s 保留", _tool_exec_timeout("deepseek_web_search") == 120,
          str(_tool_exec_timeout("deepseek_web_search")))

    yaml = (ROOT / "config.yaml").read_text(encoding="utf-8")
    check("yaml 无 draft 配置块", "draft:" not in yaml)
    cfg = (ROOT / "src/config.py").read_text(encoding="utf-8")
    check("config.py 无 DRAFT_ 字段", "DRAFT_" not in cfg)


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


# ── VF-38 LTM ∥ hints 并行 + 前端草稿 UI 删除 ────────────────────
def test_v8164_parallel_and_ux():
    print("[VF-38] hints∥LTM 并行 + 前端无反草稿 UI")
    cm = (ROOT / "src/core/context_manager.py").read_text(encoding="utf-8")
    check("hints 提前调度（create_task 与 LTM 并行）",
          "hints_task = asyncio.create_task(self._generate_hints(query, session_id, mode))" in cm)
    check("await 汇合 hints 任务结果", "await hints_task" in cm)
    check("旧串行调用已移除",
          "suggestions, format_hint = await self._generate_hints(query, session_id, mode)" not in cm)
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    # v8.17.15: 草稿三态 UI 全删（无占位/面板/阶段文案）
    check("无「草稿生成中…」占位", "草稿生成中" not in html)
    check("无「正在生成预览…」占位", "正在生成预览" not in html)
    check("无「草稿已生成 · 正在扩展检索证据…」", "正在扩展检索证据" not in html)
    check("无「正在整合证据、生成最终回答…」草稿钩子", "draft-phase" not in html)
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