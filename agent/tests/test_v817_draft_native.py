# -*- coding: utf-8 -*-
"""v8.17.15 「去草稿 + 联网回归 retrieve-agent」回归（用户决策架构重构）：

草稿全链删除（后端/前端/配置/提示词），原生联网回归 retrieve-agent 工具链：
  - deepseek_web_search 重新进入白名单（工具参数 query 即 agent 本轮 goal）
  - 每轮工具序列至多 1 次（_MAX_WEB_PER_TURN=1 + execution 层拦截）
  - 前端「联网」开关关闭时工具内短路 [DISABLED]（零网络请求、无联网融合）
  - 联网正文 → web_summaries（回执「网络综述」段）、引用 → [Wn]，
    与本地 [n] 共同支撑 supervisor 融合回答

全部离线、无模型、无网络（源码接线断言 + 纯函数语义）。
覆盖：
  VF-39  品种意图检测（_is_variety_intent）+ UCR 优先接线（回执聚拢置前，保留）
  VF-47  草稿全链删除（后端无 draft_worker/draft_store/emit_draft/DRAFT_*/
         set_draft_task；expert/light load 无 create_task；main.py 无等待草稿）
  VF-48  联网回归 retrieve-agent（白名单含 web + execution 每轮 ≤1 拦截 + 熔断保留）
  VF-49  deepseek_web_search goal 驱动 + [DISABLED] 短路保留 + URL 提取 [Wn]
  VF-50  build_evidence_report 无草稿段 + 网络综述段保留（web_summaries）
  VF-51  前端草稿 UI 删除（无 draft-panel 创建/占位/已验证标记）
  VF-52  提示词快照新语义（retrieve-agent 允许联网每轮≤1；无「草稿层」表述）
"""
import sys
import os
import inspect as _inspect
from pathlib import Path

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def check(name, cond, detail=""):
    """断言并通过 stdout 报告（与项目测试约定一致：无 pytest 断言依赖）。"""
    if not cond:
        raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  PASS  {name}")


def _src(p):
    return (ROOT / p).read_text(encoding="utf-8")


# ── VF-39 品种意图检测 + UCR 优先（保留 v8.17 能力，非草稿）─────────
def test_v81739_variety_intent_ucr():
    print("[VF-39] 品种意图检测 + UCR 聚拢置前（保留）")
    from src.tools.search import _is_variety_intent
    from src.core import agent_runner as ar

    check("品种意图命中（品种/UCR/CRC/cultivar）",
          _is_variety_intent("有哪些 UCR 品种库登记的宽皮柑橘") is True)
    check("非品种意图不误报", _is_variety_intent("柑橘黄龙病综合防治") is False)
    # 回执聚拢置前接线仍在（ucr_first 参数 + src_of 过滤）
    src = _inspect.getsource(ar.build_evidence_report)
    check("ucr_first 聚拢逻辑保留",
          'src_of(r) == "ucr"' in src and "ucr_first" in src)


# ── VF-47 草稿全链删除 ──────────────────────────────────────────────
def test_v81715_draft_removed():
    print("[VF-47] 草稿全链删除（后端接线）")
    from src.tools import deepseek_web as dw
    from src.core import progress_bus as pb
    from src.core import agent_runner as ar
    import src.config as cfg
    import src.graph.expert_graph as eg
    import src.graph.light_graph as lg
    from src.api import main as api_main

    dw_src = _inspect.getsource(dw)
    # 工具函数保留（删除的是草稿入口）
    check("deepseek_web_search 保留", "def deepseek_web_search" in dw_src)
    check("draft_worker 已删除", "def draft_worker" not in dw_src)
    check("_responses_web_search 已删除", "def _responses_web_search" not in dw_src)
    check("_call_structured_draft / 三区块解析 已删除",
          "def _call_structured_draft" not in dw_src
          and "def _parse_web_three_block" not in dw_src
          and "def _parse_structured_response" not in dw_src)
    check("_fast_llm_call 已删除（草稿/提取专用）", "def _fast_llm_call" not in dw_src)

    # progress_bus: 无 emit_draft / 任务注册表
    pb_src = _inspect.getsource(pb)
    check("emit_draft 已删除", "def emit_draft" not in pb_src)
    check("草稿任务注册表已删除", "def set_draft_task" not in pb_src
          and "def get_draft_task" not in pb_src
          and "def clear_draft_task" not in pb_src)

    # agent_runner: 无草稿证据并入 / draft_store 引用（docstring 注释提及删除历史可容忍）
    ar_src = _inspect.getsource(ar)
    check("agent_runner 无 draft_store.pop", "draft_store.pop" not in ar_src)
    check("agent_runner 无草稿并入（draft_answer 参数）",
          "draft_answer = str(_draft.get" not in ar_src
          and "草稿证据并入" not in ar_src)

    # config: 无 DRAFT_* 字段
    cfg_src = _inspect.getsource(cfg)
    check("config.py 无 DRAFT_ 字段", "DRAFT_" not in cfg_src)

    # expert/light load: 无 create_task(draft_worker)
    eg_src = _inspect.getsource(eg)
    lg_src = _inspect.getsource(lg)
    check("expert load 不再启动草稿", "draft_worker(query, session_id)" not in eg_src)
    check("light load 不再启动草稿", "draft_worker(query, session_id)" not in lg_src)

    # main.py: 无 get_draft_task / DRAFT_SSE_WAIT_SEC
    mp_src = _inspect.getsource(api_main)
    check("main.py 无等待草稿逻辑", "get_draft_task" not in mp_src
          and "DRAFT_SSE_WAIT_SEC" not in mp_src)

    # 全局源码扫描: draft_store 模块文件已删除
    import os as _os
    check("draft_store.py 文件已删除",
          not _os.path.exists(ROOT / "src/core/draft_store.py"))


# ── VF-48 联网回归 retrieve-agent ───────────────────────────────────
def test_v81715_web_back_in_agent():
    print("[VF-48] 联网回归 retrieve-agent（白名单 + 每轮≤1 预算 + 熔断）")
    from src.core import agent_runner as ar
    from src.config import settings

    names = ar._resolve_tool_names("retrieve-agent")
    check("retrieve-agent 白名单含 deepseek_web_search",
          "deepseek_web_search" in names, str(names))
    check("白名单含本地检索", "citrus_rag_search" in names)
    _aca = settings.ACADEMIC_ENABLED
    check("学术源随门控（默认关）", ("academic_search" not in names) == (not _aca))

    ar_src = _inspect.getsource(ar)
    check("per-turn web 预算 = 1（_MAX_WEB_PER_TURN）",
          "_MAX_WEB_PER_TURN = 1" in ar_src)
    check("execution 层拦截分支（每轮上限 1 次，超限占位）",
          "每轮联网搜索上限 1 次" in ar_src)
    check("连续失败熔断保留（≥2 拦截）",
          "联网搜索已连续失败" in ar_src and "_web_fail_streak" in ar_src)
    check("web_summary 进 artifact（回执网络综述素材）",
          "web_summaries" in ar_src)


# ── VF-49 deepseek_web_search goal 驱动 + 短路 + URL ────────────────
def test_v81715_web_tool_goal_driven():
    print("[VF-49] deepseek_web_search goal 驱动 + [DISABLED] 短路 + [Wn]")
    from src.tools import deepseek_web as dw
    from src.tools.deepseek_web import deepseek_web_search, _parse_response_output

    # deepseek_web_search 是 @tool StructuredTool——按模块源码断言
    src = _src("src/tools/deepseek_web.py")
    # goal 即工具参数 query（不再有 original_query contextvar 优先）
    check("输入构造 = 工具参数 query（goal 驱动）",
          '_input_prompt = f"{query}\\n\\n{_ref_cmd}"' in src)
    check("不再引用 original_query contextvar",
          "original_query" not in src and "原始问题直传" not in src)
    # [DISABLED] 短路保留（web_search_enabled contextvar）
    check("[DISABLED] 短路保留（开关关闭零网络请求）",
          "[DISABLED]" in src and "web_search_enabled" in src)
    # URL 提取 → [Wn] web_items（内联 items 构造）
    check("引用 → [Wn] web_items 构造",
          '"ref_id": f"W{idx}"' in src and '"url": c["url"]' in src)

    # 语义级：开关关闭 → 短路（不发起网络请求）
    import src.core.tracing as tr
    try:
        tr.set_web_search_enabled(False)
        content, artifact = deepseek_web_search.func("柑橘 2025 政策")
        check("开关关闭 → [DISABLED] 短路", content.startswith("[DISABLED]"))
        check("短路时无 web 融合（空 artifact）",
              artifact == {"main_results": [], "web_results": []})
    finally:
        tr.set_web_search_enabled(True)

    # 解析函数保留（与工具共用）
    psrc = _inspect.getsource(_parse_response_output)
    check("_parse_response_output 保留（工具共用解析）",
          "def _parse_response_output" in psrc)


# ── VF-50 build_evidence_report 无草稿段 + 网络综述保留 ─────────────
def test_v81715_evidence_report():
    print("[VF-50] 回执：无草稿段 + 网络综述段保留")
    from src.core import agent_runner as ar

    src = _inspect.getsource(ar.build_evidence_report)
    # 签名区（docstring 前）不再含草稿参数（docstring 内删除历史说明可容忍）
    sig = src.split('"""')[0]
    check("签名无 draft_answer / draft_extra_count",
          "draft_answer" not in sig and "draft_extra_count" not in sig)
    check("无草稿多路检索补充行", "草稿多路检索补充" not in src)
    check("网络综述段保留（v8.15.3f）", "网络综述" in src
          or "web_summaries" in src)

    # 语义级：纯函数调用不含草稿参数也不报错
    rep = ar.build_evidence_report(
        {"main_results": [], "web_results": [], "web_summaries": ["联网综述正文"]},
        "q", 1)
    check("回执可正常生成（无草稿参数）", "检索回执" in rep)
    check("网络综述正文进入回执", "联网综述正文" in rep)


# ── VF-51 前端草稿 UI 删除 ─────────────────────────────────────────
def test_v81715_frontend_cleaned():
    print("[VF-51] 前端草稿 UI 全删")
    idx = _src("index.html")
    check("无草稿面板创建（draft-panel open）", "draft-panel" not in idx)
    check("无占位「草稿生成中…」", "草稿生成中" not in idx)
    check("无占位「正在生成预览」", "正在生成预览" not in idx)
    check("无 previewShown 变量", "previewShown" not in idx)
    check("无 draftShown / draftWasShown 变量", "draftShown" not in idx)
    check("无「已通过检索验证」标记", "草稿已通过检索验证" not in idx)
    check("无深度思考草稿引用", "预检索草稿" not in idx)


# ── VF-52 提示词快照新语义 ────────────────────────────────────────
def test_v81715_prompt_snapshots():
    print("[VF-52] 提示词/快照新语义（联网回归 agent）")
    # v9.0: 提示词为"启动时固定拼接"——直接断言最终装配出的固定提示词内容与快照
    from src.prompts.loader import assemble_agent_prompt, assemble_system_prompt
    ra = assemble_agent_prompt("retrieve-agent")
    dg = assemble_system_prompt(mode="expert", format_hint=None, query=None)
    snap_ra = _src("src/prompts/snapshots/agent_retrieve-agent.txt")
    snap_dg = _src("src/prompts/snapshots/system_expert.txt")

    check("retrieve-agent 允许 deepseek_web_search",
          "deepseek_web_search" in ra and "最多调用 1 次" in ra)
    check("retrieve-agent 无「禁止调用」", "禁止调用" not in ra)
    check("retrieve-agent goal 驱动表述", "query=goal" in ra)
    check("supervisor 无草稿活性语义（草稿层执行/原生回答参考段）",
      "草稿层唯一" not in dg and "原生回答参考（草稿预答）" not in dg
      and "已由草稿层" not in dg)
    check("supervisor 含联网证据仲裁规则（替代原生回答参考）", "联网证据默认可信" in dg)
    check("快照 agent_retrieve-agent.txt 同步新语义",
          "最多调用 1 次" in snap_ra and "query=goal" in snap_ra)
    check("快照 system_expert.txt 无「原生回答参考（草稿预答）」段渲染",
      "原生回答参考（草稿预答）" not in snap_dg
      and "原生回答参考段的使用规则" not in snap_dg)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()