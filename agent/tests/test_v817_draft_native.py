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
    from src.core.evidence import src_of, is_variety_source

    check("品种意图命中（品种/UCR/CRC/cultivar）",
          _is_variety_intent("有哪些 UCR 品种库登记的宽皮柑橘") is True)
    check("非品种意图不误报", _is_variety_intent("柑橘黄龙病综合防治") is False)
    # v9.4.2: 编号池统一——回执与侧栏共用 canonical_evidence_items（含 ucr_first 聚拢）
    src_report = _inspect.getsource(ar.build_evidence_report)
    from src.core import evidence as _ev
    src_canon = _inspect.getsource(_ev.canonical_evidence_items)
    check("回执接入唯一编号池（canonical_evidence_items + ucr_first 透传）",
          "canonical_evidence_items" in src_report and "ucr_first" in src_report)
    check("编号池含品种聚拢判定（is_variety_source(src_of(r))）",
          "is_variety_source(src_of(r))" in src_canon)
    # v9.4: 品种来源判定覆盖建库口径（文件夹名 Citrus varietiesN）+ 旧 ucr 值
    check("variety 判定覆盖 Citrus varietiesN 文件夹名",
          is_variety_source("Citrus varieties1") is True
          and is_variety_source("citrus_varieties1") is True
          and is_variety_source("paper1") is False
          and is_variety_source("ucr") is True)
    check("src_of 透传批次原始来源（不再折叠 ucr）",
          src_of({"_src": "paper2"}) == "paper2"
          and src_of({"_src": "Citrus varieties1"}) == "Citrus varieties1")
    # src_of 兜底保留：无 _src 的旧 UCR chunk 仍回退 ucr
    check("src_of 兜底保留旧 UCR 判定",
          src_of({"source_type": "UCR citrus variety"}) == "ucr")


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


# ── VF-48 v9.1 联网移出 retrieve-agent（独立 web-agent 并行）─────────
def test_v81715_web_back_in_agent():
    print("[VF-48] v9.1 联网移出 retrieve-agent（白名单仅本地 + 预算在工具层）")
    from src.core import agent_runner as ar
    from src.config import settings

    names = ar._resolve_tool_names("retrieve-agent")
    check("retrieve-agent 白名单无联网工具（根除只联网不本地）",
          "deepseek_web_search" not in names, str(names))
    check("白名单含本地检索", "citrus_rag_search" in names)
    check("白名单无 academic 工具（v9.2 已全链删除）", "academic_search" not in names)

    ar_src = _inspect.getsource(ar)
    check("agent_runner 不再持有联网预算/熔断（v9.1 迁移至工具层）",
          "_web_used = 0" not in ar_src and "_web_fail_streak = 0" not in ar_src
          and "WEB_BUDGET_EXHAUSTED" not in ar_src, "残留旧联网逻辑")
    check("web_summary 收集保留（防御/兼容，artifacts 通路）",
          "web_summaries" in ar_src)

    # v9.1 新架构：supervisor 统一入口 call_search_both 并行（本地+联网）
    from src.graph import expert_graph as eg
    eg_src = _inspect.getsource(eg)
    check("supervisor 执行分支 call_search_both（并行检索）",
          "if name == \"call_search_both\":" in eg_src)


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


# ── VF-52 提示词快照新语义（v9.1：独立 web-agent 架构）──────────────
def test_v81715_prompt_snapshots():
    print("[VF-52] 提示词/快照新语义（v9.1 本地+联网并行架构）")
    from src.prompts.loader import assemble_agent_prompt, assemble_system_prompt
    ra = assemble_agent_prompt("retrieve-agent")
    dg = assemble_system_prompt(mode="expert", format_hint=None, query=None)
    snap_ra = _src("src/prompts/snapshots/agent_retrieve-agent.txt")
    snap_dg = _src("src/prompts/snapshots/system_expert.txt")

    check("retrieve-agent 无联网工具（本地唯一拆解层）",
          "deepseek_web_search" not in ra and "citrus_rag_search" in ra)
    check("retrieve-agent 唯一拆解层语义（2~4 个聚焦角度）",
          "唯一允许的拆解层" in ra and "2~4 个" in ra)
    check("retrieve-agent 不超过 4 个角度（防膨胀）", "不要超过 4 个" in ra)
    check("supervisor 含 call_search_both 统一入口", "call_search_both" in dg)
    check("supervisor 不预判不跳过 + 禁止空字符串", "不预先跳过任何一方" in dg
          and "禁止传空字符串" in dg)
    check("supervisor 含联网证据仲裁规则（时效优先 [Wn]）", "时效信息优先联网" in dg
          and "联网证据默认可信" in dg)
    check("supervisor 无 call_retrieve_agent 残留", "call_retrieve_agent" not in dg)
    check("快照 agent_retrieve-agent.txt 同步（无联网工具）",
          "deepseek_web_search" not in snap_ra and "唯一允许的拆解层" in snap_ra)
    check("快照 system_expert.txt 同步 call_search_both",
          "call_search_both" in snap_dg)


# ── VF-53 引用保存 web 条目（evidence 0 items 修复）+ historical 恢复 ──
def test_v81717_evidence_web_and_historical():
    print("[VF-53] evidence 保存 web 条目 + historical 侧栏恢复")
    from src.graph import expert_graph as eg
    from src.session import manager as sm
    import src.core.agent_loop as al   # v9.2: save 核心收敛于此

    eg_src = _inspect.getsource(eg)
    al_src = _inspect.getsource(al)
    # v9.2 重构：web 账本逻辑迁至 agent_loop.run_save_node，锚点随迁
    check("evidence 保存含 web_results（0 items 修复）",
          "web_results = state.get(\"web_results\") or []" in al_src
          or "for w in web_results[:30]" in al_src)
    check("web 条目带 url/source=web 入账本",
          '"source": "web"' in al_src and '"url": str(w.get("url", ""))' in al_src)
    check("expert save 委托共享核心并开启 web 账本",
          "run_save_node(" in eg_src and 'include_web=True' in eg_src)

    # historical 恢复接线（references_data.historical 不再为空数组）
    check("references_data 含 historical 注入（v8.17.17 恢复）",
          '"historical": historical_refs' in eg_src)

    # manager.get_evidence_refs 最近 10 轮 + url
    sm_src = _inspect.getsource(sm.SessionManager.get_evidence_refs)
    # v9.2 重构：LIMIT 字面量移入共享读取器 _load_evidence_rows，两处任一命中即保序
    check("历史引用覆盖最近 10 轮（原 4）",
          "LIMIT 10" in sm_src
          or "_load_evidence_rows(session_id, 10)" in sm_src)
    check("历史引用保留 url（web 可跳转）", '"url": str(e.get("url") or "")[:300]' in sm_src)

    # 前端 historical 链接渲染
    idx = _src("index.html")
    check("前端 historical url 可点（与 web 同级）",
          "item.type === 'historical'" in idx and "item.url" in idx)


# ── VF-54 联网摘要为空兜底 ──────────────────────────────────────────
def test_v81717_empty_summary_fallback():
    print("[VF-54] 联网 0 字摘要 → web_items 保留 + 不重试提示")
    from src.tools import deepseek_web as dw
    src = _src("src/tools/deepseek_web.py")
    check("摘要为空分支 · 明示引用但无正文",
          "返回引用条目但无正文摘要" in src)
    check("不提示「模型判断无需联网」（误导消除）",
          "模型判断无需联网，未检索" not in src)
    check("web_items 仍构造（引用可用）",
          '"ref_id": f"W{idx}"' in src)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()