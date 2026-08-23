# -*- coding: utf-8 -*-
"""v8.16.1 草稿先行特性回归：分隔符结构化解析 / 草稿事件 / 草稿仓 / 证据并入回执。

全部离线、无模型、无网络（解析/事件/仓/回执均为纯逻辑；草稿调用与多路检索
以打桩验证 fail-soft 与接线）。约定见 test_batch1.py / test_v8153_fixes.py。
覆盖：
  VF-19  分隔符结构化解析（正常/缺字段/缺区块/围栏容错/大小写/空值）
  VF-20  草稿模板加载 + config 默认值
  VF-21  emit_draft 事件 + draft_worker 全链路（打桩调用与检索，fail-soft）
  VF-22  draft_store 请求级仓（put/pop/TTL/淘汰）
  VF-23  build_evidence_report 草稿补充行（draft_extra_count）
  VF-24  源码接线断言（expert/light 启动、agent_runner 并入、前端 draft 事件）
"""
import sys
import os

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


# ── VF-19 分隔符结构化解析（计划 v8.16.1 §5.2）────────────────────
def test_v8161_parse_structured():
    print("[VF-19] _parse_structured_response 分隔符解析")
    import src.tools.deepseek_web as dw

    raw = (
        "以下是检索结果摘要。\n"
        "===STRUCTURED_START===\n"
        "DRAFT_ZH: 2026年柑橘产业政策聚焦品种改良、品牌建设。\n"
        "DRAFT_EN: In 2026, citrus policies focus on variety improvement.\n"
        "MULTI_QUERY: national citrus policy 2026|citrus HLB control measures|citrus quality standards\n"
        "SUMMARY: Variety improvement|HLB control|Brand value\n"
        "===STRUCTURED_END===\n"
        "trailing"
    )
    p = dw._parse_structured_response(raw)
    check("回答正文 = 区块前文本", p["answer"] == "以下是检索结果摘要。", p["answer"])
    check("DRAFT_ZH 提取", p["draft_zh"] == "2026年柑橘产业政策聚焦品种改良、品牌建设。", p["draft_zh"])
    check("DRAFT_EN 提取", p["draft_en"] == "In 2026, citrus policies focus on variety improvement.", p["draft_en"])
    check("MULTI_QUERY 竖线拆分 3 条", p["multi_query"] == [
        "national citrus policy 2026", "citrus HLB control measures",
        "citrus quality standards"], str(p["multi_query"]))
    check("SUMMARY 竖线拆分 3 条", p["summary"] == ["Variety improvement", "HLB control", "Brand value"],
          str(p["summary"]))

    # 字段名大小写容错（提示词要求大写，防御小写）
    raw_lower = ("===STRUCTURED_START===\n"
                 "draft_zh: 中文草稿\n"
                 "draft_en: English draft\n"
                 "multi_query: a|b|c\n"
                 "summary: x|y|z\n"
                 "===STRUCTURED_END===\n")
    p2 = dw._parse_structured_response(raw_lower)
    check("字段名大小写容错", p2["draft_zh"] == "中文草稿" and p2["summary"] == ["x", "y", "z"], str(p2))

    # Markdown 围栏容错（提示词已约束，防御模型仍包 ```）
    raw_fence = ("```text\n===STRUCTURED_START===\n"
                 "DRAFT_ZH: 中文草稿\nDRAFT_EN: English draft\n"
                 "MULTI_QUERY: a|b|c\nSUMMARY: x|y|z\n"
                 "===STRUCTURED_END===\n```")
    p3 = dw._parse_structured_response(raw_fence)
    check("代码围栏剥离后解析成功", p3["draft_zh"] == "中文草稿", str(p3))

    # 续行容错：模型把 DRAFT_EN 折行（无字段名前缀的行归属上一字段）
    raw_wrap = ("===STRUCTURED_START===\n"
                "DRAFT_ZH: 中文草稿\n"
                "DRAFT_EN: In 2026, citrus policies focus on variety improvement\n"
                "and brand building for the citrus industry.\n"
                "MULTI_QUERY: a|b|c\nSUMMARY: x|y|z\n"
                "===STRUCTURED_END===\n")
    p4 = dw._parse_structured_response(raw_wrap)
    check("DRAFT_EN 折行续行合并", p4["draft_en"] == (
        "In 2026, citrus policies focus on variety improvement\n"
        "and brand building for the citrus industry."), repr(p4["draft_en"]))
    check("续行不污染其他字段", p4["multi_query"] == ["a", "b", "c"], str(p4["multi_query"]))

    # 异常用例
    try:
        dw._parse_structured_response("没有区块的文本")
        check("缺区块 → 抛 StructuredParseError", False)
    except dw.StructuredParseError:
        check("缺区块 → 抛 StructuredParseError", True)
    raw_missing = ("===STRUCTURED_START===\nDRAFT_EN: English draft\n"
                   "MULTI_QUERY: a|b|c\n===STRUCTURED_END===\n")
    try:
        dw._parse_structured_response(raw_missing)
        check("缺 DRAFT_ZH → 抛错（v8.16.3 唯一必需字段）", False)
    except dw.StructuredParseError as e:
        check("缺 DRAFT_ZH → 抛错（v8.16.3 唯一必需字段）", "DRAFT_ZH" in str(e), str(e))
    # v8.16.3: 仅 DRAFT_ZH 必需——旧字段 SUMMARY 缺失/为空不再抛错（兼容占位）
    raw_missing_old = ("===STRUCTURED_START===\nDRAFT_ZH: 中文草稿\nDRAFT_EN: English draft\n"
                       "MULTI_QUERY: a|b|c\n===STRUCTURED_END===\n")
    p5 = dw._parse_structured_response(raw_missing_old)
    check("缺 SUMMARY → 不再抛错（兼容占位）", p5["draft_zh"] == "中文草稿"
          and p5["summary"] == [], str(p5))
    raw_empty_old = ("===STRUCTURED_START===\nDRAFT_ZH: 中文草稿\nDRAFT_EN: English draft\n"
                     "MULTI_QUERY: a|b|c\nSUMMARY: \n===STRUCTURED_END===\n")
    p6 = dw._parse_structured_response(raw_empty_old)
    check("空 SUMMARY 值 → 不再抛错", p6["draft_zh"] == "中文草稿", str(p6))
    try:
        dw._parse_structured_response("")
        check("空响应 → 抛错", False)
    except dw.StructuredParseError:
        check("空响应 → 抛错", True)

    # ── v8.17: 容忍缺失 END 标记（日志实证模型偶发在 ANSWER 后提前收尾）──
    raw_noend = ("===STRUCTURED_START===\n"
                 "ANSWER: 柑橘黄龙病防控要点：无根治手段，以综合防控为主。\n"
                 "MULTI_QUERY: citrus HLB control|HLB vector management|HLB resistant varieties\n")
    p7 = dw._parse_structured_response(raw_noend)
    check("v8.17 缺 END 标记 → 容忍解析", p7["draft_zh"].startswith("柑橘黄龙病防控要点"),
          str(p7))
    check("v8.17 缺 END 标记 → MQ 仍解析", p7["multi_query"] == [
        "citrus HLB control", "HLB vector management", "HLB resistant varieties"],
        str(p7["multi_query"]))

    # ── v8.17: ANSWER 字段 = 草稿（DRAFT_ZH 别名）──
    raw_answer = ("===STRUCTURED_START===\n"
                  "ANSWER: 品种来源综述草稿。\n"
                  "MULTI_QUERY: a|b|c\nSUMMARY: x|y|z\n"
                  "===STRUCTURED_END===\n")
    p8 = dw._parse_structured_response(raw_answer)
    check("v8.17 ANSWER 字段归一为草稿（DRAFT_ZH 别名）",
          p8["draft_zh"] == "品种来源综述草稿。", str(p8))

    # ── v8.17.6: 无包裹标记容错解析（日志实证 v4-flash 常省略 ===STRUCTURED===）──
    raw_no_wrap = ("以下是从联网回答中提炼的检索素材：\n"
                   "MULTI_QUERY: citrus HLB control 2025|ACP monitoring data|dsRNA biopesticide field\n"
                   "SUMMARY: ACP trap decline 9%|Cq 37.73 near threshold|dsRNA field validation\n补充说明…")
    p9 = dw._parse_structured_response(raw_no_wrap, require_answer=False)
    check("v8.17.6 无包裹 → 容错解析 MQ/SUMMARY", len(p9["multi_query"]) == 3
          and p9["multi_query"][0] == "citrus HLB control 2025"
          and len(p9["summary"]) == 3, str(p9))
    check("v8.17.6 容错路径 answer=标签行前正文", p9["answer"] == "以下是从联网回答中提炼的检索素材：",
          repr(p9["answer"]))

    # 加粗标签行（**MULTI_QUERY:** 形态）
    raw_bold = ("**MULTI_QUERY:** a|b|c\n**SUMMARY:** x|y|z")
    p10 = dw._parse_structured_response(raw_bold, require_answer=False)
    check("v8.17.6 加粗标签行容错", p10["multi_query"] == ["a", "b", "c"]
          and p10["summary"] == ["x", "y", "z"], str(p10))

    # [MQ]/[SUM] 短标签兜底
    raw_short = ("[MQ]\n- citrus HLB\n- ACP monitor\n[/MQ]\n"
                 "[SUM]\n- trap decline\n- Cq value\n[/SUM]")
    p11 = dw._parse_structured_response(raw_short, require_answer=False)
    check("v8.17.6 [MQ]/[SUM] 短标签兜底", p11["multi_query"] == ["citrus HLB", "ACP monitor"]
          and p11["summary"] == ["trap decline", "Cq value"], str(p11))

    # require_answer=True（草稿路径）：无包裹但含 ANSWER 行 → 直接救回草稿（不再 300 字降级）
    raw_draft_nowrap = ("下面直接回答。\n"
                        "ANSWER: 柑橘黄龙病综合防控草稿，核心是无根治手段。\n"
                        "MULTI_QUERY: a|b|c\nSUMMARY: x1|y1")
    p12 = dw._parse_structured_response(raw_draft_nowrap)
    check("v8.17.6 草稿路径无包裹 + ANSWER → 完整救回草稿（不 300 字降级）",
          p12["draft_zh"] == "柑橘黄龙病综合防控草稿，核心是无根治手段。"
          and p12["multi_query"] == ["a", "b", "c"], str(p12))
    # 纯无标签文本 + require_answer=True → 仍抛（走既有降级路径）
    try:
        dw._parse_structured_response("模型没有按模板输出,这是纯文本回答。", require_answer=True)
        check("v8.17.6 纯文本无标签 → require_answer=True 仍抛（降级路径不变）", False)
    except dw.StructuredParseError:
        check("v8.17.6 纯文本无标签 → require_answer=True 仍抛（降级路径不变）", True)


# ── VF-20 草稿模板 + config 默认值 ────────────────────────────────
def test_v8161_prompt_and_config():
    print("[VF-20] structured_output 模板 + config")
    from src.prompts.loader import assemble_structured_output_prompt, \
        assemble_structured_extract_prompt

    prompt = assemble_structured_output_prompt()
    check("模板非空", bool(prompt))
    check("模板含 START 分隔符", "===STRUCTURED_START===" in prompt)
    check("模板含 END 分隔符", "===STRUCTURED_END===" in prompt)
    # v8.17: 原生回答草稿 + 检索素材一体（从原生回答提取）
    check("模板含 ANSWER 字段（原生回答）", "ANSWER:" in prompt)
    check("模板恢复检索素材字段（v8.17 从原生回答提取）",
          "MULTI_QUERY:" in prompt and "SUMMARY:" in prompt)
    check("模板禁止 JSON/围栏", "不要输出 JSON" in prompt and "不要输出 Markdown 代码块" in prompt)
    check("模板强调 END 收尾", "不得遗漏收尾标记" in prompt or "以 `===STRUCTURED_END===` 结尾" in prompt)
    check("模板 ANSWER 300-600 字约束", "300-600" in prompt)

    ep = assemble_structured_extract_prompt()
    check("v8.17 提取模板非空且含 MQ/SUMMARY",
          bool(ep) and "MULTI_QUERY:" in ep and "SUMMARY:" in ep)
    # v8.17.6: 提取模板改为容错标签行格式（不强制包裹标记）
    check("v8.17.6 提取模板容错：不要求包裹标记",
          "不依赖包裹标记" in ep and "独立标签行" in ep
          and "解析器只取标签行" in ep)
    from pathlib import Path as _P
    _ep_file = _P(__file__).resolve().parent.parent / "src" / "prompts" / "structured_output.md"
    _ep_txt = _ep_file.read_text(encoding="utf-8")
    check("v8.17.6 草稿模板含标签行容错提示（三行齐全即可救回）",
          "容错提示（v8.17.6）" in _ep_txt and "三行标签" in _ep_txt
          and "解析器按标签行兜底提取" in _ep_txt
          and "===STRUCTURED_START===" in _ep_txt and "===STRUCTURED_END===" in _ep_txt)

    check("DRAFT_ENABLED 默认开", settings.DRAFT_ENABLED is True)
    check("DRAFT_LABEL 默认文案", settings.DRAFT_LABEL == "预检索草稿·验证中", settings.DRAFT_LABEL)
    check("DRAFT_MAX_CHARS 默认 800", settings.DRAFT_MAX_CHARS == 800)
    check("DRAFT_MAX_ANGLES 默认 3", settings.DRAFT_MAX_ANGLES == 3)
    check("DRAFT_SUMMARY_POINTS 默认 3（已确认上限）", settings.DRAFT_SUMMARY_POINTS == 3)
    check("DRAFT_EXTRA_RETRIEVAL 默认开", settings.DRAFT_EXTRA_RETRIEVAL is True)


# ── VF-21 emit_draft + draft_worker 全链路（打桩，fail-soft）──────
def test_v8161_draft_event_and_worker():
    print("[VF-21] emit_draft 事件 + draft_worker 链路")
    import asyncio
    import json
    import src.tools.deepseek_web as dw
    from src.core.progress_bus import (
        set_request_queue, clear_request_queue, emit_draft)
    from src.core.draft_store import draft_store

    # 1) emit_draft 事件形态
    q = asyncio.Queue()
    set_request_queue(q)
    try:
        emit_draft("2026年柑橘政策初判…", "预检索草稿·验证中")
        items = [q.get_nowait() for _ in range(1)]
    finally:
        clear_request_queue()
    check("draft 事件入队", items[0]["event"] == "draft", str(items[0]))
    ev = json.loads(items[0]["data"]) if isinstance(items[0].get("data"), str) else {}
    check("draft 载荷 content+label",
          ev.get("content") == "2026年柑橘政策初判…"
          and ev.get("label") == "预检索草稿·验证中", str(ev))

    # 2) draft_worker 全链路（打桩调用与检索；v8.17: 原生回答落仓 + 检索并入）
    q2 = asyncio.Queue()
    set_request_queue(q2)
    draft_store.clear()

    real_call = dw._call_structured_draft
    real_search = dw._draft_search_multi

    class _FakeCallZh:
        def __call__(self, query):
            return ("===STRUCTURED_START===\n"
                    "ANSWER: 2026年柑橘黄龙病综合防治技术现状，核心结论：无根治手段，"
                    "综合管理以三角管理（病原-寄主-媒介）为框架。主要维度：①病原靶向——"
                    "化学治疗与热疗仍为田间主要手段；②媒介防控——化学防治、经济阈值与"
                    "生物防治（寄生蜂、虫生真菌）；③寄主管理——清除病树与无病苗。"
                    "具体证据与展开将在正式回答中给出。\n"
                    "MULTI_QUERY: citrus HLB integrated management|HLB vector psyllid control|HLB resistant rootstocks\n"
                    "SUMMARY: no radical cure triangle management|vector chemical biological control|infected tree removal\n"
                    "===STRUCTURED_END===\n")

    try:
        dw._call_structured_draft = _FakeCallZh()
        dw._draft_search_multi = lambda queries: [
            {"title": "draft-h1", "doi": "10.d/1", "year": 2026, "text": "body"}]
        asyncio.run(dw.draft_worker("2026柑橘政策", "sess-1"))
    finally:
        dw._call_structured_draft = real_call
        dw._draft_search_multi = real_search
        items2 = []
        while not q2.empty():
            items2.append(q2.get_nowait())
        clear_request_queue()

    check("worker 发出 draft 事件", any(it["event"] == "draft" for it in items2), str(items2)[:200])
    draft_ev = next((it for it in items2 if it["event"] == "draft"), {})
    ddata = json.loads(draft_ev["data"]) if draft_ev.get("data") else {}
    check("草稿内容 = 原生回答完整预览", ddata.get("content", "").startswith(
        "2026年柑橘黄龙病综合防治技术现状"), str(ddata)[:120])
    check("v8.17 草稿≥140 字（完整预览）",
          len(ddata.get("content", "")) >= 140, str(len(ddata.get("content", ""))))
    check("草稿标签 = 预检索草稿·验证中", ddata.get("label") == "预检索草稿·验证中", str(ddata))

    # v8.17: 原生回答 + 检索结果并入 draft_store（供 agent_runner 融合/证据）
    payload = draft_store.pop("sess-1", "-")
    check("v8.17 草稿落仓（原生回答 + 多路检索并入）",
          payload is not None and len(payload.get("results") or []) == 1,
          str({k: (v if k != "answer_text" else str(v)[:30]) for k, v in (payload or {}).items()}))
    check("落仓含原生回答文本", bool(payload and payload.get("answer_text")))
    draft_store.clear()

    # 3) fail-soft：解析失败不抛异常、不落仓
    q3 = asyncio.Queue()
    set_request_queue(q3)

    class _FakeBad:
        def __call__(self, query):
            return "没有结构化区块的输出"

    draft_store.clear()
    try:
        dw._call_structured_draft = _FakeBad()
        asyncio.run(dw.draft_worker("2026柑橘政策", "sess-2"))
    finally:
        dw._call_structured_draft = real_call
        while not q3.empty():
            q3.get_nowait()
        clear_request_queue()
    check("解析失败 → 不落仓", draft_store.pop("sess-2", "-") is None)
    draft_store.clear()


# ── VF-22 draft_store 请求级仓 ───────────────────────────────────
def test_v8161_draft_store():
    print("[VF-22] draft_store TTL/淘汰/键隔离")
    from src.core.draft_store import DraftStore

    st = DraftStore(max_entries=4, ttl_sec=100)
    st.put("s1", "r1", {"results": [1]})
    st.put("s2", "r2", {"results": [2]})
    check("同 session 不同 request 键隔离", st.pop("s1", "r2") is None)
    check("pop 取回", st.pop("s1", "r1") == {"results": [1]})
    check("pop 后删键", st.pop("s1", "r1") is None)

    # TTL 过期
    st2 = DraftStore(max_entries=4, ttl_sec=10)
    st2.put("s", "r", {"results": [1]})
    st2._entries[st2._key("s", "r")]["ts"] -= 30  # 人为过期
    check("TTL 过期 → pop 返回 None", st2.pop("s", "r") is None)
    check("过期键被清理", st2._key("s", "r") not in st2._entries)

    # 超限淘汰最旧
    st3 = DraftStore(max_entries=3, ttl_sec=100)
    for i in range(5):
        st3.put("s", f"r{i}", {"results": [i]})
    check("超过上限 → 只留最近 3 条",
          len(st3._entries) == 3 and st3.pop("s", "r4") == {"results": [4]}
          and st3.pop("s", "r0") is None, str(len(st3._entries)))


# ── VF-23 证据回执草稿补充行 ─────────────────────────────────────
def test_v8161_evidence_report_draft_line():
    print("[VF-23] build_evidence_report 草稿补充行")
    from src.core.agent_runner import build_evidence_report

    arts = {"main_results": [{"title": "P1", "doi": "10.1/x", "year": 2023, "text": "body"}],
            "web_results": []}
    base = build_evidence_report(arts, "citrus hlb", 2)
    check("默认无草稿行", "草稿多路检索补充" not in base)
    check("默认无原生回答段", "原生回答参考" not in base)
    with_draft = build_evidence_report(arts, "citrus hlb", 2, draft_extra_count=4)
    check("draft_extra_count=4 → 回执明示补充 4 条",
          "草稿多路检索补充: 4 条证据" in with_draft, with_draft[:300])
    check("补充行含三路来源说明", "原始问题 + MULTI_QUERY + SUMMARY" in with_draft)
    # v8.17: 原生回答融合段（用户要求"最后融合原生回答进行生成"）
    with_na = build_evidence_report(arts, "citrus hlb", 2,
                                    draft_answer="原生预答内容123")
    check("v8.17 原生回答参考段渲染（v8.17.4 头文案）",
          "原生回答参考（草稿预答）" in with_na
          and "原生预答内容123" in with_na, with_na[:300])


# ── VF-24 源码接线断言（防接线回归，VF-17 同款模式）────────────────
def test_v8161_wiring_assertions():
    print("[VF-24] 源码接线断言")
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]  # agent/
    ex = (root / "src/graph/expert_graph.py").read_text(encoding="utf-8")
    lg = (root / "src/graph/light_graph.py").read_text(encoding="utf-8")
    ar = (root / "src/core/agent_runner.py").read_text(encoding="utf-8")
    dw = (root / "src/tools/deepseek_web.py").read_text(encoding="utf-8")
    idx = (root / "index.html").read_text(encoding="utf-8")
    check("expert load 启动 draft_worker（v8.16.3 提前到 load 开头）",
          "draft_worker(query, session_id)" in ex and "草稿先行启动提前到 load 开头" in ex)
    check("light load 启动 draft_worker（全场景草稿）",
          "draft_worker(query, session_id)" in lg)
    # v8.17: 草稿证据并入恢复——pop draft_store + 回执「原生回答参考」段 + 补充行
    check("agent_runner 草稿并入恢复（pop + 原生回答段）",
          "draft_store.pop" in ar and "原生回答参考" in ar and "草稿多路检索补充" in ar)
    check("deepseek_web 含解析/worker/草稿检索入口",
          "def _parse_structured_response" in dw and "async def draft_worker" in dw
          and "def _draft_search_multi" in dw)
    check("前端 draft 事件分支", "case 'draft':" in idx)
    check("前端草稿标签文案", "预检索草稿·验证中" in idx)
    check("前端首 text 清空草稿", "draftShown = false" in idx)


# ── 汇总 ──────────────────────────────────────────────────────────
def _summary():
    print(f"\n[VF-16.1] PASS {len(passed)} / FAIL {len(failed)}"
          f"  ({len(passed) + len(failed)} total)")
    for f in failed:
        print(f"  [FAIL] {f}")
    return len(failed) == 0


if __name__ == "__main__":
    import types as _types
    _funcs = [v for v in list(globals().values())
              if isinstance(v, _types.FunctionType)
              and v.__name__.startswith("test_")]
    for _f in sorted(_funcs, key=lambda x: x.__name__):
        try:
            _f()
        except Exception as _e:
            if os.environ.get("PYTEST_CURRENT_TEST"):
                raise
            print(f"  {_e}")
    ok = _summary()
    sys.exit(0 if ok else 1)