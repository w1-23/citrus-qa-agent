# -*- coding: utf-8 -*-
"""v9.2 重构回归：共享引用装配原语 / 轨迹同步判重 / light done 分支 / 检索统计接线。

全部离线、无模型、无网络（验证纯函数行为 + 关键接线点源码级锁定）。
约定与 test_batch1.py 一致（check/passed/failed/_summary）。

覆盖：
  VF-101 build_cited_refs 槽位上限与字段结构（expert 10 / light 5 参数化）
  VF-102 renumber_and_sync_trace 轨迹同步（save 判重不变量）
  VF-103 rag_stats original_query 接线（HyDE 主路径误杀修复）
  VF-104 pdf_read 统一工具出口（无越权直调 + 错参修复）
  VF-105 死代码移除锁定（web_search_enabled 键 / get_tool_elapsed / _sse_debug_enabled）
"""
import sys
import os

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


def _fake_main(i, *, source="rag"):
    return {
        "doi": f"10.1000/ref{i}",
        "title": f"文献 {i}",
        "source": source,
        "abstract": f"摘要 {i} " * 60,          # 触发 text_preview 300 截断
        "score": 0.9 - i * 0.01,
        "rerank_score": 0.8 - i * 0.01,
        "year": 2024,
        "authors": "张三",
    }


def _fake_web(i):
    return {"url": f"https://example.com/{i}", "title": f"网页 {i}",
            "snippet": f"片段 {i}", "link": f"https://example.com/{i}"}


# ── VF-101 引用回执装配（expert/light 共享原语）────────────────────
def test_v92_build_cited_refs():
    print("[VF-101] build_cited_refs 槽位上限与字段结构")
    from src.core.agent_loop import build_cited_refs

    mains = [_fake_main(i) for i in range(1, 26)]       # 25 条 → 数字槽位上限 20
    webs = [_fake_web(i) for i in range(1, 16)]         # 15 条 → web 槽位参数化

    refs_max = build_cited_refs(mains, webs, web_slot=10)
    check("expert 槽位: 20 数字 + 10 web", len(refs_max) == 30, len(refs_max))
    check("数字 ref_id 连续 1..20", [r["ref_id"] for r in refs_max[:20]] == list(range(1, 21)))
    check("web ref_id 为 W1..W10", [r["ref_id"] for r in refs_max[20:]] == [f"W{i}" for i in range(1, 11)])

    refs_light = build_cited_refs(mains, webs, web_slot=5)
    check("light 槽位: 20 数字 + 5 web", len(refs_light) == 25, len(refs_light))
    check("light web 上限 5", [r["ref_id"] for r in refs_light[20:]] == [f"W{i}" for i in range(1, 6)])

    m0 = refs_max[0]
    field_order = ["ref_id", "type", "source", "doi", "title", "section_name",
                   "text_preview", "score", "year", "authors",
                   "variety_name", "registry_id"]
    check("main 字段键序锁定（前端契约）", list(m0.keys()) == field_order, list(m0.keys()))
    check("main type/source 正确", m0["type"] == "main" and m0["source"] == "rag")
    check("text_preview 300 截断", len(m0["text_preview"]) <= 300, len(m0["text_preview"]))
    w0 = refs_max[20]
    check("web 字段键序锁定", list(w0.keys()) == ["ref_id", "type", "source", "url",
                                                  "title", "text_preview", "score"],
          list(w0.keys()))
    check("web url 回退 link", w0["url"] == "https://example.com/1")
    check("web score 恒 0", w0["score"] == 0)
    ucr = build_cited_refs([_fake_main(1, source="ucr")], [], web_slot=5)
    check("ucr 来源透传", ucr[0]["source"] == "ucr")


# ── VF-102 renumber + 轨迹同步（save 判重不变量）────────────────────
def test_v92_renumber_sync_trace():
    print("[VF-102] renumber_and_sync_trace 轨迹同步")
    from langchain_core.messages import AIMessage, HumanMessage
    from src.core.agent_loop import build_cited_refs, renumber_and_sync_trace
    from src.core.evidence import renumber_refs

    mains = [_fake_main(1), _fake_main(2), _fake_main(3), _fake_main(4)]
    cited = build_cited_refs(mains, [], web_slot=5)
    answer = "HLB 的致病机制见 [4]，防治见 [2]。"
    trace = [HumanMessage(content="问"), AIMessage(content=answer)]

    new_answer, new_cited, remap = renumber_and_sync_trace(trace, answer, cited)
    check("重排: [4][2] → [1][2]", new_answer == "HLB 的致病机制见 [1]，防治见 [2]。", new_answer)
    check("remap 正确", remap == {"2": "2", "4": "1"}, remap)
    check("轨迹内 AIMessage 已同步", trace[-1].content == new_answer, trace[-1].content)
    check("save 判重不变量: trace[-1] == 返回 answer", trace[-1].content == new_answer)

    # 已连续编号 → 文本原样、轨迹不动（identity remap 属 renumber 正常输出）
    stable = "结论见 [1] 与 [2]。"
    trace2 = [AIMessage(content=stable)]
    a2, c2, r2 = renumber_and_sync_trace(trace2, stable, cited)
    check("已连续编号文本原样", a2 == stable, a2)
    check("无改动时不触碰轨迹对象", trace2[0].content == stable)
    check("identity remap 不触发重写", r2 == {"1": "1", "2": "2"}, r2)

    # 与 evidence.renumber_refs 纯函数一致性（共享 _extract_ref_order 行为锚定）
    p_answer, p_cited, p_remap = renumber_refs(answer, cited)
    a3, c3, r3 = renumber_and_sync_trace([], answer, cited)
    check("共享重排逻辑一致", a3 == p_answer and r3 == p_remap)


# ── VF-103 检索统计 original_query 接线（HyDE 误杀修复）────────────
def test_v92_rag_stats_original_query_wiring():
    print("[VF-103] rag_stats original_query 接线")
    from src.tools.search import rag_stats_note

    # 纯函数语义不变（原 VF-9 锚定：同查询 → 输出统计）
    ok = rag_stats_note({"candidates": 20, "passed": 12, "filtered": 3,
                         "query": "citrus hlb"}, expect_query="citrus hlb")
    check("同查询 → 正常输出统计", "[检索统计]" in ok, ok)
    # 防串号语义必须保留：归属查询 ≠ 本次查询 → 空串（既有的跨会话保护）
    cross = rag_stats_note({"candidates": 20, "passed": 1, "filtered": 9,
                            "query": "another session query"},
                           expect_query="citrus hlb 黄龙病 防治")
    check("跨查询统计仍被丢弃（防串号保留）", cross == "", cross)
    # 修复语义在检索层：last_stats 归属改为记录用户原文——HyDE 主路径
    # queries[0]=生成段落，若仍记 rerank_query，工具侧 expect_query=用户原文
    # 比对恒不等 → 早停统计被误杀。以下源码级锁定接线。

    for fname, needles in [
        ("src/retrieval/multi_retriever.py", ["def search_multi(self, queries: List[str]",
                                              "original_query: str = \"\"",
                                              "original_query or rerank_query",
                                              "def search(self, query: str)",
                                              "original_query=query"]),
        ("src/tools/search.py", ["rag.search_multi(queries, original_query=query)"]),
    ]:
        src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), fname),
                   encoding="utf-8").read()
        for n in needles:
            check(f"{fname} 含接线点: {n}", n in src, n)


# ── VF-104 pdf_read 统一工具出口（错参 TypeError 修复）─────────────
def test_v92_pdf_read_unified_exit():
    print("[VF-104] pdf_read 统一出口")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "src/graph/expert_graph.py"), encoding="utf-8").read()
    check("pdf_read 分支不再直调 .func",
          "content, artifact = await asyncio.to_thread(pdf_read_func.func" not in src)
    check("pdf_read 分支走 run_tool_checked",
          "content = await run_tool_checked(pdf_read_tool, tc_dict[\"args\"])" in src)
    check("run_tool_checked 已处理 content_and_artifact",
          "content_and_artifact" in open(os.path.join(
              os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
              "src/tools/registry.py"), encoding="utf-8").read())


# ── VF-105 死代码移除锁定 ───────────────────────────────────────────
def test_v92_dead_code_removed():
    print("[VF-105] 死代码移除锁定")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    state_src = open(os.path.join(base, "src/graph/state.py"), encoding="utf-8").read()
    check("state.py 无 web_search_enabled 写而不读键", "web_search_enabled" not in state_src)

    pb_src = open(os.path.join(base, "src/core/progress_bus.py"), encoding="utf-8").read()
    check("progress_bus 无 get_tool_elapsed 定义", "def get_tool_elapsed" not in pb_src)
    check("progress_bus 无恒 True 开关定义", "_sse_debug_enabled: bool" not in pb_src)

    graph_src = open(os.path.join(base, "src/graph/graph.py"), encoding="utf-8").read()
    check("graph.py 无未用 logger", "logger = logging" not in graph_src)


# ── VF-106 证据账本读取器×4 收敛（P7）──────────────────────────────
def test_v92_evidence_readers_converged():
    print("[VF-106] 证据账本读取器 ×4 → _load_evidence_rows")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mgr_src = open(os.path.join(base, "src/session/manager.py"), encoding="utf-8").read()

    check("统一读取器存在", "def _load_evidence_rows(self, session_id" in mgr_src)
    for fn, call in [("build_evidence_block", "_load_evidence_rows(session_id,limit,"),
                     ("count_evidence_items", "_load_evidence_rows(session_id,None)"),
                     ("get_evidence_refs", "_load_evidence_rows(session_id,10)"),
                     ("get_evidence_materials", "_load_evidence_rows(session_id,4)")]:
        # 函数体内转写为委托调用（按缩进块切出函数源段，去空白后做子串校验，
        # 兼容多行调用排版）
        seg = mgr_src.split(f"def {fn}(")[1]
        seg = seg.split("\n    def ")[0]
        check(f"{fn} 委托统一读取器 ({call})", call in "".join(seg.split()))
    # 口径锚点仍在（LIMIT 10 轮 / LIMIT 4 轮 / 全表计数 / 渲染列）
    refs_seg = mgr_src.split("def get_evidence_refs(")[1].split("\n    def ")[0]
    materials_seg = mgr_src.split("def get_evidence_materials(")[1].split("\n    def ")[0]
    count_seg = mgr_src.split("def count_evidence_items(")[1].split("\n    def ")[0]
    block_seg = mgr_src.split("def build_evidence_block(")[1].split("\n    def ")[0]
    check("refs 保留 10 轮口径", "session_id, 10" in refs_seg)
    check("materials 保留 4 轮口径", "session_id, 4" in materials_seg)
    check("count 保留全表口径", "session_id, None" in count_seg)
    check("block 保留四列渲染", "turn_seq, query, evidence_json, report_text" in block_seg)


# ── VF-107 save 节点二合一（P5：只抽核心，两图行为不变）────────────
def test_v92_save_node_shared_core():
    print("[VF-107] save 节点二合一参数化")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    al_src = open(os.path.join(base, "src/core/agent_loop.py"), encoding="utf-8").read()
    check("共享核心存在", "async def run_save_node(state: dict" in al_src)
    for inv in ["trace[-1].content == answer", "startswith(\"[历史检索证据]\")",
                "no answer", "idempotency_key", "[{log_tag}:save] failed"]:
        check(f"共享核心保留不变量标记: {inv}", inv in al_src)

    eg_src = open(os.path.join(base, "src/graph/expert_graph.py"), encoding="utf-8").read()
    lg_src = open(os.path.join(base, "src/graph/light_graph.py"), encoding="utf-8").read()

    eg_seg = eg_src.split("def expert_save_node(")[1].split("\n    def ")[0]
    lg_seg = lg_src.split("def save_context_node(")[1].split("\n    def ")[0]
    eg_norm, lg_norm = "".join(eg_seg.split()), "".join(lg_seg.split())

    check("expert 包装: log_tag/IncludeWeb/ltm 门槛",
          'run_save_node(' in eg_norm and 'log_tag="ExpertGraph"' in eg_norm
          and "include_web=True" in eg_norm and "default_ltm_gate" in eg_norm)
    check("light 包装: 无 web 账本、无 LTM 额外门槛（行为不变）",
          'run_save_node' in lg_norm and 'log_tag="LightGraph"' in lg_norm
          and "include_web=False" in lg_norm and "ltm_gate" not in lg_norm)
    # 关键不变量仍在共享核心而非两图各自为政：两图包装不再含证据账本字面拼装
    check("两图包装已无账本字段拼装",
          '"chunk_id"' not in eg_norm and '"chunk_id"' not in lg_norm)


# ── 汇总 ──────────────────────────────────────────────────────────
def _summary():
    print(f"\n[VF-9.2] PASS {len(passed)} / FAIL {len(failed)}"
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