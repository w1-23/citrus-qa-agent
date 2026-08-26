# -*- coding: utf-8 -*-
"""v9.4 论文参数化回归：candidate_window / early_stop / query_mode 消融。

全部离线、无模型、无网络（仅验证配置默认值与纯函数逻辑）。约定见 test_batch1.py。
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


def _with(attr, val):
    """临时覆盖 settings 属性（try/finally 恢复）。"""
    old = getattr(settings, attr, None)
    setattr(settings, attr, val)
    return lambda: setattr(settings, attr, old)


# ── P0#1 候选窗口参数化 ─────────────────────────────────────────
def test_v94_candidate_window_default():
    print("[V94-1] candidate_window 参数化（默认 20 = 原 top_k_final*2 行为）")
    check("CANDIDATE_WINDOW 默认 20",
          settings.CANDIDATE_WINDOW == 20, settings.CANDIDATE_WINDOW)
    # 源码级锁定：multi_retriever 候选截断读 settings.CANDIDATE_WINDOW，
    # 不再出现 top_k_final*2 隐式硬编码
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "src/retrieval/multi_retriever.py"),
               encoding="utf-8").read()
    check("multi_retriever 使用 settings.CANDIDATE_WINDOW",
          "settings.CANDIDATE_WINDOW" in src)
    check("multi_retriever 无 top_k_final*2 隐式截断",
          "settings.TOP_K_FINAL * 2" not in src)
    # config.yaml 同步存在
    yaml_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 os.pardir, "config.yaml"), encoding="utf-8").read()
    check("config.yaml 含 candidate_window", "candidate_window: 20" in yaml_src)


# ── P0#2 早停参数化 ─────────────────────────────────────────────
def test_v94_early_stop_defaults():
    print("[V94-2] early_stop 参数化（默认 6 / 0.25 = 原硬编码行为）")
    check("EARLY_STOP_MIN_EVIDENCE 默认 6",
          settings.EARLY_STOP_MIN_EVIDENCE == 6, settings.EARLY_STOP_MIN_EVIDENCE)
    check("EARLY_STOP_NEW_RATIO 默认 0.25",
          abs(settings.EARLY_STOP_NEW_RATIO - 0.25) < 1e-9, settings.EARLY_STOP_NEW_RATIO)
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "src/core/agent_runner.py"),
               encoding="utf-8").read()
    check("agent_runner 使用 settings.EARLY_STOP_MIN_EVIDENCE",
          "settings.EARLY_STOP_MIN_EVIDENCE" in src)
    check("agent_runner 使用 settings.EARLY_STOP_NEW_RATIO",
          "settings.EARLY_STOP_NEW_RATIO" in src)
    check("agent_runner 无硬编码 (>= 6 and < 0.25)",
          "_prev_unique >= 6 and _new_ratio < 0.25" not in src)
    yaml_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 os.pardir, "config.yaml"), encoding="utf-8").read()
    check("config.yaml 含 early_stop_min_evidence", "early_stop_min_evidence: 6" in yaml_src)
    check("config.yaml 含 early_stop_new_ratio", "early_stop_new_ratio: 0.25" in yaml_src)


# ── P0#5 query_mode 消融开关 ────────────────────────────────────
def test_v94_query_mode_default():
    # v9.4b（2026-08-26，论文实证）：生产默认改 raw——exp1/1b/1c/2 四重互证
    # full 多路（MRR 0.256）远劣于原始查询单路（0.557 生产实测 / 0.653 对齐上界），
    # 归因 rerank 误用 HyDE 段 + 多路池稀释；config.yaml:29 已回写。
    print("[V94-3] query_mode 默认 raw（实证修正，原 full）")
    check("QUERY_MODE 默认 raw", settings.QUERY_MODE == "raw", settings.QUERY_MODE)
    yaml_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 os.pardir, "config.yaml"), encoding="utf-8").read()
    check("config.yaml 含 query_mode: raw", "query_mode: raw" in yaml_src)


def test_v94_compose_queries():
    print("[V94-4] _compose_queries 纯函数：8 模式路由（论文实验2/1 变体）")
    from src.tools.search import _compose_queries, _VALID_QUERY_MODES
    hp = {"hyde": "H" * 200,
          "multi_query": ["mq1", "mq2", "mq3", "mq4"],
          "summary": ["s1", "s2", "s3", "s4", "s5", "s6"]}
    m = _VALID_QUERY_MODES
    check("合法模式集合 8 个（含 mq_sum）", m == (
        "full", "raw", "hyde_only", "mq_only", "sum_only", "hyde_mq",
        "hyde_sum", "mq_sum"), m)
    full = _compose_queries("q", hp, "full")
    check("full = HyDE+MQ×3+SUM×3~5（9 路上限）", len(full) == 1 + 3 + 5, len(full))
    check("full 首路为 HyDE 锚", full[0] == "H" * 200)
    ho = _compose_queries("q", hp, "hyde_only")
    check("hyde_only = 1 路", len(ho) == 1 and ho[0] == "H" * 200, ho)
    mo = _compose_queries("q", hp, "mq_only")
    check("mq_only = 3 路", mo == ["mq1", "mq2", "mq3"], mo)
    so = _compose_queries("q", hp, "sum_only")
    check("sum_only = 3~5 路（取前5）", so == ["s1", "s2", "s3", "s4", "s5"], so)
    hm = _compose_queries("q", hp, "hyde_mq")
    check("hyde_mq = 4 路", len(hm) == 4 and hm[0] == "H" * 200, hm)
    hs = _compose_queries("q", hp, "hyde_sum")
    check("hyde_sum = 6 路（HyDE+SUM×5）", len(hs) == 1 + 5, hs)
    ms = _compose_queries("q", hp, "mq_sum")
    mq_sum_expected = ["mq1", "mq2", "mq3", "s1", "s2", "s3", "s4", "s5"]
    check("mq_sum = 8 路（MQ×3+SUM×5，无 HyDE）",
          ms == mq_sum_expected, ms)
    # raw 语义：无 hyde_parsed → None（调方走 rag.search 单路）
    raw = _compose_queries("q", None, "raw")
    check("raw 无 hyde → None", raw is None, raw)
    # 未知模式 → 降级 full（行为不变铁律）
    bogus = _compose_queries("q", hp, "not_a_mode")
    check("未知模式降级 full（9 路）", len(bogus) == 9, bogus)
    # 全空路源 → 原始查询保底 1 路
    empty = _compose_queries("q", {"hyde": None, "multi_query": [], "summary": []}, "full")
    check("空路源 → 原始查询保底 1 路", empty == ["q"], empty)


def test_v94_cache_key_mode_isolated():
    print("[V94-5] 缓存键纳入 query_mode（模式间缓存隔离）")
    from src.tools.search import _rag_cache_key

    class _FakeRAG:
        global_chunks = [1, 2, 3]
        batch_source = {"a": 1}

    # v9.4b: 生产默认已为 raw——显式锚定 full 作基线，避免与默认模式撞键
    restore0 = _with("QUERY_MODE", "full")
    try:
        base = _rag_cache_key("citrus hlb", True, _FakeRAG())
    finally:
        restore0()
    restore = _with("QUERY_MODE", "hyde_only")
    try:
        alt = _rag_cache_key("citrus hlb", True, _FakeRAG())
    finally:
        restore()
    check("fulll 与 hyde_only 缓存键不同（隔离）", base != alt, (base, alt))
    restore2 = _with("QUERY_MODE", "raw")
    try:
        raw_key = _rag_cache_key("citrus hlb", True, _FakeRAG())
    finally:
        restore2()
    check("full 与 raw 缓存键不同（隔离）", base != raw_key)


def test_v94_runner_callsite_preserved():
    print("[V94-6] search_multi 接线点保留（test_refactor_v92 源码锁定兼容）")
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "src/tools/search.py"), encoding="utf-8").read()
    check("search.py 保留 rag.search_multi(queries, original_query=query)",
          "rag.search_multi(queries, original_query=query)" in src)