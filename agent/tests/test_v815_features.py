# -*- coding: utf-8 -*-
"""v8.15 特性回归：学术联网默认关闭 / 证据来源体系 / 查询级缓存 / 联网开关脚手架。

全部离线、无模型、无网络（仅验证逻辑与门控纯函数）。约定见 test_batch1.py。
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.core.evidence import src_of, SOURCE_TAG, SOURCE_LABEL, SOURCE_ORDER

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


# ── F-15-1 配置默认值（学术联网关 / 缓存开 / 联网的部署展示默认）─────
def test_v815_config_defaults():
    print("[VF-1] v8.15 配置默认值")
    check("ACADEMIC_ENABLED 默认关", settings.ACADEMIC_ENABLED is False)
    check("ACADEMIC_SOURCES 默认空", settings.ACADEMIC_SOURCES == [])
    check("RAG_CACHE_ENABLED 默认开", settings.RAG_CACHE_ENABLED is True)
    check("RAG_CACHE_SIZE=300", settings.RAG_CACHE_SIZE == 300)
    check("RAG_CACHE_TTL_HOURS=24", settings.RAG_CACHE_TTL_HOURS == 24)
    # v8.15: web_search.enabled 仅部署展示默认（非启用门槛，前端开关才是总开关）；
    # 关闭时前端按钮显示关、请求不带联网标志、工具执行层短路。
    check("WEB_SEARCH_RESPONSES_PATH 默认 /v1/responses",
          settings.WEB_SEARCH_RESPONSES_PATH == "/v1/responses")


# ── F-15-2 来源判定纯函数 ──────────────────────────────────────────
def test_v815_src_of():
    print("[VF-2] 证据来源判定")
    check("UCR source_type → ucr", src_of({"source_type": "UCR citrus variety"}) == "ucr")
    check("_src=ucr → ucr", src_of({"_src": "ucr"}) == "ucr")
    check("source=ucr → ucr", src_of({"source": "ucr"}) == "ucr")
    check("无标记 → rag", src_of({"doi": "10.x"}) == "rag")
    check("非 dict → rag", src_of(None) == "rag")
    check("SOURCE_TAG rag= RAG / ucr= UCR / web= Web",
          (SOURCE_TAG["rag"], SOURCE_TAG["ucr"], SOURCE_TAG["web"]) == ("RAG", "UCR", "Web"))
    check("SOURCE_LABEL 中文名", SOURCE_LABEL["rag"] == "本地文献库"
          and SOURCE_LABEL["ucr"] == "UCR品种库" and SOURCE_LABEL["web"] == "联网搜索")
    check("SOURCE_ORDER 前端分组顺序", list(SOURCE_ORDER) == ["rag", "ucr", "web", "historical"])


# ── F-15-3 工具注册门控（academic 随开关；deepseek_web 始终注册）────────
def test_v815_tool_registry_gating():
    print("[VF-3] 工具注册门控")
    from src.tools.registry import get_tool_spec, init_tool_registry
    from src.core import agent_runner as ar

    _aca = settings.ACADEMIC_ENABLED
    try:
        # 场景 A：学术关闭默认态
        settings.ACADEMIC_ENABLED = False
        init_tool_registry()
        check("学术工具未注册", get_tool_spec("academic_search") is None)
        check("全文工具未注册", get_tool_spec("fetch_fulltext") is None)
        check("本地检索仍注册", get_tool_spec("citrus_rag_search") is not None)
        # v8.15: 联网搜索工具始终注册（前端开关是总开关，不在注册层剔除）
        check("联网搜索工具始终注册", get_tool_spec("deepseek_web_search") is not None)
        names = ar._resolve_tool_names("retrieve-agent")
        check("retrieve-agent 含本地+联网", names == ["citrus_rag_search", "deepseek_web_search"],
              str(names))

        # 场景 B：学术开启恢复
        settings.ACADEMIC_ENABLED = True
        init_tool_registry()
        check("学术工具恢复注册", get_tool_spec("academic_search") is not None)
        check("全文工具恢复注册", get_tool_spec("fetch_fulltext") is not None)
        names2 = ar._resolve_tool_names("retrieve-agent")
        expect = ["citrus_rag_search", "academic_search", "fetch_fulltext", "deepseek_web_search"]
        check("retrieve-agent 工具列表齐全", names2 == expect, str(names2))
    finally:
        settings.ACADEMIC_ENABLED = _aca
        init_tool_registry()


# ── F-15-4 工具级 DISABLED 守卫（请求 contextvar 短路）───────────────
def test_v815_tool_guards():
    print("[VF-4] 工具关闭守卫")
    from src.tools.search import academic_search
    from src.tools.fulltext import fetch_fulltext
    from src.tools.deepseek_web import deepseek_web_search
    from src.core.tracing import set_web_search_enabled, web_search_enabled

    _aca = settings.ACADEMIC_ENABLED
    try:
        settings.ACADEMIC_ENABLED = False
        # 联网开关（请求 contextvar）默认关闭 → deepseek_web_search 短路
        set_web_search_enabled(False)
        c, a = academic_search.func("citrus huanglongbing 2025")
        check("academic_search 短路 [DISABLED]", isinstance(c, str) and c.startswith("[DISABLED]"))
        check("academic_search artifact 双键空", a == {"main_results": [], "web_results": []}, str(a))
        c2, a2 = fetch_fulltext.func("10.1038/nature25447", "hlb titer")
        check("fetch_fulltext 短路 [DISABLED]", c2.startswith("[DISABLED]"))
        c3, a3 = deepseek_web_search.func("最新柑橘品种")
        check("deepseek_web_search 开关关 → [DISABLED]", c3.startswith("[DISABLED]"))
        check("deepseek_web_search artifact 空", a3 == {"main_results": [], "web_results": []})
        # contextvar 隔离：开了 ≠ 走短路，且上下文按请求隔离
        check("web_search_enabled 默认 False", web_search_enabled() is False)
        set_web_search_enabled(True)
        check("web_search_enabled 设置后 True", web_search_enabled() is True)
        set_web_search_enabled(False)
        check("web_search_enabled 复位 False", web_search_enabled() is False)
    finally:
        settings.ACADEMIC_ENABLED = _aca
        set_web_search_enabled(False)


# ── F-15-5 查询级结果缓存（纯函数 + 命中/失效）────────────────────
def test_v815_rag_cache():
    print("[VF-5] 查询级结果缓存")
    from src.tools.search import (
        _norm_cache_query, _corpus_fingerprint, _rag_cache_key,
        _rag_cache_get, _rag_cache_put, _RAG_CACHE, _RAG_CACHE_LOCK,
    )

    class FakeRAG:
        global_chunks = list(range(100))
        batch_source = {"1-50": "rag", "categories-cn": "ucr"}

    old_enabled = settings.RAG_CACHE_ENABLED
    old_size = settings.RAG_CACHE_SIZE
    try:
        settings.RAG_CACHE_ENABLED = True
        settings.RAG_CACHE_SIZE = 3
        check("规范化查询（空格/大小写）", _norm_cache_query(" Citrus   HLB ") == "citrus hlb")
        fp1 = _corpus_fingerprint(FakeRAG())
        check("语料指纹含块数与批次数", fp1 == "100:1-50,categories-cn", fp1)
        k1 = _rag_cache_key("citrus hlb", True, FakeRAG())
        k2 = _rag_cache_key("citrus  hlb", True, FakeRAG())
        k3 = _rag_cache_key("citrus hlb", False, FakeRAG())
        check("同 query 同键（规范化后命中）", k1 == k2)
        check("HyDE 开关不同则键不同", k1 != k3)
        # 写入 → 命中
        fake_results = [{"title": "P1", "doi": "10.1/x", "text": "body"}]
        _rag_cache_put(k1, fake_results)
        got = _rag_cache_get(k1)
        check("写入后命中", got is not None and len(got) == 1)
        if got:
            check("命中返回浅拷贝（改副本不污染缓存）", got[0] is not fake_results[0])
        # LRU 淘汰
        _rag_cache_put(k3, [{"title": "P2"}])
        check("LRU 淘汰后旧键失效", len(_RAG_CACHE) <= 3)
    finally:
        settings.RAG_CACHE_ENABLED = old_enabled
        settings.RAG_CACHE_SIZE = old_size
        with _RAG_CACHE_LOCK:
            _RAG_CACHE.clear()


# ── F-15-6 参数解析（Responses web_search_call 防御式解析，含实测结构）───
def test_v815_web_response_parse():
    print("[VF-6] Responses 引用解析")
    from src.tools.deepseek_web import _parse_response_output
    out = [
        {"type": "message", "content": [{"type": "output_text", "text": "摘要…"}]},
        {"type": "web_search_call", "status": "completed", "title": "T1",
         "url": "https://example.com/1", "content": "snippet1"},
        {"type": "web_search_call", "title": "T2", "url": "", "content": {"url": "https://x/2", "title": "T2"}},
        {"type": "other"},
    ]
    summary, calls, meta = _parse_response_output(out)
    check("text 块聚合", summary == "摘要…", summary)
    check("引用解析 2 条", len(calls) == 2, str(calls))
    check("引用字段 title/url/abstract", calls[0]["title"] == "T1"
          and calls[0]["url"] == "https://example.com/1")
    check("兼容 content 内嵌 url", calls[1]["url"] == "https://x/2")


def test_v815_web_response_parse_real_shape():
    """复刻 2026-08 真机探测的真实结构：web_search_call 只含 action.queries，
    URL 来自 message 行内 MD 链接 / 裸 URL / content.annotations。"""
    print("[VF-7] Responses 引用解析（真机结构）")
    from src.tools.deepseek_web import _parse_response_output
    out = [
        {"type": "reasoning",
         "content": [{"type": "reasoning_text", "text": "thinking…"}]},
        {"type": "web_search_call", "id": "c1", "status": "completed",
         "action": {"type": "search", "queries": ["柑橘黄龙病 防治 2024", "citrus hlb 2025"]}},
        {"type": "message", "content": [
            {"type": "output_text",
             "text": "最新进展见 [中国农学通报综述](https://www.cjab.org.cn/x) "
                     "和 https://pubmed.ncbi.nlm.nih.gov/12345 …"}]},
        {"type": "message", "content": [
            {"type": "output_text", "text": "华中农大纳米酶研究…", "annotations": [
                {"type": "url_citation", "title": "纳米酶综述", "url": "https://nature.com/articles/123"}]}]},
    ]
    summary, calls, meta = _parse_response_output(out)
    check("检索关键词提取", meta.get("queries", [])[:2] == ["柑橘黄龙病 防治 2024", "citrus hlb 2025"],
          str(meta.get("queries")))
    check("URL 去重 3 条", len(calls) == 3, str([c["url"] for c in calls]))
    urls = [c["url"] for c in calls]
    check("MD 链接标题+url", any(u == "https://www.cjab.org.cn/x" for u in urls))
    check("裸 URL 兜底", any(u == "https://pubmed.ncbi.nlm.nih.gov/12345" for u in urls))
    check("annotations 深扫", any(u == "https://nature.com/articles/123" for u in urls))
    check("text 聚合", len(summary) > 10, summary[:30])


# ── 汇总 ──────────────────────────────────────────────────────────
def _summary():
    print(f"\n[VF-15] PASS {len(passed)} / FAIL {len(failed)}"
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
