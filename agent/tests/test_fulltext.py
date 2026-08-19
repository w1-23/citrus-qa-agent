# -*- coding: utf-8 -*-
"""fetch_fulltext 验证（离线纯函数必跑；联网抓取经 --live 可选）。

用法:
    python tests/test_fulltext.py           # 仅离线纯函数
    python tests/test_fulltext.py --live    # 附加联网抓取（需外网）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed, failed = [], []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
        print(f"  PASS  {name}")
    else:
        failed.append(name)
        print(f"  FAIL  {name}  {detail}")


from src.tools.fulltext import (
    _sentence_spans,
    _chunk_blocks,
    _stitch,
    _xml_to_blocks,
    _fetch_blocks,
    _CHUNK_MAX_CHARS,
)
from src.core.agent_runner import _dedup_evidence_items


def test_sentence_spans():
    print("[Fulltext] 复句级细切块")
    text = ("Candidatus Liberibacter asiaticus infects citrus. "
            "The mean titer was 1.2e6 copies per gram. "
            "This increase was significant (p = 0.003). Next, qPCR confirmed it.")
    spans = _sentence_spans(text, 60)
    check("切出多块", len(spans) >= 2, f"got {len(spans)}")
    check("每块 ≤ max_chars", all(len(s) <= 60 for s in spans),
          [len(s) for s in spans])

    long_sentence = "A" * 900
    hard = _sentence_spans(long_sentence, 300)
    check("超长句硬切 3 块且各 ≤300", len(hard) == 3 and all(len(s) <= 300 for s in hard),
          [len(s) for s in hard])

    check("空输入 → 空列表", _sentence_spans("   ", 100) == [])


def test_chunk_blocks():
    print("[Fulltext] 块 → 细块（上下文贯通 + 文档序）")
    blocks = [
        ("摘要", "First sentence. Second sentence here."),
        ("Results", "Another result. With numbers 12.4 mM."),
    ]
    chunks = _chunk_blocks(blocks, 30)
    check("chunk_index 连续", [c["chunk_index"] for c in chunks] == list(range(len(chunks))))
    check("上下文继承", all(c["ctx"] in ("摘要", "Results") for c in chunks),
          [c["ctx"] for c in chunks])


def test_stitch():
    print("[Fulltext] 文档序拼接 + 预算截断")
    kept = [
        {"chunk_index": 2, "ctx": "Results", "text": "RESULT2", "rerank_score": 0.9},
        {"chunk_index": 0, "ctx": "摘要", "text": "ABS", "rerank_score": 0.8},
    ]
    s = _stitch(kept, 1000)
    check("按文档序（摘要在前）", s.index("ABS") < s.index("RESULT2"))
    check("含分数字段", "0.80" in s and "0.90" in s)

    s2 = _stitch(kept, 20)
    check("预算截断", len(s2) < 40, f"len={len(s2)}")


def test_xml_to_blocks():
    print("[Fulltext] JATS XML 解析（无图、含表、命名空间无关）")
    xml = """<?xml version="1.0"?>
    <article xmlns:xlink="http://www.w3.org/1999/xlink">
      <front><article-meta><title-group>
        <article-title>Citrus HLB Test</article-title>
      </title-group><abstract><p>We measured 3.5 mM citrate.</p></abstract>
      </article-meta></front>
      <body>
        <sec><title>Introduction</title><p>HLB is caused by CLas.</p></sec>
        <sec><title>Results</title><p>p-value was 0.003.</p>
          <table-wrap><label>Table 1</label><caption>Metabolite levels.</caption>
            <tr><td>citrate</td><td>12.4 mM</td></tr></table-wrap>
          <fig><label>Figure 1</label><caption>Pathway map.</caption></fig>
        </sec>
      </body>
    </article>"""
    blocks = _xml_to_blocks(xml.encode("utf-8"))
    texts = "\n".join(t for _, t in blocks)
    ctxs = [c for c, _ in blocks]
    check("标题提取", any("Citrus HLB Test" in b[1] for b in blocks))
    check("摘要提取", any("3.5 mM" in b[1] for b in blocks))
    check("表格 cell 数值保留", "12.4 mM" in texts)
    check("图题保留", any("Pathway map" in b[1] for b in blocks))
    check("章节上下文标注", "Results" in ctxs, str(ctxs))
    check("表题作为上下文", any("Table 1" in c for c in ctxs), str(ctxs))


def test_dedup():
    print("[Fulltext] 证据去重碰撞 → 保留正文更丰富")
    abstract = {"doi": "10.1/x", "title": "T", "abstract": "short abstract"}
    fulltext = {"doi": "10.1/x", "title": "T", "text": "x" * 500}
    out = _dedup_evidence_items([abstract, fulltext])
    check("同 DOI 仅 1 条", len(out) == 1)
    check("正文在摘要后仍取代摘要", len(out[0].get("text", "")) == 500)

    out2 = _dedup_evidence_items([fulltext, abstract])
    check("先到正文不被摘要覆盖", len(out2[0].get("text", "")) == 500)

    out3 = _dedup_evidence_items([
        {"doi": "10.1/a", "title": "A", "text": "a"},
        {"doi": "10.1/b", "title": "B", "text": "b"},
    ])
    check("不同 DOI 均保留", len(out3) == 2)

    out4 = _dedup_evidence_items([
        {"title": "Same Title", "text": "short"},
        {"title": "Same Title", "text": "much longer " + "y" * 200},
    ])
    check("无 DOI 同标题去重且留更长者", len(out4) == 1 and len(out4[0]["text"]) > 100)


def test_live_fetch():
    """联网抓取一条真实 OA 柑橘文献（--live 触发；失败不判 FAIL）。"""
    import requests
    try:
        r = requests.get(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
            params={"query": "OPEN_ACCESS:y AND (citrus huanglongbing)",
                    "format": "json", "resultType": "core", "pageSize": "1"},
            headers={"User-Agent": "citrus-qa-agent/1.0 (mailto:citrus-agent@localhost)"},
            timeout=20)
        if r.status_code != 200:
            print("  SKIP  联网搜索失败 status=%s（沙箱可能无外网）" % r.status_code)
            return
        hits = (r.json().get("resultList") or {}).get("result") or []
        if not hits:
            print("  SKIP  无 OA 命中")
            return
        h = hits[0]
        doi = h.get("doi") or ""
        pmcid = h.get("pmcid") or ""
        print("  命中: %s | %s | %s" % (doi, pmcid, (h.get("title") or "")[:60]))
        blocks, source = _fetch_blocks(doi or pmcid,
                                       {"pmcid": pmcid, "title": h.get("title"),
                                        "year": h.get("pubYear"), "doi": doi})
        check("联网抓取 OA 全文块", bool(blocks), f"source={source} blocks={len(blocks)}")
        if not blocks:
            return
        chunks = _chunk_blocks(blocks, _CHUNK_MAX_CHARS)
        check("全文切成细块", len(chunks) > 0, f"{len(chunks)} 块")

        # 端到端（含重排/阈值/拼接），Reranker 需本地模型缓存
        try:
            from src.tools.fulltext import fetch_fulltext
            content, artifact = fetch_fulltext.func(doi or pmcid, "huanglongbing bacterial titer qPCR")
            items = artifact.get("main_results") or []
            check("端到端产生 1 条证据", len(items) == 1, f"items={len(items)}")
            check("证据文本非空", bool(items and items[0].get("text")))
        except Exception as e:
            print("  SKIP  端到端重排失败（可能无模型缓存/无 GPU）: %r" % e)
    except Exception as e:
        print("  SKIP  联网异常: %r" % e)


if __name__ == "__main__":
    test_sentence_spans()
    test_chunk_blocks()
    test_stitch()
    test_xml_to_blocks()
    test_dedup()
    if "--live" in sys.argv:
        test_live_fetch()
    print("\n" + "=" * 50)
    print(f"结果: {len(passed)} PASS / {len(failed)} FAIL")
    if failed:
        print("失败项: " + ", ".join(failed))
        sys.exit(1)
    print("全部通过 ✓")