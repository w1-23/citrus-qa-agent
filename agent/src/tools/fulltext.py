# -*- coding: utf-8 -*-
"""fetch_fulltext — 开放全文即时证据链路（v8.14，全程确定性代码，无 LLM）。

背景：academic_search 只回题录+摘要，CrossRef 更是多数连摘要都没有；数值/机制/
表格/统计类问题「摘要不够」。本工具把「找论文」升级成「抠正文证据」：

    DOI/PMID ──► 取 OA 全文 ──► 复句级细切块 ──► 复用 Reranker 重排
             ──► 动态阈值 ──► 按文档序拼接 top 块 ──► 单篇证据块进检索回执

设计要点（与既有业务严格兼容）：
  - 无 LLM：抓取/切块/重排/取块全是代码，LLM 只读最终证据块，token 最小化。
  - 无图：JATS XML / PDF 只取文字层，图天然丢弃；表格文字保留（数值富矿）。
  - 即时文本不入向量库：重排走 cross-encoder（Reranker.rerank 与索引无关），
    只有超大文档（>1000 块）才用 Embedder 内存 cosine 粗筛，用完即弃。
  - 粒度比离线语料更细：离线 ingest 按段落窗口（~800 字符）为召回保完整；
    这里复句级 ~500 字符，为「抓具体数值」保精度。
  - 复用 Reranker 单例（MultiBatchRetriever 已预热），不新增模型加载。
"""
import logging
import re
import time
from typing import List, Optional, Tuple

import requests

from langchain_core.tools import tool

from src.config import settings
from src.core.evidence import EVIDENCE_RENDER_MAX_CHARS
from src.engine.reranker import Reranker
from src.tools.search import _format_tool_result

logger = logging.getLogger(__name__)

# ── 可调常量（复句级细块：上限 500 字符；比离线 ~800 更细，利于抠数值）──
_CHUNK_MAX_CHARS = 500
_TOP_K = 8
_HTTP_TIMEOUT = 15
_PDF_TIMEOUT = 30
_COARSE_MIN_CHUNKS = 1000      # 超过才启用 Embedder 粗筛
_COARSE_KEEP = 200             # 粗筛保留候选数

_EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_EPMC_REST = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_PMC_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_UNPAYWALL = "https://api.unpaywall.org/v2"

_SENT_SPLIT = re.compile(r'(?<=[.!?。！？])\s+(?=[A-Z0-9])')
_WHITESPACE = re.compile(r"\s+")


def _email() -> str:
    return settings.NCBI_ENTREZ_EMAIL or "citrus-agent@localhost"


def _headers() -> dict:
    return {"User-Agent": f"citrus-qa-agent/1.0 (mailto:{_email()})"}


# ───────────────────────── 1. 全文抓取 ─────────────────────────


def _europepmc_meta(doi: str) -> dict:
    """Europe PMC 按 DOI 查元数据（标题/年份/PMCID/PMID），供全文定位与证据标注。"""
    try:
        r = requests.get(
            _EPMC_SEARCH,
            params={"query": f'DOI:"{doi}"', "format": "json",
                    "resultType": "core", "pageSize": "1"},
            headers=_headers(), timeout=_HTTP_TIMEOUT,
        )
        if r.status_code != 200:
            return {}
        hits = (r.json().get("resultList") or {}).get("result") or []
        if not hits:
            return {}
        h = hits[0]
        return {
            "title": h.get("title") or "",
            "year": h.get("pubYear") or "",
            "pmcid": h.get("pmcid") or "",
            "pmid": h.get("pmid") or "",
            "doi": h.get("doi") or doi,
        }
    except Exception as e:
        logger.warning(f"[fetch_fulltext] europepmc meta 失败: {e}")
        return {}


def _xml_to_blocks(xml_bytes: bytes) -> List[Tuple[str, str]]:
    """JATS XML → 有序 (上下文, 文本) 块列表；命名空间无关，仅取文字层（无图）。

    上下文用于给每个证据块标注章/节/表/图题来源。表格 cell 文字并入文本（数值富矿）。
    """
    import xml.etree.ElementTree as ET

    def local(e):
        return e.tag.rsplit("}", 1)[-1]

    def text_of(e):
        return " ".join(e.itertext())

    def sec_title(sec):
        for c in sec:
            if local(c) == "title":
                t = text_of(c).strip()
                if t:
                    return t
        return ""

    blocks: List[Tuple[str, str]] = []

    def walk(elem, section):
        for child in elem:
            name = local(child)
            if name == "article-title":
                t = text_of(child).strip()
                if t:
                    blocks.append(("标题", t))
            elif name == "abstract":
                walk(child, "摘要")
            elif name == "sec":
                walk(child, sec_title(child) or section)
            elif name == "p":
                t = text_of(child).strip()
                if t:
                    blocks.append((section or "正文", t))
            elif name in ("table-wrap", "fig"):
                label = next((text_of(e).strip() for e in child.iter()
                              if local(e) == "label"), "")
                caption = " ".join(text_of(e).strip() for e in child.iter()
                                   if local(e) == "caption").strip()
                heading = " ".join(x for x in (label, caption) if x).strip() or \
                    ("表" if name == "table-wrap" else "图")
                kind = "表" if name == "table-wrap" else "图"
                body = ""
                if name == "table-wrap":
                    body = " | ".join(text_of(e).strip() for e in child.iter()
                                      if local(e) in ("td", "th") and text_of(e).strip())
                full = " ".join(x for x in (caption, body) if x).strip()
                if full:
                    blocks.append((heading, full))
            elif name in ("front", "body", "back", "title-group", "article-meta"):
                walk(child, section)
            # 其余节点（ref-list、pub-history 等）忽略

    try:
        root = ET.fromstring(xml_bytes)
    except Exception as e:
        logger.warning(f"[fetch_fulltext] XML 解析失败，退化为去标签: {e}")
        return [("正文", _strip_tags(xml_bytes.decode("utf-8", "ignore")))]
    walk(root, "正文")
    # 去空、合并相邻同上下文（正文段落常被打散成多个 p）
    merged: List[Tuple[str, str]] = []
    for ctx, txt in blocks:
        txt = _WHITESPACE.sub(" ", txt).strip()
        if not txt:
            continue
        if merged and merged[-1][0] == ctx:
            merged[-1] = (ctx, f"{merged[-1][1]} {txt}")
        else:
            merged.append((ctx, txt))
    return merged


def _strip_tags(xml_text: str) -> str:
    return re.sub(r"<[^>]+>", " ", xml_text or "")


def _pdf_to_text(pdf_bytes: bytes) -> str:
    import fitz  # PyMuPDF（ingest.py 同款）
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def _fetch_blocks(doi: str, meta: dict) -> Tuple[List[Tuple[str, str]], str]:
    """按顺序尝试三条 OA 通道返回 (块列表, 来源标识)；全部失败返回 ([], "")。"""
    pmcid = meta.get("pmcid") or ""
    if pmcid:
        # 1) Europe PMC fullTextXML
        try:
            r = requests.get(f"{_EPMC_REST}/{pmcid}/fullTextXML",
                             headers=_headers(), timeout=_HTTP_TIMEOUT)
            if r.status_code == 200 and r.content:
                blocks = _xml_to_blocks(r.content)
                if blocks:
                    return blocks, "europepmc"
        except Exception as e:
            logger.warning(f"[fetch_fulltext] europepmc fullText 失败: {e}")
        # 2) NCBI PMC efetch（同 JATS，同一提取器）
        try:
            r = requests.get(_PMC_EFETCH,
                             params={"db": "pmc", "id": pmcid, "retmode": "xml"},
                             timeout=_HTTP_TIMEOUT)
            if r.status_code == 200 and r.content:
                blocks = _xml_to_blocks(r.content)
                if blocks:
                    return blocks, "pmc"
        except Exception as e:
            logger.warning(f"[fetch_fulltext] pmc efetch 失败: {e}")
    # 3) Unpaywall → OA PDF
    try:
        r = requests.get(f"{_UNPAYWALL}/{doi}", params={"email": _email()},
                         headers=_headers(), timeout=_HTTP_TIMEOUT)
        if r.status_code == 200:
            pdf_url = ((r.json().get("best_oa_location") or {}).get("url_for_pdf")) or ""
            if pdf_url:
                pr = requests.get(pdf_url, headers=_headers(), timeout=_PDF_TIMEOUT)
                if pr.status_code == 200 and pr.content:
                    txt = _pdf_to_text(pr.content)
                    if txt.strip():
                        return [("全文", txt)], "unpaywall-pdf"
    except Exception as e:
        logger.warning(f"[fetch_fulltext] unpaywall 失败: {e}")
    return [], ""


# ───────────────────────── 2. 复句级细切块 ─────────────────────────


def _sentence_spans(text: str, max_chars: int) -> List[str]:
    """按句子边界聚合到 max_chars 窗口；超长句子硬切。复句级、保上下文。

    不变量：任一产出块长度 ≤ max_chars —— 仅当「当前句 + 新句 ≤ max_chars」时才
    并入，否则先落当前块再起新块（单句已由硬切保证 ≤ max_chars）。
    """
    text = _WHITESPACE.sub(" ", text or "").strip()
    if not text:
        return []

    def joined_len(chunks: List[str]) -> int:
        return sum(len(x) for x in chunks) + max(0, len(chunks) - 1)

    spans: List[str] = []
    cur: List[str] = []
    for raw in _SENT_SPLIT.split(text):
        s = raw.strip()
        if not s:
            continue
        # 罕见超长句：硬切，保证单句 ≤ max_chars
        while len(s) > max_chars:
            if cur:
                spans.append(" ".join(cur))
                cur = []
            spans.append(s[:max_chars].strip())
            s = s[max_chars:].lstrip()
        if not s:
            continue
        if cur and joined_len(cur) + 1 + len(s) > max_chars:
            spans.append(" ".join(cur))
            cur = [s]
        else:
            cur.append(s)
    if cur:
        spans.append(" ".join(cur))
    return [sp for sp in spans if sp.strip()]


def _chunk_blocks(blocks: List[Tuple[str, str]], max_chars: int) -> List[dict]:
    """块列表 → 细块（每块带上下文标签与文档序 chunk_index）。"""
    chunks: List[dict] = []
    for ctx, text in blocks:
        for sp in _sentence_spans(text, max_chars):
            chunks.append({"ctx": ctx or "正文", "text": sp, "chunk_index": len(chunks)})
    return chunks


# ───────────────────────── 3. 重排 + 阈值 + 拼接 ─────────────────────────


def _coarse_filter(query: str, chunks: List[dict], keep: int) -> List[dict]:
    """超大文档先用 Embedder 内存 cosine 粗筛（不留库），再交 Reranker 精排。"""
    try:
        import numpy as np
        from src.engine.embedder import Embedder
        emb = Embedder()
        qv = emb.embed_query(query)
        if not qv:
            return chunks[:keep]
        dv = emb.embed_docs([c["text"] for c in chunks])
        if not dv or len(dv) != len(chunks):
            return chunks[:keep]
        q = np.asarray(qv, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        D = np.asarray(dv, dtype=np.float32)
        D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
        idx = np.argsort(-(D @ q))[:keep]
        return [chunks[int(i)] for i in idx]
    except Exception as e:
        logger.warning(f"[fetch_fulltext] coarse filter 失败，退化为截断: {e}")
        return chunks[:keep]


def _stitch(kept: List[dict], budget: int) -> str:
    """按文档序拼接重排后保留下来的块，每块带「章节 | 分数」来源标注。"""
    ordered = sorted(kept, key=lambda c: c.get("chunk_index", 0))
    parts: List[str] = []
    used = 0
    for c in ordered:
        head = f"[{c.get('ctx', '正文')} | {c.get('rerank_score', 0):.2f}] {c['text']}"
        if used + len(head) + 2 > budget:
            break
        parts.append(head)
        used += len(head) + 2
    return "\n\n".join(parts)


# ───────────────────────── 4. 工具入口 ─────────────────────────


@tool(response_format="content_and_artifact")
def fetch_fulltext(doi: str, query: str, pmid: str = "", top_k: int = 8) -> Tuple[str, dict]:
    """按 DOI 抓取开放全文并返回与 query 最相关的正文证据块（确定性代码，无 LLM）。

    用于「摘要不够」的问题（具体数值/机制/基因/表格/统计结论）：先在 academic_search
    命中后，对 top 命中文献传入其 DOI 调用本工具，得到正文细块（~500 字符/块，带
    章节/表图题上下文），已按 query 重排并过滤低相关块。

    Args:
        doi: 论文 DOI（如 10.1038/s41438-...）或 PMCID（如 PMC12345）。
        query: 用于重排的自然语言问题/关键词（英文更佳，5-15 词）。
        pmid: 可选 PMID，仅作元数据兜底。
        top_k: 返回证据块数上限（默认 8）。

    返回格式: content_and_artifact —— content 为拼接后的正文证据文本；
              artifact["main_results"] 为该篇文献的单条证据（结构与检索回执一致）。
    """
    t0 = time.perf_counter()
    doi = (doi or "").strip()
    query = (query or "").strip()
    empty = {"main_results": [], "web_results": []}

    if not doi:
        return ("[ERR_PARSE] doi 不能为空。建议: 从 academic_search 结果中复制 DOI 传入。",
                empty)
    if not query:
        logger.warning("[fetch_fulltext] query 为空，用 DOI 兜底重排")

    rerank_q = query or doi

    try:
        meta = _europepmc_meta(doi) or {}
        if not meta and doi.upper().startswith("PMC"):
            meta = {"pmcid": doi.upper(), "title": "", "year": "", "doi": ""}
        if pmid and not meta.get("pmid"):
            meta["pmid"] = pmid

        blocks, source = _fetch_blocks(doi, meta)
        elapsed = (time.perf_counter() - t0) * 1000
        if not blocks:
            content = _format_tool_result(
                "fetch_fulltext", f"doi={doi}",
                "[ERR_NETWORK] 未获取到该文献的开放全文（可能非 OA / 无全文权限 / 网络失败）。"
                "建议: 依赖学术摘要或本地 RAG；或换一篇 OA 文献。",
                status="error", elapsed_ms=elapsed)
            return content, empty

        chunks = _chunk_blocks(blocks, _CHUNK_MAX_CHARS)
        if not chunks:
            content = _format_tool_result(
                "fetch_fulltext", f"doi={doi}",
                "[ERR_PARSE] 全文文本为空或无法解析。建议: 换一个 DOI 或源。",
                status="empty", elapsed_ms=elapsed)
            return content, empty

        if len(chunks) > _COARSE_MIN_CHUNKS:
            chunks = _coarse_filter(rerank_q, chunks, _COARSE_KEEP)

        ranked = Reranker().rerank(rerank_q, chunks, top_k=max(int(top_k), 1))
        kept: List[dict] = []
        top_score = 0.0
        if ranked:
            top_score = ranked[0].get("rerank_score", 0)
            # 动态阈值与本地 RAG 同口径（max(绝对下限, top×比例)）
            threshold = max(settings.RERANK_THRESHOLD,
                            top_score * settings.DYNAMIC_THRESHOLD_RATIO)
            kept = [c for c in ranked if c.get("rerank_score", 0) >= threshold]

        if not kept:
            content = _format_tool_result(
                "fetch_fulltext", f"doi={doi}",
                f"已取得全文（{len(chunks)} 块，来源 {source}）但无段落与问题相关"
                f"（最高分 {top_score:.2f} 低于阈值 {threshold if ranked else 0:.2f}）。"
                "建议: 换检索角度或依赖摘要。",
                status="empty", elapsed_ms=elapsed)
            return content, empty

        stitched = _stitch(kept, EVIDENCE_RENDER_MAX_CHARS)
        title = meta.get("title") or doi
        year = meta.get("year") or ""
        paper_id = "ft:" + re.sub(r"[^\w.-]+", "_", meta.get("doi") or doi)

        item = {
            "title": title,
            "year": year,
            "doi": meta.get("doi") or doi,
            "source": f"fulltext:{source}",
            "text": stitched,
            "score": top_score,
            "rerank_score": top_score,
            "paper_id": paper_id,
            "chunk_index": 0,
        }

        content = _format_tool_result(
            "fetch_fulltext", f"doi={doi} q={query[:60]}",
            stitched, status="ok", results_count=len(kept), elapsed_ms=elapsed)
        return content, {"main_results": [item], "web_results": []}

    except Exception as e:
        logger.error(f"[fetch_fulltext] 失败: {e}")
        content = _format_tool_result(
            "fetch_fulltext", f"doi={doi}",
            f"[ERR_PARSE] 全文证据抓取异常: {e}。建议: 依赖学术摘要或本地 RAG。",
            status="error",
            elapsed_ms=(time.perf_counter() - t0) * 1000)
        return content, empty