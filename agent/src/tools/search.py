"""Search tools — web, academic, RAG, PDF, paper trace"""
from abc import ABC, abstractmethod
import json
import logging
import os
import re
import threading
import time
import urllib.error

from pathlib import Path

import urllib.parse

import urllib.request

import xml.etree.ElementTree as ET

from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Optional

import requests

from langchain_core.tools import tool

from src.config import settings, PROJECT_ROOT

logger = logging.getLogger(__name__)

# ── Web result quality filters (spam + domain whitelist) ──

SPAM_PATTERNS = [
    "checking your browser", "cloudflare", "ray id:",
    "enable javascript", "browser verification",
    "强烈推荐", "最新网址", "谷歌浏览器", "赌博",
    "casino", "click here", "subscribe now",
]

CONTENT_NOISE_PATTERNS = [
    "pest management", "害虫管理",
    "front page", "table of contents", "目录",
    "fruit crops and their management",
    "articles?searchType=journalSearch", "articles?sort=",
    "/volumes/", "/issues/", "volume ", "issue ",
]

RETRACTED_PATTERNS = [
    "this article has been retracted", "retraction notice",
    "retracted article", "retraction of",
]

CITRUS_FALSE_POSITIVE_PATTERNS = [
    "citrus county", "citrus clerk", "citrus public records",
    "citrus court records", "citrus property records",
    "citrus tax records", "citrus voter", "citrus heights",
    "citrus park", "citrus springs",
    "citrus based cleaner", "citrus degreaser",
    "citrus industrial cleaner", "citrus air freshener",
    "citrus scented", "citrus candle", "citrus soap",
    "citrus detergent", "citrus wipes", "citrus disinfectant",
    "citrus air care", "3m citrus", "cleaning citrus",
]

AUTHORITATIVE_DOMAINS = [
    "nature.com", "science.org", "springer.com", "wiley.com",
    "elsevier.com", "sciencedirect.com", "mdpi.com",
    "frontiersin.org", "plos.org", "oup.com", "academic.oup.com",
    "cambridge.org", "tandfonline.com", "biorxiv.org",
    "arxiv.org", "cell.com", "plantphysiol.org",
    "plantcell.org", "genetics.org", "theplantjournal.com",
    "newphytologist.org", "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "researchgate.net", "zenodo.org", "osf.io",
    "who.int", "fao.org", "cabi.org",
]

def _is_valid_web_result(r: dict) -> tuple[bool, str]:
    content = (r.get("content") or "").strip().lower()
    title = (r.get("title") or "").strip().lower()
    url = (r.get("url") or "").lower()

    # Spam check first (even short spam should be caught)
    for pat in SPAM_PATTERNS:
        if pat in content or pat in title:
            return False, f"spam: {pat}"

    # Content noise (off-topic results like nematode books)
    for pat in CONTENT_NOISE_PATTERNS:
        if pat in title:
            return False, f"noise: {pat}"

    # Retracted papers
    combined = (title + " " + content)[:500]
    for pat in RETRACTED_PATTERNS:
        if pat in combined:
            return False, f"retracted: {pat}"

    # Citrus false positives (e.g. "Citrus County" geographic matches)
    for pat in CITRUS_FALSE_POSITIVE_PATTERNS:
        if pat in combined:
            return False, f"false_positive: {pat}"

    if not content or len(content) < 50:
        return False, "too_short"

    for d in AUTHORITATIVE_DOMAINS:
        if d in url:
            return True, "primary"

    if url.endswith(".edu") or ".edu/" in url or url.endswith(".gov") or ".gov/" in url:
        return True, "primary"

    return True, "supplementary"

def _filter_web_results(results: list[dict]) -> tuple[list[dict], list[dict]]:
    primary, supplementary = [], []
    for r in results:
        valid, reason = _is_valid_web_result(r)
        if not valid:
            continue
        if reason == "primary":
            primary.append(r)
        else:
            supplementary.append(r)
    return primary, supplementary

def format_rag_context(results: list, source: str = "main") -> str:

    """Format RAG search results into readable context text."""

    if not results:

        return "未检索到相关文献。"

    return (f"检索到 {len(results)} 条{'文献' if source == 'main' else '互联网资讯'}。\n"

            + "\n".join(

                f"- {r.get('title', '?')} (DOI: {r.get('doi', '?')}, 信心度: {r.get('score', 0):.2f})"

                for r in results[:10]

            ))

# ── Shared helpers ────────────────────────────

def _format_tool_result(
    tool_name: str,
    query: str,
    content: str,
    status: str = "ok",
    results_count: int = 0,
    elapsed_ms: float = 0,
) -> str:
    """Standard header for all tool results. LLM sees consistent format."""
    header = (
        f"## [ToolResult] {tool_name}\n"
        f"**status**: {status}\n"
        f"**query**: \"{query[:120]}\"\n"
        f"**results_count**: {results_count}"
    )
    if elapsed_ms > 0:
        header += f"\n**elapsed_ms**: {elapsed_ms:.0f}"
    return f"{header}\n\n{content}"

_WORKSPACE_ROOT = PROJECT_ROOT / settings.WORKSPACE_DIR

_CITRUS_KEYWORDS = [

    "citrus", "orange", "lemon", "grapefruit", "mandarin", "pomelo",

    "黄龙病", "溃疡病", "柑橘", "sweet orange", "Citrus sinensis",

]

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"

CROSSREF_API = "https://api.crossref.org/works"

PUBMED_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

OPALEX_API = "https://api.openalex.org/works"

def _is_citrus_related(text: str) -> bool:

    return any(kw.lower() in text.lower() for kw in _CITRUS_KEYWORDS)

def _extract_doi(text: str) -> str | None:

    match = re.search(r'''(10\.\d{4,}/[^\s"'\]\)]+)''', text)

    return match.group(0) if match else None

# 1. citrus_rag_search

_RAG_INSTANCE = None

_RAG_LOCK = threading.Lock()

def _get_rag():

    global _RAG_INSTANCE

    if _RAG_INSTANCE is None:

        with _RAG_LOCK:

            if _RAG_INSTANCE is None:

                from src.retrieval.multi_retriever import MultiBatchRetriever

                _RAG_INSTANCE = MultiBatchRetriever()  # singleton __init__ auto-loads Qdrant+BM25+Models

                logger.info("[citrus_rag_search] MultiBatchRetriever 单例已预热")

    return _RAG_INSTANCE

@tool(response_format="content_and_artifact")

def citrus_rag_search(query: str) -> tuple[str, dict]:

    """在柑橘科研文献库中检索与问题相关的论文片段。支持中文输入，系统自动生成英文假想答案提升召回。返回带 DOI 和置信度的结果"""

    if not query or not query.strip():

        return "[ERR_PARSE] 查询词不能为空", {"main_results": []}

    query = query.strip()

    if len(query) > 500:

        return f"[ERR_PARSE] 查询词过长 ({len(query)}字符)，请精简至 500 字符以内", {"main_results": []}

    if any(ch in query for ch in "<>\"'\\;"):

        return f"[ERR_PARSE] 查询词包含非法字符（<>\"'\\;），请移除后重试", {"main_results": []}

    hyde_answer = None
    if getattr(settings, 'RAG_HYDE_ENABLED', True):
        try:
            from src.core.progress_bus import emit_progress
            emit_progress("tool_progress", {"message": "正在构建语义假设 (HyDE)...", "tool_call_id": ""})
        except Exception: pass
        hyde_answer = _cached_hyde(query)
        if hyde_answer:
            logger.info(f"[HyDE] generated {len(hyde_answer)} chars for: {query[:60]}...")
            try:
                from src.core.progress_bus import emit_progress
                emit_progress("tool_progress", {"message": f"HyDE 生成完毕 ({len(hyde_answer)}字), 开始向量检索...", "tool_call_id": ""})
            except Exception: pass
        else:
            logger.debug(f"[HyDE] fallback for: {query[:60]}...")
            try:
                from src.core.progress_bus import emit_progress
                emit_progress("tool_progress", {"message": "HyDE 超时/失败, 降级为基础检索...", "tool_call_id": ""})
            except Exception: pass

    try:

        rag = _get_rag()
        t0 = time.perf_counter()

        if hyde_answer:
            results = rag.search_hyde(original_query=query, hyde_answer=hyde_answer)
        else:
            results = rag.search(query)

        elapsed = (time.perf_counter() - t0) * 1000

        if not results:
            content = _format_tool_result("citrus_rag_search", query, "未检索到相关文献。", status="empty", elapsed_ms=elapsed)
        else:
            raw = format_rag_context(results, "main")
            content = _format_tool_result(
                "citrus_rag_search", query, raw,
                status="ok", results_count=len(results), elapsed_ms=elapsed,
            )
        artifact = {"main_results": results} if results else {"main_results": []}

        return content, artifact

    except Exception as e:

        logger.error(f"[citrus_rag_search] 检索失败: {e}")

        return f"文献检索服务暂时不可用。错误: {e}", {"main_results": []}

# ── HyDE LRU Cache ──

import hashlib
_hyde_cache: dict[str, str] = {}
_HYDE_CACHE_MAX = 500

def _cached_hyde(query: str) -> str | None:
    key = hashlib.md5(query.strip().encode("utf-8")).hexdigest()
    if key in _hyde_cache:
        return _hyde_cache[key]
    answer = _generate_hyde_answer(query)
    if answer and len(_hyde_cache) >= _HYDE_CACHE_MAX:
        _hyde_cache.pop(next(iter(_hyde_cache)))
    if answer:
        _hyde_cache[key] = answer
    return answer

_HYDE_PROMPT = (
    "You are a scientific literature retrieval assistant for citrus research.\n"
    "Given a user question, generate a short hypothetical answer paragraph "
    "that might appear in a citrus research paper.\n\n"
    "Rules:\n"
    "1. Write in English.\n"
    "2. Keep it under 180 words.\n"
    "3. Do not invent specific numbers, p-values, gene IDs, accession numbers, or citations.\n"
    "4. If uncertain, use generic academic phrasing.\n"
    "5. Include likely biological mechanisms, gene families, pathways, and domain terms.\n"
    "6. Output only the paragraph, no preamble.\n"
)

def _generate_hyde_answer(query: str) -> str | None:
    """Generate a HyDE (Hypothetical Document Embedding) answer via fast LLM."""
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=settings.RESOLVED_FAST_API_KEY,
            base_url=settings.RESOLVED_FAST_BASE_URL,
            timeout=3,
        )
        resp = client.chat.completions.create(
            model=settings.FAST_MODEL,
            messages=[
                {"role": "system", "content": _HYDE_PROMPT},
                {"role": "user", "content": f"User question:\n{query}\n\nHypothetical answer paragraph:"},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        answer = resp.choices[0].message.content.strip()
        return answer if len(answer) >= 30 else None
    except Exception as e:
        logger.debug(f"[HyDE] generation failed: {e}")
        return None

# 2. Web search provider


# ── PDF read: Abstract → Results/Conclusion, regex dedup, no references/images ──

_CITATION_BOILERPLATE_PATTERNS = [
    r"Citation:\s*[^\n]+",
    r"Academic Editor:\s*[^\n]*",
    r"Received:\s*[^\n]*",
    r"Revised:\s*[^\n]*",
    r"Accepted:\s*[^\n]*",
    r"Published:\s*[^\n]*",
    r"©\s*\d{4}\s+by the authors\..*",
    r"Licensee MDPI.*",
    r"This article is an open access article.*",
    r"Correspondence:\s*[^\n]*",
    r"\d+\.\s+Department of\s[^\n]+",
    r"\*+Correspondence:\s*[^\n]+",
]

_RESULTS_SECTION_MARKERS = [
    r"\b(Results|Result|实验结果|结果与分析|结果)\b",
    r"\b(Discussion|讨论)\b",
    r"\b(Conclusion|Conclusions|结论|总结)\b",
]

_REFERENCE_START_MARKER = re.compile(
    r"^\s*(References|参考文献|Bibliography|文献)\s*$", re.IGNORECASE
)

_DEDUP_MIN_LINE = 30

def _clean_abstract_text(raw_abstract: str) -> str:
    cleaned = raw_abstract
    for pat in _CITATION_BOILERPLATE_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(Abstract|摘要)\s*[:\-]?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_clean_pdf_text(doc) -> tuple[str, list[str]]:
    all_text = []
    body_paras = []
    for page in doc:
        blocks = page.get_text("blocks")
        page_h = page.rect.height
        page_w = page.rect.width
        for b in blocks:
            if b[6] != 0:  # skip images
                continue
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4].strip()
            if not text:
                continue
            if len(text) < 10 and y1 > page_h * 0.85 and abs(x0 + x1 - page_w) < page_w * 0.3:
                continue
            if len(text) < 40 and y0 < page_h * 0.08:
                continue
            if len(text) < 20 and y1 > page_h * 0.92:
                continue
            if len(text) < 60 and y0 < page_h * 0.06 and any(
                kw in text.lower() for kw in ["molecules", "frontiers", "journal",
                                              "article", "www.", "doi:", "copyright"]
            ):
                continue
            all_text.append(text)
            if len(text) > 100:
                body_paras.append(text)
    return "\n".join(all_text), body_paras


def _find_abstract(text: str, paragraphs: list[str]) -> str:
    abstract_raw = ""
    for i, p in enumerate(paragraphs):
        if re.search(r"\b(abstract|摘要)\b", p[:60], re.IGNORECASE):
            abstract_raw = p
            for j in range(i + 1, min(i + 5, len(paragraphs))):
                nxt = paragraphs[j]
                if re.search(r"\b(introduction|keywords|引言|关键词|1\.\s*[A-Z])\b", nxt[:60], re.IGNORECASE):
                    break
                abstract_raw += " " + nxt
            break
    if abstract_raw:
        abstract_raw = _clean_abstract_text(abstract_raw)
    return abstract_raw[:3000]


def _find_results_conclusion(full_text: str, paragraphs: list[str]) -> str:
    """从全文中定位 Results/Discussion/Conclusion 段落，截至 References 前。"""
    pattern = re.compile("|".join(_RESULTS_SECTION_MARKERS), re.IGNORECASE)

    # 逐全文行扫描找 Results 起始位置
    lines = full_text.split("\n")
    result_start = -1
    ref_start = len(lines)
    for i, line in enumerate(lines):
        s = line.strip()
        if result_start < 0 and pattern.search(s[:120]):
            result_start = i
        if _REFERENCE_START_MARKER.match(s):
            ref_start = i
            break

    if result_start < 0:
        return ""

    # 收集 result_start ~ ref_start 之间的长段落
    section_text = "\n".join(lines[result_start:ref_start])
    # 从 paragraphs 中筛选落在该区域的长段落
    collected = []
    for p in paragraphs:
        if p[:80] in section_text or section_text.find(p[:80]) >= 0:
            collected.append(p)

    if not collected:
        # fallback: 直接用该区域全文截断
        return section_text[:8000]

    return "\n\n".join(collected)


def _deduplicate_text(text: str) -> str:
    """Remove near-duplicate lines from extracted text."""
    lines = text.split("\n")
    seen = set()
    unique = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) < _DEDUP_MIN_LINE:
            unique.append(line)
            continue
        norm = re.sub(r"\s+", " ", stripped).lower()[:120]
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(line)
    return "\n".join(unique)


@tool(response_format="content_and_artifact")
def pdf_read(file_path: str, cross_reference: bool = True) -> tuple[str, dict]:
    """读取本地 PDF 论文文件，提取 摘要 → Results/Conclusion，去重去参文献。

    Args:
        file_path: PDF 文件路径（绝对路径或相对 workspace/ 的路径）
        cross_reference: 是否与文献库比对关联信息
    """
    import fitz

    resolved_path = file_path
    if not os.path.isabs(file_path):
        candidate = PROJECT_ROOT / "workspace" / file_path
        if candidate.exists():
            resolved_path = str(candidate)
    else:
        workspace_root = (PROJECT_ROOT / "workspace").resolve()
        abs_path = str(Path(file_path).resolve())
        if not abs_path.startswith(str(workspace_root)):
            return f"Access denied: 路径不在 workspace/ 内: {file_path}", {"pdf_data": None}
        resolved_path = abs_path

    if not os.path.exists(resolved_path):
        return f"文件不存在: {resolved_path}", {"pdf_data": None}

    try:
        doc = fitz.open(resolved_path)
        meta = doc.metadata or {}
        total_pages = len(doc)

        full_text, body_paras = _extract_clean_pdf_text(doc)
        doc.close()

        lines = [l.strip() for l in full_text.split("\n") if l.strip()]
        title = meta.get("title", "") or (lines[0] if lines else "未知标题")
        doi = _extract_doi(full_text)
        abstract = _find_abstract(full_text, body_paras)

        # 从 body_paras 提取 Results → Conclusion，去重
        results_text = _find_results_conclusion(full_text, body_paras)
        if results_text:
            results_text = _deduplicate_text(results_text)
            results_text = re.sub(r"(?:参考文献|References)[\s\S]*$", "", results_text, flags=re.IGNORECASE)

        is_citrus = _is_citrus_related(full_text)

        parts = [f"## {title}"]
        if doi:
            parts.append(f"DOI: {doi}")
        parts.append("")
        parts.append(f"### 摘要\n{abstract[:2000] if abstract else '(未提取到摘要)'}")

        if results_text:
            parts.append("")
            parts.append(f"### Results & Conclusion\n{results_text[:5000]}")

        text_result = "\n".join(parts)

        artifact = {"pdf_data": {"title": title, "doi": doi,
                                 "abstract": abstract[:2000] if abstract else "",
                                 "total_pages": total_pages,
                                  "is_citrus_related": is_citrus,
                                  "text_body": full_text[:8000]}}

        content = _format_tool_result(
            "pdf_read", file_path[:80], text_result,
            status="ok", results_count=total_pages,
        )
        return content, artifact

    except Exception as e:
        logger.error(f"[PDFRead] 读取失败: {e}")
        return _format_tool_result(
            "pdf_read", file_path[:80],
            f"PDF 读取失败: {e}",
            status="error",
        ), {"pdf_data": None}

# 8. Multi-source search

def _reconstruct_abstract(inverted_index: dict | None) -> str:

    if not inverted_index:

        return ""

    words = {}

    for word, positions in inverted_index.items():

        for pos in positions:

            words[pos] = word

    return " ".join(words[i] for i in sorted(words.keys()))[:500]

def _search_semantic_scholar_multi(query: str, limit: int = 5, timeout: int = 12) -> list[dict]:

    try:

        resp = requests.get(S2_API, params={"query": query, "limit": limit, "fields": "title,authors,year,venue,externalIds,abstract,citationCount"},

                            timeout=timeout, headers={"User-Agent": "CitrusQA/1.0"})

        if resp.status_code != 200:

            return []

        data = resp.json()

        results = []

        for paper in data.get("data", []):

            authors = [a.get("name", "") for a in (paper.get("authors") or [])[:4]]

            results.append({"source": "Semantic Scholar", "title": paper.get("title", ""), "authors": ", ".join(authors),

                            "year": paper.get("year"), "venue": paper.get("venue", ""),

                            "doi": (paper.get("externalIds") or {}).get("DOI", ""),

                            "citations": paper.get("citationCount", 0), "abstract": (paper.get("abstract") or "")[:500],

                            "url": f"https://doi.org/{(paper.get('externalIds') or {}).get('DOI', '')}" if (paper.get('externalIds') or {}).get('DOI') else ""})

        return results

    except Exception as e:

        logger.warning(f"[MultiSearch] Semantic Scholar 失败: {e}")

        return []

def _search_crossref_multi(query: str, limit: int = 5, timeout: int = 12) -> list[dict]:

    try:

        resp = requests.get(CROSSREF_API, params={"query.title": query, "rows": limit, "sort": "relevance", "mailto": settings.NCBI_ENTREZ_EMAIL},

                            timeout=timeout, headers={"User-Agent": "CitrusQA/1.0"})

        if resp.status_code != 200:

            return []

        data = resp.json()

        items = (data.get("message") or {}).get("items", [])

        results = []

        for item in items:

            authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in (item.get("author") or [])[:4]]

            doi = item.get("DOI", "")

            results.append({"source": "CrossRef", "title": (item.get("title") or [""])[0] if item.get("title") else "",

                            "authors": ", ".join(authors),

                            "year": (item.get("published-print") or {}).get("date-parts", [[None]])[0][0] or (item.get("created") or {}).get("date-parts", [[None]])[0][0],

                            "venue": (item.get("container-title") or [""])[0], "doi": doi,

                            "citations": item.get("is-referenced-by-count", 0), "abstract": (item.get("abstract") or "")[:500],

                            "url": f"https://doi.org/{doi}" if doi else ""})

        return results

    except Exception as e:

        logger.warning(f"[MultiSearch] CrossRef 失败: {e}")

        return []

def _search_pubmed_multi(query: str, limit: int = 5, timeout: int = 12) -> list[dict]:

    try:

        search_resp = requests.get(PUBMED_SEARCH, params={"db": "pubmed", "term": f"({query}) AND (plant OR crop OR citrus OR agriculture)", "retmax": limit, "retmode": "json"}, timeout=timeout)

        if search_resp.status_code != 200:

            return []

        search_data = search_resp.json()

        id_list = (search_data.get("esearchresult") or {}).get("idlist", [])

        if not id_list:

            return []

        fetch_resp = requests.get(PUBMED_FETCH, params={"db": "pubmed", "id": ",".join(id_list), "retmode": "json"}, timeout=timeout)

        if fetch_resp.status_code != 200:

            return []

        fetch_data = fetch_resp.json()

        results = fetch_data.get("result", {})

        papers = []

        for pmid in id_list:

            info = results.get(pmid, {})

            papers.append({"source": "PubMed", "title": (info.get("title") or [""])[0] if isinstance(info.get("title"), list) else (info.get("title") or ""),

                           "authors": ", ".join(a.get("name", "") for a in (info.get("authors") or [])[:4]),

                           "year": info.get("pubdate", "")[:4] if info.get("pubdate") else None,

                           "venue": (info.get("source") or ""), "doi": (info.get("elocationid") or "").replace("doi: ", ""),

                           "citations": 0, "abstract": (info.get("attributes") or [""])[0] if info.get("attributes") else "",

                           "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "pmid": pmid})

        return papers

    except Exception as e:

        logger.warning(f"[MultiSearch] PubMed 失败: {e}")

        return []

def _search_openalex_multi(query: str, limit: int = 5, timeout: int = 12) -> list[dict]:

    try:

        resp = requests.get(OPALEX_API, params={"search": query, "per_page": limit, "sort": "cited_by_count:desc"},

                            timeout=timeout, headers={"User-Agent": "CitrusQA/1.0"})

        if resp.status_code != 200:

            return []

        data = resp.json()

        results = []

        for item in (data.get("results") or []):

            authors = [a.get("author", {}).get("display_name", "") for a in (item.get("authorships") or [])[:4]]

            doi = (item.get("ids") or {}).get("doi", "")

            if doi:

                doi = doi.replace("https://doi.org/", "")

            results.append({"source": "OpenAlex", "title": item.get("display_name", ""), "authors": ", ".join(authors),

                            "year": item.get("publication_year"), "venue": (item.get("primary_location") or {}).get("source", {}).get("display_name", ""),

                            "doi": doi, "citations": item.get("cited_by_count", 0),

                            "abstract": _reconstruct_abstract(item.get("abstract_inverted_index")),

                            "url": f"https://doi.org/{doi}" if doi else ""})

        return results

    except Exception as e:

        logger.warning(f"[MultiSearch] OpenAlex 失败: {e}")

        return []

_SOURCES = {

    "semantic_scholar": _search_semantic_scholar_multi,

    "crossref": _search_crossref_multi,

    "pubmed": _search_pubmed_multi,

    "openalex": _search_openalex_multi,

}

def _deduplicate(results: list[dict]) -> list[dict]:

    seen_dois = set()

    seen_titles = set()

    deduped = []

    for r in results:

        doi = (r.get("doi") or "").lower().strip()

        title = (r.get("title") or "").lower().strip()[:80]

        if doi and doi in seen_dois:

            continue

        if title and title in seen_titles:

            continue

        if doi:

            seen_dois.add(doi)

        if title:

            seen_titles.add(title)

        deduped.append(r)

    return deduped

@tool(response_format="content_and_artifact")

def academic_search(query: str, limit_per_source: int = 3, focus: str = "auto", _timeout: int = 5, limit_per_string: int = None) -> tuple[str, dict]:

    """多源学术论文检索（Semantic Scholar / CrossRef / PubMed / OpenAlex）。返回每篇论文的标题、作者、期刊、年份、DOI、引用数、摘要（500字）。阅读摘要判断论文是否匹配问题。学术API是关键词匹配，请用英文核心关键词（非长句子）。

    Args:

        query: 检索关键词（英文更佳）

        limit_per_source: 每个来源最多返回条数（默认3）

        focus: 检索聚焦领域，"auto"自动检测，"citrus"柑橘，"plant"植物科学

        limit_per_string: 兼容 LLM 可能误用的参数名（映射到 limit_per_source）

    """
    if limit_per_string is not None and limit_per_source == 3:
        limit_per_source = limit_per_string

    t0_ms = time.perf_counter()

    # Academic keyword extraction: stopword-based, not case-based
    _STOPWORDS = {"the", "a", "an", "in", "of", "for", "to", "and",
                  "what", "how", "why", "which", "does", "is", "are",
                  "was", "were", "been", "have", "has", "had",
                  "this", "that", "these", "those", "with", "from",
                  "about", "into", "through", "during", "before",
                  "after", "above", "below", "between", "under",
                  "also", "not", "only", "very", "just", "than",
                  "then", "now", "here", "there", "when", "where"}
    
    if len(query) > 80:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query)
        keywords = [w for w in words if w.lower() not in _STOPWORDS]
        if keywords:
            query = " ".join(keywords[:6])

    # Remove generic citrus/plant words to improve academic API specificity
    _GENERIC_DOMAIN = {"citrus", "orange", "lemon", "grapefruit", "mandarin",
                       "pomelo", "fruit", "fruits", "spp", "plant", "crop",
                       "cultivar", "species", "genus", "lime", "clementine",
                       "柑橘", "tangerine"}
    topic_words = [w for w in query.split()
                   if w.lower() not in _GENERIC_DOMAIN and len(w) > 2]
    if topic_words:
        # Keep citrus as domain limiter (append at end for lower weight)
        citrus_kw_in_q = any(kw in query.lower() for kw in
                             ["citrus", "orange", "lemon", "grapefruit",
                              "柑橘", "黄龙病", "溃疡病"])
        if citrus_kw_in_q:
            query = " ".join(topic_words[:5]) + " citrus"
        else:
            query = " ".join(topic_words[:6])

    domain_boost = {"citrus": "citrus orange lemon grapefruit mandarin", "plant": "plant crop agriculture horticulture"}

    if focus in domain_boost:

        boosted_query = f"({query}) AND ({domain_boost[focus]})"

    elif focus == "auto":

        citrus_kw = ["citrus", "orange", "lemon", "grapefruit", "黄龙病", "溃疡病"]

        boosted_query = f"({query}) AND (citrus)" if any(kw in query.lower() for kw in citrus_kw) else query

    else:

        boosted_query = query

    all_results = []
    source_times = {}

    enabled_sources = getattr(settings, 'ACADEMIC_SOURCES', None) or ["crossref"]
    active_sources = {name: fn for name, fn in _SOURCES.items() if name in enabled_sources}
    if not active_sources:
        return "学术搜索引擎未启用，请检查 config.yaml", {"main_results": []}

    with ThreadPoolExecutor(max_workers=len(active_sources)) as executor:
        src_timeout = _timeout if _timeout > 0 else getattr(settings, 'ACADEMIC_TIMEOUT', 8)
        futures = {executor.submit(fn, boosted_query, limit_per_source, src_timeout): name
                   for name, fn in active_sources.items()}

        for future in as_completed(futures):

            name = futures[future]
            t_src = time.perf_counter()

            try:

                results = future.result()

                all_results.extend(results)
                dt_src = (time.perf_counter() - t_src) * 1000
                source_times[name] = dt_src
                logger.info(f"[MultiSearch] {name}: {len(results)} 条 ({dt_src:.0f}ms)")

            except Exception as e:
                dt_src = (time.perf_counter() - t_src) * 1000
                source_times[name] = dt_src
                logger.warning(f"[MultiSearch] {name} 异常 ({dt_src:.0f}ms): {e}")

    deduped = _deduplicate(all_results)

    # Citrus relevance filter: only when query itself is citrus-specific
    citrus_kw_in_query = any(kw in query.lower() for kw in
        ["citrus", "orange", "lemon", "grapefruit", "mandarin",
         "柑橘", "黄龙病", "溃疡病"])
    if citrus_kw_in_query:
        citrus_filtered = []
        for r in deduped:
            title = r.get("title", "") or ""
            abstract = r.get("abstract", "") or ""
            text = title + " " + abstract
            if _is_citrus_related(text):
                citrus_filtered.append(r)
        dropped_count = len(deduped) - len(citrus_filtered)
        if dropped_count > 0:
            logger.info(f"[MultiSearch] citrus filter: kept {len(citrus_filtered)}, dropped {dropped_count}")
        deduped = citrus_filtered

    deduped.sort(key=lambda x: x.get("citations", 0), reverse=True)

    source_counts = {}

    for r in deduped:

        src = r.get("source", "unknown")

        source_counts[src] = source_counts.get(src, 0) + 1

    dt_total = (time.perf_counter() - t0_ms) * 1000
    time_str = ", ".join(f"{k}={v:.0f}ms" for k, v in sorted(source_times.items()))
    logger.info(f"[MultiSearch] total: {len(deduped)} results, {time_str} (total {dt_total:.0f}ms)")

    text_result = f"## 学术论文检索结果\n检索词: {query}\n共 {len(deduped)} 条结果 (去重后) | 来源: {', '.join(f'{k}={v}' for k, v in source_counts.items())}\n\n"

    for i, paper in enumerate(deduped[:15], 1):

        text_result += (

            f"[{i}] {paper.get('title', '')[:100]}\n"

            f"  作者: {paper.get('authors', 'N/A')[:80]}\n"

            f"  期刊: {paper.get('venue', 'N/A')} ({paper.get('year', 'N/A')})\n"

            f"  引用: {paper.get('citations', 0)} 次 | 来源: {paper.get('source', '')}\n"

        )

        if paper.get("doi"):

            text_result += f"  DOI: {paper['doi']}\n"

        if paper.get("abstract"):

            text_result += f"  摘要: {paper['abstract'][:300]}...\n"

        text_result += "\n"

    artifact = {"main_results": deduped[:20], "source_distribution": source_counts, "total_results": len(deduped), "query": query}

    content = _format_tool_result(
        "academic_search", query, text_result,
        status="ok" if deduped else "empty",
        results_count=len(deduped), elapsed_ms=dt_total,
    )
    return content, artifact

# 9. Encyclopedia search (Wikipedia API → Baidu Baike fallback)

@tool(response_format="content_and_artifact")
def encyclopedia_search(query: str) -> tuple[str, dict]:
    """搜索百科类权威知识（Wikipedia中文站，不可用时降级百度百科）。返回结构化正文和来源URL。适合术语解释、概念定义、品种背景、病害描述等确定性知识查询。"""

    if not query or not query.strip():
        return "查询词不能为空", {"web_results": []}

    query = query.strip()
    url = ""
    extract = ""
    source_label = "Wikipedia"

    # ── Level 1: zh.wikipedia.org ──
    try:
        search_resp = requests.get(
            "https://zh.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search",
                "srsearch": query, "srlimit": 1,
                "format": "json",
            },
            timeout=8,
            headers={"User-Agent": "CitrusQA/1.0"},
        )
        if search_resp.status_code == 200:
            search_data = search_resp.json()
            pages = (search_data.get("query") or {}).get("search", [])
            if pages:
                page_title = pages[0]["title"]
                page_id = pages[0]["pageid"]
                content_resp = requests.get(
                    "https://zh.wikipedia.org/w/api.php",
                    params={
                        "action": "query", "prop": "extracts",
                        "exintro": False, "exchars": 2000,
                        "explaintext": True, "pageids": page_id,
                        "format": "json",
                    },
                    timeout=8,
                    headers={"User-Agent": "CitrusQA/1.0"},
                )
                if content_resp.status_code == 200:
                    content_data = content_resp.json()
                    page_info = (content_data.get("query") or {}).get("pages", {}).get(str(page_id), {})
                    extract = page_info.get("extract", "")[:2000]
                    url = f"https://zh.wikipedia.org/wiki/{urllib.parse.quote(page_title)}"
                    if extract:
                        logger.info(f"[encyclopedia] zh.wikipedia: {page_title} ({len(extract)} chars)")
    except Exception:
        pass  # silent fallthrough to next level

    # ── Level 2: en.wikipedia.org ──
    if not extract:
        try:
            search_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "list": "search",
                    "srsearch": query, "srlimit": 1,
                    "format": "json",
                },
                timeout=8,
                headers={"User-Agent": "CitrusQA/1.0"},
            )
            if search_resp.status_code == 200:
                search_data = search_resp.json()
                pages = (search_data.get("query") or {}).get("search", [])
                if pages:
                    page_title = pages[0]["title"]
                    page_id = pages[0]["pageid"]
                    content_resp = requests.get(
                        "https://en.wikipedia.org/w/api.php",
                        params={
                            "action": "query", "prop": "extracts",
                            "exintro": False, "exchars": 2000,
                            "explaintext": True, "pageids": page_id,
                            "format": "json",
                        },
                        timeout=8,
                        headers={"User-Agent": "CitrusQA/1.0"},
                    )
                    if content_resp.status_code == 200:
                        content_data = content_resp.json()
                        page_info = (content_data.get("query") or {}).get("pages", {}).get(str(page_id), {})
                        extract = page_info.get("extract", "")[:2000]
                        url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_title)}"
                        if extract:
                            source_label = "Wikipedia-EN"
                            logger.info(f"[encyclopedia] en.wikipedia: {page_title} ({len(extract)} chars)")
        except Exception:
            pass

    # ── Level 3: DDGS web search for Baidu Baike / encyclopedia content ──
    if not extract:
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(f"{query} 百科", max_results=3))
                for r in results:
                    title = r.get("title", "")
                    snippet = r.get("body", "") or r.get("snippet", "")
                    url = r.get("link", "") or r.get("href", "")
                    baike_kw = any(kw in title for kw in ["百科", "百度", "Wiki", "wiki"])
                    if len(snippet) > 60 or (baike_kw and len(snippet) > 30):
                        extract = f"{title}: {snippet}"[:2000]
                        source_label = "Search"
                        logger.info(f"[encyclopedia] DDGS: {query} ({len(extract)} chars)")
                        break
        except Exception as e:
            logger.info(f"[encyclopedia] DDGS failed (non-critical): {e}")

    if not extract:
        return _format_tool_result(
            "encyclopedia_search", query,
            f"未在百科中找到'{query}'相关内容。",
            status="empty",
        ), {"web_results": []}

    raw = (
        f"## 百科[{source_label}]: {query}\n"
        f"URL: {url}\n\n"
        f"{extract}"
    )
    content = _format_tool_result(
        "encyclopedia_search", query, raw,
        status="ok", results_count=1,
    )
    artifact = {"web_results": [{
        "title": query,
        "url": url,
        "snippet": extract[:300],
        "content": extract,
        "source": source_label,
    }]}

    return content, artifact