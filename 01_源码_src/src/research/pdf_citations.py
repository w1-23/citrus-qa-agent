"""Request-scoped PDF citation extraction, highlighting and OA lookup."""
from __future__ import annotations

import base64
import re
from typing import Any


def _reference_start(pages: list[str]) -> tuple[int, int] | None:
    pattern = re.compile(r"^\s*(references|bibliography|literature cited|参考文献)\s*$", re.I)
    for page_no, text in enumerate(pages):
        for line_no, line in enumerate(text.splitlines()):
            if pattern.match(line):
                return page_no, line_no
    return None


def review_pdf(name: str, data_url: str, session_id: str) -> dict[str, Any]:
    raw = str(data_url or "")
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        content = base64.b64decode(raw, validate=True)
    except Exception as exc:
        raise ValueError("PDF 内容无法解码") from exc
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("PDF 超过 20 MB 限制")
    if not content.startswith(b"%PDF"):
        raise ValueError("文件不是有效 PDF")

    import fitz
    doc = fitz.open(stream=content, filetype="pdf")
    pages = [doc.load_page(i).get_text("text") for i in range(min(len(doc), 300))]
    if not any(text.strip() for text in pages):
        doc.close()
        return {"status": "incomplete", "reason": "未提取到文本，可能是扫描版 PDF。",
                "references": [], "highlighted_pdf_url": "", "pages": len(pages)}
    start = _reference_start(pages)
    refs: list[dict[str, Any]] = []
    if start:
        page_no, line_no = start
        lines = "\n".join(pages[page_no].splitlines()[line_no + 1:])
        lines += "\n" + "\n".join("\n".join(pages[p].splitlines()) for p in range(page_no + 1, len(pages)))
        entries = re.split(r"(?m)(?=^\s*(?:\[\d{1,3}\]|\d{1,3}[.)、]\s+|[A-Z][^\n]{1,90}\(?(?:19|20)\d{2}[a-z]?\)?))", lines)
        for entry in entries:
            text = re.sub(r"\s+", " ", entry).strip()
            if len(text) < 20:
                continue
            idx = re.match(r"^\[?(\d{1,3})[\].、)]?\s*", text)
            doi_match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.I)
            title = text[idx.end():] if idx else text
            title = re.split(r"\b(?:doi\s*:|https?://doi\.org/|10\.\d{4,9}/)\b", title, maxsplit=1, flags=re.I)[0].strip(" .;,")
            refs.append({"number": int(idx.group(1)) if idx else len(refs) + 1,
                         "text": text[:1200], "title": title[:500],
                         "doi": doi_match.group(0).rstrip(".,;)" ) if doi_match else ""})
            if len(refs) >= 300:
                break

    # Add visible PDF highlight annotations to numbered in-text citation markers.
    citation_pattern = re.compile(
        r"\[(?:\d{1,3})(?:\s*[,–-]\s*\d{1,3})*\]"
        r"|\([A-Z][A-Za-z'’.-]+(?:\s+et\s+al\.)?[,]?\s*(?:19|20)\d{2}[a-z]?(?:\s*;[^)]*)?\)"
    )
    for page_no, text in enumerate(pages):
        if start and page_no > start[0]:
            continue
        page = doc.load_page(page_no)
        for match in citation_pattern.finditer(text):
            if start and page_no == start[0]:
                lines = text.splitlines(keepends=True)
                heading_offset = sum(len(line) for line in lines[:start[1]])
                if match.start() >= heading_offset:
                    continue
            for rect in page.search_for(match.group(0))[:10]:
                annot = page.add_highlight_annot(rect)
                annot.set_info(title="CitrusQA 引用标记", content="正文中检测到的编号引用，供人工核对。")
                annot.update()
    from src.research.tools import _save_artifact
    output = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    artifact_url = _save_artifact(_safe_pdf_name(name), output, "application/pdf")
    return {"status": "ok" if start else "partial",
            "reason": "" if start else "没有识别到独立的参考文献标题；仅返回提取到的引用项。",
            "pages": len(pages), "references": refs,
            "highlighted_pdf_url": artifact_url}


def _safe_pdf_name(name: str) -> str:
    base = re.sub(r"[^\w.-]+", "_", str(name or "paper.pdf"), flags=re.UNICODE).strip("._") or "paper.pdf"
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base[:100]


def verify_reference(doi: str = "", title: str = "", citation_text: str = "") -> dict[str, Any]:
    """Compare a citation against Crossref registration metadata; never infer nonexistence."""
    import requests
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", str(doi or "").strip(), flags=re.I).rstrip(".,;)")
    citation_text = str(citation_text or title or "").strip()
    if not doi and not citation_text:
        return {"status": "unverified", "message": "缺少 DOI 和标题，无法核验。"}
    try:
        if doi:
            response = requests.get("https://api.crossref.org/works/" + requests.utils.quote(doi, safe="/"),
                                    timeout=12, headers={"User-Agent": "CitrusQA/1.0 (citation metadata check)"})
        else:
            response = requests.get("https://api.crossref.org/works", params={"query.bibliographic": citation_text[:400], "rows": 1},
                                    timeout=12, headers={"User-Agent": "CitrusQA/1.0 (citation metadata check)"})
        if response.status_code == 404:
            return {"status": "not_found_in_registry", "message": "登记库未查到；这不能证明文献不存在。"}
        response.raise_for_status()
        payload = response.json().get("message", {})
        if "items" in payload:
            payload = (payload.get("items") or [{}])[0]
        candidate_title = " ".join((payload.get("title") or [])[:1]).strip()
        candidate_doi = str(payload.get("DOI") or "")
        year = ""
        for field in ("published-print", "published-online", "created"):
            parts = (payload.get(field) or {}).get("date-parts") or []
            if parts and parts[0]:
                year = str(parts[0][0]); break
        if not candidate_title:
            return {"status": "not_found_in_registry", "message": "登记库未查到；这不能证明文献不存在。"}
        from difflib import SequenceMatcher
        normalized_citation = re.sub(r"\W", "", citation_text.lower())
        normalized_title = re.sub(r"\W", "", candidate_title.lower())
        similarity = SequenceMatcher(None, normalized_citation, normalized_title).ratio() if citation_text else 1.0
        title_covered = bool(normalized_title and normalized_title in normalized_citation)
        expected_year_match = re.search(r"\b(?:19|20)\d{2}\b", citation_text)
        expected_year = expected_year_match.group(0) if expected_year_match else ""
        title_match = not citation_text or title_covered or similarity >= 0.45
        doi_match = not doi or candidate_doi.lower() == doi.lower()
        year_match = not expected_year or not year or expected_year == year
        authors = payload.get("author") or []
        author_names = [" ".join(str(v) for v in (a.get("given"), a.get("family")) if v).strip()
                        for a in authors[:8] if isinstance(a, dict)]
        first_family = str((authors[0] or {}).get("family") or "") if authors and isinstance(authors[0], dict) else ""
        author_match = not citation_text or not first_family or re.search(r"\b" + re.escape(first_family) + r"\b", citation_text, re.I) is not None
        containers = payload.get("container-title") or []
        journal = str(containers[0]) if containers else ""
        journal_match = not citation_text or not journal or re.search(re.escape(journal), citation_text, re.I) is not None
        matched = doi_match and year_match and title_match and author_match
        return {"status": "metadata_match" if matched else "metadata_mismatch",
                "registered_title": candidate_title, "registered_doi": candidate_doi,
                "authors": author_names, "journal": journal, "year": year,
                "checks": {"doi": ("not_checked" if not doi else "match" if doi_match else "mismatch"),
                           "title": ("not_checked" if not citation_text else "match" if title_match else "mismatch"),
                           "year": ("not_checked" if not expected_year else "match" if year_match else "mismatch"),
                           "first_author": ("not_checked" if not citation_text else "match" if author_match else "mismatch"),
                           "journal_text_present": ("not_checked" if not citation_text else
                                                    "match" if journal_match else "not_found_in_citation")},
                "title_similarity": round(similarity, 3),
                "message": "Crossref 元数据匹配。" if matched else "DOI、标题、年份或首位作者与登记元数据不一致，请人工核对。"}
    except Exception as exc:
        return {"status": "unverified", "message": f"联网失败，未核验（{type(exc).__name__}）。"}


def fetch_open_access_pdf(doi: str, session_id: str) -> dict[str, Any]:
    """Find and download only a PDF exposed by Europe PMC's OA service."""
    import requests
    from xml.etree import ElementTree as ET
    from urllib.parse import urljoin
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", str(doi or "").strip(), flags=re.I)
    if not re.fullmatch(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", doi, re.I):
        return {"status": "unavailable", "message": "没有有效 DOI，无法按 DOI 查找开放全文。"}
    try:
        result = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                              params={"query": f'DOI:"{doi}"', "format": "json", "resultType": "core", "pageSize": 1}, timeout=12)
        result.raise_for_status()
        rows = (result.json().get("resultList") or {}).get("result") or []
        pmcid = next((str(row.get("pmcid") or "") for row in rows if row.get("pmcid")), "")
        if not pmcid:
            return {"status": "not_open_access", "message": "未找到 Europe PMC 开放全文；可能是付费文献或尚未收录。"}
        metadata = requests.get(f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}", timeout=12)
        metadata.raise_for_status()
        root = ET.fromstring(metadata.content)
        href = next((node.attrib.get("href", "") for node in root.iter() if node.tag.lower().endswith("link") and node.attrib.get("format") == "pdf"), "")
        if not href:
            return {"status": "not_open_access", "message": "该记录没有可直接下载的开放 PDF。"}
        pdf_url = urljoin("https://ftp.ncbi.nlm.nih.gov/", href.replace("ftp://ftp.ncbi.nlm.nih.gov/", "https://ftp.ncbi.nlm.nih.gov/"))
        parsed = __import__("urllib.parse", fromlist=["urlsplit"]).urlsplit(pdf_url)
        if parsed.scheme != "https" or parsed.hostname not in {"ftp.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov"}:
            return {"status": "unavailable", "message": "开放全文来源不在允许的公共仓储域名内。"}
        allowed_hosts = {"ftp.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov",
                         "ncbi.nlm.nih.gov", "cdn.ncbi.nlm.nih.gov", "www.ebi.ac.uk"}
        response = None
        for _ in range(4):
            parsed = __import__("urllib.parse", fromlist=["urlsplit"]).urlsplit(pdf_url)
            if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
                return {"status": "unavailable", "message": "开放全文重定向到未允许的域名，已停止下载。"}
            response = requests.get(pdf_url, timeout=30, stream=True, allow_redirects=False)
            if response.is_redirect or response.is_permanent_redirect:
                target = response.headers.get("Location", "")
                if not target:
                    break
                pdf_url = urljoin(pdf_url, target)
                continue
            response.raise_for_status()
            break
        if response is None or response.is_redirect or response.is_permanent_redirect:
            return {"status": "unavailable", "message": "开放全文重定向次数过多，未下载。"}
        chunks = []; size = 0
        for chunk in response.iter_content(1024 * 1024):
            if not chunk: continue
            size += len(chunk)
            if size > 30 * 1024 * 1024:
                return {"status": "too_large", "message": "开放全文超过 30 MB，未下载。"}
            chunks.append(chunk)
        content = b"".join(chunks)
        if not content.startswith(b"%PDF"):
            return {"status": "unavailable", "message": "开放仓储返回的文件不是有效 PDF。"}
        from src.research.tools import _save_artifact
        filename = re.sub(r"[^A-Za-z0-9._-]+", "_", pmcid) + ".pdf"
        url = _save_artifact(filename, content, "application/pdf")
        return {"status": "downloaded", "message": "已从 Europe PMC 开放仓储获取 PDF。", "download_url": url}
    except Exception as exc:
        return {"status": "unverified", "message": f"开放全文查询失败（{type(exc).__name__}）。"}
