"""Shared pipeline utilities — pure logic, zero I/O, zero LLM calls."""
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Reference:
    id: int
    source_type: str  # "rag" | "web"
    title: str
    snippet: str
    doi: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None


def build_retrieval_context(main_results: list[dict], web_results: list[dict], max_main: int = 10, max_web: int = 10, mode: str = "SIMPLE") -> tuple[str, list[Reference]]:
    if mode == "COMPLEX":
        max_main = 50
        max_web = 20
    """Build shared context string and Reference list.

    Returns:
        (context_string_for_prompt, list_of_Reference)
    """
    refs: list[Reference] = []
    parts: list[str] = []
    idx = 1

    if main_results:
        lines = [f"## 学术文献（共 {len(main_results)} 篇）"]
        for i, p in enumerate(main_results[:max_main]):
            ref = Reference(
                id=idx,
                source_type="rag",
                title=p.get("title") or p.get("chunk_title", ""),
                snippet=(p.get("abstract") or p.get("snippet") or p.get("content") or p.get("text", ""))[:800],
                doi=p.get("doi"),
                authors=p.get("authors", "")[:80],
                year=p.get("year"),
            )
            refs.append(ref)
            lines.append(f"[{idx}] {ref.title}\n{ref.snippet}")
            idx += 1
        parts.append("\n\n".join(lines))

    if web_results:
        lines = [f"## 网络资讯（共 {len(web_results)} 条）"]
        for i, p in enumerate(web_results[:max_web]):
            ref = Reference(
                id=idx,
                source_type="web",
                title=p.get("title", ""),
                snippet=(p.get("abstract") or p.get("content") or p.get("snippet", ""))[:800],
                url=p.get("url"),
                doi=p.get("doi"),
            )
            refs.append(ref)
            lines.append(f"[{idx}] {ref.title}\n{ref.snippet}")
            idx += 1
        parts.append("\n\n".join(lines))

    context_str = "\n\n".join(parts) if parts else ""
    return context_str, refs
