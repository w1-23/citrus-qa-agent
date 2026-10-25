"""Language guard for queries sent to the local English-language corpus."""
from __future__ import annotations

import re
import logging


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323AF]")
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\]\[<>\"']+", re.I)
logger = logging.getLogger(__name__)


class LocalQueryLanguageError(ValueError):
    """Raised before local BM25/vector retrieval when an English query is absent."""


def contains_cjk(text: object) -> bool:
    return bool(_CJK_RE.search(str(text or "")))


def normalize_local_query(query: object) -> tuple[str | None, dict[str, object]]:
    """Return an English-only retrieval query, translating Chinese fail-closed."""
    original = str(query or "").strip()
    info: dict[str, object] = {
        "original_query": original,
        "translation_used": False,
        "local_query_language": "english",
    }
    if not original:
        info.update(status="empty_query", error="query must not be empty")
        return None, info
    if not contains_cjk(original):
        info.update(status="ready", query=original)
        return original, info

    try:
        # Import lazily to keep query validation independent of model startup.
        from src.retrieval.query_translation import cached_translation
        translated = str(cached_translation(original) or "").strip()
    except Exception as exc:  # noqa: BLE001
        translated = ""
        info["error"] = f"translation failed: {type(exc).__name__}: {exc}"

    if not translated:
        logger.warning("Local query translation failed closed (input_chars=%d)", len(original))
        info.update(status="translation_error",
                    error=info.get("error") or "translation returned no query")
        return None, info
    if contains_cjk(translated):
        logger.warning("Local query translation retained CJK; retrieval blocked (input_chars=%d)",
                       len(original))
        info.update(status="translation_error",
                    error="translation still contains CJK characters")
        return None, info

    # Preserve stable identifiers that translation models may alter or omit.
    doi_match = _DOI_RE.search(original)
    if doi_match and doi_match.group(0).lower() not in translated.lower():
        translated = f"{translated} {doi_match.group(0)}"
    translated = re.sub(r"\s+", " ", translated).strip()
    if not translated or contains_cjk(translated):
        logger.warning("Normalized local query failed English-only validation")
        info.update(status="translation_error",
                    error="normalized query is empty or contains CJK characters")
        return None, info

    info.update(status="ready", query=translated, translation_used=True)
    return translated, info


def require_english_local_query(query: object, *, field: str = "query") -> str:
    """Translate direct public retrieval calls or reject them before indexing."""
    normalized, info = normalize_local_query(query)
    if normalized is None:
        raise LocalQueryLanguageError(str(info.get("error") or "English local query required"))
    return normalized
