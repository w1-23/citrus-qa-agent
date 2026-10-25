"""Strict, cached translation for the English-only local retrieval corpus."""
from __future__ import annotations

import hashlib
import logging
import re
import threading

from src.config import settings
from src.core.llm_pool import is_thinking_rejected

logger = logging.getLogger(__name__)
_CACHE: dict[str, str] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 500

_TRANSLATE_PROMPT = (
    "You are a scientific literature retrieval assistant for citrus research.\n"
    "Translate the user question into standard English scientific terms.\n"
    "Rules:\n"
    "1. Output ONLY the English translation, one line, no preamble, no markdown.\n"
    "2. Preserve all entities: cultivar names, gene IDs, disease names, chemical names.\n"
    "3. Use standard English scientific terminology (e.g., HLB for huanglongbing).\n"
    "4. Never leave Chinese characters in the translation. Translate Chinese common names to\n"
    "   established English names where known; otherwise use Latin names or pinyin.\n"
    "5. Do not invent numbers, p-values, gene IDs, or citations.\n"
)
_BAD_TRANSLATION_RE = re.compile(
    r"\bplease\s+search\b|\bplease\s+find\b|\bplease\s+look\b|"
    r"\btell\s+me\b|\bfind\s+me\b|\bshow\s+me\b|"
    r"\bsearch\s+for\b|\bi\s+(?:want|need)\s+(?:you\s+)?to\b|"
    r"\blet\s+me\b|\bcan\s+you\b|\bcould\s+you\b",
    re.IGNORECASE,
)


def _is_english(text: str) -> bool:
    return bool(text) and not re.search(
        r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323AF]",
        text,
    )


def parse_translation(raw: str) -> str | None:
    if not raw or not raw.strip():
        return None
    text = re.sub(r"^```[^\n]*\n?", "", raw.strip())
    text = re.sub(r"\n```", "", text)
    for line in text.splitlines():
        candidate = line.strip().strip('"').strip("'").strip()
        for prefix in ("translation:", "english:", "translated:"):
            if candidate.lower().startswith(prefix):
                candidate = candidate[len(prefix):].strip().strip('"').strip("'").strip()
                break
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if candidate:
            return candidate if _is_english(candidate) else None
    return None


def _generate_english_translation(text: str) -> str | None:
    if not text or not str(text).strip():
        return None
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=settings.RESOLVED_FAST_API_KEY,
            base_url=settings.RESOLVED_FAST_BASE_URL,
            timeout=15,
        )
        off_body = {}
        if getattr(settings, "HYDE_THINKING_OFF", True):
            configured = getattr(settings, "MODEL_REASONING_OFF_BODY", None) or {}
            if isinstance(configured, dict):
                off_body = dict(configured)
        answer = ""
        for attempt in (1, 2):
            kwargs = {
                "model": settings.RESOLVED_FAST_MODEL,
                "messages": [
                    {"role": "system", "content": _TRANSLATE_PROMPT},
                    {"role": "user", "content": f"User question:\n{text}\n\nEnglish translation:"},
                ],
                "temperature": 0.0,
                "max_tokens": max(64, int(getattr(settings, "HYDE_MAX_TOKENS", 512))),
            }
            if off_body:
                kwargs["extra_body"] = off_body
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                if not off_body or not is_thinking_rejected(exc):
                    raise
                logger.warning("Translation provider rejected reasoning flag; retrying without it")
                off_body = {}
                kwargs.pop("extra_body", None)
                response = client.chat.completions.create(**kwargs)
            answer = str(response.choices[0].message.content or "").strip()
            if answer:
                break
            logger.warning("Local-query translation returned empty content (attempt %d)", attempt)
        translated = parse_translation(answer)
        if not translated:
            return None
        if _BAD_TRANSLATION_RE.search(translated):
            logger.warning("Local-query translation remained imperative; retrying once")
            retry = client.chat.completions.create(**kwargs)
            translated = parse_translation(str(retry.choices[0].message.content or ""))
            if not translated or _BAD_TRANSLATION_RE.search(translated):
                return None
        return translated
    except Exception as exc:  # noqa: BLE001
        logger.warning("Local-query translation failed: %s", type(exc).__name__)
        return None


def cached_translation(text: str) -> str | None:
    """Translate once per distinct input and keep a bounded process-local LRU-ish cache."""
    normalized = str(text or "").strip()
    if not normalized:
        return None
    key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    with _CACHE_LOCK:
        value = _CACHE.get(key)
        if value is not None:
            _CACHE.pop(key, None)
            _CACHE[key] = value
            return value
    value = _generate_english_translation(normalized)
    if value:
        with _CACHE_LOCK:
            _CACHE[key] = value
            while len(_CACHE) > _CACHE_MAX:
                _CACHE.pop(next(iter(_CACHE)))
    return value
