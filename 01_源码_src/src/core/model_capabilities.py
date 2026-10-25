"""Capabilities inferred from the selected provider/model pair.

OpenAI-compatible ``/models`` endpoints commonly return model IDs without
modality metadata. Keep image support in one conservative registry so stale
per-install booleans cannot override the selected model's known capability.
"""
from __future__ import annotations

import re


_DEEPSEEK_VISION_MODELS = {
    "deepseek-flash",
    "deepseek-v4.1-flash",
    "deepseek-v4.1-flash-1017",
    "deepseek-v4-flash",
    "deepseek-v4-flash-vision-exp",
}

_VISION_MARKERS = (
    "-vl", "_vl", "vision", "glm-4v", "glm-4.1v", "glm-4.5v", "qvq",
    "gpt-4o", "gpt-4.1", "gemini-", "claude-3", "pixtral", "llava",
    "internvl", "cogvlm",
)


def model_supports_images(provider_id: str, model_id: str) -> bool:
    """Return whether a selected model is known to accept image input.

    Unknown IDs fail closed. The settings UI must not ask users to maintain a
    second, potentially stale image-capability switch.
    """
    provider = str(provider_id or "").strip().lower()
    model = str(model_id or "").strip().lower().rsplit("/", 1)[-1]
    model = re.sub(r"@[a-z0-9._-]+$", "", model)
    if not model:
        return False

    if model in _DEEPSEEK_VISION_MODELS:
        return provider in {"deepseek", "opencode_go", "aliyun_bailian", "zhipu", "custom_openai"}

    if any(marker in model for marker in _VISION_MARKERS):
        return True

    # Aliyun and other OpenAI-compatible catalogs sometimes expose versioned
    # Qwen-VL names without a separator immediately before ``vl``.
    return bool(re.search(r"\bqwen\d*(?:\.\d+)?-?vl\b", model))
