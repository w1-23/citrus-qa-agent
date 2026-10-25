"""Request-scoped access to the current turn's image attachments.

``api/main.py`` stores the prepared attachments in this contextvar right after
``_prepare_chat_attachments``.  ``src/tools/registry._invoke_tool_with_ctx`` runs
sync tools through ``contextvars.copy_context()``, so tools executed on the
executor thread can still read the images of the request they belong to —
without writing uploads to disk (they stay in memory for this turn only).
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass

MAX_IMAGES = 8


@dataclass(frozen=True)
class RequestImage:
    name: str
    url: str  # data:<mime>;base64,<payload>


_CURRENT: contextvars.ContextVar[tuple[RequestImage, ...]] = contextvars.ContextVar(
    "vision_request_images", default=())


def set_request_images(images) -> contextvars.Token:
    """Publish this turn's images. Returns a token for :func:`reset_request_images`."""
    normalized: list[RequestImage] = []
    for item in list(images or []):
        if len(normalized) >= MAX_IMAGES:
            break
        url = str((item or {}).get("url") or "")
        if not url.startswith("data:"):
            continue
        normalized.append(RequestImage(name=str(item.get("name") or "image")[:180], url=url))
    return _CURRENT.set(tuple(normalized))


def reset_request_images(token: contextvars.Token) -> None:
    try:
        _CURRENT.reset(token)
    except (ValueError, LookupError):  # token from another context; best effort
        _CURRENT.set(())


def get_request_images() -> tuple[RequestImage, ...]:
    return _CURRENT.get()


def clear_request_images() -> None:
    _CURRENT.set(())
