"""Request-local uploaded table payloads for research tools."""
from __future__ import annotations

import contextvars
from typing import Any

_tables: contextvars.ContextVar[list[dict[str, Any]]] = contextvars.ContextVar(
    "research_uploaded_tables", default=[])


def set_request_tables(tables: list[dict[str, Any]]):
    return _tables.set(list(tables or []))


def reset_request_tables(token) -> None:
    _tables.reset(token)


def get_request_tables() -> list[dict[str, Any]]:
    return list(_tables.get() or [])

