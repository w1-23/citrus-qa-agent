"""Bounded chat-attachment parsing regression tests."""
from __future__ import annotations

import base64
import io
import struct
import tempfile
import zlib
import atexit
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image

import src.config as config
_test_runtime = tempfile.TemporaryDirectory(
    prefix=".test-attachment-safety-", dir=Path(__file__).resolve().parents[2])
config.PROJECT_ROOT = Path(_test_runtime.name)

from src.api.main import _estimated_attachment_bytes, _prepare_chat_attachments

atexit.register(_test_runtime.cleanup)


def _png_bytes(size: tuple[int, int] = (2, 2)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (40, 150, 90)).save(output, format="PNG")
    return output.getvalue()


def _oversized_png_header(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


def _item(name: str, payload: bytes, mime: str) -> dict[str, str]:
    encoded = base64.b64encode(payload).decode("ascii")
    return {"name": name, "type": mime, "data": f"data:{mime};base64,{encoded}"}


def test_image_is_decoded_before_it_is_forwarded():
    context, images, tables = _prepare_chat_attachments([_item("leaf.png", _png_bytes(), "image/png")])
    assert not context and not tables
    assert len(images) == 1 and images[0]["url"].startswith("data:image/png;base64,")


def test_image_magic_bytes_do_not_bypass_decoder_and_pixel_limit():
    with pytest.raises(HTTPException) as corrupt:
        _prepare_chat_attachments([_item("broken.png", b"\x89PNG\r\n\x1a\nnot-an-image", "image/png")])
    assert corrupt.value.status_code == 400

    too_large = _oversized_png_header(6000, 5001)  # >30 megapixels
    with pytest.raises(HTTPException) as large:
        _prepare_chat_attachments([_item("huge.png", too_large, "image/png")])
    assert large.value.status_code == 413
    assert "像素" in str(large.value.detail)


def test_image_extension_and_declared_mime_must_match_real_format():
    png = _png_bytes()
    with pytest.raises(HTTPException) as suffix:
        _prepare_chat_attachments([_item("leaf.jpg", png, "image/jpeg")])
    assert suffix.value.status_code == 400

    with pytest.raises(HTTPException) as declared:
        _prepare_chat_attachments([_item("leaf.png", png, "image/jpeg")])
    assert declared.value.status_code == 400


def test_attachment_count_and_encoded_byte_preflight_are_bounded():
    with pytest.raises(HTTPException) as too_many:
        _estimated_attachment_bytes([{"name": "x", "data": ""}] * 9)
    assert too_many.value.status_code == 413

    with pytest.raises(HTTPException) as too_large:
        _estimated_attachment_bytes([{"name": "x", "data": "A" * (28 * 1024 * 1024)}])
    assert too_large.value.status_code == 413


def test_pdf_text_extraction_reports_page_cap(monkeypatch):
    import fitz
    import src.api.main as api_main

    document = fitz.open()
    for text in ("first page text", "second page text"):
        page = document.new_page()
        page.insert_text((72, 72), text)
    payload = document.tobytes()
    document.close()

    monkeypatch.setattr(api_main, "_MAX_CHAT_PDF_PAGES", 1)
    extracted, images = api_main._extract_pdf_attachment(payload, "paper.pdf")
    assert "first page text" in extracted
    assert "second page text" not in extracted
    assert "最多 1 页" in extracted
    assert images == []
