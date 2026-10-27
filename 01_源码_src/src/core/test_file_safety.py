"""Archive expansion and member-count guards for uploaded office documents."""
from __future__ import annotations

import io
import zipfile

import pytest

from src.core import file_safety


def _archive(entries: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return output.getvalue()


def test_zip_member_count_and_expanded_size_limits_fail_closed(monkeypatch):
    monkeypatch.setattr(file_safety, "MAX_ARCHIVE_MEMBERS", 1)
    with pytest.raises(ValueError, match="过多压缩包成员"):
        file_safety.validate_zip_bytes(_archive({"one": b"a", "two": b"b"}), label="book.xlsx")

    monkeypatch.setattr(file_safety, "MAX_ARCHIVE_MEMBERS", 10)
    monkeypatch.setattr(file_safety, "MAX_ARCHIVE_EXPANDED_BYTES", 32)
    with pytest.raises(ValueError, match="解压后超过"):
        file_safety.validate_zip_bytes(_archive({"large": b"x" * 1000}), label="book.xlsx")


def test_invalid_zip_is_reported_as_invalid_office_archive():
    with pytest.raises(ValueError, match="不是有效的 ZIP/OOXML 文件"):
        file_safety.validate_zip_bytes(b"not a zip", label="broken.docx")
