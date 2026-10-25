"""Bounded ZIP inspection for uploaded OOXML and spreadsheet documents."""
from __future__ import annotations

import io
import zipfile


MAX_ARCHIVE_MEMBERS = 4096
MAX_ARCHIVE_EXPANDED_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 64 * 1024 * 1024


def validate_zip_bytes(data: bytes, *, label: str = "文件") -> int:
    """Return total uncompressed size or raise ValueError before parsing XML."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_MEMBERS:
                raise ValueError(f"{label}包含过多压缩包成员，已停止读取")
            total = 0
            for info in entries:
                if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    raise ValueError(f"{label}单个解压文件超过 64 MB，已停止读取")
                total += info.file_size
                if total > MAX_ARCHIVE_EXPANDED_BYTES:
                    raise ValueError(f"{label}解压后超过 100 MB，已停止读取")
            return total
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{label}不是有效的 ZIP/OOXML 文件") from exc
