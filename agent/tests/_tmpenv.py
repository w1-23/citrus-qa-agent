# -*- coding: utf-8 -*-
"""测试临时目录统一入口（v8.4.14）。

DSH 执行沙箱（workspace-write）只保证工作区可写：`tempfile.mkdtemp()`
指向系统 TEMP 时 sqlite/文件写入被拒（历史 16 个"环境性失败"的根因——
用户真实 rag-agent 环境无沙箱，那些测试本可全过）。
统一改用工作区内 tests/.tmp_runner/，沙箱内外行为一致。
"""
import uuid
from pathlib import Path

_TMP_ROOT = Path(__file__).resolve().parent / ".tmp_runner"
_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def tmp_path(name: str = "") -> Path:
    """返回工作区内唯一临时路径（UUID 后缀，避免并发/残留干扰）。

    Args:
        name: 语义前缀（如 "db"、"csv"），便于排查残留文件。
    """
    return _TMP_ROOT / (f"{name}_{uuid.uuid4().hex[:10]}" if name
                        else uuid.uuid4().hex[:12])


def tmp_dir(name: str = "") -> Path:
    """返回工作区内唯一临时目录（已创建）。"""
    p = tmp_path(name)
    p.mkdir(parents=True, exist_ok=True)
    return p
