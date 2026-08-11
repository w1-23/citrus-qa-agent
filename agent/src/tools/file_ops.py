"""File operations — sandboxed file write"""
import os
import threading
from pathlib import Path
from typing import Literal
from langchain_core.tools import tool

from src.config import PROJECT_ROOT

_WORKSPACE_ROOT = (PROJECT_ROOT / "workspace" / "output").resolve()
_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

# 进程内按路径分桶的写锁，防止同文件并发 read-modify-write 丢失更新 (AG-9)
_write_locks: dict = {}
_write_locks_guard = threading.Lock()


def _get_write_lock(path: str) -> threading.Lock:
    with _write_locks_guard:
        lock = _write_locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _write_locks[path] = lock
        return lock


@tool
def write_local_file(path: str, content: str, mode: Literal["write", "append"] = "write") -> str:
    """将内容写入沙箱工作区。仅限 workspace/output/ 目录。

    Args:
        path: 相对路径，例如 'reviews/hlb_review_2024.md'
        content: 要写入的 Markdown 或文本内容
        mode: 'write' 覆盖写入（默认）；'append' 尾部追加。写长篇综述时用 append 分块。
    """
    try:
        normalized = Path(path).as_posix()
        for prefix in ("workspace/output/", "output/"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        target_path = (_WORKSPACE_ROOT / normalized).resolve()
        if not str(target_path).startswith(str(_WORKSPACE_ROOT)):
            return f"Error: Access denied. Path '{path}' is outside workspace/output/."

        target_path.parent.mkdir(parents=True, exist_ok=True)

        new_content = content  # 本轮新增内容（append 合并前），用于预览
        lock = _get_write_lock(str(target_path))
        with lock:
            if mode == "write":
                actual_mode = "write"
            elif mode == "append" and target_path.exists():
                existing = target_path.read_text(encoding="utf-8")
                content = existing + "\n\n" + content
                actual_mode = "append"
            else:
                actual_mode = "write"

            # 原子写: 先写临时文件再 os.replace，避免崩溃产生半截文件
            tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
            tmp_path.write_text(content, encoding="utf-8")
            os.replace(tmp_path, target_path)

        total_chars = len(content)
        size_kb = target_path.stat().st_size / 1024
        # v8.3.1: 内容预览——append 时显示【本轮新增块】而非文件开头，
        # 否则 LLM 每轮看到相同预览，无法确认本轮写了什么（防重写失效）
        preview_src = new_content if actual_mode == "append" else content
        preview = preview_src[:200].replace("\n", " ")
        return (f"Success: {actual_mode} to {normalized}. Total file size now: "
                f"{total_chars} chars ({size_kb:.1f} KB).\n内容预览: {preview}")
    except Exception as e:
        return f"Error writing file: {e}"
