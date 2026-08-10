"""File operations — sandboxed file write"""
from pathlib import Path
from typing import Literal
from langchain_core.tools import tool

from src.config import PROJECT_ROOT

_WORKSPACE_ROOT = (PROJECT_ROOT / "workspace" / "output").resolve()
_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)


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

        if mode == "write" and target_path.exists():
            # 防御: 文件已存在 → 强制追加，防止 LLM 误覆盖
            existing = target_path.read_text(encoding="utf-8")
            content = existing + "\n\n" + content
            actual_mode = "append (forced: file existed)"
        elif mode == "append" and target_path.exists():
            existing = target_path.read_text(encoding="utf-8")
            content = existing + "\n\n" + content
            actual_mode = "append"
        else:
            actual_mode = "write"

        target_path.write_text(content, encoding="utf-8")

        total_chars = len(content)
        size_kb = target_path.stat().st_size / 1024
        return f"Success: {actual_mode} to {normalized}. Total file size now: {total_chars} chars ({size_kb:.1f} KB)."
    except Exception as e:
        return f"Error writing file: {e}"
