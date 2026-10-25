"""PathPolicy — 集中式路径安全判定（v8.13 第四批第一步）。

背景（审计 D 簇 A-01/A-10/A-11 / E 簇 API-1）: 文件类工具各自的路径校验
"resolve + 白名单"是分头实现、各自打补丁的——readfile/file_ops/pdf_read/
api workspace-files 四处曾各自残留 startswith 前缀漏洞或 CWD 相对锚点，
修复节奏不同步。收敛为单一策略：resolve + is_relative_to + 根列表。

锚点统一为 PROJECT_ROOT（不再有 Path.cwd() 兜底），判据统一 is_relative_to
（杜绝 "output_evil" 同前缀绕过）。
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.config import PROJECT_ROOT, settings

WORKSPACE_ROOT = (PROJECT_ROOT / "workspace").resolve()
OUTPUT_ROOT = (WORKSPACE_ROOT / "output").resolve()


def _current_output_root() -> Path:
    try:
        from src.core.tracing import get_user_id
        user_id = get_user_id()
    except Exception:
        user_id = None
    if user_id is None:
        return OUTPUT_ROOT
    users_root = (OUTPUT_ROOT / "users").resolve()
    user_root = (users_root / str(int(user_id))).resolve()
    if not user_root.is_relative_to(users_root):
        raise PermissionError("invalid account workspace")
    users_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    user_root.mkdir(mode=0o700, exist_ok=True)
    if __import__("os").name != "nt":
        for directory in (users_root, user_root):
            try:
                directory.chmod(0o700)
            except OSError:
                pass
    return user_root


def _is_output_reference(path: str) -> bool:
    raw = Path(path)
    if raw.is_absolute():
        try:
            return raw.resolve().is_relative_to(OUTPUT_ROOT)
        except (OSError, ValueError):
            return False
    normalized = raw.as_posix().lstrip("./")
    return normalized.startswith(("output/", "workspace/output/"))


def read_roots() -> List[Path]:
    """读取允许根：项目根 + 配置的额外读取根（FILE_READ_EXTRA_ROOTS）。"""
    extra = [Path(r).resolve()
             for r in (getattr(settings, "FILE_READ_EXTRA_ROOTS", None) or [])]
    return [PROJECT_ROOT.resolve()] + extra


def is_within(path: Path, roots: List[Path]) -> bool:
    """严格前缀判定（is_relative_to），无 startswith 同前缀漏洞。"""
    for root in roots:
        try:
            if path.is_relative_to(root):
                return True
        except ValueError:
            continue
    return False


def resolve_within(path: str, base: Path, roots: List[Path]) -> Path:
    """解析并校验：绝对路径直接 resolve；相对路径锚定 base 再 resolve。
    越界抛出 PermissionError（fail-closed）。
    """
    p = Path(path)
    full = p.resolve() if p.is_absolute() else (base / p).resolve()
    if not is_within(full, roots):
        raise PermissionError(f"拒绝访问允许目录外的路径: {full}")
    return full


def resolve_read(path: str) -> Path:
    """读文件路径：相对路径锚定 workspace/，允许根 = 项目根 + 额外根。"""
    if _current_output_root() != OUTPUT_ROOT and _is_output_reference(path):
        raw = Path(path)
        if raw.is_absolute():
            candidate = raw.resolve()
            if not candidate.is_relative_to(_current_output_root()):
                raise PermissionError("拒绝读取其他账户的工作区文件")
            return candidate
        return resolve_write(path)
    return resolve_within(path, WORKSPACE_ROOT, read_roots())


def resolve_workspace_read(path: str) -> Path:
    """仅 workspace/ 内读取（pdf_read 等限定型工具）。"""
    if _current_output_root() != OUTPUT_ROOT and _is_output_reference(path):
        raw = Path(path)
        if raw.is_absolute():
            candidate = raw.resolve()
            if not candidate.is_relative_to(_current_output_root()):
                raise PermissionError("拒绝读取其他账户的工作区文件")
            return candidate
        return resolve_write(path)
    return resolve_within(path, WORKSPACE_ROOT, [WORKSPACE_ROOT])


def resolve_write(path: str) -> Path:
    """写文件路径：仅 workspace/output/（剥离 output/ 或 workspace/output/ 前缀）。"""
    output_root = _current_output_root()
    normalized = Path(path).as_posix()
    for prefix in ("workspace/output/", "output/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    full = (output_root / normalized).resolve()
    if not is_within(full, [output_root]):
        raise PermissionError(f"拒绝写入 workspace/output 外的路径: {path}")
    return full


def is_output_path(path: str, root: Optional[Path] = None) -> bool:
    """路径是否落在 workspace/output 内（auto_workspace 免询问依据）。"""
    try:
        if not path:
            return False
        full = resolve_write(path)
        return full.is_relative_to(root or OUTPUT_ROOT)
    except Exception:
        return False
