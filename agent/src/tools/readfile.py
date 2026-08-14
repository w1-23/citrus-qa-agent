"""File reading tool — read_local_file (system-wide read, workspace-relative as fallback)

v8.3.0: max_chars parameter — default 30000 for PDF, unlimited for others.
Supervisor controls read depth; user can say "读全文" or "读前3页".
"""
import logging
import os
from pathlib import Path

from langchain_core.tools import tool

from src.config import settings, PROJECT_ROOT

logger = logging.getLogger(__name__)

_WORKSPACE_ROOT = PROJECT_ROOT / settings.WORKSPACE_DIR

PDF_DEFAULT_MAX_CHARS = 30000


def _is_path_allowed(full: Path) -> bool:
    """v8.3.3/v8.4.14 路径白名单: 项目根目录 + 配置的额外读取根目录。

    v8.4.14: 修复 commonpath 同盘前缀漏洞（E:\anywhere 与 E:\agent 同盘前缀
    会被误放行）→ is_relative_to 严格判定。
    """
    try:
        if full.is_relative_to(PROJECT_ROOT.resolve()):
            return True
    except ValueError:
        pass
    for root in getattr(settings, "FILE_READ_EXTRA_ROOTS", None) or []:
        try:
            if full.is_relative_to(Path(root).resolve()):
                return True
        except ValueError:
            continue
    return False


def _resolve_read_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        full = p.resolve()
        # v8.3.3 安全: 绝对路径仅允许项目根目录内（防 LLM 读取系统任意文件，
        # 与 write_local_file/pdf_read 的 workspace 校验对称）
        if not _is_path_allowed(full):
            raise PermissionError(f"拒绝读取项目目录外的文件: {full}")
    else:
        full = (_WORKSPACE_ROOT / p).resolve()
        if not full.exists():
            full = Path.cwd() / p
            full = full.resolve()

    if not full.exists():
        raise FileNotFoundError(f"文件不存在: {full}")

    return full


def _read_pdf(path: Path, max_chars: int = 0) -> str:
    """Read PDF pages. If max_chars > 0, stop accumulating once threshold exceeded."""
    try:
        import fitz
        doc = fitz.open(path)
        texts = []
        total_chars = 0
        for i, page in enumerate(doc):
            text = page.get_text()
            if text and text.strip():
                page_block = f"--- Page {i+1} ---\n{text}"
                texts.append(page_block)
                total_chars += len(page_block)
                if max_chars > 0 and total_chars >= max_chars:
                    break
        total = len(doc)
        doc.close()
        valid = len(texts)
        if valid:
            header = f"[PDF: {path.name} | {valid} of {total} pages | fitz]\n\n"
            return header + "\n".join(texts)
        return f"[PDF: {path.name} | {total} pages | no extractable text]"
    except Exception:
        pass
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            texts = []
            total_chars = 0
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    page_block = f"--- Page {i+1} ---\n{text}"
                    texts.append(page_block)
                    total_chars += len(page_block)
                    if max_chars > 0 and total_chars >= max_chars:
                        break
            total = len(pdf.pages)
            valid = len(texts)
            if valid:
                header = f"[PDF: {path.name} | {valid} of {total} pages | pdfplumber]\n\n"
                return header + "\n".join(texts)
            return f"[PDF: {path.name} | {total} pages | no extractable text]"
    except Exception:
        pass
    return f"[PDF: {path.name} | failed to read with both fitz and pdfplumber]"


def _read_excel(path: Path) -> str:
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    parts = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        parts.append(f"## Sheet: {sheet_name} ({len(rows)} 行)")
        for i, row in enumerate(rows[:50]):
            parts.append(f"  Row {i+1}: {[str(c)[:60] if c else '' for c in row]}")
        if len(rows) > 50:
            parts.append(f"  ... (省略 {len(rows) - 50} 行)")
    wb.close()
    return "\n".join(parts)


def _read_csv(path: Path) -> str:
    import pandas as pd
    total_rows = sum(1 for _ in open(path, encoding="utf-8", errors="replace")) - 1
    df = pd.read_csv(path, nrows=1000)
    preview_json = df.to_json(orient="records", force_ascii=False)
    lines = [
        f"[CSV Preview] Total rows: {total_rows}. Showing first {len(df)} rows.\n",
        preview_json,
        f"\n\n*** Full dataset ({total_rows} rows) available in backend. "
        f"Use statistical_analysis(file_path='{path.name}') for full computations. ***",
    ]
    return "\n".join(lines)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    encodings = ["utf-8", "gbk", "gb2312", "latin-1", "shift_jis"]
    try:
        import chardet
        detected = chardet.detect(raw)
        enc = detected.get("encoding", "utf-8") or "utf-8"
        encodings.insert(0, enc)
    except ImportError:
        pass
    for enc in encodings:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1", errors="replace")


@tool
async def read_local_file(path: str, max_chars: int = 0) -> str:
    """读取本地文件。PDF 默认读前 30000 字符（约8-12页），其它格式默认读全文。

    绝对路径：仅允许项目根目录内的文件。
    相对路径：从 workspace/ 或当前工作目录查找。

    Args:
        path: 文件的绝对路径或相对路径
        max_chars: 最大读取字符数（0 = PDF默认30000, 其他不限）
    """
    logger.info(f"[read_local_file] path={path} max_chars={max_chars}")

    if not path or not path.strip():
        return "[ERR_PARSE] 文件路径不能为空"

    try:
        full_path = _resolve_read_path(path)
    except FileNotFoundError as e:
        return str(e)
    except Exception as e:
        return f"[ERR_PARSE] 路径解析失败: {e}"

    if not full_path.exists():
        return f"[ERR_FILE_NOT_FOUND] 文件不存在: {full_path}"

    size_mb = full_path.stat().st_size / (1024 * 1024)
    if size_mb > settings.FILE_READ_MAX_SIZE_MB:
        return f"[ERR_FILE_TOO_LARGE] 文件过大: {size_mb:.1f}MB > {settings.FILE_READ_MAX_SIZE_MB}MB 限制"

    ext = full_path.suffix.lower()

    try:
        if ext == ".pdf":
            if max_chars <= 0:
                max_chars = PDF_DEFAULT_MAX_CHARS
            content = _read_pdf(full_path, max_chars)
        elif ext in (".xlsx", ".xls"):
            content = _read_excel(full_path)
        elif ext == ".csv":
            content = _read_csv(full_path)
        else:
            content = _read_text(full_path)
        return content
    except Exception as e:
        logger.error(f"[read_local_file] 读取失败: {e}")
        return f"读取失败: {e}"
