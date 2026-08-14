"""
Structured retrieval logger for traceability and manual review.
Writes query + reranked chunks to timestamped log files.
"""
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from src.config import PROJECT_ROOT, settings

logger = logging.getLogger(__name__)


def _log_dir() -> Path:
    """v8.4.3 工单10: 日志目录可被 CITRUS_LOG_DIR 覆盖（测试独立 sink，防测试帧污染生产日志）。"""
    override = os.environ.get("CITRUS_LOG_DIR", "").strip()
    if override:
        return Path(override)
    return PROJECT_ROOT / settings.LOG_DIR


class RequestIdFilter(logging.Filter):
    """注入 request_id（contextvar 贯穿请求链路，无请求时为 '-'）。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from src.core.tracing import get_request_id
            record.request_id = get_request_id()
        except Exception:
            record.request_id = "-"
        return True


class MaskingFormatter(logging.Formatter):
    """v8.4.6 B6: 日志脱敏（PII 正则过滤）——输出到文件/控制台前统一掩码。"""

    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        try:
            from src.core.pii_mask import mask_sensitive
            return mask_sensitive(s)
        except Exception:
            return s


def setup_logging():
    """Configure root logger: INFO+ to file, WARNING+ to console."""
    log_level = getattr(logging, (settings.LOG_LEVEL or "INFO").upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(log_level)

    # File handler
    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        str(log_dir / "agent.log"), when="midnight",
        encoding="utf-8", backupCount=7
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(MaskingFormatter(
        "%(asctime)s | %(levelname)-5s | %(name)s | req=%(request_id)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler: WARNING+ only
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    rid_filter = RequestIdFilter()
    file_handler.addFilter(rid_filter)
    console.addFilter(rid_filter)

    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console)

    # Suppress noisy library logs
    for lib in ("openai", "httpx", "httpcore", "markdown_it", "fastembed",
                "qdrant_client", "urllib3", "asyncio"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    logger.info(f"[Logger] initialized: file={logging.getLevelName(log_level)}, console=WARNING+")


class RetrievalLogger:
    """Writes retrieval search records to structured log files."""

    def __init__(self):
        self._log_dir = _log_dir() / "retrieval"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _log_path(self) -> Path:
        return self._log_dir / f"retrieval_{datetime.now().strftime('%Y-%m-%d')}.log"

    def log_search(
        self,
        query: str,
        results: List[Dict],
        elapsed: Optional[float] = None,
    ):
        """Append a structured search record to today's log file."""
        if not results:
            return

        now = datetime.now()
        lines = []
        lines.append(f"[{now.isoformat()}] QUERY: {query}")
        if elapsed is not None:
            lines.append(f"  Elapsed: {elapsed:.2f}s | Results: {len(results)}")
        lines.append(f"  {'─' * 80}")

        for i, r in enumerate(results, 1):
            paper_id = r.get("paper_id", "N/A")
            doi = r.get("doi", "N/A")
            section = r.get("section_name", "N/A")
            batch = r.get("_batch", "N/A")
            score = r.get("rerank_score", 0)
            text = r.get("text", "").replace("\n", " ").strip()

            lines.append(f"  [{i:2d}] CE={score:.4f} | {paper_id}")
            lines.append(f"        Batch: {batch} | DOI: {doi}")
            lines.append(f"        Section: {section}")
            lines.append(f"        Text: {text[:200]}{'...' if len(text) > 200 else ''}")
            lines.append("")

        lines.append("")

        with open(self._log_path(), "a", encoding="utf-8") as f:
            f.write("\n".join(lines))


_search_logger = RetrievalLogger()


def log_search(query: str, results: List[Dict], elapsed: Optional[float] = None):
    """Module-level convenience function."""
    _search_logger.log_search(query, results, elapsed)
