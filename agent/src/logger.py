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
    """检索日志（v8.7 合并重构）——统一记录检索过滤全过程。

    原 retrieval/（只记通过结果）与 debug_filter/（过滤明细但 filtered 恒空）
    两个文件功能重叠且都不完整，合并为单个 logs/retrieval/retrieval_YYYY-MM-DD.log：
    query / 模式 / 候选数 / 动态阈值 / 通过·被过滤明细 / 耗时 / 预算·去重拦截。
    日志目录跟随 CITRUS_LOG_DIR 覆盖（测试独立 sink）。
    """

    def __init__(self):
        self._log_dir = _log_dir() / "retrieval"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _log_path(self) -> Path:
        return self._log_dir / f"retrieval_{datetime.now().strftime('%Y-%m-%d')}.log"

    def log_retrieval(
        self,
        query: str,
        reranked: List[Dict],
        threshold: float,
        passed: List[Dict],
        filtered: List[Dict],
        elapsed: Optional[float] = None,
        extra: Optional[Dict] = None,
    ):
        """Append a complete retrieval-filter record to today's log file.

        reranked: 重排序候选全集；passed: 过阈值保留；filtered: 被动态阈值拦截。
        """
        now = datetime.now()
        lines = []
        lines.append(f"[{now.isoformat()}] QUERY: {query}")
        head = (f"  candidates={len(reranked)} | threshold={threshold:.4f} "
                f"| passed={len(passed)} | filtered={len(filtered)}")
        if elapsed is not None:
            head += f" | elapsed={elapsed:.2f}s"
        for k, v in (extra or {}).items():
            head += f" | {k}={v}"
        lines.append(head)
        lines.append(f"  {'─' * 80}")

        if passed:
            lines.append("  ✅ PASSED:")
            for i, r in enumerate(passed, 1):
                lines.append(self._fmt_chunk(i, r))
        if filtered:
            lines.append("  ❌ FILTERED (dynamic threshold):")
            for i, r in enumerate(filtered, 1):
                lines.append(self._fmt_chunk(i, r))
        if not passed and not filtered:
            lines.append("  (no candidates)")
        lines.append("")

        with open(self._log_path(), "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _fmt_chunk(i: int, r: dict) -> str:
        paper_id = r.get("paper_id", "N/A")
        doi = r.get("doi", "N/A")
        section = r.get("section_name", "N/A")
        batch = r.get("_batch", "N/A")
        score = r.get("rerank_score", 0)
        text = str(r.get("text", "") or "").replace("\n", " ").strip()
        head = f"    [{i:2d}] CE={score:.4f} | {paper_id} | {section} | batch={batch} | DOI: {doi}"
        if text:
            head += f"\n        Text: {text[:160]}{'...' if len(text) > 160 else ''}"
        return head


_search_logger = RetrievalLogger()


def log_retrieval(query: str, reranked: List[Dict], threshold: float,
                  passed: List[Dict], filtered: List[Dict],
                  elapsed: Optional[float] = None,
                  extra: Optional[Dict] = None):
    """Module-level convenience function (v8.7 统一检索过滤日志入口)."""
    _search_logger.log_retrieval(query, reranked, threshold, passed, filtered,
                                 elapsed=elapsed, extra=extra)
