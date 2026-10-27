"""Small, provider-agnostic status envelope for tool results."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any


@dataclass(frozen=True)
class ToolOutcome:
    status: str
    reason_code: str = ""
    coverage: str = ""
    artifacts: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_FAILURE_PREFIXES = (
    "[TOOL_ERROR]", "[WEB_TOOL_ERROR]", "[WEB_FETCH_ERROR]", "[WEB_FETCH_FAILED]",
    "[WEB_DISABLED]", "[WEB_FETCH_BLOCKED]", "[LOCAL_SEARCH_ERROR]",
    "[LOCAL_HYBRID_ERROR]", "[LOCAL_HYBRID_TRANSLATION_ERROR]",
    "[LOCAL_SEARCH_TRANSLATION_ERROR]", "[ERR_",
)


def normalize_tool_outcome(tool_name: str, content: str, artifact: dict | None = None) -> ToolOutcome:
    """Normalize explicit machine signals; never infer success from no-hit prose."""
    text = str(content or "").strip()
    data = artifact if isinstance(artifact, dict) else {}
    links = tuple(dict.fromkeys(re.findall(
        r"\[[^\]]+\]\((/api/v2/session/[A-Za-z0-9_-]+/artifacts/[a-f0-9]{32})\)", text)))

    if text.startswith("[ERR_REQUEST_DEADLINE]") or text.startswith("[ERR_TIMEOUT]"):
        return ToolOutcome("failed", "timeout", artifacts=links)
    if text.startswith("[WEB_DISABLED]"):
        return ToolOutcome("not_run", "disabled", artifacts=links)
    if text.startswith("[WEB_FETCH_BLOCKED]"):
        return ToolOutcome("not_run", "blocked_destination", artifacts=links)
    if text.startswith("[WEB_FETCH_EMPTY]"):
        return ToolOutcome("no_results", "empty_page", artifacts=links)
    if text.startswith("[ERR_QUOTA]") or text.startswith("[ERR_CREDITS_EXHAUSTED]"):
        return ToolOutcome("failed", "quota", artifacts=links)
    if text.startswith(_FAILURE_PREFIXES):
        lowered = text[:500].lower()
        reason = "tool_error"
        if "quota" in lowered or "credit" in lowered or "额度" in lowered or "余额" in lowered:
            reason = "quota"
        elif "401" in lowered or "403" in lowered or "unauthorized" in lowered or "认证" in lowered:
            reason = "auth"
        elif "timeout" in lowered or "timed out" in lowered or "超时" in lowered:
            reason = "timeout"
        elif "network" in lowered or "connect" in lowered or "网络" in lowered:
            reason = "network"
        elif "MISSING_DEP" in text[:40]:
            reason = "dependency"
        return ToolOutcome("failed", reason, artifacts=links)

    if tool_name == "audit_uploaded_data":
        if text.startswith("审查未启动"):
            return ToolOutcome("not_run", "no_selected_attachment", artifacts=links)
        if text.startswith("数据审查未完整完成"):
            return ToolOutcome("partial", "coverage_incomplete", "未完成范围见报告", links)
        if text.startswith("数据审查已完成"):
            return ToolOutcome("complete", coverage="按报告注明范围完成", artifacts=links)

    if tool_name == "verify_uploaded_references":
        if "未自动提取到 DOI" in text or "当前请求没有可提取的来源" in text:
            return ToolOutcome("not_run", "no_reference_identifiers", artifacts=links)
        if "文本提取失败" in text or "联网失败" in text:
            # A mixed verification can contain successful rows alongside failures.
            match = re.search(r"(?:metadata_matched|metadata_mismatch|title_candidate|not_found_in_checked_registries)\s+(\d+)\s*条", text)
            return ToolOutcome("partial" if match and int(match.group(1)) else "failed",
                               "network" if not match or not int(match.group(1)) else "some_records_unverified",
                               artifacts=links)
        match = re.search(r"unverified\s+(\d+)\s*条", text, re.I)
        if match and int(match.group(1)):
            completed = re.search(r"(?:metadata_matched|metadata_mismatch|title_candidate|not_found_in_checked_registries)\s+(\d+)\s*条", text, re.I)
            return ToolOutcome("partial" if completed and int(completed.group(1)) else "failed",
                               "some_records_unverified" if completed and int(completed.group(1)) else "network",
                               artifacts=links)
        if "not_checked" in text:
            return ToolOutcome("partial", "verification_limit", artifacts=links)

    if tool_name == "fetch_and_audit_open_supplements":
        if text.startswith("公开补充材料获取未完成") or "归档解析失败" in text:
            return ToolOutcome("failed", "network_or_archive_error", artifacts=links)
        if "没有可下载归档" in text or "不在可信 PMC OA 主机白名单" in text:
            return ToolOutcome("no_results" if "没有可下载归档" in text else "not_run",
                               "no_open_archive" if "没有可下载归档" in text else "blocked_source",
                               artifacts=links)
        if "未能审查" in text or "没有识别到可审查的表格材料" in text:
            return ToolOutcome("partial", "some_material_unreviewed", artifacts=links)

    if tool_name == "local_hybrid_search":
        local_status = str(data.get("local_status") or "")
        if local_status == "skipped_no_matching_family":
            return ToolOutcome("not_run", "no_matching_corpus")
        if local_status == "hit":
            count = len(data.get("main_results") or [])
            partial = bool(data.get("errors") or data.get("reranker_fallback"))
            return ToolOutcome("partial" if partial else ("complete" if count else "no_results"),
                               "" if count else "no_candidates",
                               f"{count} 条本地候选")
        if local_status == "zero_hit":
            errors = data.get("errors") or {}
            return ToolOutcome("partial" if errors else "no_results",
                               "retrieval_channel_error" if errors else "no_candidates",
                               "零候选；有通道失败" if errors else "零候选")
        if local_status in {"error", "failed"}:
            return ToolOutcome("failed", "local_search_error")

    if tool_name in {"deepseek_search", "deepseek_fetch", "deepseek_fetch_paper"}:
        rows = data.get("web_results")
        if isinstance(rows, list) and not rows:
            return ToolOutcome("no_results", "no_web_results", artifacts=links)
        if isinstance(rows, list):
            return ToolOutcome("complete" if rows else "no_results",
                               "" if rows else "no_web_results",
                               f"{len(rows)} 条联网结果", links)

    return ToolOutcome("complete", artifacts=links)
