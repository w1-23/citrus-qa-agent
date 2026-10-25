"""Request-scoped adapter for the MIT-licensed PaperConan numeric audit engine.

Uploaded files are copied only into a unique temporary directory under CitrusQA's
configured data directory. The directory is removed before this function returns.
No image inventory or image diagnostics are enabled here.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)
_SUPPORTED = {".xlsx", ".xls", ".xlsm", ".xlsb", ".csv", ".tsv", ".pdf", ".docx"}
_UPSTREAM = Path(__file__).resolve().parent / "vendor"


def _load_engine():
    package_parent = str(_UPSTREAM.resolve())
    if package_parent not in sys.path:
        sys.path.insert(0, package_parent)
    from paperconan import scan_dir, write_html_report, distill_findings_for_review
    from paperconan import BLOCK_FINDING_GROUPS
    return scan_dir, write_html_report, distill_findings_for_review, BLOCK_FINDING_GROUPS


def _jsonable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _location_for(block: dict[str, Any], finding: dict[str, Any]) -> str:
    pieces = [str(block.get("file") or "")]
    if block.get("sheet"):
        pieces.append(f"工作表 {block['sheet']}")
    if block.get("block"):
        pieces.append(f"区块 {block['block'].get('rows', '?')} 行 × {block['block'].get('cols', '?')} 列")
    cells = finding.get("example_cells") or finding.get("cells")
    if cells:
        pieces.append("示例位置 " + _jsonable(cells[:8] if isinstance(cells, list) else cells))
    elif finding.get("row") is not None:
        pieces.append(f"第 {finding['row']} 行")
    return " · ".join(x for x in pieces if x)


def _restore_input_names(scan: dict[str, Any], original_names: dict[str, str],
                         temporary_root: Path) -> None:
    temporary_prefixes = {str(temporary_root), str(temporary_root).replace("\\", "/")}

    def visit(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            for child_key in list(value):
                if child_key == "path":
                    # The engine records absolute temporary paths for CLI re-reading.
                    # Web reports must not expose server filesystem paths or imply
                    # the request-scoped temporary file persists.
                    value.pop(child_key, None)
                    continue
                value[child_key] = visit(value[child_key], child_key)
            return value
        if isinstance(value, list):
            return [visit(item, key) for item in value]
        if isinstance(value, str):
            if key in {"file", "file_a", "file_b"}:
                return original_names.get(value, value)
            for temporary, original in original_names.items():
                value = value.replace(temporary, original)
            for prefix in temporary_prefixes:
                value = value.replace(prefix, "本轮附件临时目录")
        return value
    visit(scan)


def flatten_scan(scan: dict[str, Any], groups: tuple[str, ...]) -> list[dict[str, Any]]:
    """Convert PaperConan's full structured result into report rows without dropping rules."""
    rows: list[dict[str, Any]] = []
    for block in scan.get("relations_blocks", []) or []:
        for group in groups:
            for finding in block.get(group, []) or []:
                kind = str(finding.get("kind") or group)
                rows.append({
                    "severity": str(finding.get("severity") or "review"),
                    "check": kind,
                    "file": str(block.get("file") or ""),
                    "sheet": str(block.get("sheet") or ""),
                    "location": _location_for(block, finding),
                    "field": str(finding.get("col_a") or finding.get("col") or finding.get("row_a") or ""),
                    "detail": str(finding.get("rule") or finding.get("detail") or "PaperConan numeric signal"),
                    "evidence": _jsonable(finding),
                    "prefilter": str(finding.get("prefilter") or finding.get("profile_action") or ""),
                })
    for finding in scan.get("cross_sheet_findings", []) or []:
        rows.append({
            "severity": str(finding.get("severity") or "review"),
            "check": str(finding.get("kind") or "cross_table_signal"),
            "file": str(finding.get("file_a") or finding.get("file") or ""),
            "sheet": " ↔ ".join(str(finding.get(key) or "") for key in ("sheet_a", "sheet_b")).strip(" ↔"),
            "location": _jsonable({key: finding.get(key) for key in
                                    ("file_a", "sheet_a", "block_a", "file_b", "sheet_b", "block_b") if finding.get(key)}),
            "field": "",
            "detail": str(finding.get("rule") or finding.get("detail") or "跨表数值关系信号"),
            "evidence": _jsonable(finding),
            "prefilter": str(finding.get("prefilter") or finding.get("profile_action") or ""),
        })
    for section, label in (("digit_distribution", "last_digit_distribution"),
                           ("decimal_endings", "repeated_decimal_endings"),
                           ("decimal_tail_clusters", "decimal_tail_clustering")):
        for finding in scan.get(section, []) or []:
            rows.append({
                "severity": str(finding.get("severity") or ("review" if finding.get("significant") else "info")),
                "check": str(finding.get("kind") or label),
                "file": str(finding.get("file") or ""),
                "sheet": str(finding.get("sheet") or ""),
                "location": str(finding.get("label") or finding.get("sheet") or "整表数字分布"),
                "field": "",
                "detail": str(finding.get("rule") or finding.get("summary") or _jsonable(finding)),
                "evidence": _jsonable(finding),
                "prefilter": str(finding.get("prefilter") or ""),
            })
    return rows


def scan_request_tables(tables: list[dict[str, Any]], indexes: list[int]) -> dict[str, Any]:
    """Run the complete upstream numeric scan over selected request-local attachments."""
    selected: list[tuple[int, dict[str, Any], str]] = []
    unsupported: list[str] = []
    for index in indexes:
        if index < 0 or index >= len(tables):
            raise ValueError(f"附件序号无效：{index}")
        item = tables[index]
        name = Path(str(item.get("name") or f"attachment_{index + 1}")).name
        suffix = Path(name).suffix.lower()
        if suffix not in _SUPPORTED:
            unsupported.append(f"{name}（格式不在本次数值审查范围）")
            continue
        content = item.get("data") or b""
        if not isinstance(content, (bytes, bytearray)):
            raise ValueError(f"附件数据格式无效：{name}")
        if not content:
            raise ValueError(f"附件为空：{name}")
        selected.append((index, item, name))
    if not selected:
        detail = "；".join(unsupported[:8])
        raise ValueError("没有可审查的表格或结构化文档附件。" + (f"已跳过：{detail}" if detail else ""))

    scan_dir, write_html_report, distill_findings, groups = _load_engine()
    temp_base = Path(settings.DATA_DIR).resolve() / "tmp" / "numeric-audit"
    temp_base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="request-", dir=temp_base) as request_dir:
        input_dir = Path(request_dir) / "input"
        input_dir.mkdir()
        original_names: dict[str, str] = {}
        for index, item, name in selected:
            source = Path(name)
            # Prefixing the input index preserves duplicate original filenames.
            safe_stem = re.sub(r"[^\w.-]+", "_", source.stem, flags=re.UNICODE)[:100] or "attachment"
            temp_name = f"{index + 1:02d}_{safe_stem}{source.suffix.lower()}"
            target = input_dir / temp_name
            original_names[target.name] = name
            target.write_bytes(bytes(item["data"]))
        scan = scan_dir(str(input_dir), str(Path(request_dir) / "unused-output"),
                        write_md=False, write_html=False, write_json=False,
                        evidence=True, images=False, image_diagnostics=False,
                        profile="review")
        # Do not expose the ephemeral server path in reports or artifacts.
        scan["input_dir"] = "本轮上传附件（临时处理，扫描后清理）"
        _restore_input_names(scan, original_names, Path(request_dir))
        source_names = [name for _, _, name in selected]
        scan["name"] = "、".join(source_names[:3]) + (
            f" 等 {len(source_names)} 个文件" if len(source_names) > 3 else "")
        if unsupported:
            scan.setdefault("scan_errors", []).extend(unsupported)
            scan["scan_status"] = "partial"
            scan.setdefault("coverage", {}).setdefault("limitations", []).append(
                f"{len(unsupported)} 个所选附件格式不在数值审查范围")
        packet = distill_findings(scan)
        report_path = Path(request_dir) / "numeric-audit.html"
        write_html_report(scan, str(report_path))
        html_report = report_path.read_bytes()
        json_report = _jsonable(scan).encode("utf-8")
    return {
        "scan": scan,
        "packet": packet,
        "findings": flatten_scan(scan, groups),
        "html_report": html_report,
        "json_report": json_report,
        "upstream_version": "0.9.0",
        "upstream_commit": "8d116455b23e858651250d8ed23ba41c1a87ab0d",
    }
