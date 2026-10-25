from __future__ import annotations

import io
import zipfile

import pandas as pd
from openpyxl import Workbook

from src.research.context import reset_request_tables, set_request_tables
from src.research.tools import (
    _audit_frame,
    _decode_csv,
    _json_arg,
    _load_frame,
    _metadata_candidate,
    _normalize_doi,
    audit_uploaded_data,
    export_paper_analysis,
)
from src.tools.draw_svg import draw_svg


def test_numeric_relationships_and_review_only_outliers():
    frame = pd.DataFrame({
        "control": [float(x) for x in range(10, 20)],
        "derived": [float(x + 4) for x in range(10, 20)],
        "yield": [1, 1.1, 1.2, 1.1, 1.3, 1.2, 1.4, 1.1, 1.2, 40],
        "note": ["ok"] * 9 + [None],
    })
    findings, _ = _audit_frame(frame, {}, [], [])
    checks = {row["check"] for row in findings}
    assert "constant_offset_columns" in checks
    assert "missing_values" in checks
    assert "iqr_outlier" in checks
    assert all(row["severity"] != "error" for row in findings)


def test_nonstandard_group_mapping_and_integer_summary_are_review_flags():
    frame = pd.DataFrame({
        "处理": ["CK", "CK", "T1", "T1", "T1"],
        "重复号": [1, 2, 1, 2, 3],
        "均值": [0.27, 1.0, 0.5, 1.0, 1.25],
        "SD": [0.4, 0.5, 0.2, 0.5, 0.6],
        "n": [4, 4, 4, 4, 4],
        "备注": ["unit: count"] * 5,
    })
    mapping = {"treatment": "处理", "replicate": "重复号", "mean": "均值", "sd": "SD", "n": "n"}
    findings, _ = _audit_frame(frame, mapping, ["处理"], ["均值"],
                               integer_data_confirmed=True, sd_kind="sample")
    checks = {row["check"] for row in findings}
    assert "unequal_group_size" in checks
    assert "unequal_replicates" in checks
    assert "integer_mean_inconsistent" in checks
    assert "integer_summary_inconsistent" in checks
    assert all(row["severity"] == "review" for row in findings if row["check"].startswith("integer_"))


def test_csv_and_tsv_are_decoded_without_rewriting_values():
    csv_frame, encoding, *_ = _decode_csv("处理组,样本,值\nA,001,1.25\n".encode("utf-8"))
    assert csv_frame.iloc[0, 1] == "001"
    assert encoding == "utf-8-sig"
    tsv_frame, *_ = _decode_csv("处理组\t重复\t值\nA\t1\t2.5\n".encode("utf-8"))
    assert tsv_frame.shape == (1, 3)


def test_xlsx_lists_and_reads_multiple_sheets_in_request_context():
    wb = Workbook()
    wb.active.title = "raw"
    wb.active.append(["group", "rep", "value"])
    wb.active.append(["A", 1, 2.4])
    copied = wb.create_sheet("copy")
    copied.append(["group", "rep", "value"])
    copied.append(["B", 1, 2.4])
    payload = io.BytesIO()
    wb.save(payload)
    token = set_request_tables([{"name": "trial.xlsx", "data": payload.getvalue()}])
    try:
        frame, meta = _load_frame(0)
        assert meta["sheets"] == ["raw", "copy"]
        assert frame.shape == (1, 3)
        copied_frame, copied_meta = _load_frame(0, sheet_name="copy")
        assert copied_meta["sheet"] == "copy"
        assert copied_frame.iloc[0]["value"] == 2.4
    finally:
        reset_request_tables(token)


def test_docx_structured_table_is_extractable():
    document_xml = b'''<?xml version="1.0" encoding="UTF-8"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>
    <w:tbl><w:tr><w:tc><w:p><w:r><w:t>group</w:t></w:r></w:p></w:tc>
    <w:tc><w:p><w:r><w:t>value</w:t></w:r></w:p></w:tc></w:tr>
    <w:tr><w:tc><w:p><w:r><w:t>A</w:t></w:r></w:p></w:tc>
    <w:tc><w:p><w:r><w:t>2.5</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
    </w:body></w:document>'''
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
    token = set_request_tables([{"name": "paper.docx", "data": payload.getvalue()}])
    try:
        frame, meta = _load_frame(0)
        assert meta["encoding"] == "docx-table"
        assert frame.iloc[0]["value"] == "2.5"
    finally:
        reset_request_tables(token)


def test_pdf_table_extraction_uses_structured_tables(monkeypatch):
    class Page:
        def extract_tables(self):
            return [[['group', 'value'], ['A', '2.5'], ['B', '3.5']]]

    class Pdf:
        pages = [Page()]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr("pdfplumber.open", lambda *_args, **_kwargs: Pdf())
    token = set_request_tables([{"name": "paper.pdf", "data": b"%PDF-fake"}])
    try:
        frame, meta = _load_frame(0)
        assert meta["encoding"] == "pdf-table"
        assert frame.shape == (2, 2)
        assert frame.iloc[1]["value"] == "3.5"
    finally:
        reset_request_tables(token)


def test_doi_normalization_and_registry_not_found(monkeypatch):
    assert _normalize_doi("https://doi.org/10.1234/ABC.1.") == "10.1234/abc.1"

    class Response:
        status_code = 404

    monkeypatch.setattr("httpx.get", lambda *args, **kwargs: Response())
    status, record, _ = _metadata_candidate("10.1234/missing")
    assert status == "not_found_in_checked_registries"
    assert record is None


def test_doi_network_failure_is_not_mislabeled_as_nonexistent(monkeypatch):
    def fail(*args, **kwargs):
        raise TimeoutError("network timeout")

    monkeypatch.setattr("httpx.get", fail)
    status, record, _ = _metadata_candidate("10.1234/example")
    assert status == "unverified"
    assert record is None


def test_data_audit_accepts_native_json_tool_arguments():
    assert _json_arg([0, 2], list, []) == [0, 2]
    assert _json_arg({"treatment": "处理"}, dict, {}) == {"treatment": "处理"}
    fields = audit_uploaded_data.args_schema.model_fields
    assert "list[int]" in str(fields["file_indexes_json"].annotation)
    assert "dict[str, str]" in str(fields["column_mapping_json"].annotation)


def test_export_paper_analysis_can_infer_title_from_markdown():
    fields = export_paper_analysis.args_schema.model_fields
    assert not fields["title"].is_required()
    assert not fields["analysis_markdown"].is_required()


def test_svg_tools_handle_missing_or_fenced_input_and_export_svg_only(monkeypatch):
    valid_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="30" viewBox="0 0 40 30"><title>测试流程图</title><rect width="40" height="30" fill="#fff"/></svg>'
    assert "SVG_INPUT_MISSING" in draw_svg.invoke({})
    assert "SVG 已通过安全校验" in draw_svg.invoke({"svg": f"```svg\n{valid_svg}\n```"})
    saved = []

    def fake_save(filename, content, mime):
        saved.append({"filename": filename, "content": content, "mime": mime})
        return f"/download/{len(saved)}"

    monkeypatch.setattr("src.research.tools._save_artifact", fake_save)
    result = export_paper_analysis.invoke({"svg_code": valid_svg})
    assert "已生成 HTML" in result and "下载 SVG" in result
    html_file = next(item for item in saved if item["filename"].endswith(".html"))
    svg_file = next(item for item in saved if item["filename"].endswith(".svg"))
    assert html_file["filename"].startswith("测试流程图")
    assert b"<svg" in html_file["content"]
    assert svg_file["content"].startswith(b"<svg")

