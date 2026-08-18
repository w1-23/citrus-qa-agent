# -*- coding: utf-8 -*-
"""v8.13-b4c 证据单例 + render_evidence 单元测试."""
from src.core.evidence import (
    render_evidence, evidence_id,
    EVIDENCE_TOOL_MAX_CHARS, EVIDENCE_SNIPPET_MAX_CHARS, EVIDENCE_RENDER_MAX_CHARS,
)


def test_evidence_id_stable_key():
    assert evidence_id({"paper_id": "P1", "chunk_index": 3}) == "P1:3"
    # 跨批次重复索引同 (paper_id, chunk_index) → 同一 evidence_id，可据此去重收敛
    assert evidence_id({"paper_id": "P1", "chunk_index": 3, "_global_idx": 99}) == "P1:3"


def test_evidence_id_hash_fallback():
    r = {"title": "Citrus", "text": "mechanism detail"}
    eid = evidence_id(r)
    assert eid.startswith("h:") and len(eid) == len("h:") + 16
    assert evidence_id({"title": "Citrus", "text": "mechanism detail"}) == eid
    assert evidence_id({"title": "Other", "text": "different"}) != eid


def test_render_evidence_field_priority():
    assert render_evidence({"text": "chunk 正文机制细节", "abstract": "摘要"}) == "chunk 正文机制细节"
    assert render_evidence({"abstract": "摘要", "snippet": "片段"}) == "摘要"
    assert render_evidence({"snippet": "片段"}) == "片段"
    assert render_evidence({}) == ""


def test_render_evidence_truncation_marker():
    r = {"text": "x" * 40}
    out = render_evidence(r, max_chars=10)
    assert out == "x" * 10 + " …[超长片段截断: 原文 40 字符]"
    # 未超长零截断
    assert render_evidence(r, max_chars=100) == "x" * 40


def test_render_evidence_named_budgets():
    # 具名常量各自对齐用途（工具 1000 / 账本 2000 / 回执材料 3000）
    assert (EVIDENCE_TOOL_MAX_CHARS, EVIDENCE_SNIPPET_MAX_CHARS,
            EVIDENCE_RENDER_MAX_CHARS) == (1000, 2000, 3000)