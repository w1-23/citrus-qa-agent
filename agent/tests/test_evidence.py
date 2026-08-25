# -*- coding: utf-8 -*-
"""v8.13-b4c 证据单例 + render_evidence 单元测试."""
from src.core.evidence import (
    render_evidence, evidence_id, renumber_refs,
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


# ── v9.2 引用编号统一重排（后端收敛：数字 1..k / W W1..Wm / H H1..Hp）──
def _cited():
    return [
        {"ref_id": "1", "source": "rag", "title": "P1"},
        {"ref_id": "2", "source": "rag", "title": "P2"},
        {"ref_id": "3", "source": "rag", "title": "P3"},
        {"ref_id": "4", "source": "rag", "title": "P4"},
        {"ref_id": "9", "source": "ucr", "title": "U9"},
        {"ref_id": "W1", "source": "web", "title": "W1"},
        {"ref_id": "W3", "source": "web", "title": "W3"},
        {"ref_id": "W7", "source": "web", "title": "W7"},
        {"ref_id": "H1", "source": "historical", "title": "H1"},
        {"ref_id": "H5", "source": "historical", "title": "H5"},
    ]


def test_renumber_refs_local_jumps_fixed():
    # 用户实测场景 [1][2][4][9]：只引 1,2,4,9 → 连续 1..4
    ans = "结论见 [1] 与 [2]，机制在 [4]（与 [9] 对比），另见 [1] 重复引用"
    out_ans, out_cited, remap = renumber_refs(ans, _cited())
    assert [r["ref_id"] for r in out_cited] == ["1", "2", "3", "4"], \
        str([r["ref_id"] for r in out_cited])
    assert remap == {"1": "1", "2": "2", "4": "3", "9": "4"}, str(remap)
    # 正文同步重写：未引用编号 [3] 保持原样（乱引防御），[9]→[4]
    assert "[4]" in out_ans and "[1]" in out_ans and "[2]" in out_ans
    assert out_ans.count("[9]") == 0 and out_ans.count("[4]") == 1


def test_renumber_refs_web_group_compact():
    # 前端 v8.17.3 只压 W 的场景：引用 W7, W1, W3 → 连续 W1..W3（出现顺序）
    ans = "联网见 [W7]，早期 [W1]，最新 [W3]"
    out_ans, out_cited, remap = renumber_refs(ans, _cited())
    assert [r["ref_id"] for r in out_cited] == ["W1", "W2", "W3"], \
        str([r["ref_id"] for r in out_cited])
    assert remap == {"W7": "W1", "W1": "W2", "W3": "W3"}, str(remap)
    assert out_ans == "联网见 [W1]，早期 [W2]，最新 [W3]", out_ans


def test_renumber_refs_history_group():
    ans = "历史见 [H5] 与 [H1]"
    out_ans, out_cited, remap = renumber_refs(ans, _cited())
    assert [r["ref_id"] for r in out_cited] == ["H1", "H2"], \
        str([r["ref_id"] for r in out_cited])
    assert out_ans == "历史见 [H1] 与 [H2]", out_ans


def test_renumber_refs_mixed_groups_independent():
    # 三组独立编号：数字/ W / H 各自从 1 起，互不串号
    ans = "本地 [9] 与联网 [W3]、历史 [H5] 并列；再看本地 [2]"
    out_ans, _, remap = renumber_refs(ans, _cited())
    assert remap == {"9": "1", "W3": "W1", "H5": "H1", "2": "2"}, str(remap)
    assert out_ans == "本地 [1] 与联网 [W1]、历史 [H1] 并列；再看本地 [2]", out_ans


def test_renumber_refs_defensive():
    c = _cited()
    a1, c1, r1 = renumber_refs("", c)
    assert a1 == "" and c1 == c and r1 == {}
    a2, c2, r2 = renumber_refs("无引用编号的自然回答", c)
    assert a2 == "无引用编号的自然回答" and c2 == c and r2 == {}
    a3, c3, r3 = renumber_refs("引用 [99] 和 [1]", c)
    assert c3 == [c[0]] and r3 == {"1": "1"}, (a3, [x["ref_id"] for x in c3])
    # 不存在的编号保持原文（[99] 未重写）
    assert a3 == "引用 [99] 和 [1]", a3