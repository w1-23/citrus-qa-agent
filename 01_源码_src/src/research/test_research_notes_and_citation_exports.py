from __future__ import annotations

import sqlite3

import pytest
from src.core.tracing import set_session_id, set_user_id
from src.research import tools as research_tools


@pytest.fixture(autouse=True)
def clear_request_identity():
    yield
    set_user_id(None)
    set_session_id("")


def test_project_notes_are_account_scoped_across_sessions_and_migrate_owned_legacy_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(research_tools, "PROJECT_ROOT", tmp_path)
    state = tmp_path / "state"
    state.mkdir()

    sessions_path = state / "sessions.db"
    with sqlite3.connect(sessions_path) as conn:
        conn.execute("CREATE TABLE sessions(session_id TEXT PRIMARY KEY, owner_user_id INTEGER)")
        conn.executemany("INSERT INTO sessions(session_id,owner_user_id) VALUES(?,?)",
                         [("legacy-owned", 101), ("legacy-orphan", None)])

    # Simulate the old schema: records have only session ownership.
    artifact_path = state / "research_artifacts.sqlite3"
    with sqlite3.connect(artifact_path) as conn:
        conn.execute("""CREATE TABLE research_notes(
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            project_name TEXT NOT NULL, note TEXT NOT NULL, created_at REAL NOT NULL)""")
        conn.executemany("INSERT INTO research_notes(session_id,project_name,note,created_at) VALUES(?,?,?,?)",
                         [("legacy-owned", "Citrus", "owned legacy note", 1.0),
                          ("legacy-orphan", "Citrus", "orphan legacy note", 2.0)])

    set_user_id(101)
    set_session_id("new-session-a")
    recalled = research_tools.recall_research_notes.func("Citrus")
    assert "owned legacy note" in recalled
    assert "orphan legacy note" not in recalled

    saved = research_tools.save_research_note.func("Citrus", "new decision")
    assert "账号内课题" in saved
    set_session_id("new-session-b")
    assert "new decision" in research_tools.recall_research_notes.func("Citrus")

    set_user_id(202)
    assert "没有课题" in research_tools.recall_research_notes.func("Citrus")
    with sqlite3.connect(artifact_path) as conn:
        rows = dict((sid, owner) for sid, owner in conn.execute(
            "SELECT session_id,owner_user_id FROM research_notes"))
    assert rows["legacy-owned"] == 101
    assert rows["legacy-orphan"] is None
    assert rows["new-session-a"] == 101


def test_ris_and_bibtex_exports_only_emit_supplied_metadata():
    rows = [{
        "title": "Citrus response to heat",
        "authors": ["A. Author", "B. Author"],
        "journal": "Plant Science",
        "year": "2025",
        "doi": "10.1000/example_1",
        "url": "https://example.org/paper",
        "claim": "Heat response varies",
        "evidence": "Reported in the results",
    }]

    ris, extension, mime = research_tools._render_reference_export(rows, "ris")
    assert extension == ".ris"
    assert mime.startswith("application/x-research-info-systems")
    assert "TY  - JOUR" in ris
    assert "DO  - 10.1000/example_1" in ris
    assert "AU  - A. Author" in ris

    bib, extension, mime = research_tools._render_reference_export(rows, "bibtex")
    assert extension == ".bib"
    assert mime.startswith("application/x-bibtex")
    assert "title = {Citrus response to heat}" in bib
    assert "doi = {10.1000/example\\_1}" in bib
    assert "abstract =" not in bib


def test_legacy_unregistered_audit_tools_are_removed():
    assert not hasattr(research_tools, "audit_uploaded_table")
    assert not hasattr(research_tools, "audit_uploaded_data_bundle")
