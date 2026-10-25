"""Extract tabular data locked inside supplementary PDF / Word files.

Many data inconsistencies are visible in numbers that are presented *in the
paper itself* — supplementary PDF tables, Word appendix tables — rather than in
a downloadable .xlsx source-data file. This module pulls those tables out
and normalizes them into the same ``{sheet_name: rows}`` shape the rest of
paperconan already consumes, so every existing numeric detector applies with no
change.

Scope: real ruled/structured tables only. It does NOT digitize data points off
bar charts or curves (pixel digitization introduces error that would itself trip
the arithmetic/duplication detectors), and it does not OCR scanned images.

The heavy parsers (pdfplumber, python-docx) are optional extras, imported lazily
so the base install (xlsx/csv/tsv) never depends on them.
"""
from __future__ import annotations

import os

from ._audit import _coerce_cell


def tables_to_sheets(stem, labeled_tables):
    """Normalize extracted tables into ``{sheet_name: rows}``.

    ``labeled_tables`` is a list of ``(label, table)`` where ``table`` is a list
    of rows of raw string/None cells (as pdfplumber / python-docx hand them
    over). Each table becomes one sheet named ``"<stem>!<label>"`` (e.g.
    ``"supp!p3_t1"``), so cross-sheet detectors stay meaningful and every
    finding is traceable back to the page/table it came from.

    Cells are coerced to int/float/text via the same conservative parser used
    for CSV input; ragged rows are padded to the widest row. Tables with no
    content at all are dropped.
    """
    sheets = {}
    for label, table in labeled_tables:
        rows = [[_coerce_cell(_as_text(c)) for c in row] for row in (table or [])]
        if not any(c is not None for row in rows for c in row):
            continue  # nothing in this table — drop it rather than emit noise
        maxc = max((len(r) for r in rows), default=0)
        for r in rows:
            if len(r) < maxc:
                r.extend([None] * (maxc - len(r)))
        sheets[f"{stem}!{label}"] = rows
    return sheets


def _as_text(cell):
    """Adapters hand us str or None; make that explicit for _coerce_cell."""
    if cell is None:
        return None
    return cell if isinstance(cell, str) else str(cell)


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def load_pdf_tables(path):
    """Extract every table from a PDF as ``{sheet_name: rows}``.

    Sheets are named ``<stem>!p<page>_t<table>`` (1-based page and table index).
    """
    try:
        import pdfplumber
    except ImportError as e:  # pragma: no cover - exercised via message only
        raise ImportError(
            "reading .pdf tables needs the optional extra: "
            "pip install 'paperconan[pdf]'"
        ) from e

    labeled = []
    with pdfplumber.open(path) as pdf:
        for pi, page in enumerate(pdf.pages, start=1):
            for ti, table in enumerate(page.extract_tables(), start=1):
                labeled.append((f"p{pi}_t{ti}", table))
    return tables_to_sheets(_stem(path), labeled)


def load_docx_tables(path):
    """Extract every table from a Word .docx as ``{sheet_name: rows}``.

    Sheets are named ``<stem>!t<table>`` (1-based table index).
    """
    try:
        import docx
    except ImportError as e:  # pragma: no cover - exercised via message only
        raise ImportError(
            "reading .docx tables needs the optional extra: "
            "pip install 'paperconan[docx]'"
        ) from e

    doc = docx.Document(path)
    labeled = []
    for ti, table in enumerate(doc.tables, start=1):
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        labeled.append((f"t{ti}", rows))
    return tables_to_sheets(_stem(path), labeled)
