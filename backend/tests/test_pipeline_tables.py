"""T-004: pdfplumber tables must reach the chunker through the processing pipeline.

Requires pymupdf + pdfplumber (skipped otherwise, like the extractor
integration tests). Builds a real PDF with a ruled table so pdfplumber's
default line-based detector finds it, then drives the same two pipeline
helpers ``_stage_chunking`` uses.
"""
from __future__ import annotations

import io
import os

import pytest

pymupdf = pytest.importorskip("pymupdf")
pytest.importorskip("pdfplumber")

os.environ.setdefault("POSTGRESQL_URI", "sqlite:///:memory:")
os.environ.setdefault("MONGODB_URI", "mongodb://u:p@localhost/x")

from app.processing import pipeline  # noqa: E402
from app.services import pdf_extractor  # noqa: E402

TABLE_ROWS = [
    ["Account", "Balance"],
    ["Cash", "QX-7781-ZED"],
    ["Receivables", "1,234.56"],
]
PIPE_ROW = "Cash | QX-7781-ZED"


def _build_ruled_table_pdf(rows: list[list[str]], prose: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 90), prose, fontsize=11)
    x0, y0, col_w, row_h = 72, 144, 140, 24
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(
                x0 + c * col_w, y0 + r * row_h, x0 + (c + 1) * col_w, y0 + (r + 1) * row_h
            )
            page.draw_rect(rect, color=(0, 0, 0), width=1)
            page.insert_text((rect.x0 + 4, rect.y1 - 7), cell, fontsize=10)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _joined_text(chunks: list) -> str:
    return "\n".join(c.chunk_text for c in chunks)


def test_pipeline_chunks_carry_pdfplumber_table_rows():
    pdf_bytes = _build_ruled_table_pdf(TABLE_ROWS, prose="Balance sheet summary for the quarter.")
    extraction = pdf_extractor.extract(pdf_bytes)
    assert extraction.tables, "pdfplumber did not detect the ruled table"

    chunks = pipeline._chunk(pipeline._chunker_pages(extraction), doc_id="doc-tables")
    assert chunks
    assert PIPE_ROW in _joined_text(chunks)
    assert {c.page_num for c in chunks} == {1}

    # Control: the pre-fix text-only path never produced the pipe-delimited row.
    text_only = pipeline._chunk(list(extraction.pages), doc_id="doc-tables")
    assert PIPE_ROW not in _joined_text(text_only)


def test_chunker_pages_fails_closed_when_table_merge_raises(monkeypatch):
    def boom(_result):
        raise RuntimeError("bridge exploded")

    monkeypatch.setattr(pdf_extractor, "to_chunker_pages", boom)
    extraction = pdf_extractor.ExtractionResult(
        pages=[pdf_extractor.PageText(page_num=1, text="Body")],
        tables=[pdf_extractor.PageTable(page_num=1, rows=[["a", "b"]])],
        page_count=1,
    )
    with pytest.raises(pipeline.PipelineStageError) as exc_info:
        pipeline._chunker_pages(extraction)
    assert exc_info.value.stage == "chunking"
    assert "table merge failed" in str(exc_info.value)
    assert "bridge exploded" in str(exc_info.value)
