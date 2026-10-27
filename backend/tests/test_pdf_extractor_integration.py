"""End-to-end: real PDF bytes → extract → bridge → chunker.

These tests require pymupdf + pdfplumber to be installed. They are skipped
otherwise so CI boxes without the native deps still green.
"""
from __future__ import annotations

import io

import pytest

pymupdf = pytest.importorskip("pymupdf")
pytest.importorskip("pdfplumber")

from app.rag.chunker import Chunk, chunk_pages
from app.services.pdf_extractor import extract, to_chunker_pages


def _build_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        page.insert_text((72, 72), body)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def test_extract_two_page_pdf_round_trips_through_chunker():
    pdf_bytes = _build_pdf(
        [
            "Introduction. Alpha beta gamma delta epsilon zeta eta theta iota kappa.",
            "Second page carries different vocabulary: lambda mu nu xi omicron pi rho sigma.",
        ]
    )

    result = extract(pdf_bytes)
    assert result.page_count == 2
    assert len(result.pages) == 2
    assert result.truncated is False

    pages = to_chunker_pages(result)
    assert [p.page_num for p in pages] == [1, 2]

    chunks = chunk_pages(pages, doc_id="doc-smoke", chunk_size=400, chunk_overlap=40)
    assert chunks, "expected non-empty chunks from a two-page PDF"
    assert all(isinstance(c, Chunk) for c in chunks)
    page_nums = {c.page_num for c in chunks}
    assert page_nums == {1, 2}

    # Page citations must stay accurate — vocabulary unique to each page
    # should never appear in chunks tagged with the OTHER page's number.
    for c in chunks:
        if c.page_num == 1:
            assert "lambda" not in c.chunk_text
        else:
            assert "Alpha" not in c.chunk_text


def test_extract_empty_pdf_produces_no_chunks():
    pdf_bytes = _build_pdf([""])  # one page with no visible text
    result = extract(pdf_bytes)
    assert result.page_count == 1
    pages = to_chunker_pages(result)
    chunks = chunk_pages(pages, doc_id="doc-empty")
    assert chunks == []


def test_extract_single_page_metadata_fields():
    pdf_bytes = _build_pdf(["Single page body text for metadata assertion."])
    result = extract(pdf_bytes)
    pages = to_chunker_pages(result)
    chunks = chunk_pages(pages, doc_id="doc-meta")
    assert chunks
    first = chunks[0]
    assert first.doc_id == "doc-meta"
    assert first.page_num == 1
    assert first.char_count == len(first.chunk_text)
    assert first.created_at.endswith("Z")
