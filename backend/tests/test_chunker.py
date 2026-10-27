"""Tests for app.rag.chunker (T-006)."""
from __future__ import annotations

import pytest

from app.rag.chunker import Chunk, Page, chunk_pages, chunk_text


SAMPLE_PARAGRAPHS = (
    "Introduction paragraph one. It has multiple sentences, and some commas, to "
    "ensure the splitter has fallback separators to use.\n\n"
    "Second paragraph with more detail about the subject matter at hand. "
    "This paragraph is deliberately longer so we exercise the overlap behaviour "
    "when the single paragraph exceeds the chunk size.\n\n"
    "Third paragraph closes out the page."
)


def test_single_page_chunking_preserves_page_number():
    chunks = chunk_pages(
        [Page(page_num=7, text=SAMPLE_PARAGRAPHS)],
        doc_id="doc-abc",
        chunk_size=120,
        chunk_overlap=30,
    )
    assert chunks, "expected at least one chunk"
    assert all(c.page_num == 7 for c in chunks)
    assert all(c.doc_id == "doc-abc" for c in chunks)
    assert all(c.char_count == len(c.chunk_text) for c in chunks)


def test_metadata_fields_populated():
    chunks = chunk_text("Hello world. " * 50, doc_id="doc-xyz", page_num=3)
    assert chunks
    first = chunks[0]
    assert isinstance(first, Chunk)
    assert first.page_num == 3
    assert first.doc_id == "doc-xyz"
    assert first.created_at.endswith("Z")
    d = first.as_dict()
    assert set(d.keys()) >= {"chunk_text", "page_num", "doc_id", "created_at"}


def test_overlap_actually_overlaps():
    text = "ABCDE " * 400  # ~2400 chars
    chunks = chunk_text(text, doc_id="d", chunk_size=200, chunk_overlap=50)
    assert len(chunks) >= 2
    # The splitter guarantees chunk size as an upper bound; overlap means
    # consecutive chunks share trailing/leading content for a smooth join.
    tail = chunks[0].chunk_text[-30:]
    head = chunks[1].chunk_text[:60]
    assert any(ch in head for ch in tail.split()), "expected shared tokens across chunks"


def test_pages_do_not_bleed_across_boundaries():
    pages = [
        Page(page_num=1, text="PageOneContent " * 20),
        Page(page_num=2, text="PageTwoContent " * 20),
    ]
    chunks = chunk_pages(pages, doc_id="d", chunk_size=200, chunk_overlap=40)
    page_one = [c for c in chunks if c.page_num == 1]
    page_two = [c for c in chunks if c.page_num == 2]
    assert page_one and page_two
    assert all("PageTwoContent" not in c.chunk_text for c in page_one)
    assert all("PageOneContent" not in c.chunk_text for c in page_two)


def test_empty_and_whitespace_pages_are_skipped():
    pages = [
        Page(page_num=1, text=""),
        Page(page_num=2, text="   \n\n  "),
        Page(page_num=3, text="Real content on page three."),
    ]
    chunks = chunk_pages(pages, doc_id="d")
    assert len(chunks) == 1
    assert chunks[0].page_num == 3


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_text("x" * 100, doc_id="d", chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        chunk_text("x" * 100, doc_id="d", chunk_size=0, chunk_overlap=0)


def test_chunk_index_is_monotonic_per_page():
    pages = [Page(page_num=p, text=SAMPLE_PARAGRAPHS) for p in (1, 2)]
    chunks = chunk_pages(pages, doc_id="d", chunk_size=100, chunk_overlap=20)
    for page_num in (1, 2):
        page_chunks = [c for c in chunks if c.page_num == page_num]
        indices = [c.chunk_index for c in page_chunks]
        assert indices == sorted(indices)
        assert indices[0] == 0
