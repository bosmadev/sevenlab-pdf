"""Tests for the extractor → chunker bridge (`to_chunker_pages`).

These tests exercise ONLY the pure-Python bridge logic. They build
`ExtractionResult` instances by hand so they run even when PyMuPDF /
pdfplumber are not installed — the native deps are only needed by the
`extract()` entry point, which these tests do not call.
"""
from __future__ import annotations

import importlib
import sys
import types

import pytest


def _import_bridge():
    """Import `pdf_extractor`, stubbing fitz/pdfplumber ONLY when missing.

    The extractor imports both native libs at module scope. For a pure-logic
    test we only need the bridge functions, so we stub the deps when they
    are not installed. We MUST attempt the real import first — a blind
    `sys.modules[name] = ModuleType(name)` would poison later tests that
    actually use `fitz.open()`.
    """
    for name in ("fitz", "pdfplumber"):
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except Exception:
            sys.modules[name] = types.ModuleType(name)
    return importlib.import_module("app.services.pdf_extractor")


ext = _import_bridge()
Page = importlib.import_module("app.rag.chunker").Page


def _make(pages, tables=()):
    return ext.ExtractionResult(
        pages=list(pages), tables=list(tables), page_count=len(pages), truncated=False
    )


def test_text_only_page_passes_through():
    pages = [ext.PageText(page_num=1, text="Hello world.")]
    out = ext.to_chunker_pages(_make(pages))
    assert out == [Page(page_num=1, text="Hello world.")]


def test_table_only_page_renders_as_pipe_rows():
    tables = [ext.PageTable(page_num=1, rows=[["a", "b"], ["1", "2"]])]
    out = ext.to_chunker_pages(_make([ext.PageText(page_num=1, text="")], tables))
    assert len(out) == 1
    assert out[0].page_num == 1
    assert out[0].text == "a | b\n1 | 2"


def test_mixed_page_concatenates_text_and_tables_with_blank_line():
    pages = [ext.PageText(page_num=2, text="Intro paragraph.")]
    tables = [ext.PageTable(page_num=2, rows=[["col1", "col2"], ["x", "y"]])]
    out = ext.to_chunker_pages(_make(pages, tables))
    assert out[0].text == "Intro paragraph.\n\ncol1 | col2\nx | y"


def test_empty_pages_are_dropped():
    pages = [
        ext.PageText(page_num=1, text=""),
        ext.PageText(page_num=2, text="   \n  "),
        ext.PageText(page_num=3, text="Real content."),
    ]
    out = ext.to_chunker_pages(_make(pages))
    assert [p.page_num for p in out] == [3]


def test_tables_sorted_by_page_number_match():
    pages = [
        ext.PageText(page_num=1, text="One"),
        ext.PageText(page_num=2, text="Two"),
    ]
    tables = [
        ext.PageTable(page_num=2, rows=[["t2"]]),
        ext.PageTable(page_num=1, rows=[["t1"]]),
    ]
    out = ext.to_chunker_pages(_make(pages, tables))
    assert out[0].page_num == 1 and "t1" in out[0].text and "t2" not in out[0].text
    assert out[1].page_num == 2 and "t2" in out[1].text and "t1" not in out[1].text


def test_table_with_all_empty_rows_is_dropped():
    pages = [ext.PageText(page_num=1, text="")]
    tables = [ext.PageTable(page_num=1, rows=[["", ""], [None, ""]])]  # type: ignore[list-item]
    out = ext.to_chunker_pages(_make(pages, tables))
    assert out == []


def test_multiple_tables_on_same_page_concatenate():
    pages = [ext.PageText(page_num=5, text="")]
    tables = [
        ext.PageTable(page_num=5, rows=[["a"]]),
        ext.PageTable(page_num=5, rows=[["b"]]),
    ]
    out = ext.to_chunker_pages(_make(pages, tables))
    assert len(out) == 1
    assert out[0].text == "a\n\nb"


def test_output_pages_use_chunker_page_type():
    out = ext.to_chunker_pages(
        _make([ext.PageText(page_num=1, text="Hi")])
    )
    assert all(isinstance(p, Page) for p in out)
