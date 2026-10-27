"""PDF text + table extraction — PyMuPDF for text, pdfplumber for tables.

Decision 2026-10-27: hybrid approach. PyMuPDF (fitz) wins on text speed
and font handling; pdfplumber wins on table structure recovery.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from typing import Final

import fitz  # PyMuPDF
import pdfplumber

MAX_PAGES: Final[int] = int(os.getenv("MAX_PDF_PAGES", "500"))


class PdfExtractionError(Exception):
    pass


@dataclass
class PageText:
    page_num: int
    text: str
    bbox_image_regions: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class PageTable:
    page_num: int
    rows: list[list[str]]


@dataclass
class ExtractionResult:
    pages: list[PageText] = field(default_factory=list)
    tables: list[PageTable] = field(default_factory=list)
    page_count: int = 0
    truncated: bool = False


def _extract_text_pymupdf(pdf_bytes: bytes) -> tuple[list[PageText], int, bool]:
    pages: list[PageText] = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise PdfExtractionError(f"PyMuPDF open failed: {exc}") from exc

    total = doc.page_count
    truncated = total > MAX_PAGES
    limit = min(total, MAX_PAGES)
    try:
        for i in range(limit):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            image_bboxes: list[tuple[float, float, float, float]] = []
            try:
                for item in page.get_images(full=True):
                    xref = item[0]
                    for rect in page.get_image_rects(xref):
                        image_bboxes.append(tuple(rect))  # (x0,y0,x1,y1)
            except Exception:
                pass
            pages.append(
                PageText(page_num=i + 1, text=text, bbox_image_regions=image_bboxes)
            )
    finally:
        doc.close()
    return pages, total, truncated


def _extract_tables_pdfplumber(pdf_bytes: bytes, max_pages: int) -> list[PageTable]:
    tables: list[PageTable] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as plumb:
            for i, page in enumerate(plumb.pages[:max_pages]):
                try:
                    extracted = page.extract_tables()
                except Exception:
                    continue
                for raw_table in extracted or []:
                    rows = [
                        [(cell or "").strip() for cell in row]
                        for row in raw_table
                        if row
                    ]
                    if rows:
                        tables.append(PageTable(page_num=i + 1, rows=rows))
    except Exception as exc:  # noqa: BLE001
        raise PdfExtractionError(f"pdfplumber failed: {exc}") from exc
    return tables


def extract(pdf_bytes: bytes) -> ExtractionResult:
    """Run both extractors, return merged structured result."""
    pages, total, truncated = _extract_text_pymupdf(pdf_bytes)
    tables = _extract_tables_pdfplumber(pdf_bytes, max_pages=min(total, MAX_PAGES))
    return ExtractionResult(
        pages=pages, tables=tables, page_count=total, truncated=truncated
    )


def _format_table(rows: list[list[str]]) -> str:
    """Render a 2D cell matrix as pipe-delimited markdown-ish text.

    Empty rows and all-empty columns are dropped. Returned string has no
    trailing newline so it concatenates cleanly with surrounding page text.
    """
    cleaned: list[list[str]] = []
    for row in rows:
        trimmed = [(cell or "").strip() for cell in row]
        if any(trimmed):
            cleaned.append(trimmed)
    if not cleaned:
        return ""
    return "\n".join(" | ".join(row) for row in cleaned)


def to_chunker_pages(result: ExtractionResult) -> list:
    """Collapse ExtractionResult into the chunker's `Page` list.

    Each output `Page` carries the PyMuPDF text first, then any tables
    extracted for that page rendered as pipe-delimited rows. Pages with no
    text and no tables are dropped so the chunker does not have to re-filter.

    Lazy-imports `app.rag.chunker.Page` to avoid a circular dependency and
    to keep this module importable even when LangChain is not installed.
    """
    from app.rag.chunker import Page  # local import — avoids chunker ↔ extractor cycle

    tables_by_page: dict[int, list[PageTable]] = {}
    for tbl in result.tables:
        tables_by_page.setdefault(tbl.page_num, []).append(tbl)

    out: list = []
    for page in result.pages:
        parts: list[str] = []
        body = (page.text or "").strip()
        if body:
            parts.append(body)
        for tbl in tables_by_page.get(page.page_num, []):
            rendered = _format_table(tbl.rows)
            if rendered:
                parts.append(rendered)
        if not parts:
            continue
        out.append(Page(page_num=page.page_num, text="\n\n".join(parts)))
    return out
