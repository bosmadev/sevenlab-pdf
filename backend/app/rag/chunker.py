"""Text chunking for the RAG pipeline (T-006).

Splits per-page PDF text into overlapping chunks using LangChain's
RecursiveCharacterTextSplitter. Paragraph boundaries are preserved first
(``\\n\\n``), then line, sentence, word boundaries. Each emitted chunk
carries ``chunk_text``, ``page_num``, ``doc_id`` and ``created_at``
metadata so T-008 can upsert straight into MongoDB Atlas.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Iterable, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


# Paragraph-first separators. The empty string at the end lets the splitter
# fall back to a hard character cut when a run of text has no whitespace.
DEFAULT_SEPARATORS: list[str] = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass(frozen=True)
class Page:
    """One page of extracted PDF text."""

    page_num: int
    text: str


@dataclass(frozen=True)
class Chunk:
    """One chunk ready for embedding + MongoDB upsert."""

    chunk_text: str
    page_num: int
    doc_id: str
    created_at: str
    chunk_index: int = 0
    char_count: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
        keep_separator=True,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_pages(
    pages: Sequence[Page] | Iterable[Page],
    doc_id: str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    created_at: str | None = None,
) -> list[Chunk]:
    """Split per-page text into overlapping chunks with page-number metadata.

    Chunks are produced per page so each chunk maps unambiguously to a single
    source page. Overlap is handled inside a page; pages do not bleed into
    each other (page citations must stay accurate for T-012's RAG prompts).
    """
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    stamp = created_at or _utc_now_iso()

    splitter = _build_splitter(size, overlap)

    out: list[Chunk] = []
    for page in pages:
        text = (page.text or "").strip()
        if not text:
            continue
        parts = splitter.split_text(text)
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            out.append(
                Chunk(
                    chunk_text=part,
                    page_num=page.page_num,
                    doc_id=doc_id,
                    created_at=stamp,
                    chunk_index=i,
                    char_count=len(part),
                )
            )
    return out


def chunk_text(
    text: str,
    doc_id: str,
    *,
    page_num: int = 1,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Convenience wrapper for a single blob of text (one logical page)."""
    return chunk_pages(
        [Page(page_num=page_num, text=text)],
        doc_id=doc_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


__all__ = [
    "Page",
    "Chunk",
    "chunk_pages",
    "chunk_text",
    "DEFAULT_SEPARATORS",
]
