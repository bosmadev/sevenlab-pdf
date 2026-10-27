"""RAG orchestrator (T-012): retrieve → assemble prompt → stream LLM.

Public surface:
    `answer_stream(workspace_id, question)` → async iterator of SSE frames.

Each yielded value is a complete `data: {...}\n\n` SSE frame. Frames:
    {"type": "citations", "citations": [...]}  — sent once before any token
    {"type": "token", "text": "<delta>"}       — one per LLM delta
    {"type": "done"}                            — final frame before close
    {"type": "error", "message": "..."}        — on any failure

The citation list carries `{doc_id, page, chunk_id, score, preview}` so the
Next.js chat UI (T-011) can render page-number links alongside the stream.

Hybrid search contract (owned by T-010, hybrid-impl-1):
    app.rag.retriever.hybrid_search(workspace_id, query, top_k) -> list[Chunk]
    where Chunk = {doc_id, chunk_id, chunk_text, page_num, score, filename?}

If T-010 has not landed yet, this module returns an SSE error frame instead
of a traceback, so the chat UI can show a graceful message.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from app.config import get_settings
from app.rag.llm import LLMError, stream_chat

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are the SevenLab PDF assistant. Answer the user's question using ONLY the "
    "numbered CONTEXT passages below. Cite the sources inline as [1], [2], etc. — "
    "the numbers correspond to the passage order. If the context does not contain "
    "the answer, say so plainly; never invent facts or citations. Keep answers "
    "concise and faithful to the source."
)


def _sse(event: dict[str, Any]) -> str:
    """Serialize one event as a complete SSE `data:` frame."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _format_context(chunks: list[dict[str, Any]], max_chars: int) -> tuple[str, list[dict[str, Any]]]:
    """Pack ranked chunks into the prompt, truncating at max_chars.

    Returns (context_block, citations) where citations mirror the included
    chunks 1:1 in order, so [n] in the model output maps to citations[n-1].
    """
    lines: list[str] = []
    citations: list[dict[str, Any]] = []
    used = 0
    for idx, c in enumerate(chunks, start=1):
        text = (c.get("chunk_text") or "").strip()
        if not text:
            continue
        page = c.get("page_num")
        filename = c.get("filename") or "document"
        header = f"[{idx}] (p.{page}, {filename})" if page is not None else f"[{idx}] ({filename})"
        snippet = f"{header}\n{text}\n"
        if used + len(snippet) > max_chars:
            break
        lines.append(snippet)
        used += len(snippet)

        preview = text[:200] + ("..." if len(text) > 200 else "")
        citations.append({
            "n": idx,
            "doc_id": c.get("doc_id"),
            "chunk_id": c.get("chunk_id"),
            "page": page,
            "filename": filename,
            "score": c.get("score"),
            "preview": preview,
        })
    return ("\n".join(lines) if lines else "(no relevant passages found)", citations)


async def _retrieve(
    workspace_id: str,
    question: str,
    top_k: int,
    doc_id: str | None = None,
) -> list[dict[str, Any]]:
    """Call the hybrid retriever (T-010). The import is isolated so a
    missing module surfaces as a clean SSE error, not a 500. ``doc_id`` is
    always forwarded — silently dropping it would widen a per-document
    chat to the whole workspace.
    """
    try:
        from app.rag.retriever import hybrid_search  # type: ignore[attr-defined]
    except ImportError as e:  # T-010 not landed yet
        raise LLMError(
            "hybrid_search not available — T-010 (hybrid-impl-1) must land first"
        ) from e

    result = hybrid_search(
        workspace_id=workspace_id, query=question, top_k=top_k, doc_id=doc_id
    )
    if hasattr(result, "__await__"):
        result = await result  # type: ignore[assignment]
    return list(result)


def _lookup_filenames(doc_ids: list[str]) -> dict[str, str]:
    """Map Mongo chunk ``doc_id`` values to the uploaded filename via Postgres.

    Chunks store only the parent ``doc_id`` (``documents.id`` as a UUID
    string); the human-readable name lives in ``documents.original_filename``.
    One indexed ``IN`` query per chat turn covers every cited document.
    """
    ids: list[uuid.UUID] = []
    for raw in doc_ids:
        try:
            ids.append(uuid.UUID(str(raw)))
        except (ValueError, TypeError, AttributeError):
            continue
    if not ids:
        return {}

    from sqlalchemy import select

    from app.db.models import Document
    from app.db.postgres import SessionLocal

    db = SessionLocal()
    try:
        rows = db.execute(
            select(Document.id, Document.original_filename, Document.sanitized_filename)
            .where(Document.id.in_(ids))
        ).all()
    finally:
        db.close()
    return {
        str(doc_id): (original or sanitized)
        for doc_id, original, sanitized in rows
        if original or sanitized
    }


async def _attach_filenames(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill ``filename`` on chunks that lack it. Best-effort: a Postgres
    failure degrades citations to the literal "document", never the turn."""
    missing = sorted(
        {str(c["doc_id"]) for c in chunks if c.get("doc_id") and not c.get("filename")}
    )
    if not missing:
        return chunks
    try:
        names = await asyncio.to_thread(_lookup_filenames, missing)
    except Exception as e:  # noqa: BLE001 — citations are cosmetic, the answer is not
        logger.warning("rag: filename lookup failed, citations fall back to 'document': %s", e)
        return chunks
    for c in chunks:
        if not c.get("filename"):
            name = names.get(str(c.get("doc_id")))
            if name:
                c["filename"] = name
    return chunks


async def answer_stream(
    workspace_id: str,
    question: str,
    doc_id: str | None = None,
) -> AsyncIterator[str]:
    """Yield SSE frames for one chat turn. Never raises — errors are frames."""
    s = get_settings()
    question = (question or "").strip()
    if not question:
        yield _sse({"type": "error", "message": "question is empty"})
        return

    # 1. Retrieve
    try:
        chunks = await _retrieve(workspace_id, question, s.rag_top_k, doc_id=doc_id)
    except Exception as e:
        logger.exception("rag: retrieval failed")
        yield _sse({"type": "error", "message": f"retrieval failed: {e}"})
        return

    chunks = await _attach_filenames(chunks)
    context_block, citations = _format_context(chunks, s.rag_max_context_chars)
    yield _sse({"type": "citations", "citations": citations})

    # 2. Assemble prompt
    user_prompt = (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer using only the context above. Cite as [n]."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # 3. Stream LLM
    try:
        async for token in stream_chat(messages):
            yield _sse({"type": "token", "text": token})
    except LLMError as e:
        logger.warning("rag: LLM stream failed: %s", e)
        yield _sse({"type": "error", "message": str(e)})
        return
    except Exception as e:
        logger.exception("rag: unexpected LLM failure")
        yield _sse({"type": "error", "message": f"llm error: {e}"})
        return

    yield _sse({"type": "done"})
