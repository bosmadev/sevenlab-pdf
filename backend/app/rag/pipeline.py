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

import json
import logging
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
    """Call the hybrid retriever owned by T-010. Isolates the import so a
    missing module during interleaved Ralph iterations surfaces as a clean
    SSE error, not a 500. If the retriever's signature does not accept
    `doc_id` yet (interim version), fall back to a workspace-wide search.
    """
    try:
        from app.rag.retriever import hybrid_search  # type: ignore[attr-defined]
    except ImportError as e:  # T-010 not landed yet
        raise LLMError(
            "hybrid_search not available — T-010 (hybrid-impl-1) must land first"
        ) from e

    try:
        result = hybrid_search(
            workspace_id=workspace_id, query=question, top_k=top_k, doc_id=doc_id
        )
    except TypeError:
        result = hybrid_search(workspace_id=workspace_id, query=question, top_k=top_k)
    if hasattr(result, "__await__"):
        result = await result  # type: ignore[assignment]
    return list(result)


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
