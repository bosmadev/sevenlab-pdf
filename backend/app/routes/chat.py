"""Chat / RAG endpoints (T-012).

POST /api/chat/ask/stream
    Body: {"question": "<str>", "doc_id"?: "<str>"}
    Response: text/event-stream with frames defined in ``app/rag/pipeline.py``.

Auth: HTTP-only ``app_session`` cookie, verified by HMAC via
``require_owner`` (``app/security/deps.py``). The ``workspace_id`` always
comes from the signed owner id, never from the request body — a tenant
cannot query another tenant's PDFs. Enforced at the retriever layer by
the ``workspace_id`` filter in Atlas ``$search`` / ``$vectorSearch``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.rag.pipeline import answer_stream
from app.security.deps import Owner, require_owner

logger = logging.getLogger(__name__)
router = APIRouter()


class AskRequest(BaseModel):
    """Chat question payload. ``doc_id`` is optional — when omitted the
    retriever searches every PDF in the workspace."""

    question: str = Field(min_length=1, max_length=4000)
    doc_id: str | None = Field(default=None, max_length=128)


SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # Tell nginx / Caddy to flush frames immediately rather than buffering.
    "X-Accel-Buffering": "no",
}


@router.post("/ask/stream")
async def ask_stream(
    body: AskRequest,
    owner: Owner = Depends(require_owner),
) -> StreamingResponse:
    """Stream the answer to one question as SSE frames.

    The workspace is taken from the signed session cookie, not the body,
    so a tenant can never read another tenant's PDFs even if they forge
    the payload. 401 is raised upstream by ``require_owner`` when the
    cookie is missing or the HMAC fails to verify.
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="question must not be empty",
        )

    logger.info(
        "chat.ask_stream workspace=%s doc_id=%s qlen=%d",
        owner.workspace_id,
        body.doc_id,
        len(question),
    )

    return StreamingResponse(
        answer_stream(
            workspace_id=owner.workspace_id,
            question=question,
            doc_id=body.doc_id,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
