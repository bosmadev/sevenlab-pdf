"""Processing status + retry endpoints (T-014).

Mounted alongside ``routes/documents.py`` at the same ``/api/documents``
prefix so the frontend can poll:

    GET  /api/documents/{doc_id}/status  → current pipeline state
    POST /api/documents/{doc_id}/retry   → re-run after a prior error

Both routes are scoped to the caller's workspace via ``require_owner`` +
``document_service.get_document``: a document owned by another workspace
(or soft-deleted) answers HTTP 404, the same convention as
``routes/document_mgmt.py``, so ids are not oracle-able across tenants. The
admin owner may read / retry any workspace.

The upload endpoint (T-004) calls :func:`app.processing.enqueue` directly;
it is not re-exported here to keep the HTTP surface read-only.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.postgres import get_db
from app.processing import get_status, pipeline
from app.security.deps import Owner, require_owner
from app.services import document_service

router = APIRouter()


def _parse_doc_id(doc_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(doc_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"invalid document id: {doc_id!r}",
        ) from exc


def _scoped_doc_id(doc_id: str, owner: Owner, db: Session) -> uuid.UUID:
    """Parse the id and prove the caller may see it, else 400 / 404."""
    doc_uuid = _parse_doc_id(doc_id)
    document_service.get_document(
        db,
        document_id=doc_uuid,
        workspace_id=owner.workspace_id,
        is_admin=owner.is_admin,
    )
    return doc_uuid


def _not_found(doc_id: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"document {doc_id} not found",
    )


@router.get("/{doc_id}/status")
async def get_document_status(
    doc_id: str,
    owner: Owner = Depends(require_owner),
    db: Session = Depends(get_db),
) -> dict:
    """Return the latest status snapshot for a document.

    Response shape::

        {
          "doc_id": "<uuid>",
          "status": "pending|parsing|chunking|embedding|ready|error",
          "message": "<nullable error message>",
          "progress_pct": 0..100,
          "chunk_count": int,
          "page_count": int | null,
          "ocr_tier_used": "proximity|tesseract|azure_cv" | null,
          "created_at": "<iso timestamp>",
          "processed_at": "<iso timestamp> | null",
          "is_terminal": bool
        }
    """
    doc_uuid = _scoped_doc_id(doc_id, owner, db)
    report = get_status(doc_uuid)
    if report is None:
        raise _not_found(doc_id)
    return report.as_dict()


@router.post("/{doc_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_processing(
    doc_id: str,
    owner: Owner = Depends(require_owner),
    db: Session = Depends(get_db),
) -> dict:
    """Re-queue a document for processing (only valid in the ``error`` state)."""
    doc_uuid = _scoped_doc_id(doc_id, owner, db)
    report = get_status(doc_uuid)
    if report is None:
        raise _not_found(doc_id)
    if report.status == "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="document already in terminal state 'ready'; nothing to retry",
        )

    scheduled = pipeline.retry(doc_uuid)
    if not scheduled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="retry refused by pipeline",
        )
    return {"doc_id": doc_id, "status": "pending", "scheduled": True}
