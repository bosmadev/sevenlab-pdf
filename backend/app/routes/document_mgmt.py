"""Document management endpoints (T-013).

Isolated from ``documents.py`` (which the T-004 upload route owns) so the
two Ralph agents never contend for the same file. All routes mount under
``/api/documents`` from ``main.py``.

Endpoints
---------
GET    /api/documents/                       list the caller's documents
GET    /api/documents/{id}                   full metadata
DELETE /api/documents/{id}                   soft delete (sets ``deleted_at``)
GET    /api/documents/{id}/chat              chat history (chronological)
POST   /api/documents/{id}/chat              append a chat message

Admin owner (``is_admin``) may read any workspace. All other callers are
scoped by ``Owner.workspace_id``; a cross-workspace probe is answered with
HTTP 404 so document ids are not oracle-able.
"""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, Query, Request, status
from sqlalchemy.orm import Session

from app.db.postgres import get_db
from app.schemas.documents import (
    ChatHistoryResponse,
    ChatMessage,
    ChatMessageCreate,
    DocumentDeleteResponse,
    DocumentDetail,
    DocumentSummary,
)
from app.security.deps import Owner, require_owner
from app.services import chat_history, document_service
from app.services.audit import audit_document_deleted, extract_client_ip

log = logging.getLogger(__name__)

router = APIRouter()


def _summary(doc) -> DocumentSummary:
    return DocumentSummary(
        id=doc.id,
        original_filename=doc.original_filename,
        size_bytes=doc.size_bytes,
        page_count=doc.page_count,
        status=doc.status,
        chunk_count=doc.chunk_count,
        created_at=doc.created_at,
        processed_at=doc.processed_at,
    )


def _detail(doc) -> DocumentDetail:
    return DocumentDetail(
        id=doc.id,
        original_filename=doc.original_filename,
        sanitized_filename=doc.sanitized_filename,
        sha256=doc.sha256,
        mime_type=doc.mime_type,
        size_bytes=doc.size_bytes,
        page_count=doc.page_count,
        status=doc.status,
        status_message=doc.status_message,
        chunk_count=doc.chunk_count,
        ocr_tier_used=doc.ocr_tier_used,
        created_at=doc.created_at,
        processed_at=doc.processed_at,
    )


@router.get("/", response_model=list[DocumentSummary])
def list_documents(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    include_deleted: bool = Query(default=False),
    owner: Owner = Depends(require_owner),
    db: Session = Depends(get_db),
) -> list[DocumentSummary]:
    """List this owner's documents, newest first.

    Soft-deleted rows are hidden unless ``include_deleted=true``. Admin
    owners see every workspace's documents.
    """
    docs = document_service.list_documents(
        db,
        workspace_id=owner.workspace_id,
        is_admin=owner.is_admin,
        include_deleted=include_deleted,
        limit=limit,
        offset=offset,
    )
    return [_summary(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(
    document_id: uuid.UUID,
    owner: Owner = Depends(require_owner),
    db: Session = Depends(get_db),
) -> DocumentDetail:
    """Full metadata for one document (404 across workspace boundaries)."""
    doc = document_service.get_document(
        db,
        document_id=document_id,
        workspace_id=owner.workspace_id,
        is_admin=owner.is_admin,
    )
    return _detail(doc)


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    status_code=status.HTTP_200_OK,
)
def delete_document(
    document_id: uuid.UUID,
    request: Request,
    owner: Owner = Depends(require_owner),
    db: Session = Depends(get_db),
) -> DocumentDeleteResponse:
    """Soft delete the document — Phase-2 sweeper reclaims the bytes.

    Idempotent: deleting an already-deleted document returns the original
    ``deleted_at`` timestamp.
    """
    doc = document_service.soft_delete_document(
        db,
        document_id=document_id,
        workspace_id=owner.workspace_id,
        is_admin=owner.is_admin,
    )
    assert doc.deleted_at is not None  # soft_delete_document guarantees this
    doc_id, deleted_at = doc.id, doc.deleted_at

    # ISO 27001 A.8.15 — every delete carries actor workspace + client IP.
    audit_document_deleted(
        session=db,
        workspace_id=owner.workspace_id,
        document_id=doc_id,
        client_ip=extract_client_ip(request),
    )
    return DocumentDeleteResponse(id=doc_id, deleted_at=deleted_at)


@router.get("/{document_id}/chat", response_model=ChatHistoryResponse)
def get_chat_history(
    document_id: uuid.UUID,
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    owner: Owner = Depends(require_owner),
    db: Session = Depends(get_db),
) -> ChatHistoryResponse:
    """Return chronological chat messages for a document.

    Reading history requires the document itself to be visible to the
    caller — a soft-deleted document returns 404, which also hides its
    chat log from unauthorised peeks.
    """
    document_service.get_document(
        db,
        document_id=document_id,
        workspace_id=owner.workspace_id,
        is_admin=owner.is_admin,
    )
    rows, total = chat_history.list_messages(
        workspace_id=owner.workspace_id,
        document_id=str(document_id),
        limit=limit,
        offset=offset,
        is_admin=owner.is_admin,
    )
    return ChatHistoryResponse(
        document_id=document_id,
        messages=[ChatMessage(**r) for r in rows],
        total=total,
    )


@router.post(
    "/{document_id}/chat",
    response_model=ChatMessage,
    status_code=status.HTTP_201_CREATED,
)
def append_chat_message(
    document_id: uuid.UUID,
    body: ChatMessageCreate,
    owner: Owner = Depends(require_owner),
    db: Session = Depends(get_db),
) -> ChatMessage:
    """Append a chat message to the document's history.

    Called by the RAG pipeline (T-012) after each turn to persist both the
    user prompt and the assistant reply with citations.
    """
    document_service.get_document(
        db,
        document_id=document_id,
        workspace_id=owner.workspace_id,
        is_admin=owner.is_admin,
    )
    row = chat_history.append_message(
        workspace_id=owner.workspace_id,
        document_id=str(document_id),
        role=body.role,
        content=body.content,
        citations=[c.model_dump() for c in body.citations],
        metadata=body.metadata,
    )
    return ChatMessage(**row)
