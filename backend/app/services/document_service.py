"""Document management service — list, get, soft-delete (T-013).

Workspace isolation: every query filters by ``workspace_id`` unless the
caller is the admin owner (``is_admin_owner``).

Soft delete sets ``Document.deleted_at`` to now, then purges the derived
data the row points at: chunk embeddings in the ``pdfs`` Mongo collection
(``app.db.vector.delete_document_chunks``) and the sanitized PDF bytes in
GridFS (``app.services.gridfs_storage.delete_pdf``). The Postgres row stays
as the ISO A.8.15 audit trail. A purge failure is logged, never raised: the
row is already marked deleted, and ``app.rag.retriever`` drops every chunk
whose ``doc_id`` is soft-deleted as a second guard.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import vector as _vector_mod
from app.db.models import Document
from app.services import gridfs_storage as _gridfs_mod

log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_uuid(doc_id: str | uuid.UUID) -> uuid.UUID:
    """Validate the id is a UUID. Returns a ``uuid.UUID`` the ORM accepts.

    Postgres' ``UUID`` column type binds a ``uuid.UUID`` directly; SQLite
    (used in tests) treats the column as ``String`` and auto-str-converts.
    """
    if isinstance(doc_id, uuid.UUID):
        return doc_id
    try:
        return uuid.UUID(str(doc_id))
    except (ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="document id is not a valid UUID",
        ) from exc


def _id_bind(doc_uuid: uuid.UUID) -> uuid.UUID | str:
    """Pick the correct bind shape for ``Document.id``.

    When the ORM column is swapped to ``String`` (SQLite tests) the stored
    rows are plain strings, so compare on ``str``. Postgres' native UUID
    column binds a ``uuid.UUID`` cleanly.
    """
    try:
        from sqlalchemy import String

        if isinstance(Document.__table__.c.id.type, String):
            return str(doc_uuid)
    except Exception:  # pragma: no cover — defensive
        pass
    return doc_uuid


def list_documents(
    db: Session,
    *,
    workspace_id: str,
    is_admin: bool = False,
    include_deleted: bool = False,
    limit: int = 100,
    offset: int = 0,
) -> list[Document]:
    """Return workspace documents, newest first; soft-deleted hidden by default."""
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))

    stmt = select(Document)
    if not is_admin:
        stmt = stmt.where(Document.workspace_id == workspace_id)
    if not include_deleted:
        stmt = stmt.where(Document.deleted_at.is_(None))
    stmt = stmt.order_by(Document.created_at.desc()).limit(limit).offset(offset)
    return list(db.execute(stmt).scalars().all())


def get_document(
    db: Session,
    *,
    document_id: str | uuid.UUID,
    workspace_id: str,
    is_admin: bool = False,
    include_deleted: bool = False,
) -> Document:
    """Return one document or raise 404 (own-row 404, never leak existence)."""
    doc_uuid = _as_uuid(document_id)
    stmt = select(Document).where(Document.id == _id_bind(doc_uuid))
    if not is_admin:
        stmt = stmt.where(Document.workspace_id == workspace_id)
    if not include_deleted:
        stmt = stmt.where(Document.deleted_at.is_(None))

    doc = db.execute(stmt).scalar_one_or_none()
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="document not found",
        )
    return doc


def _purge_derived_data(doc: Document) -> bool:
    """Drop the Mongo chunks and GridFS bytes behind a deleted row.

    Returns True when the chunk purge succeeded, so the caller can zero
    ``chunk_count``. Each step is independent and best-effort: a Mongo
    outage must not roll back a delete the user already saw succeed.
    """
    workspace_id = doc.workspace_id
    doc_id = str(doc.id)
    ctx = {"workspace_id": workspace_id, "document_id": doc_id}

    chunks_purged = False
    try:
        removed = _vector_mod.delete_document_chunks(workspace_id, doc_id)
        chunks_purged = True
        log.info("document.purge_chunks", extra={**ctx, "removed": removed})
    except Exception:  # noqa: BLE001 — pymongo raises many types
        log.exception("document.purge_chunks_failed", extra=ctx)

    try:
        _gridfs_mod.delete_pdf(doc.gridfs_file_id, workspace_id=workspace_id)
        log.info(
            "document.purge_pdf",
            extra={**ctx, "gridfs_file_id": doc.gridfs_file_id},
        )
    except _gridfs_mod.PdfNotFoundError:
        pass  # bytes already gone — purge is idempotent
    except Exception:  # noqa: BLE001
        log.exception("document.purge_pdf_failed", extra=ctx)

    return chunks_purged


def soft_delete_document(
    db: Session,
    *,
    document_id: str | uuid.UUID,
    workspace_id: str,
    is_admin: bool = False,
) -> Document:
    """Mark a document deleted and purge its chunks + PDF bytes.

    Idempotent — a second DELETE returns the row without purging again.
    """
    doc = get_document(
        db,
        document_id=document_id,
        workspace_id=workspace_id,
        is_admin=is_admin,
        include_deleted=True,
    )
    if doc.deleted_at is None:
        doc.deleted_at = _utcnow()
        doc.status = "deleted"
        db.add(doc)
        db.commit()
        db.refresh(doc)
        log.info(
            "document.soft_delete",
            extra={
                "workspace_id": doc.workspace_id,
                "document_id": str(doc.id),
                "sha256": doc.sha256,
            },
        )
        if _purge_derived_data(doc):
            doc.chunk_count = 0
            db.add(doc)
            db.commit()
            db.refresh(doc)
    return doc
