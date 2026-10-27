"""Document management service — list, get, soft-delete (T-013).

Workspace isolation: every query filters by ``workspace_id`` unless the
caller is the admin owner (``is_admin_owner``).

Soft delete sets ``Document.deleted_at`` to now. Original PDF bytes in
GridFS and chunk embeddings in the ``pdfs`` Mongo collection are left in
place; a Phase-2 cleanup task sweeps rows whose ``deleted_at`` is older
than the retention window.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document

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


def soft_delete_document(
    db: Session,
    *,
    document_id: str | uuid.UUID,
    workspace_id: str,
    is_admin: bool = False,
) -> Document:
    """Mark a document deleted. Idempotent — a second DELETE returns the row."""
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
    return doc
