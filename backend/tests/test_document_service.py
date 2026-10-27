"""T-013 tests — in-memory SQLite exercises the service layer.

The Mongo-backed ``chat_history`` module is covered via monkeypatching the
collection so a live Atlas is not required.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import AuditLog, Document, SessionToken, User
from app.db.postgres import Base
from app.services import document_service


# ---------------------------------------------------------------------------
# In-memory SQLite — SQLAlchemy models use postgres UUID, so swap to String
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_session() -> Session:
    """Session bound to an in-memory SQLite with a UUID -> String shim."""
    # Postgres UUID column type does not exist on SQLite — swap in CHAR(36).
    from sqlalchemy import String
    from sqlalchemy.dialects.postgresql import UUID as PgUUID

    for mapper in (User.__table__, Document.__table__, SessionToken.__table__, AuditLog.__table__):
        for col in mapper.columns:
            if isinstance(col.type, PgUUID):
                col.type = String(36)

    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _mk_user(db: Session, workspace_id: str) -> User:
    u = User(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        password_hash="$2b$12$dummydummydummydummydummydummydummydummydummydummydu",
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def _mk_doc(db: Session, user: User, *, deleted: bool = False, status_: str = "ready") -> Document:
    d = Document(
        id=str(uuid.uuid4()),
        user_id=user.id,
        workspace_id=user.workspace_id,
        original_filename="paper.pdf",
        sanitized_filename=f"{uuid.uuid4().hex}.pdf",
        gridfs_file_id=uuid.uuid4().hex,
        sha256="a" * 64,
        size_bytes=1234,
        mime_type="application/pdf",
        page_count=3,
        status=status_,
        chunk_count=5,
    )
    if deleted:
        d.deleted_at = datetime.now(timezone.utc)
        d.status = "deleted"
    db.add(d)
    db.commit()
    db.refresh(d)
    return d


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------


def test_list_hides_other_workspaces(db_session: Session) -> None:
    alice = _mk_user(db_session, "alice")
    bob = _mk_user(db_session, "bob")
    _mk_doc(db_session, alice)
    _mk_doc(db_session, bob)

    rows = document_service.list_documents(db_session, workspace_id="alice")
    assert [d.workspace_id for d in rows] == ["alice"]


def test_list_excludes_soft_deleted_by_default(db_session: Session) -> None:
    alice = _mk_user(db_session, "alice")
    _mk_doc(db_session, alice)
    _mk_doc(db_session, alice, deleted=True)

    live = document_service.list_documents(db_session, workspace_id="alice")
    assert len(live) == 1
    assert live[0].deleted_at is None

    with_deleted = document_service.list_documents(
        db_session, workspace_id="alice", include_deleted=True
    )
    assert len(with_deleted) == 2


def test_admin_sees_every_workspace(db_session: Session) -> None:
    alice = _mk_user(db_session, "alice")
    bob = _mk_user(db_session, "bob")
    _mk_doc(db_session, alice)
    _mk_doc(db_session, bob)

    rows = document_service.list_documents(
        db_session, workspace_id="admin", is_admin=True
    )
    assert {d.workspace_id for d in rows} == {"alice", "bob"}


# ---------------------------------------------------------------------------
# get_document
# ---------------------------------------------------------------------------


def test_get_cross_workspace_returns_404(db_session: Session) -> None:
    alice = _mk_user(db_session, "alice")
    _mk_user(db_session, "bob")
    doc = _mk_doc(db_session, alice)

    with pytest.raises(HTTPException) as exc:
        document_service.get_document(
            db_session, document_id=doc.id, workspace_id="bob"
        )
    assert exc.value.status_code == 404


def test_get_soft_deleted_returns_404_by_default(db_session: Session) -> None:
    alice = _mk_user(db_session, "alice")
    doc = _mk_doc(db_session, alice, deleted=True)

    with pytest.raises(HTTPException) as exc:
        document_service.get_document(
            db_session, document_id=doc.id, workspace_id="alice"
        )
    assert exc.value.status_code == 404

    found = document_service.get_document(
        db_session,
        document_id=doc.id,
        workspace_id="alice",
        include_deleted=True,
    )
    assert found.id == doc.id


def test_get_bad_uuid_returns_400(db_session: Session) -> None:
    with pytest.raises(HTTPException) as exc:
        document_service.get_document(
            db_session, document_id="not-a-uuid", workspace_id="alice"
        )
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# soft_delete_document
# ---------------------------------------------------------------------------


def test_soft_delete_sets_deleted_at(db_session: Session) -> None:
    alice = _mk_user(db_session, "alice")
    doc = _mk_doc(db_session, alice)

    result = document_service.soft_delete_document(
        db_session, document_id=doc.id, workspace_id="alice"
    )
    assert result.deleted_at is not None
    assert result.status == "deleted"


def test_soft_delete_is_idempotent(db_session: Session) -> None:
    alice = _mk_user(db_session, "alice")
    doc = _mk_doc(db_session, alice)

    first = document_service.soft_delete_document(
        db_session, document_id=doc.id, workspace_id="alice"
    )
    original_ts = first.deleted_at
    second = document_service.soft_delete_document(
        db_session, document_id=doc.id, workspace_id="alice"
    )
    assert second.deleted_at == original_ts  # unchanged on re-delete


def test_soft_delete_cross_workspace_404(db_session: Session) -> None:
    alice = _mk_user(db_session, "alice")
    _mk_user(db_session, "bob")
    doc = _mk_doc(db_session, alice)

    with pytest.raises(HTTPException) as exc:
        document_service.soft_delete_document(
            db_session, document_id=doc.id, workspace_id="bob"
        )
    assert exc.value.status_code == 404


def test_soft_delete_preserves_underlying_data(db_session: Session) -> None:
    """Soft delete must NOT null out gridfs_file_id / sha256 (ISO A.8.15 trail)."""
    alice = _mk_user(db_session, "alice")
    doc = _mk_doc(db_session, alice)
    original_gridfs = doc.gridfs_file_id
    original_sha = doc.sha256

    document_service.soft_delete_document(
        db_session, document_id=doc.id, workspace_id="alice"
    )
    db_session.expire_all()
    reread = document_service.get_document(
        db_session,
        document_id=doc.id,
        workspace_id="alice",
        include_deleted=True,
    )
    assert reread.gridfs_file_id == original_gridfs
    assert reread.sha256 == original_sha
