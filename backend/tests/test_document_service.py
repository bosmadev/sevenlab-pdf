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
from app.services.gridfs_storage import PdfNotFoundError


# ---------------------------------------------------------------------------
# Mongo + GridFS purge stubs — soft delete calls both; no live Atlas here
# ---------------------------------------------------------------------------


class _PurgeRecorder:
    """Records the purge calls soft delete makes; flags flip them to failures."""

    def __init__(self) -> None:
        self.chunk_calls: list[tuple[str, str]] = []
        self.pdf_calls: list[tuple[str, str]] = []
        self.chunks_error: Exception | None = None
        self.pdf_error: Exception | None = None

    def delete_document_chunks(self, workspace_id: str, doc_id: str) -> int:
        self.chunk_calls.append((workspace_id, doc_id))
        if self.chunks_error is not None:
            raise self.chunks_error
        return 5

    def delete_pdf(self, file_id: str, *, workspace_id: str) -> None:
        self.pdf_calls.append((file_id, workspace_id))
        if self.pdf_error is not None:
            raise self.pdf_error


@pytest.fixture(autouse=True)
def purge(monkeypatch: pytest.MonkeyPatch) -> _PurgeRecorder:
    rec = _PurgeRecorder()
    monkeypatch.setattr("app.db.vector.delete_document_chunks", rec.delete_document_chunks)
    monkeypatch.setattr("app.services.gridfs_storage.delete_pdf", rec.delete_pdf)
    return rec


# ---------------------------------------------------------------------------
# In-memory SQLite — SQLAlchemy models use postgres UUID, so swap to String
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_session() -> Session:
    """Session bound to an in-memory SQLite.

    UUID columns use `app.db.types.UUIDType`, which transparently renders
    as CHAR(36) on SQLite, so no column-type patching is needed.
    """
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


# ---------------------------------------------------------------------------
# soft_delete_document — derived-data purge (Mongo chunks + GridFS bytes)
# ---------------------------------------------------------------------------


def test_soft_delete_purges_chunks_and_gridfs(
    db_session: Session, purge: _PurgeRecorder
) -> None:
    alice = _mk_user(db_session, "alice")
    doc = _mk_doc(db_session, alice)

    result = document_service.soft_delete_document(
        db_session, document_id=doc.id, workspace_id="alice"
    )

    assert purge.chunk_calls == [("alice", str(doc.id))]
    assert purge.pdf_calls == [(doc.gridfs_file_id, "alice")]
    assert result.chunk_count == 0
    assert result.deleted_at is not None


def test_soft_delete_purges_only_once(
    db_session: Session, purge: _PurgeRecorder
) -> None:
    alice = _mk_user(db_session, "alice")
    doc = _mk_doc(db_session, alice)

    document_service.soft_delete_document(
        db_session, document_id=doc.id, workspace_id="alice"
    )
    document_service.soft_delete_document(
        db_session, document_id=doc.id, workspace_id="alice"
    )

    assert len(purge.chunk_calls) == 1
    assert len(purge.pdf_calls) == 1


def test_soft_delete_survives_mongo_outage(
    db_session: Session, purge: _PurgeRecorder
) -> None:
    """A failed chunk purge must not roll back the delete the user saw."""
    alice = _mk_user(db_session, "alice")
    doc = _mk_doc(db_session, alice)
    purge.chunks_error = RuntimeError("mongo down")

    result = document_service.soft_delete_document(
        db_session, document_id=doc.id, workspace_id="alice"
    )

    assert result.deleted_at is not None
    assert result.status == "deleted"
    # chunk_count is only zeroed once the purge is confirmed
    assert result.chunk_count == 5
    # the GridFS purge still runs — the two steps are independent
    assert purge.pdf_calls == [(doc.gridfs_file_id, "alice")]


def test_soft_delete_tolerates_missing_gridfs_file(
    db_session: Session, purge: _PurgeRecorder
) -> None:
    alice = _mk_user(db_session, "alice")
    doc = _mk_doc(db_session, alice)
    purge.pdf_error = PdfNotFoundError("pdf not found")

    result = document_service.soft_delete_document(
        db_session, document_id=doc.id, workspace_id="alice"
    )

    assert result.deleted_at is not None
    assert result.chunk_count == 0


def test_soft_delete_survives_gridfs_outage(
    db_session: Session, purge: _PurgeRecorder
) -> None:
    alice = _mk_user(db_session, "alice")
    doc = _mk_doc(db_session, alice)
    purge.pdf_error = RuntimeError("gridfs down")

    result = document_service.soft_delete_document(
        db_session, document_id=doc.id, workspace_id="alice"
    )

    assert result.deleted_at is not None
    assert result.chunk_count == 0
