"""HTTP-level tenant-isolation tests for ``app.routes.processing``.

Closes plan-scout-2 Gap B: ``GET /{id}/status`` and ``POST /{id}/retry``
must answer 404 for a document owned by another workspace, exactly like
``routes/document_mgmt.py`` does, so ids are not oracle-able across tenants.

Mounts only the processing router. ``require_owner`` is overridden with a
swappable ``Owner``; ``get_db`` and ``pipeline.SessionLocal`` share one
in-memory SQLite; ``pipeline._schedule_task`` is stubbed so a retry never
spawns work.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest

os.environ.setdefault("POSTGRESQL_URI", "sqlite:///:memory:")
os.environ.setdefault("MONGODB_URI", "mongodb://u:p@localhost/x")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import postgres as pg_mod
from app.db.models import AuditLog, Document, User  # noqa: F401 — register with Base
from app.db.postgres import Base, get_db
from app.processing import pipeline
from app.routes.processing import router as processing_router
from app.security.deps import Owner, require_owner

OWNER_A = "alice"
OWNER_B = "bob"


@pytest.fixture()
def client_setup(monkeypatch: pytest.MonkeyPatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(pg_mod, "SessionLocal", SessionLocal)
    monkeypatch.setattr(pipeline, "SessionLocal", SessionLocal)
    monkeypatch.setattr(pipeline, "_schedule_task", lambda *_a, **_k: None)

    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    with SessionLocal() as db:
        db.add(User(id=user_id, workspace_id=OWNER_A, password_hash="x"))
        db.add(
            Document(
                id=doc_id,
                user_id=user_id,
                workspace_id=OWNER_A,
                original_filename="a.pdf",
                sanitized_filename="a.pdf",
                gridfs_file_id="gf-a",
                sha256="0" * 64,
                size_bytes=1234,
                status="error",
                status_message="[parsing] boom",
            )
        )
        db.commit()

    app = FastAPI()
    app.include_router(processing_router, prefix="/api/documents")

    def _db_override():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    current_owner: dict[str, Owner] = {"owner": Owner(owner_id=OWNER_A, is_admin=False)}
    app.dependency_overrides[get_db] = _db_override
    app.dependency_overrides[require_owner] = lambda: current_owner["owner"]

    client = TestClient(app)
    yield client, current_owner, doc_id, SessionLocal


def _status_in_db(SessionLocal, doc_id: uuid.UUID) -> str:
    with SessionLocal() as db:
        return db.get(Document, doc_id).status


def test_owner_can_poll_status(client_setup):
    client, _, doc_id, _ = client_setup
    r = client.get(f"/api/documents/{doc_id}/status")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "error"
    assert body["doc_id"] == str(doc_id)


def test_cross_tenant_status_is_404(client_setup):
    client, current_owner, doc_id, _ = client_setup
    current_owner["owner"] = Owner(owner_id=OWNER_B, is_admin=False)
    r = client.get(f"/api/documents/{doc_id}/status")
    assert r.status_code == 404


def test_cross_tenant_retry_is_404_and_does_not_requeue(client_setup):
    client, current_owner, doc_id, SessionLocal = client_setup
    current_owner["owner"] = Owner(owner_id=OWNER_B, is_admin=False)
    r = client.post(f"/api/documents/{doc_id}/retry")
    assert r.status_code == 404
    assert _status_in_db(SessionLocal, doc_id) == "error"


def test_owner_retry_is_202_and_requeues(client_setup):
    client, _, doc_id, SessionLocal = client_setup
    r = client.post(f"/api/documents/{doc_id}/retry")
    assert r.status_code == 202
    assert r.json()["scheduled"] is True
    assert _status_in_db(SessionLocal, doc_id) == "pending"


def test_admin_can_poll_any_workspace(client_setup):
    client, current_owner, doc_id, _ = client_setup
    current_owner["owner"] = Owner(owner_id="admin", is_admin=True)
    r = client.get(f"/api/documents/{doc_id}/status")
    assert r.status_code == 200


def test_soft_deleted_doc_is_404_for_owner(client_setup):
    client, _, doc_id, SessionLocal = client_setup
    with SessionLocal() as db:
        db.get(Document, doc_id).deleted_at = datetime.now(timezone.utc)
        db.commit()
    assert client.get(f"/api/documents/{doc_id}/status").status_code == 404
    assert client.post(f"/api/documents/{doc_id}/retry").status_code == 404


def test_unknown_id_is_404(client_setup):
    client, _, _, _ = client_setup
    r = client.get(f"/api/documents/{uuid.uuid4()}/status")
    assert r.status_code == 404


def test_bad_uuid_is_400(client_setup):
    client, _, _, _ = client_setup
    r = client.get("/api/documents/not-a-uuid/status")
    assert r.status_code == 400
