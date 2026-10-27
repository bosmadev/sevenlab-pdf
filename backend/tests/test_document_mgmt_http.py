"""T-013 HTTP-level tests — mount document_mgmt router only.

Bypasses the T-004 upload module (and its pikepdf requirement) by building a
tiny FastAPI app that mounts only our router. The ``get_db`` dependency is
overridden with an in-memory SQLite session; ``require_owner`` is overridden
to inject a fixed ``Owner``; ``chat_history.get_chats_collection`` is
monkeypatched to a dict-backed stub.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from bson import ObjectId
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import String, create_engine
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import AuditLog, Document, SessionToken, User
from app.db.postgres import Base, get_db
from app.routes.document_mgmt import router as mgmt_router
from app.security.deps import Owner, require_owner
from app.services import chat_history


# ---------------------------------------------------------------------------
# One-time: swap Postgres UUID columns for String so SQLite can host them
# ---------------------------------------------------------------------------
for table in (User.__table__, Document.__table__, SessionToken.__table__, AuditLog.__table__):
    for col in table.columns:
        if isinstance(col.type, PgUUID):
            col.type = String(36)


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def sort(self, *_, **__):
        return self

    def skip(self, n):
        self._rows = self._rows[n:]
        return self

    def limit(self, n):
        self._rows = self._rows[:n]
        return self

    def __iter__(self):
        return iter(self._rows)


class _FakeCol:
    def __init__(self):
        self._rows: list[dict] = []

    def _match(self, row, q):
        return all(row.get(k) == v for k, v in q.items())

    def create_index(self, *_, **__):
        return None

    def insert_one(self, row):
        row["_id"] = ObjectId()
        self._rows.append(dict(row))
        return type("R", (), {"inserted_id": row["_id"]})

    def count_documents(self, q):
        return sum(1 for r in self._rows if self._match(r, q))

    def find(self, q):
        matched = [r for r in self._rows if self._match(r, q)]
        matched.sort(key=lambda r: r.get("created_at"))
        return _FakeCursor(matched)

    def find_one(self, q, sort=None):
        matched = [r for r in self._rows if self._match(r, q)]
        return matched[0] if matched else None

    def delete_many(self, q):
        before = len(self._rows)
        self._rows = [r for r in self._rows if not self._match(r, q)]
        return type("R", (), {"deleted_count": before - len(self._rows)})


@pytest.fixture()
def client_setup(monkeypatch):
    # StaticPool keeps the single in-memory DB alive across sessions.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    fake_col = _FakeCol()
    monkeypatch.setattr(chat_history, "get_chats_collection", lambda: fake_col)
    chat_history._INDEXED = False

    app = FastAPI()
    app.include_router(mgmt_router, prefix="/api/documents")

    def _db_override():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    current_owner: dict[str, Owner] = {
        "owner": Owner(owner_id="alice", is_admin=False),
    }

    def _owner_override() -> Owner:
        return current_owner["owner"]

    app.dependency_overrides[get_db] = _db_override
    app.dependency_overrides[require_owner] = _owner_override

    # Seed Alice + one document
    seed = SessionLocal()
    alice = User(
        id=str(uuid.uuid4()),
        workspace_id="alice",
        password_hash="$2b$12$dummydummydummydummydummydummydummydummydummydummydu",
    )
    seed.add(alice)
    seed.commit()
    seed.refresh(alice)
    doc = Document(
        id=str(uuid.uuid4()),
        user_id=alice.id,
        workspace_id="alice",
        original_filename="paper.pdf",
        sanitized_filename=f"{uuid.uuid4().hex}.pdf",
        gridfs_file_id=uuid.uuid4().hex,
        sha256="a" * 64,
        size_bytes=4096,
        mime_type="application/pdf",
        page_count=4,
        status="ready",
        chunk_count=12,
    )
    seed.add(doc)
    seed.commit()
    seed.refresh(doc)
    doc_id = doc.id
    seed.close()

    client = TestClient(app)
    yield client, current_owner, doc_id


def test_list_returns_owner_documents(client_setup):
    client, _, doc_id = client_setup
    r = client.get("/api/documents/")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["id"] == str(doc_id)
    assert rows[0]["original_filename"] == "paper.pdf"


def test_get_returns_full_detail(client_setup):
    client, _, doc_id = client_setup
    r = client.get(f"/api/documents/{doc_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["sha256"] == "a" * 64
    assert body["chunk_count"] == 12


def test_get_unknown_id_is_404(client_setup):
    client, _, _ = client_setup
    r = client.get(f"/api/documents/{uuid.uuid4()}")
    assert r.status_code == 404


def test_get_bad_uuid_is_422(client_setup):
    """FastAPI path-param validation handles non-UUIDs before service runs."""
    client, _, _ = client_setup
    r = client.get("/api/documents/not-a-uuid")
    assert r.status_code == 422


def test_delete_is_soft(client_setup):
    client, _, doc_id = client_setup
    r = client.delete(f"/api/documents/{doc_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "soft_deleted"
    assert body["deleted_at"]

    # subsequent GET must now 404 (soft-deleted hidden)
    r2 = client.get(f"/api/documents/{doc_id}")
    assert r2.status_code == 404

    # list hides it unless include_deleted=true
    r3 = client.get("/api/documents/")
    assert r3.json() == []

    r4 = client.get("/api/documents/?include_deleted=true")
    assert len(r4.json()) == 1


def test_delete_is_idempotent(client_setup):
    client, _, doc_id = client_setup
    r1 = client.delete(f"/api/documents/{doc_id}")
    r2 = client.delete(f"/api/documents/{doc_id}")
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["deleted_at"] == r2.json()["deleted_at"]


def test_chat_roundtrip_through_http(client_setup):
    client, _, doc_id = client_setup
    r_post = client.post(
        f"/api/documents/{doc_id}/chat",
        json={"role": "user", "content": "What's on page 2?"},
    )
    assert r_post.status_code == 201
    posted = r_post.json()
    assert posted["role"] == "user"
    assert posted["content"] == "What's on page 2?"

    client.post(
        f"/api/documents/{doc_id}/chat",
        json={
            "role": "assistant",
            "content": "The abstract.",
            "citations": [{"page_num": 2, "chunk_id": "c7", "chunk_index": 0}],
        },
    )

    r_get = client.get(f"/api/documents/{doc_id}/chat")
    assert r_get.status_code == 200
    body = r_get.json()
    assert body["total"] == 2
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["messages"][1]["citations"][0]["page_num"] == 2


def test_chat_rejects_posting_to_soft_deleted(client_setup):
    client, _, doc_id = client_setup
    client.delete(f"/api/documents/{doc_id}")
    r = client.post(
        f"/api/documents/{doc_id}/chat",
        json={"role": "user", "content": "still there?"},
    )
    assert r.status_code == 404


def test_cross_workspace_access_404(client_setup):
    client, current_owner, doc_id = client_setup
    current_owner["owner"] = Owner(owner_id="bob", is_admin=False)
    r = client.get(f"/api/documents/{doc_id}")
    assert r.status_code == 404


def test_admin_sees_everyones_documents(client_setup):
    client, current_owner, doc_id = client_setup
    current_owner["owner"] = Owner(owner_id="admin", is_admin=True)
    r = client.get("/api/documents/")
    assert r.status_code == 200
    assert len(r.json()) == 1  # Alice's doc visible to admin


def test_list_pagination_caps(client_setup):
    client, _, _ = client_setup
    # bogus limit — Pydantic/Query validation rejects >500
    r = client.get("/api/documents/?limit=1000")
    assert r.status_code == 422
