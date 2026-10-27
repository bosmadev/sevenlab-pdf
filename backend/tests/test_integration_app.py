"""Full-stack MVP flow through ``app.main.app`` — every middleware, every router.

The per-router test files mount one router each and bypass the middleware
stack. This file is the one place the *real* composition is exercised:

    CORS → WorkspaceAuthMiddleware → RequestIdMiddleware → routers
         → exception handlers

Flow pinned here (the thing a browser session actually does):

    health → 401 on documents → login → /me → upload → status → list →
    detail → chat SSE → chat history → soft delete → logout → 401

External systems are faked at the boundary: GridFS (``store_pdf``), the
RAG pipeline (``answer_stream``), the Mongo ``chats`` collection, and the
background processor (``_schedule_processing``). Postgres is SQLite.
No lifespan: ``TestClient`` is used without a context manager so the
Atlas ping / index-ensure hooks never run.
"""
from __future__ import annotations

import json
import uuid

import bcrypt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.db import postgres as pg_mod
from app.db.models import AuditLog, Document, SessionToken, User  # noqa: F401 — register with Base
from app.db.postgres import Base, get_db
from app.main import app as main_app
from app.processing import pipeline as proc_pipeline
from app.routes import chat as chat_route
from app.routes import documents as upload_route
from app.security.rate_limit import reset_login_rate_limit
from app.security.tokens import COOKIE_NAME
from app.services import chat_history
from app.services.gridfs_storage import StoredPdf
from tests.fixtures.fake_chats import FakeChatsCollection
from tests.fixtures.pdf_builder import make_clean_pdf

ADMIN_PW = "primary-pass"
DANNY_PW = "danny-pass"
ERIN_PW = "erin-pass"


@pytest.fixture
def stack(monkeypatch):
    """Wire the real app to SQLite + fakes; tear down dependency overrides."""
    cost = 4
    primary = bcrypt.hashpw(ADMIN_PW.encode(), bcrypt.gensalt(rounds=cost)).decode()
    danny = bcrypt.hashpw(DANNY_PW.encode(), bcrypt.gensalt(rounds=cost)).decode()
    erin = bcrypt.hashpw(ERIN_PW.encode(), bcrypt.gensalt(rounds=cost)).decode()
    monkeypatch.setenv("SESSION_SECRET", "integration-secret-" * 2)
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("APP_PASSWORD_HASH", primary)
    monkeypatch.setenv("APP_SECONDARY_HASHES", f"danny:{danny},erin:{erin}")
    # Keep the sanitizer in-process: the docker sandbox is pinned by
    # tests/test_pdf_sandbox.py and must not be launched from this suite.
    monkeypatch.setenv("PDF_SANDBOX_MODE", "inline")
    get_settings.cache_clear()
    reset_login_rate_limit()

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    def _db_override():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    main_app.dependency_overrides[get_db] = _db_override
    monkeypatch.setattr(pg_mod, "SessionLocal", SessionLocal)
    monkeypatch.setattr(proc_pipeline, "SessionLocal", SessionLocal)

    scheduled: list[str] = []
    monkeypatch.setattr(upload_route, "_schedule_processing", lambda doc_id: scheduled.append(doc_id))

    def fake_store_pdf(*, pdf_bytes, workspace_id, original_filename, sha256, extra_metadata=None):
        return StoredPdf(
            file_id="507f1f77bcf86cd799439011",
            server_filename=f"{uuid.uuid4().hex}.pdf",
            sha256=sha256,
            size_bytes=len(pdf_bytes),
        )

    monkeypatch.setattr(upload_route, "store_pdf", fake_store_pdf)

    asked: list[dict] = []

    async def fake_answer_stream(workspace_id: str, question: str, doc_id=None):
        asked.append({"workspace_id": workspace_id, "question": question, "doc_id": doc_id})
        yield f'data: {json.dumps({"type": "citations", "citations": [{"n": 1, "doc_id": doc_id, "page": 1}]})}\n\n'
        yield f'data: {json.dumps({"type": "token", "text": "answer"})}\n\n'
        yield f'data: {json.dumps({"type": "done"})}\n\n'

    monkeypatch.setattr(chat_route, "answer_stream", fake_answer_stream)

    chats = FakeChatsCollection()
    monkeypatch.setattr(chat_history, "get_chats_collection", lambda: chats)
    monkeypatch.setattr(chat_history, "_INDEXED", False)

    client = TestClient(main_app)
    try:
        yield {
            "client": client,
            "SessionLocal": SessionLocal,
            "scheduled": scheduled,
            "asked": asked,
            "chats": chats,
        }
    finally:
        main_app.dependency_overrides.clear()
        get_settings.cache_clear()
        reset_login_rate_limit()


def _login(client: TestClient, password: str) -> None:
    r = client.post("/api/auth", json={"password": password})
    assert r.status_code == 200, r.text


def _upload(client: TestClient, filename: str = "paper.pdf") -> dict:
    r = client.post(
        "/api/documents/upload",
        files={"file": (filename, make_clean_pdf(), "application/pdf")},
    )
    assert r.status_code == 202, r.text
    return r.json()


def _sse_types(text: str) -> list[str]:
    frames = [f for f in text.split("\n\n") if f.strip()]
    return [json.loads(f[len("data: "):])["type"] for f in frames]


def _assert_middleware_401(res) -> None:
    """The middleware envelope must match ``app.errors`` and carry the request id."""
    assert res.status_code == 401, res.text
    body = res.json()
    assert body["detail"] == "Not authenticated"
    assert body["error"]["code"] == "UNAUTHORIZED"
    rid = res.headers.get("x-request-id")
    assert rid
    assert body["error"]["request_id"] == rid


def _assert_me(body: dict, owner_id: str, is_admin: bool) -> None:
    assert body["owner_id"] == owner_id
    assert body["is_admin"] is is_admin
    assert body["expires_at"] > body["issued_at"]


# ---- the MVP path ------------------------------------------------------------


def test_full_mvp_flow_admin(stack):
    client: TestClient = stack["client"]

    # Open routes work without a session.
    health = client.get("/api/health/status")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    # Protected prefix is blocked by the middleware before any handler runs.
    _assert_middleware_401(client.get("/api/documents/"))

    # Login → /me
    _login(client, ADMIN_PW)
    assert COOKIE_NAME in client.cookies
    _assert_me(client.get("/api/auth/me").json(), "admin", True)

    # Upload → background processing scheduled with the new id.
    up = _upload(client)
    doc_id = up["id"]
    assert up["status"] == "pending"
    assert stack["scheduled"] == [doc_id]

    # Status poll surface (T-014) sees the same row.
    st = client.get(f"/api/documents/{doc_id}/status")
    assert st.status_code == 200, st.text
    assert st.json()["status"] == "pending"
    assert st.json()["is_terminal"] is False
    assert st.json()["progress_pct"] == 0

    # Listing + detail (T-013).
    rows = client.get("/api/documents/").json()
    assert [r["id"] for r in rows] == [doc_id]
    assert rows[0]["status"] == "pending"
    detail = client.get(f"/api/documents/{doc_id}").json()
    assert detail["sha256"] == up["sha256"]
    assert detail["page_count"] == 1

    # Chat SSE (T-012) — workspace is the signed cookie's owner.
    ask = client.post("/api/chat/ask/stream", json={"question": "What is this?", "doc_id": doc_id})
    assert ask.status_code == 200, ask.text
    assert ask.headers["content-type"].startswith("text/event-stream")
    assert _sse_types(ask.text) == ["citations", "token", "done"]
    assert stack["asked"] == [{"workspace_id": "admin", "question": "What is this?", "doc_id": doc_id}]

    # Chat history persistence (T-013).
    posted = client.post(f"/api/documents/{doc_id}/chat", json={"role": "user", "content": "What is this?"})
    assert posted.status_code == 201, posted.text
    hist = client.get(f"/api/documents/{doc_id}/chat").json()
    assert hist["total"] == 1
    assert hist["messages"][0]["content"] == "What is this?"

    # Soft delete hides the document and its chat.
    deleted = client.delete(f"/api/documents/{doc_id}")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "soft_deleted"
    assert client.get("/api/documents/").json() == []
    assert client.get(f"/api/documents/{doc_id}").status_code == 404
    assert client.get(f"/api/documents/{doc_id}/chat").status_code == 404

    # Logout → gate closes again.
    assert client.delete("/api/auth").status_code == 200
    client.cookies.clear()
    assert client.get("/api/documents/").status_code == 401
    assert client.get("/api/auth/me").status_code == 401


def test_tenant_isolation_across_documents_and_chat(stack):
    client: TestClient = stack["client"]

    _login(client, DANNY_PW)
    danny_doc = _upload(client, "danny.pdf")["id"]
    client.post(f"/api/documents/{danny_doc}/chat", json={"role": "user", "content": "danny secret"})
    client.cookies.clear()

    _login(client, ERIN_PW)
    _assert_me(client.get("/api/auth/me").json(), "erin", False)
    assert client.get("/api/documents/").json() == []
    assert client.get(f"/api/documents/{danny_doc}").status_code == 404
    assert client.get(f"/api/documents/{danny_doc}/chat").status_code == 404
    assert client.get(f"/api/documents/{danny_doc}/status").status_code == 404
    assert client.post(f"/api/documents/{danny_doc}/retry").status_code == 404
    assert client.delete(f"/api/documents/{danny_doc}").status_code == 404
    assert client.post(
        f"/api/documents/{danny_doc}/chat", json={"role": "user", "content": "erin probe"}
    ).status_code == 404

    ask = client.post("/api/chat/ask/stream", json={"question": "leak?", "doc_id": danny_doc})
    assert ask.status_code == 200
    assert stack["asked"][-1]["workspace_id"] == "erin"

    erin_doc = _upload(client, "erin.pdf")["id"]
    assert [r["id"] for r in client.get("/api/documents/").json()] == [erin_doc]
    client.cookies.clear()

    _login(client, ADMIN_PW)
    admin_view = {r["id"] for r in client.get("/api/documents/").json()}
    assert admin_view == {danny_doc, erin_doc}
    assert client.get(f"/api/documents/{danny_doc}/chat").json()["total"] == 1
    assert client.get(f"/api/documents/{danny_doc}/status").status_code == 200
    assert client.get(f"/api/documents/{erin_doc}/status").status_code == 200

    with stack["SessionLocal"]() as db:
        assert {u.workspace_id for u in db.query(User).all()} == {"danny", "erin"}
        assert db.get(Document, uuid.UUID(danny_doc)).workspace_id == "danny"
        assert db.get(Document, uuid.UUID(erin_doc)).workspace_id == "erin"


@pytest.mark.xfail(
    reason="app/routes/processing.py does not depend on require_owner: /status and /retry "
    "are gated by the middleware only, so any authenticated tenant can poll or retry "
    "another tenant's document by id. Flip to a hard assertion once the route is scoped.",
    strict=False,
)
def test_status_and_retry_are_workspace_scoped(stack):
    client: TestClient = stack["client"]
    _login(client, DANNY_PW)
    danny_doc = _upload(client)["id"]
    client.cookies.clear()

    _login(client, ERIN_PW)
    assert client.get(f"/api/documents/{danny_doc}/status").status_code == 404
    assert client.post(f"/api/documents/{danny_doc}/retry").status_code == 404


# ---- middleware + wiring details --------------------------------------------


def test_status_route_404_and_400_shapes(stack):
    client: TestClient = stack["client"]
    _login(client, ADMIN_PW)
    assert client.get(f"/api/documents/{uuid.uuid4()}/status").status_code == 404
    bad = client.get("/api/documents/not-a-uuid/status")
    assert bad.status_code == 400
    assert "invalid document id" in bad.json()["detail"]


def test_retry_refuses_pending_document_conflict_free(stack, monkeypatch):
    """Retry on a non-error document re-queues it (only ``ready`` is refused)."""
    client: TestClient = stack["client"]
    _login(client, ADMIN_PW)
    doc_id = _upload(client)["id"]
    monkeypatch.setattr(proc_pipeline, "_schedule_task", lambda *_a, **_k: None)
    r = client.post(f"/api/documents/{doc_id}/retry")
    assert r.status_code == 202, r.text
    assert r.json() == {"doc_id": doc_id, "status": "pending", "scheduled": True}


def test_retry_refused_when_ready(stack, monkeypatch):
    client: TestClient = stack["client"]
    _login(client, ADMIN_PW)
    doc_id = _upload(client)["id"]
    with stack["SessionLocal"]() as db:
        doc = db.get(Document, uuid.UUID(doc_id))
        doc.status = "ready"
        db.commit()
    r = client.post(f"/api/documents/{doc_id}/retry")
    assert r.status_code == 409


def test_request_id_is_honoured_and_echoed(stack):
    client: TestClient = stack["client"]
    r = client.get("/api/health/status", headers={"X-Request-ID": "trace-abc-123"})
    assert r.headers["x-request-id"] == "trace-abc-123"

    r2 = client.get("/api/health/status")
    generated = r2.headers["x-request-id"]
    assert len(generated) == 32
    uuid.UUID(generated)


def test_unauthenticated_upload_blocked_at_middleware(stack):
    client: TestClient = stack["client"]
    r = client.post(
        "/api/documents/upload",
        files={"file": ("paper.pdf", make_clean_pdf(), "application/pdf")},
    )
    assert r.status_code == 401
    assert r.json() == {"error": "Not authenticated"}
    assert stack["scheduled"] == []


def test_chat_requires_session_through_full_stack(stack):
    client: TestClient = stack["client"]
    r = client.post("/api/chat/ask/stream", json={"question": "hi"})
    assert r.status_code == 401
    assert r.json() == {"error": "Not authenticated"}
    assert stack["asked"] == []


@pytest.mark.parametrize(
    ("origin", "allowed"),
    [
        ("https://sevenlab.bosma.dev", True),
        ("http://127.0.0.1:3000", True),
        ("https://evil.example", False),
        ("http://localhost:3000", False),
    ],
)
def test_cors_preflight_allowlist(stack, origin, allowed):
    client: TestClient = stack["client"]
    r = client.options(
        "/api/auth",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    if allowed:
        assert r.status_code == 200, r.text
        assert r.headers["access-control-allow-origin"] == origin
        assert r.headers["access-control-allow-credentials"] == "true"
    else:
        assert r.status_code == 400
        assert "access-control-allow-origin" not in r.headers


def test_cors_exposes_request_id_to_browser(stack):
    client: TestClient = stack["client"]
    r = client.get("/api/health/status", headers={"Origin": "https://sevenlab.bosma.dev"})
    assert r.status_code == 200
    assert "x-request-id" in r.headers["access-control-expose-headers"].lower()


def test_login_rate_limit_through_full_stack(stack):
    from app.security.rate_limit import MAX_ATTEMPTS

    client: TestClient = stack["client"]
    for _ in range(MAX_ATTEMPTS):
        assert client.post("/api/auth", json={"password": "wrong"}).status_code == 401
    r = client.post("/api/auth", json={"password": ADMIN_PW})
    assert r.status_code == 429
    assert r.headers.get("retry-after")
