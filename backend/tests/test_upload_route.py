"""HTTP tests for ``POST /api/documents/upload`` (T-004).

The validator, sanitizer and GridFS wrapper have their own unit tests; this
file pins the *route* contract — the thing the Next.js upload form talks to:

* 401 before any bytes are read when the session cookie is missing/forged
* 400 + ``{"code", "reason"}`` detail for every validator / sanitizer reject
* 413 when the streamed body passes the size cap
* 503 when GridFS is unavailable (no internal detail leaks)
* 202 on success: Document row persisted under the *cookie's* workspace,
  placeholder User auto-created once, background processing scheduled with
  the new document id, response carries the sanitized sha256 + page count

GridFS is replaced by an in-memory fake; Postgres by SQLite ``StaticPool``.
"""
from __future__ import annotations

import hashlib
import uuid

import bcrypt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import get_settings
from app.db.models import AuditLog, Document, SessionToken, User  # noqa: F401 — register with Base
from app.db.postgres import Base, get_db
from app.routes import documents as upload_route
from app.security.tokens import COOKIE_NAME, create_session_token
from app.services.gridfs_storage import StorageError, StoredPdf
from tests.fixtures.pdf_builder import (
    make_clean_pdf,
    make_encrypted_pdf,
    make_malicious_pdf,
    make_owner_encrypted_pdf,
    make_pdf_with_leading_junk,
)

UPLOAD_URL = "/api/documents/upload"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("SESSION_SECRET", "upload-route-secret-" * 2)
    monkeypatch.setenv("APP_ENV", "development")
    primary = bcrypt.hashpw(b"pw", bcrypt.gensalt(rounds=4)).decode()
    monkeypatch.setenv("APP_PASSWORD_HASH", primary)
    # Route contract tests run the sanitizer in-process; the docker worker
    # argv and failure modes are pinned by tests/test_pdf_sandbox.py.
    monkeypatch.setenv("PDF_SANDBOX_MODE", "inline")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def harness(monkeypatch):
    """Mount only the upload router with SQLite + fake GridFS + recorders."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

    stored: list[dict] = []
    scheduled: list[str] = []
    rejected: list[dict] = []
    accepted: list[dict] = []
    storage_failure = {"raise": False}

    def fake_store_pdf(*, pdf_bytes, workspace_id, original_filename, sha256, extra_metadata=None):
        if storage_failure["raise"]:
            raise StorageError("GridFS put failed: connection refused")
        stored.append(
            {
                "pdf_bytes": pdf_bytes,
                "workspace_id": workspace_id,
                "original_filename": original_filename,
                "sha256": sha256,
                "extra_metadata": dict(extra_metadata or {}),
            }
        )
        return StoredPdf(
            file_id="507f1f77bcf86cd799439011",
            server_filename=f"{uuid.uuid4().hex}.pdf",
            sha256=sha256,
            size_bytes=len(pdf_bytes),
        )

    monkeypatch.setattr(upload_route, "store_pdf", fake_store_pdf)
    monkeypatch.setattr(upload_route, "_schedule_processing", lambda doc_id: scheduled.append(doc_id))
    monkeypatch.setattr(upload_route, "audit_upload_rejected", lambda **kw: rejected.append(kw))
    monkeypatch.setattr(upload_route, "audit_upload_accepted", lambda **kw: accepted.append(kw))

    app = FastAPI()
    app.include_router(upload_route.router, prefix="/api/documents")

    def _db_override():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _db_override

    return {
        "client": TestClient(app),
        "SessionLocal": SessionLocal,
        "stored": stored,
        "scheduled": scheduled,
        "rejected": rejected,
        "accepted": accepted,
        "storage_failure": storage_failure,
    }


def _login(client: TestClient, owner_id: str = "alice") -> None:
    client.cookies.set(COOKIE_NAME, create_session_token(owner_id))


def _post(client: TestClient, data: bytes, *, filename: str = "paper.pdf", mime: str = "application/pdf"):
    return client.post(UPLOAD_URL, files={"file": (filename, data, mime)})


# ---- auth gate ---------------------------------------------------------------


def test_upload_requires_cookie(harness):
    r = _post(harness["client"], make_clean_pdf())
    assert r.status_code == 401
    assert harness["stored"] == []
    assert harness["scheduled"] == []


def test_upload_rejects_forged_cookie(harness):
    client = harness["client"]
    client.cookies.set(COOKIE_NAME, "v2.alice." + "ab" * 32 + "." + "00" * 32)
    r = _post(client, make_clean_pdf())
    assert r.status_code == 401
    assert harness["stored"] == []


# ---- validator rejects -------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "mime", "payload", "code"),
    [
        ("paper.pdf", "text/plain", make_clean_pdf(), "MIME_REJECTED"),
        ("paper.pdf", "application/pdf", b"this is not a pdf at all", "MAGIC_MISSING"),
        ("paper.pdf.exe", "application/pdf", make_clean_pdf(), "FILENAME_DOUBLE_EXT"),
        ("notes.txt", "application/pdf", make_clean_pdf(), "FILENAME_NOT_PDF"),
        ("paper.pdf", "application/pdf", b"", "SIZE_EMPTY"),
    ],
)
def test_validator_rejects_map_to_400_with_code(harness, filename, mime, payload, code):
    client = harness["client"]
    _login(client)
    r = _post(client, payload, filename=filename, mime=mime)
    assert r.status_code == 400, r.text
    assert r.json()["detail"]["code"] == code
    assert harness["stored"] == []
    assert harness["scheduled"] == []
    assert harness["rejected"][-1]["code"] == code
    assert harness["rejected"][-1]["workspace_id"] == "alice"


def test_mime_parameters_are_tolerated(harness):
    client = harness["client"]
    _login(client)
    r = _post(client, make_clean_pdf(), mime="application/pdf; charset=binary")
    assert r.status_code == 202, r.text
    assert r.json()["mime_type"] == "application/pdf"


def test_leading_junk_within_window_is_accepted(harness):
    client = harness["client"]
    _login(client)
    r = _post(client, make_pdf_with_leading_junk(512))
    assert r.status_code == 202, r.text


def test_oversize_body_is_413_before_validation(harness, monkeypatch):
    monkeypatch.setattr(upload_route, "MAX_PDF_SIZE_BYTES", 64)
    client = harness["client"]
    _login(client)
    r = _post(client, make_clean_pdf())
    assert r.status_code == 413
    assert r.json()["detail"]["code"] == "SIZE_LIMIT"
    assert harness["rejected"][-1]["code"] == "SIZE_LIMIT"
    assert harness["rejected"][-1]["size_bytes"] > 64
    assert harness["stored"] == []


# ---- sanitizer + storage -----------------------------------------------------


def test_encrypted_pdf_is_400_sanitize_failed(harness):
    client = harness["client"]
    _login(client)
    r = _post(client, make_encrypted_pdf())
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "SANITIZE_FAILED"
    # pdf_sandbox prefixes the reason ("sanitize failed: " inline,
    # "PdfSanitizationError: " docker) so match the sanitizer's message.
    assert "Password-protected PDFs are not allowed" in detail["reason"]
    assert harness["stored"] == []


def test_owner_only_encrypted_pdf_is_400_sanitize_failed(harness):
    # Owner password, empty user password: opens without PasswordError,
    # so only the explicit is_encrypted check rejects it.
    client = harness["client"]
    _login(client)
    r = _post(client, make_owner_encrypted_pdf())
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "SANITIZE_FAILED"
    # pdf_sandbox prefixes the reason ("sanitize failed: " inline,
    # "PdfSanitizationError: " docker) so match the sanitizer's message.
    assert "Password-protected PDFs are not allowed" in detail["reason"]
    assert harness["stored"] == []


def test_storage_outage_is_503_without_internal_detail(harness):
    harness["storage_failure"]["raise"] = True
    client = harness["client"]
    _login(client)
    r = _post(client, make_clean_pdf())
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert detail["code"] == "STORAGE_ERROR"
    assert "connection refused" not in r.text
    assert harness["scheduled"] == []
    with harness["SessionLocal"]() as db:
        assert db.query(Document).count() == 0


# ---- happy path --------------------------------------------------------------


def test_clean_upload_is_persisted_and_scheduled(harness):
    client = harness["client"]
    _login(client, "alice")
    r = _post(client, make_clean_pdf())
    assert r.status_code == 202, r.text
    body = r.json()

    assert body["status"] == "pending"
    assert body["page_count"] == 1
    assert body["stripped_keys"] == []
    assert body["mime_type"] == "application/pdf"
    assert body["sanitized_filename"] == "paper.pdf"
    uuid.UUID(body["id"])

    stored = harness["stored"]
    assert len(stored) == 1
    assert stored[0]["workspace_id"] == "alice"
    assert stored[0]["sha256"] == hashlib.sha256(stored[0]["pdf_bytes"]).hexdigest()
    assert body["sha256"] == stored[0]["sha256"]
    assert body["size_bytes"] == len(stored[0]["pdf_bytes"])

    assert harness["scheduled"] == [body["id"]]
    assert harness["accepted"][-1]["workspace_id"] == "alice"
    assert harness["accepted"][-1]["sha256"] == body["sha256"]

    with harness["SessionLocal"]() as db:
        doc = db.get(Document, uuid.UUID(body["id"]))
        assert doc is not None
        assert doc.workspace_id == "alice"
        assert doc.status == "pending"
        assert doc.gridfs_file_id == "507f1f77bcf86cd799439011"
        assert doc.original_filename == "paper.pdf"
        user = db.query(User).filter(User.workspace_id == "alice").one()
        assert doc.user_id == user.id


def test_malicious_pdf_is_stripped_before_storage(harness):
    client = harness["client"]
    _login(client)
    raw = make_malicious_pdf()
    r = _post(client, raw)
    assert r.status_code == 202, r.text
    body = r.json()

    stripped = body["stripped_keys"]
    assert stripped, "sanitizer must report the removed active-content keys"
    assert any("OpenAction" in k for k in stripped)
    assert any("Launch" in k for k in stripped)
    assert any("JavaScript" in k for k in stripped)

    stored = harness["stored"][0]
    assert stored["pdf_bytes"] != raw
    assert stored["extra_metadata"]["original_sha256"] == hashlib.sha256(raw).hexdigest()
    assert stored["extra_metadata"]["original_sha256"] != stored["sha256"]
    assert b"calc.exe" not in stored["pdf_bytes"]


def test_workspace_comes_from_cookie_and_user_row_is_reused(harness):
    client = harness["client"]
    _login(client, "danny")
    r1 = _post(client, make_clean_pdf(), filename="a.pdf")
    r2 = _post(client, make_clean_pdf(), filename="b.pdf")
    assert r1.status_code == r2.status_code == 202

    with harness["SessionLocal"]() as db:
        assert db.query(User).count() == 1
        assert db.query(User).one().workspace_id == "danny"
        docs = db.query(Document).all()
        assert len(docs) == 2
        assert {d.workspace_id for d in docs} == {"danny"}
    assert {s["workspace_id"] for s in harness["stored"]} == {"danny"}


def test_two_tenants_land_in_separate_workspaces(harness):
    client = harness["client"]
    _login(client, "alice")
    assert _post(client, make_clean_pdf()).status_code == 202
    _login(client, "bob")
    assert _post(client, make_clean_pdf()).status_code == 202

    with harness["SessionLocal"]() as db:
        by_ws = {d.workspace_id: d for d in db.query(Document).all()}
        assert set(by_ws) == {"alice", "bob"}
        assert by_ws["alice"].user_id != by_ws["bob"].user_id


def test_path_traversal_filename_is_sanitized(harness):
    client = harness["client"]
    _login(client)
    r = _post(client, make_clean_pdf(), filename="../../etc/passwd.pdf")
    assert r.status_code == 202, r.text
    safe = r.json()["sanitized_filename"]
    assert safe == "passwd.pdf"
    assert "/" not in safe and ".." not in safe
    assert harness["stored"][0]["original_filename"] == "passwd.pdf"


def test_unicode_filename_is_normalised(harness):
    client = harness["client"]
    _login(client)
    r = _post(client, make_clean_pdf(), filename="rapport été ∑.pdf")
    assert r.status_code == 202, r.text
    safe = r.json()["sanitized_filename"]
    assert safe.endswith(".pdf")
    assert safe.isascii()


# ---- sandbox wiring (ISO 27001 A.8.7) ----------------------------------------


def test_upload_hands_raw_bytes_to_sandbox_process(harness, monkeypatch):
    """The route must route attacker bytes through pdf_sandbox, never parse them itself."""
    from app.services.pdf_sandbox import SandboxResult

    raw = make_clean_pdf()
    calls: list[bytes] = []

    def fake_sandbox_process(pdf_bytes: bytes) -> SandboxResult:
        calls.append(pdf_bytes)
        return SandboxResult(
            sanitized_pdf=b"%PDF-1.4\n%sandboxed\n",
            sanitizer={},
            extraction={},
            stripped_keys=["/OpenAction"],
            pages=3,
            original_sha256="a" * 64,
            sanitized_sha256="b" * 64,
            mode="docker",
        )

    monkeypatch.setattr(upload_route, "sandbox_process", fake_sandbox_process)
    client = harness["client"]
    _login(client)
    r = _post(client, raw)
    assert r.status_code == 202, r.text
    assert calls == [raw]

    body = r.json()
    assert body["page_count"] == 3
    assert body["stripped_keys"] == ["/OpenAction"]
    assert body["sha256"] == "b" * 64

    stored = harness["stored"][0]
    assert stored["pdf_bytes"] == b"%PDF-1.4\n%sandboxed\n"
    assert stored["sha256"] == "b" * 64
    assert stored["extra_metadata"]["original_sha256"] == "a" * 64
    assert stored["extra_metadata"]["pages"] == 3
    assert harness["accepted"][-1]["pages"] == 3


def test_sandbox_error_is_400_sanitize_failed(harness, monkeypatch):
    from app.services.pdf_sandbox import SandboxError

    def boom(_pdf_bytes: bytes):
        raise SandboxError("sandbox timed out after 300s")

    monkeypatch.setattr(upload_route, "sandbox_process", boom)
    client = harness["client"]
    _login(client)
    r = _post(client, make_clean_pdf())
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "SANITIZE_FAILED"
    assert "timed out" in detail["reason"]
    assert harness["stored"] == []
    assert harness["scheduled"] == []
    assert harness["rejected"][-1]["code"] == "SANITIZE_FAILED"


def test_route_module_does_not_bind_in_process_sanitizer():
    """Regression pin: a future import of sanitize_pdf here re-opens A.8.7."""
    assert not hasattr(upload_route, "sanitize_pdf")
    assert upload_route.sandbox_process.__module__ == "app.services.pdf_sandbox"
