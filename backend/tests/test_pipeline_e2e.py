"""Upload → sanitize → GridFS → parse (+OCR) → chunk → embed → Mongo.

Layer 1 (always runs, no services): unit checks on the wiring that used to
be broken — the upload route's background hook, the no-loop fallback, the
GridFS id coercion and the table merge.

Layer 2 (opt-in): a real end-to-end run against a reachable Mongo. Enable
with::

    SEVENLAB_E2E_MONGO_URI=mongodb://127.0.0.1:27017 pytest tests/test_pipeline_e2e.py

It creates a throwaway database and drops it on teardown. Postgres is
replaced by in-memory SQLite, Cohere embeddings by deterministic vectors
(no Azure key needed) and the Tesseract / Azure OCR tiers are disabled so
only the free tier-1 proximity captions fire.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import io
import os
import threading
import uuid
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRESQL_URI", "sqlite:///:memory:")
os.environ.setdefault("MONGODB_URI", "mongodb://u:p@localhost/x")

from app.processing import pipeline  # noqa: E402
from app.routes import documents as docs_route  # noqa: E402

E2E_URI = os.environ.get("SEVENLAB_E2E_MONGO_URI", "").strip()


# ---------------------------------------------------------------------------
# Layer 1 — wiring units
# ---------------------------------------------------------------------------


def test_upload_background_hook_is_async_and_enqueues(monkeypatch):
    """Starlette runs sync background tasks off-loop; the hook must be async."""
    assert inspect.iscoroutinefunction(docs_route._schedule_processing)
    seen: list[uuid.UUID] = []
    monkeypatch.setattr(pipeline, "_schedule_task", lambda d: seen.append(d) or None)
    doc_id = uuid.uuid4()
    asyncio.run(docs_route._schedule_processing(str(doc_id)))
    assert seen == [doc_id]


def test_schedule_task_without_loop_runs_pipeline_on_thread(monkeypatch):
    done = threading.Event()
    got: dict[str, uuid.UUID] = {}

    def fake_run(doc_id, *, policy=None):
        got["doc_id"] = doc_id
        done.set()
        return pipeline.STATUS_READY

    monkeypatch.setattr(pipeline, "run_pipeline", fake_run)
    doc_id = uuid.uuid4()
    assert pipeline._schedule_task(doc_id) is None
    assert done.wait(5), "pipeline thread never ran"
    assert got["doc_id"] == doc_id


def test_load_pdf_bytes_uses_workspace_scoped_reader(monkeypatch):
    from app.services import gridfs_storage

    calls: dict[str, tuple] = {}

    def fake_read(file_id, *, workspace_id):
        calls["args"] = (file_id, workspace_id)
        return b"%PDF-x"

    monkeypatch.setattr(gridfs_storage, "read_pdf", fake_read)
    doc = SimpleNamespace(gridfs_file_id="0" * 24, workspace_id="ws-a")
    assert pipeline._load_pdf_bytes(doc) == b"%PDF-x"
    assert calls["args"] == ("0" * 24, "ws-a")


def test_chunker_pages_merges_tables_into_page_text():
    from app.services.pdf_extractor import ExtractionResult, PageTable, PageText

    res = ExtractionResult(
        pages=[PageText(page_num=1, text="Body")],
        tables=[PageTable(page_num=1, rows=[["a", "b"], ["1", "2"]])],
        page_count=1,
    )
    pages = pipeline._chunker_pages(res)
    assert len(pages) == 1
    assert pages[0].text == "Body\n\na | b\n1 | 2"


def test_chunker_pages_tolerates_stub_without_tables():
    stub = SimpleNamespace(pages=[SimpleNamespace(page_num=1, text="x")])
    assert pipeline._chunker_pages(stub) == stub.pages


# ---------------------------------------------------------------------------
# Layer 2 — real Mongo end-to-end (opt-in)
# ---------------------------------------------------------------------------


def _build_two_page_pdf() -> bytes:
    """Page 1: prose + caption + a small image (tier-1 OCR target). Page 2: prose."""
    import pymupdf
    from PIL import Image

    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text(
        (72, 72), "Quarterly revenue grew twelve percent on strong demand.", fontsize=11
    )
    p1.insert_text((72, 100), "Figure 1: Revenue by region, fiscal year 2026.", fontsize=10)
    img = Image.new("RGB", (120, 60), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    p1.insert_image(pymupdf.Rect(72, 110, 192, 170), stream=buf.getvalue())
    p2 = doc.new_page()
    p2.insert_text(
        (72, 72), "Second page body text about operating margins and cash flow.", fontsize=11
    )
    out = doc.tobytes()
    doc.close()
    return out


def _fake_embed_documents(texts, **_):
    dim = 2048
    out = []
    for t in texts:
        seed = hashlib.sha256(t.encode("utf-8")).digest()
        vec = [((seed[i % 32] / 255.0) - 0.5) for i in range(dim)]
        out.append(vec)
    return out


@pytest.mark.skipif(not E2E_URI, reason="set SEVENLAB_E2E_MONGO_URI to run")
def test_pipeline_end_to_end_mongo(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from app.config import get_settings
    from app.db import mongodb as mongo_mod
    from app.db import postgres as pg_mod
    from app.db.models import AuditLog, Document, SessionToken, User  # noqa: F401
    from app.db.postgres import Base, get_db
    from app.rag import embeddings as emb_mod
    from app.rag import vision as vision_mod
    from app.routes.processing import router as processing_router
    from app.security.deps import Owner, require_owner
    from app.services import audit as audit_mod
    from app.services.gridfs_storage import read_pdf

    db_name = f"sevenlab_e2e_{uuid.uuid4().hex[:8]}"
    monkeypatch.setenv("MONGODB_URI", E2E_URI)
    monkeypatch.setenv("MONGODB_DB_NAME", db_name)
    monkeypatch.setattr(mongo_mod, "_client", None)

    # Postgres → in-memory SQLite shared by the route, the pipeline and audit.
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

    # No network: deterministic embeddings, no Tesseract, no Azure.
    monkeypatch.setattr(emb_mod, "embed_documents", _fake_embed_documents)
    monkeypatch.setattr(vision_mod, "run_tesseract", lambda img, **_: ("", 0.0))
    monkeypatch.setattr(get_settings(), "ocr_enable_azure", False, raising=False)
    audit_events: list[tuple] = []
    for name in ("log_event", "log_audit"):
        if hasattr(audit_mod, name):
            monkeypatch.setattr(
                audit_mod, name, lambda *a, **k: audit_events.append((a, k))
            )

    # Capture the background hand-off instead of racing a real task.
    scheduled: list[uuid.UUID] = []
    monkeypatch.setattr(pipeline, "_schedule_task", lambda d: scheduled.append(d) or None)

    app = FastAPI()
    app.include_router(docs_route.router, prefix="/api/documents")
    app.include_router(processing_router, prefix="/api/documents")

    def _db_override():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _db_override
    app.dependency_overrides[require_owner] = lambda: Owner(owner_id="ws-e2e", is_admin=False)

    client = mongo_mod.get_mongo_client()
    try:
        pdf_bytes = _build_two_page_pdf()
        with TestClient(app) as http:
            resp = http.post(
                "/api/documents/upload",
                files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
            )
            assert resp.status_code == 202, resp.text
            body = resp.json()
            doc_id = body["id"]
            assert body["status"] == "pending"
            assert body["page_count"] == 2

            # Upload handed the id to the processor inside the loop.
            assert scheduled == [uuid.UUID(doc_id)]

            # Sanitized bytes landed in GridFS under the workspace.
            with SessionLocal() as db:
                row = db.get(Document, uuid.UUID(doc_id))
                gridfs_id = row.gridfs_file_id
            stored = read_pdf(gridfs_id, workspace_id="ws-e2e")
            assert stored[:5] == b"%PDF-"
            with pytest.raises(Exception):
                read_pdf(gridfs_id, workspace_id="ws-other")

            # Drive the pipeline synchronously (what the scheduled task does).
            assert pipeline.run_pipeline(doc_id) == pipeline.STATUS_READY

            status = http.get(f"/api/documents/{doc_id}/status").json()
            assert status["status"] == "ready", status
            assert status["is_terminal"] is True
            assert status["page_count"] == 2
            assert status["chunk_count"] >= 2
            assert status["ocr_tier_used"] == "proximity"

        coll = mongo_mod.get_pdfs_collection()
        rows = list(coll.find({"workspace_id": "ws-e2e", "doc_id": doc_id}))
        assert len(rows) == status["chunk_count"]
        assert {r["_id"] for r in rows} == {f"{doc_id}:{r['chunk_index']}" for r in rows}
        assert sorted(r["chunk_index"] for r in rows) == list(range(len(rows)))
        assert {r["page_num"] for r in rows} == {1, 2}
        assert all(len(r["embedding"]) == 2048 for r in rows)
        assert any("tier=proximity" in r["chunk_text"] for r in rows)
        assert any("operating margins" in r["chunk_text"] for r in rows)

        # Re-running (retry path) must not duplicate or orphan rows.
        assert pipeline.run_pipeline(doc_id) == pipeline.STATUS_READY
        assert coll.count_documents({"doc_id": doc_id}) == len(rows)
    finally:
        client.drop_database(db_name)
        client.close()
        monkeypatch.setattr(mongo_mod, "_client", None)
