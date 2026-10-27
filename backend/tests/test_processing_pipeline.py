"""Unit tests for app.processing.pipeline (T-014).

Does not hit Postgres, Mongo, or Cohere. Mocks:
    - SessionLocal → in-memory SQLite with the same models
    - stage adapters (_load_pdf_bytes / _extract / _chunk / _embed_and_store)
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

import pytest

os.environ.setdefault("POSTGRESQL_URI", "sqlite:///:memory:")
os.environ.setdefault("MONGODB_URI", "mongodb://u:p@localhost/x")

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import postgres as pg_mod
from app.db.postgres import Base
from app.db.models import Document, User  # noqa: F401 — register with Base
from app.processing import pipeline


# ---------------------------------------------------------------------------
# In-memory DB fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_db(monkeypatch: pytest.MonkeyPatch):
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    monkeypatch.setattr(pg_mod, "SessionLocal", SessionLocal)
    monkeypatch.setattr(pipeline, "SessionLocal", SessionLocal)
    return SessionLocal


@pytest.fixture
def seeded_doc(memory_db) -> uuid.UUID:
    user_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    with memory_db() as db:
        user = User(
            id=user_id,
            workspace_id="ws-test",
            password_hash="x",
        )
        doc = Document(
            id=doc_id,
            user_id=user_id,
            workspace_id="ws-test",
            original_filename="t.pdf",
            sanitized_filename="t.pdf",
            gridfs_file_id="gf-1",
            sha256="0" * 64,
            size_bytes=1234,
            status="pending",
        )
        db.add(user)
        db.add(doc)
        db.commit()
    return doc_id


# ---------------------------------------------------------------------------
# Stage stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubPage:
    page_num: int
    text: str


@dataclass
class _StubExtraction:
    pages: list[_StubPage]
    page_count: int


@dataclass
class _StubChunk:
    chunk_index: int
    chunk_text: str
    page_num: int
    doc_id: str
    created_at: str = "2026-10-27T00:00:00Z"
    char_count: int = 0


def _happy_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    counters: dict[str, int] = {"parse": 0, "chunk": 0, "embed": 0}

    def fake_load(doc):
        counters["parse"] += 1
        return b"%PDF-stub"

    def fake_extract(pdf_bytes):
        return _StubExtraction(
            pages=[_StubPage(1, "hello"), _StubPage(2, "world")],
            page_count=2,
        )

    def fake_chunk(pages, doc_id):
        counters["chunk"] += 1
        return [
            _StubChunk(chunk_index=i, chunk_text=p.text, page_num=p.page_num, doc_id=doc_id)
            for i, p in enumerate(pages)
        ]

    def fake_embed_and_store(workspace_id, doc_id, chunks):
        counters["embed"] += 1
        return len(chunks)

    monkeypatch.setattr(pipeline, "_load_pdf_bytes", fake_load)
    monkeypatch.setattr(pipeline, "_extract", fake_extract)
    monkeypatch.setattr(pipeline, "_chunk", fake_chunk)
    monkeypatch.setattr(pipeline, "_embed_and_store", fake_embed_and_store)
    return counters


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_pipeline_happy_path(
    memory_db, seeded_doc: uuid.UUID, monkeypatch: pytest.MonkeyPatch
):
    counters = _happy_stubs(monkeypatch)

    final = pipeline.run_pipeline(seeded_doc)
    assert final == pipeline.STATUS_READY

    with memory_db() as db:
        doc = db.get(Document, seeded_doc)
        assert doc.status == "ready"
        assert doc.chunk_count == 2
        assert doc.page_count == 2
        assert doc.processed_at is not None
        assert doc.status_message is None

    assert counters == {"parse": 1, "chunk": 1, "embed": 1}


def test_status_report_shape(
    memory_db, seeded_doc: uuid.UUID, monkeypatch: pytest.MonkeyPatch
):
    _happy_stubs(monkeypatch)
    pipeline.run_pipeline(seeded_doc)
    report = pipeline.get_status(seeded_doc)
    assert report is not None
    data = report.as_dict()
    assert data["status"] == "ready"
    assert data["is_terminal"] is True
    assert data["progress_pct"] == 100
    assert data["chunk_count"] == 2
    assert data["doc_id"] == str(seeded_doc)


def test_status_report_pending_at_start(memory_db, seeded_doc: uuid.UUID):
    report = pipeline.get_status(seeded_doc)
    assert report is not None
    assert report.status == "pending"
    assert report.progress_pct == 0
    assert report.is_terminal is False


# ---------------------------------------------------------------------------
# Error + retry
# ---------------------------------------------------------------------------


def test_fatal_error_lands_in_error_state(
    memory_db, seeded_doc: uuid.UUID, monkeypatch: pytest.MonkeyPatch
):
    def boom_extract(_):
        raise pipeline.PipelineStageError("parsing", "pdf corrupt")

    monkeypatch.setattr(pipeline, "_load_pdf_bytes", lambda d: b"x")
    monkeypatch.setattr(pipeline, "_extract", boom_extract)

    final = pipeline.run_pipeline(seeded_doc)
    assert final == pipeline.STATUS_ERROR

    report = pipeline.get_status(seeded_doc)
    assert report is not None
    assert report.status == "error"
    assert "pdf corrupt" in (report.message or "")


def test_retryable_error_retries_then_succeeds(
    memory_db, seeded_doc: uuid.UUID, monkeypatch: pytest.MonkeyPatch
):
    counters = _happy_stubs(monkeypatch)
    calls = {"n": 0}

    def flaky_embed(workspace_id, doc_id, chunks):
        calls["n"] += 1
        if calls["n"] < 2:
            raise pipeline.PipelineStageError("embedding", "HTTP 429 rate limit")
        counters["embed"] += 1
        return len(chunks)

    monkeypatch.setattr(pipeline, "_embed_and_store", flaky_embed)
    # Zero backoff so the test runs instantly.
    monkeypatch.setattr(
        pipeline, "DEFAULT_POLICY", pipeline.RetryPolicy(
            attempts=3, initial_delay_s=0.0, backoff_multiplier=1.0,
        )
    )

    final = pipeline.run_pipeline(seeded_doc)
    assert final == pipeline.STATUS_READY
    assert calls["n"] == 2


def test_retry_endpoint_resets_status(
    memory_db, seeded_doc: uuid.UUID, monkeypatch: pytest.MonkeyPatch
):
    def boom(_):
        raise pipeline.PipelineStageError("parsing", "boom")

    monkeypatch.setattr(pipeline, "_load_pdf_bytes", lambda d: b"x")
    monkeypatch.setattr(pipeline, "_extract", boom)
    pipeline.run_pipeline(seeded_doc)
    assert pipeline.get_status(seeded_doc).status == "error"

    # Prevent the retry's enqueue from actually spawning work (no loop here).
    monkeypatch.setattr(pipeline, "_schedule_task", lambda *_a, **_k: None)
    assert pipeline.retry(seeded_doc) is True

    report = pipeline.get_status(seeded_doc)
    assert report.status == "pending"
    assert report.message is None


def test_retry_refused_on_ready(
    memory_db, seeded_doc: uuid.UUID, monkeypatch: pytest.MonkeyPatch
):
    _happy_stubs(monkeypatch)
    pipeline.run_pipeline(seeded_doc)
    monkeypatch.setattr(pipeline, "_schedule_task", lambda *_a, **_k: None)
    assert pipeline.retry(seeded_doc) is False


def test_retry_missing_doc_returns_false(memory_db):
    assert pipeline.retry(uuid.uuid4()) is False


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


def test_get_status_missing_doc_returns_none(memory_db):
    assert pipeline.get_status(uuid.uuid4()) is None


def test_set_status_rejects_unknown(memory_db, seeded_doc: uuid.UUID):
    with pytest.raises(ValueError):
        with memory_db() as db:
            pipeline._set_status(db, seeded_doc, status="bogus")


def test_is_retryable_detects_rate_limit():
    assert pipeline._is_retryable(RuntimeError("HTTP 429 Too Many Requests"))
    assert pipeline._is_retryable(RuntimeError("timeout waiting for upstream"))
    assert not pipeline._is_retryable(RuntimeError("KeyError 'foo'"))


# ---------------------------------------------------------------------------
# Parsing stage goes through the Docker sandbox (ISO 27001 A.8.7)
# ---------------------------------------------------------------------------


def _worker_payload() -> dict[str, Any]:
    import base64

    return {
        "sanitizer": {
            "original_sha256": "a" * 64,
            "sanitized_sha256": "b" * 64,
            "stripped_keys": [],
            "pages": 1,
            "was_encrypted": False,
            "original_bytes": 5,
            "sanitized_bytes": 5,
        },
        "sanitized_pdf_b64": base64.b64encode(b"%PDF-").decode("ascii"),
        "extraction": {
            "page_count": 1,
            "truncated": False,
            "pages": [{"page_num": 1, "text": "hello", "bbox_image_regions": []}],
            "tables": [{"page_num": 1, "rows": [["k", "v"]]}],
        },
    }


def test_extract_stage_routes_bytes_through_docker_sandbox(
    monkeypatch: pytest.MonkeyPatch,
):
    """In docker mode the parsing stage must spawn the worker, not PyMuPDF."""
    import json
    import subprocess

    monkeypatch.setenv("PDF_SANDBOX_MODE", "docker")
    argv_seen: list[list[str]] = []
    stdin_seen: list[bytes] = []

    def fake_run(argv, **kw):
        argv_seen.append(list(argv))
        stdin_seen.append(kw.get("input", b""))
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=json.dumps(_worker_payload()).encode(),
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = pipeline._extract(b"%PDF-1.4 untrusted")

    assert len(argv_seen) == 1
    assert argv_seen[0][:2] == ["docker", "run"]
    assert "--network=none" in argv_seen[0]
    assert stdin_seen == [b"%PDF-1.4 untrusted"]
    assert result.page_count == 1
    assert [p.text for p in result.pages] == ["hello"]
    assert result.tables[0].rows == [["k", "v"]]


def test_extract_stage_wraps_sandbox_failure_as_parsing_error(
    monkeypatch: pytest.MonkeyPatch,
):
    import subprocess

    monkeypatch.setenv("PDF_SANDBOX_MODE", "docker")

    def hang(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd=["docker"], timeout=1)

    monkeypatch.setattr(subprocess, "run", hang)

    with pytest.raises(pipeline.PipelineStageError) as excinfo:
        pipeline._extract(b"%PDF-1.4")
    assert excinfo.value.stage == "parsing"
    assert "timed out" in str(excinfo.value)
