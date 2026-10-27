"""Smoke tests — modules import, env-less constructors fail cleanly.

These tests do NOT require a live Postgres or Mongo; they verify module
layout and config-error paths. Live-connection tests live in
test_db_integration.py and are skipped without env vars.
"""

from __future__ import annotations

import os

import pytest


def test_models_module_imports() -> None:
    from app.db import models

    assert models.User.__tablename__ == "users"
    assert models.Document.__tablename__ == "documents"
    assert models.SessionToken.__tablename__ == "sessions"
    assert models.AuditLog.__tablename__ == "audit_log"


def test_postgres_module_imports_and_requires_uri(monkeypatch) -> None:
    monkeypatch.delenv("POSTGRESQL_URI", raising=False)

    from app.db import postgres

    # Reset cached engine so the next call re-reads env
    postgres._engine = None
    postgres._SessionLocal = None

    with pytest.raises(RuntimeError, match="POSTGRESQL_URI"):
        postgres.engine()


def test_mongodb_module_imports_and_requires_uri(monkeypatch) -> None:
    monkeypatch.delenv("MONGODB_URI", raising=False)

    from app.db import mongodb

    mongodb._client = None

    with pytest.raises(RuntimeError, match="MONGODB_URI"):
        mongodb.get_mongo_client()


def test_indices_constants() -> None:
    from app.db import indices

    assert indices.VECTOR_DIM == 2048, "must match Cohere V5-Pro default"
    assert indices.VECTOR_INDEX_NAME == "vector_index"
    assert indices.SEARCH_INDEX_NAME == "bm25_index"


@pytest.mark.skipif(
    not os.environ.get("MONGODB_URI"),
    reason="live MongoDB Atlas connection required",
)
def test_mongo_ping_live() -> None:
    from app.db.mongodb import ping

    result = ping()
    assert result.get("ok") is True, f"Atlas unreachable: {result}"
