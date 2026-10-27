"""T-003 database-setup specific checks.

Verifies the pieces T-003 is answerable for independently of the live cloud:
    - the `mongo.py` legacy shim re-exports the real builders from `mongodb.py`
      so `from app.db.mongo import pdfs_collection` (used by `app.rag.bm25`
      and `scripts/create_search_indexes.py`) keeps resolving.
    - the shim's getters surface the same `RuntimeError` text as the canonical
      module when `MONGODB_URI` is unset (callers can rely on one failure mode).
    - the SQLAlchemy models carry the fields and indexes T-003 promises.
    - `init_schema()` requires `POSTGRESQL_URI` and fails cleanly without it.

Live-connection coverage lives in `test_db_imports.py::test_mongo_ping_live`
and only runs when `MONGODB_URI` is exported.
"""

from __future__ import annotations

import pytest


def test_mongo_shim_reexports_canonical_module() -> None:
    """`app.db.mongo` is a shim; its callables delegate to `app.db.mongodb`.

    Both halves of the codebase (bm25-impl-1's `app.rag.bm25` imports from the
    shim; db-impl-1's `__init__.py` imports from the canonical module) must
    reach the SAME MongoClient at runtime so pool/timeout tweaks never diverge.
    """
    from app.db import mongo, mongodb

    assert callable(mongo.mongo_client)
    assert callable(mongo.mongo_db)
    assert callable(mongo.pdfs_collection)
    # Delegation sanity: the shim pulls builders out of `mongodb` directly.
    assert mongo.get_mongo_client is mongodb.get_mongo_client  # type: ignore[attr-defined]
    assert mongo.get_mongo_db is mongodb.get_mongo_db  # type: ignore[attr-defined]
    assert mongo.get_pdfs_collection is mongodb.get_pdfs_collection  # type: ignore[attr-defined]


def test_mongo_shim_raises_without_uri(monkeypatch) -> None:
    monkeypatch.delenv("MONGODB_URI", raising=False)
    from app.db import mongo, mongodb

    mongodb._client = None  # reset cached client so env is re-read

    with pytest.raises(RuntimeError, match="MONGODB_URI"):
        mongo.mongo_client()


def test_models_carry_t003_fields() -> None:
    from app.db.models import AuditLog, Document, SessionToken, User

    user_cols = {c.name for c in User.__table__.columns}
    assert {"id", "workspace_id", "password_hash", "created_at", "is_active"} <= user_cols

    doc_cols = {c.name for c in Document.__table__.columns}
    assert {
        "id",
        "user_id",
        "workspace_id",
        "sha256",
        "size_bytes",
        "gridfs_file_id",
        "status",
    } <= doc_cols

    sess_cols = {c.name for c in SessionToken.__table__.columns}
    assert {"id", "user_id", "workspace_id", "token_hash", "expires_at"} <= sess_cols

    audit_cols = {c.name for c in AuditLog.__table__.columns}
    assert {"id", "event_type", "created_at"} <= audit_cols


def test_documents_has_composite_indexes() -> None:
    from app.db.models import Document

    names = {idx.name for idx in Document.__table__.indexes}
    assert "ix_documents_workspace_status" in names
    assert "ix_documents_workspace_created" in names


def test_init_schema_requires_postgresql_uri(monkeypatch) -> None:
    monkeypatch.delenv("POSTGRESQL_URI", raising=False)

    from app.db import postgres

    postgres._engine = None
    postgres._SessionLocal = None

    with pytest.raises(RuntimeError, match="POSTGRESQL_URI"):
        postgres.init_schema()


def test_init_postgres_script_imports() -> None:
    """Smoke-test that the T-003 bootstrap script loads without side-effects."""
    import importlib

    mod = importlib.import_module("scripts.init_postgres")
    assert callable(mod.main)
