"""Idempotent PostgreSQL schema bootstrap for SEVENLAB-PDFCHAT (T-003).

Creates every table defined on `app.db.postgres.Base.metadata`:
    - users         (workspace owners, bcrypt hash lives here, not in `.env`)
    - documents     (PDF upload metadata; bytes live in GridFS)
    - sessions      (HMAC opaque tokens, 7-day expiry)
    - audit_log     (ISO 27001 A.8.15 upload/access events)

Design notes:
    - Phase 1 does not use Alembic — `create_all` is enough for a greenfield
      schema. Migrations get added when the first production change ships
      (then-live schema becomes the Alembic baseline).
    - Script is safe to re-run: `create_all` only emits DDL for missing tables.
    - Reads `POSTGRESQL_URI` from `backend/.env` via `app.config` so it uses
      the same connection string as the running FastAPI process.

Run from `backend/`:
    python -m scripts.init_postgres
"""

from __future__ import annotations

import os
import sys

# Make `app.*` importable when invoked from the backend/ root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import inspect

from app.db import models  # noqa: F401 — register ORM classes with Base
from app.db.postgres import Base, engine


def main() -> None:
    eng = engine()
    print(f"target: {eng.url.render_as_string(hide_password=True)}")

    inspector = inspect(eng)
    before = set(inspector.get_table_names())
    print(f"tables before: {sorted(before) or '(none)'}")

    Base.metadata.create_all(bind=eng)

    inspector = inspect(eng)
    after = set(inspector.get_table_names())
    created = sorted(after - before)
    print(f"tables after:  {sorted(after)}")
    if created:
        print(f"created:       {created}")
    else:
        print("created:       (nothing new — schema was already current)")
    print("done")


if __name__ == "__main__":
    main()
