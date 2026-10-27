"""Idempotent PostgreSQL schema bootstrap for SEVENLAB-PDFCHAT (T-003).

Applies every Alembic migration under `backend/alembic/versions` so the
database reaches the head revision:
    - users         (workspace owners, bcrypt hash lives here, not in `.env`)
    - documents     (PDF upload metadata; bytes live in GridFS)
    - sessions      (HMAC opaque tokens, 7-day expiry)
    - audit_log     (ISO 27001 A.8.15 upload/access events)

Design notes:
    - Alembic is the source of truth. `app.db.postgres.Base.metadata` is kept
      in lock-step by `tests/test_alembic_migrations.py`; `alembic check`
      must stay clean.
    - Safe to re-run: at head it is a no-op.
    - Reads `POSTGRESQL_URI` from the environment or `backend/.env` via
      `app.config`, the same source the FastAPI process uses.

Run from `backend/`:
    python -m scripts.init_postgres
    # equivalent: alembic upgrade head
"""

from __future__ import annotations

import os
import sys

# Make `app.*` importable when invoked from the backend/ root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import inspect

from app.db import models  # noqa: F401 — register ORM classes with Base
from app.db.migrate import current_revision, head_revision, upgrade_to_head
from app.db.postgres import engine


def main() -> None:
    eng = engine()
    print(f"target:   {eng.url.render_as_string(hide_password=True)}")

    with eng.connect() as conn:
        before_rev = current_revision(conn)
        before = set(inspect(conn).get_table_names())
    print(f"revision before: {before_rev or '(none)'}")
    print(f"tables before:   {sorted(before) or '(none)'}")

    with eng.connect() as conn:
        upgrade_to_head(connection=conn)

    with eng.connect() as conn:
        after_rev = current_revision(conn)
        after = set(inspect(conn).get_table_names())
    created = sorted(after - before)
    print(f"revision after:  {after_rev} (head={head_revision()})")
    print(f"tables after:    {sorted(after)}")
    if created:
        print(f"created:         {created}")
    else:
        print("created:         (nothing new — schema was already at head)")
    print("done")


if __name__ == "__main__":
    main()
