"""Bootstrap script: migrate Postgres schema (Alembic head) + Mongo indexes.

Usage:
    python -m app.db.init_db                # alembic upgrade head + indexes
    python -m app.db.init_db --wait         # also wait for Atlas Search READY
    python -m app.db.init_db --mongo-only
    python -m app.db.init_db --postgres-only
"""

from __future__ import annotations

import json
import logging
import sys

from app.db.indices import ensure_all_indexes
from app.db.migrate import current_revision, head_revision
from app.db.mongodb import ping as mongo_ping
from app.db.postgres import engine, init_schema

log = logging.getLogger(__name__)


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    wait = "--wait" in argv
    mongo_only = "--mongo-only" in argv
    pg_only = "--postgres-only" in argv

    out: dict = {}

    if not mongo_only:
        log.info("migrating PostgreSQL schema to alembic head")
        init_schema()
        with engine().connect() as conn:
            out["postgres"] = {
                "ok": True,
                "driver": conn.engine.name,
                "revision": current_revision(conn),
                "head": head_revision(),
            }

    if not pg_only:
        log.info("pinging MongoDB Atlas")
        out["mongo_ping"] = mongo_ping()
        if not out["mongo_ping"].get("ok"):
            print(json.dumps(out, default=str, indent=2), file=sys.stderr)
            return 2
        log.info("creating MongoDB indexes (regular + vector + search)")
        out["mongo_indexes"] = ensure_all_indexes(wait=wait)

    print(json.dumps(out, default=str, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
