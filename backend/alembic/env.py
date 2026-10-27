"""Alembic environment — binds migrations to `app.db.postgres.Base.metadata`.

URL resolution order (first non-empty wins):
    1. `config.attributes["sqlalchemy_url"]`  — set by app.db.migrate helpers
    2. `POSTGRESQL_URI` env var
    3. `app.config.get_settings().postgresql_uri` (reads backend/.env)
    4. `sqlalchemy.url` in alembic.ini (placeholder only)

A caller may also hand in a live `Connection` via
`config.attributes["connection"]`; the test-suite uses this to migrate an
in-memory SQLite engine, where every new connection is a fresh database.
"""

from __future__ import annotations

import os
import sys
from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Connection

from alembic import context

# Make `app.*` importable when the CLI is invoked from backend/.
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.db import models  # noqa: E402,F401 — register ORM tables on Base
from app.db.postgres import Base  # noqa: E402

config = context.config

if config.config_file_name is not None and not config.attributes.get("skip_logging"):
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _resolve_url() -> str:
    url = config.attributes.get("sqlalchemy_url")
    if url:
        return str(url)
    url = os.environ.get("POSTGRESQL_URI")
    if url:
        return url
    try:
        from app.config import get_settings

        url = get_settings().postgresql_uri
    except Exception:  # pragma: no cover — settings import is best-effort here
        url = ""
    if url:
        return url
    url = config.get_main_option("sqlalchemy.url") or ""
    if "placeholder" in url:
        raise RuntimeError(
            "POSTGRESQL_URI is missing — export it or set it in backend/.env "
            "before running alembic"
        )
    return url


def _configure(connection: Connection | None = None, url: str | None = None) -> None:
    dialect_name = connection.dialect.name if connection is not None else ""
    context.configure(
        connection=connection,
        url=url,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=False,
        render_as_batch=(dialect_name == "sqlite"),
        literal_binds=connection is None,
        dialect_opts={"paramstyle": "named"} if connection is None else None,
    )


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live connection (`alembic upgrade head --sql`)."""
    _configure(url=_resolve_url())
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = config.attributes.get("connection")
    if isinstance(connectable, Connection):
        _configure(connection=connectable)
        with context.begin_transaction():
            context.run_migrations()
        return

    section = config.get_section(config.config_ini_section) or {}
    section["sqlalchemy.url"] = _resolve_url()
    engine = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with engine.connect() as connection:
        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
