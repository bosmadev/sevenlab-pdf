"""PostgreSQL connection + SQLAlchemy session factory.

Env vars:
    POSTGRESQL_URI — SQLAlchemy URL, e.g. postgresql+psycopg://user:pass@host/db
    POSTGRESQL_POOL_SIZE — default 10
    POSTGRESQL_MAX_OVERFLOW — default 20
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""


def _build_engine() -> Engine:
    uri = os.environ.get("POSTGRESQL_URI")
    if not uri:
        raise RuntimeError(
            "POSTGRESQL_URI is missing — set it in backend/.env"
        )
    if not (uri.startswith("postgresql://") or uri.startswith("postgresql+")):
        raise RuntimeError(
            "POSTGRESQL_URI must start with postgresql:// or postgresql+<driver>://"
        )
    pool_size = int(os.environ.get("POSTGRESQL_POOL_SIZE", "10"))
    max_overflow = int(os.environ.get("POSTGRESQL_MAX_OVERFLOW", "20"))
    return create_engine(
        uri,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        pool_recycle=1800,
        future=True,
    )


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def SessionLocal() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )
    return _SessionLocal()


def get_db() -> Iterator[Session]:
    """FastAPI dependency — yields a scoped session, closes on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_schema() -> None:
    """Bring the schema to the Alembic head revision (idempotent).

    Replaces the Phase-1 `Base.metadata.create_all` so a fresh database is
    stamped in `alembic_version` and later `alembic upgrade head` runs do not
    collide with pre-existing tables.
    """
    from app.db import models  # noqa: F401 — register models with Base
    from app.db.migrate import upgrade_to_head

    with engine().connect() as conn:
        upgrade_to_head(connection=conn)
