"""Cross-dialect column types.

`UUIDType` renders as native `UUID` on PostgreSQL (production) and as
`CHAR(36)` on every other dialect — notably SQLite, which the test suite
uses as an in-memory stand-in. Values round-trip as `uuid.UUID` instances
regardless of the backing store, so callers never have to care which
dialect is live.

Before this type was introduced, two tests (`test_document_mgmt_http.py`
and `test_document_service.py`) hand-swapped the Postgres UUID columns
for `String(36)` at import or fixture scope. That swap leaked into other
tests because the SQLAlchemy mapper caches `Comparator.type` separately
from `Column.type`, so `SELECT ... WHERE id = ?` still bound `uuid.hex`
while `INSERT` bound `str(uuid)` — mismatch, zero rows. One source of
truth here fixes the leak and keeps the production schema unchanged.
"""
from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import CHAR
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.types import TypeDecorator


class UUIDType(TypeDecorator):
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PgUUID(as_uuid=True))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return value
        if isinstance(value, uuid.UUID):
            return str(value)
        return value

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(value)
