"""Pytest shared fixtures.

Reserved for cross-file setup. UUID round-tripping on SQLite is handled by
`app.db.types.UUIDType`, so no sqlite3 adapter registration is needed here.
"""
from __future__ import annotations
