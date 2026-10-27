"""Backwards-compatibility shim for ``app.auth`` imports.

Canonical home for auth primitives is ``app.security``. This package exists
only so lazy imports written against the earlier ``app.auth`` path (e.g.
``app.routes.documents``) continue to resolve.
"""
from app.security import (
    ADMIN_OWNER_ID,
    COOKIE_NAME,
    Owner,
    SESSION_MAX_AGE,
    TOKEN_VERSION,
    create_session_token,
    is_valid_owner_id,
    resolve_owner_id,
    verify_session_token,
)

__all__ = [
    "ADMIN_OWNER_ID",
    "COOKIE_NAME",
    "Owner",
    "SESSION_MAX_AGE",
    "TOKEN_VERSION",
    "create_session_token",
    "is_valid_owner_id",
    "resolve_owner_id",
    "verify_session_token",
]
