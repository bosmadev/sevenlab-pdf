"""Authentication primitives for sevenlab-pdf.

HMAC-signed opaque session tokens (NOT JWT), bcrypt password verification,
rate limiting, and a FastAPI dependency that gates `/api/documents/*` and
`/api/chat/*`. Mirrors `~/source/scraper/main/lib/auth.ts` conventions.
"""
from app.security.password import resolve_owner_id, is_valid_owner_id, ADMIN_OWNER_ID
from app.security.tokens import (
    COOKIE_NAME,
    SESSION_MAX_AGE,
    TOKEN_VERSION,
    create_session_token,
    verify_session_token,
)
from app.security.rate_limit import (
    RateVerdict,
    check_login_rate,
    client_key_for,
    record_login_failure,
    record_login_success,
    reset_login_rate_limit,
)
from app.security.deps import Owner, require_owner, require_admin, optional_owner
from app.security.session_store import resolve_session

__all__ = [
    "ADMIN_OWNER_ID",
    "COOKIE_NAME",
    "Owner",
    "RateVerdict",
    "SESSION_MAX_AGE",
    "TOKEN_VERSION",
    "check_login_rate",
    "client_key_for",
    "create_session_token",
    "is_valid_owner_id",
    "optional_owner",
    "record_login_failure",
    "record_login_success",
    "require_admin",
    "require_owner",
    "reset_login_rate_limit",
    "resolve_owner_id",
    "resolve_session",
    "verify_session_token",
]
