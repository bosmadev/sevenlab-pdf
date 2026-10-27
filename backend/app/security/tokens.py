"""HMAC-signed opaque session tokens (NOT JWT).

Token format (v2): ``v2.<ownerId>.<random>.<hmac>``
The HMAC covers ``v2.<ownerId>.<random>``, so the owner id is signed and a
tenant cannot rewrite its cookie to impersonate another tenant or admin.

``<random>`` is 56 hex chars: an 8-hex-char issued-at (unix seconds) followed
by a 48-hex-char nonce. Because the issued-at sits inside the signed body, the
server enforces the 7-day lifetime itself — a cookie that outlives its
``Max-Age`` (copied out of a browser, replayed from a log) is still rejected.

Port of ``~/source/scraper/main/lib/auth.ts``. Uses ``hmac.compare_digest`` for
timing-safe signature comparison, and ``secrets.token_hex`` for the nonce.
"""
from __future__ import annotations

import hmac
import secrets
import time
from dataclasses import dataclass
from hashlib import sha256

from app.config import get_settings
from app.security.password import is_valid_owner_id

COOKIE_NAME = "app_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 7 days (default; ``SESSION_MAX_AGE`` env overrides)

TOKEN_VERSION = "v2"

_ISSUED_AT_HEX_LEN = 8
_NONCE_BYTES = 24
_RANDOM_SEGMENT_LEN = _ISSUED_AT_HEX_LEN + _NONCE_BYTES * 2  # 56
# Tolerate this much client/server clock drift before calling a token "from the future".
CLOCK_SKEW_SEC = 60


@dataclass(frozen=True)
class SessionClaims:
    """What a verified token asserts: who, and for how long."""

    owner_id: str
    issued_at: int
    expires_at: int


def _now() -> int:
    return int(time.time())


def _secret_bytes() -> bytes:
    secret = get_settings().session_secret
    return secret.encode("utf-8") if secret else b""


def session_max_age() -> int:
    """Configured session lifetime in seconds (``SESSION_MAX_AGE`` env, default 7 days)."""
    configured = get_settings().session_max_age_seconds
    return configured if configured and configured > 0 else SESSION_MAX_AGE


def _sign(body: str, secret: bytes) -> str:
    return hmac.new(secret, body.encode("utf-8"), sha256).hexdigest()


def create_session_token(owner_id: str) -> str:
    """Create a signed session token bound to an owner id.

    Raises:
        ValueError: when ``owner_id`` fails the owner-id charset check.
        RuntimeError: when ``SESSION_SECRET`` is empty — refuse to mint
            unsigned-equivalent tokens.
    """
    if not is_valid_owner_id(owner_id):
        raise ValueError(f"Refusing to sign a token for invalid owner id: {owner_id!r}")
    secret = _secret_bytes()
    if not secret:
        raise RuntimeError("SESSION_SECRET is not configured; refusing to mint tokens")

    random_segment = f"{_now():0{_ISSUED_AT_HEX_LEN}x}{secrets.token_hex(_NONCE_BYTES)}"
    body = f"{TOKEN_VERSION}.{owner_id}.{random_segment}"
    return f"{body}.{_sign(body, secret)}"


def parse_session_token(token: str | None) -> SessionClaims | None:
    """Verify a token's HMAC and lifetime; return its claims.

    Returns:
        ``SessionClaims`` on success, or ``None`` when the token is absent,
        malformed, forged, uses a legacy v1 format, carries an invalid owner
        id, was issued in the future, or is older than ``session_max_age()``.
    """
    if not token:
        return None
    secret = _secret_bytes()
    if not secret:
        return None

    parts = token.split(".")
    # v1 tokens (`<payload>.<hmac>`) carried no identity — reject, do not upgrade.
    if len(parts) != 4:
        return None

    version, owner_id, random_segment, signature = parts
    if version != TOKEN_VERSION:
        return None
    if not owner_id or not random_segment or not signature:
        return None
    if not is_valid_owner_id(owner_id):
        return None

    expected = _sign(f"{version}.{owner_id}.{random_segment}", secret)
    try:
        if not hmac.compare_digest(signature, expected):
            return None
    except (TypeError, ValueError):
        return None

    # Signature is good — only now trust the embedded issued-at.
    if len(random_segment) != _RANDOM_SEGMENT_LEN:
        return None
    try:
        issued_at = int(random_segment[:_ISSUED_AT_HEX_LEN], 16)
    except ValueError:
        return None

    now = _now()
    if issued_at > now + CLOCK_SKEW_SEC:
        return None
    expires_at = issued_at + session_max_age()
    if now >= expires_at:
        return None

    return SessionClaims(owner_id=owner_id, issued_at=issued_at, expires_at=expires_at)


def verify_session_token(token: str | None) -> str | None:
    """Verify a session token and extract the signed owner id (``None`` on any failure)."""
    claims = parse_session_token(token)
    return claims.owner_id if claims else None
