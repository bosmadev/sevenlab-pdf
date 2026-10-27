"""HMAC-signed opaque session tokens (NOT JWT).

Token format (v2): ``v2.<ownerId>.<random>.<hmac>``
The HMAC covers ``v2.<ownerId>.<random>``, so the owner id is signed and a
tenant cannot rewrite its cookie to impersonate another tenant or admin.

Port of ``~/source/scraper/main/lib/auth.ts``. Uses ``hmac.compare_digest`` for
timing-safe signature comparison, and ``secrets.token_hex`` for the nonce.
"""
from __future__ import annotations

import hmac
import secrets
from hashlib import sha256

from app.config import get_settings
from app.security.password import is_valid_owner_id

COOKIE_NAME = "app_session"
SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 7 days

TOKEN_VERSION = "v2"


def _secret_bytes() -> bytes:
    secret = get_settings().session_secret
    return secret.encode("utf-8") if secret else b""


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

    nonce = secrets.token_hex(32)
    body = f"{TOKEN_VERSION}.{owner_id}.{nonce}"
    signature = hmac.new(secret, body.encode("utf-8"), sha256).hexdigest()
    return f"{body}.{signature}"


def verify_session_token(token: str | None) -> str | None:
    """Verify a session token's HMAC and extract the signed owner id.

    Returns:
        The signed owner id, or ``None`` when the token is absent, malformed,
        forged, uses a legacy v1 format, or carries an invalid owner id.
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

    version, owner_id, nonce, signature = parts
    if version != TOKEN_VERSION:
        return None
    if not owner_id or not nonce or not signature:
        return None
    if not is_valid_owner_id(owner_id):
        return None

    expected = hmac.new(
        secret,
        f"{version}.{owner_id}.{nonce}".encode("utf-8"),
        sha256,
    ).hexdigest()

    try:
        if hmac.compare_digest(signature, expected):
            return owner_id
    except (TypeError, ValueError):
        return None
    return None
