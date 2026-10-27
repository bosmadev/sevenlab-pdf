"""Server-side session registry: revocation + expiry on top of the HMAC token.

``app.security.tokens`` proves a token was minted by this server and is inside
its signed lifetime. It cannot say whether the user has since logged out. This
module records every minted token (as a SHA-256 hash, never the token itself)
in the ``sessions`` table and consults that row on every verification.

Policy, fail-closed on every branch:
  * a token with no row is rejected (tokens minted before this registry
    existed simply need one re-login),
  * a row with ``revoked_at`` set is rejected,
  * a row whose ``expires_at`` has passed is rejected,
  * when the database cannot be reached, verification returns ``None`` and
    login refuses to mint.

Tests swap the Postgres session factory for SQLite via ``configure``.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import sha256

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.security.tokens import SessionClaims, parse_session_token

log = logging.getLogger(__name__)

_USER_AGENT_MAX = 512

_factory: Callable[[], Session] | None = None


class SessionStoreError(RuntimeError):
    """The ``sessions`` table could not be read or written."""


def configure(factory: Callable[[], Session] | None) -> None:
    """Install a session factory (tests); ``None`` restores the Postgres default."""
    global _factory
    _factory = factory


def _open() -> Session:
    if _factory is not None:
        return _factory()
    from app.db.postgres import SessionLocal

    return SessionLocal()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    """SQLite hands back naive datetimes; treat those as UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def hash_token(token: str) -> str:
    """SHA-256 hex of the full token, the only form that touches the database."""
    return sha256(token.encode("utf-8")).hexdigest()


def _get_or_create_user(db: Session, owner_id: str):
    from app.db.models import User

    user = db.query(User).filter(User.workspace_id == owner_id).one_or_none()
    if user is not None:
        return user
    user = User(workspace_id=owner_id, password_hash="")
    db.add(user)
    try:
        db.flush()
    except IntegrityError:
        # Two first logins for the same owner raced; the other one won.
        db.rollback()
        user = db.query(User).filter(User.workspace_id == owner_id).one()
    return user


def record_session(
    token: str,
    *,
    client_ip: str | None = None,
    user_agent: str | None = None,
) -> SessionClaims:
    """Insert the ``sessions`` row for a freshly minted token.

    Raises:
        SessionStoreError: the token does not verify, or the database refused
            the write. The caller must NOT hand the token to the client.
    """
    claims = parse_session_token(token)
    if claims is None:
        raise SessionStoreError("refusing to register a token that does not verify")

    from app.db.models import SessionToken

    try:
        db = _open()
    except Exception as exc:  # noqa: BLE001 — missing URI, driver, pool exhaustion
        raise SessionStoreError(f"session store unavailable: {exc}") from exc
    try:
        user = _get_or_create_user(db, claims.owner_id)
        db.add(
            SessionToken(
                user_id=user.id,
                workspace_id=claims.owner_id,
                token_hash=hash_token(token),
                created_at=datetime.fromtimestamp(claims.issued_at, tz=timezone.utc),
                expires_at=datetime.fromtimestamp(claims.expires_at, tz=timezone.utc),
                client_ip=client_ip,
                user_agent=(user_agent or "")[:_USER_AGENT_MAX] or None,
            )
        )
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        raise SessionStoreError(f"could not record session: {exc}") from exc
    finally:
        db.close()
    return claims


def revoke_session(token: str) -> bool:
    """Stamp ``revoked_at`` on the row for ``token``.

    Returns ``True`` when a live row was revoked, ``False`` when there was no
    row or it was already revoked (idempotent).

    Raises:
        SessionStoreError: the database refused the write.
    """
    from app.db.models import SessionToken

    try:
        db = _open()
    except Exception as exc:  # noqa: BLE001
        raise SessionStoreError(f"session store unavailable: {exc}") from exc
    try:
        row = (
            db.query(SessionToken)
            .filter(SessionToken.token_hash == hash_token(token))
            .one_or_none()
        )
        if row is None or row.revoked_at is not None:
            return False
        row.revoked_at = _utcnow()
        db.commit()
        return True
    except SQLAlchemyError as exc:
        db.rollback()
        raise SessionStoreError(f"could not revoke session: {exc}") from exc
    finally:
        db.close()


def resolve_session(token: str | None) -> SessionClaims | None:
    """Full verification: HMAC + signed lifetime, then the ``sessions`` row.

    Never raises. Any database failure is logged and treated as "not
    authenticated" so an outage cannot widen access.
    """
    claims = parse_session_token(token)
    if claims is None or token is None:
        return None

    from app.db.models import SessionToken

    try:
        db = _open()
    except Exception as exc:  # noqa: BLE001
        log.warning("session store unavailable, rejecting session: %s", exc)
        return None
    try:
        row = (
            db.query(SessionToken)
            .filter(SessionToken.token_hash == hash_token(token))
            .one_or_none()
        )
    except SQLAlchemyError as exc:
        log.warning("session store lookup failed, rejecting session: %s", exc)
        return None
    finally:
        db.close()

    if row is None or row.revoked_at is not None:
        return None
    if _as_utc(row.expires_at) <= _utcnow():
        return None
    return claims


__all__ = [
    "SessionStoreError",
    "configure",
    "hash_token",
    "record_session",
    "resolve_session",
    "revoke_session",
]
