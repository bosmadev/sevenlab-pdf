"""FastAPI dependencies — session cookie → ``Owner`` identity.

Gates ``/api/documents/*`` and ``/api/chat/*``. The HMAC-signed cookie is
verified by ``verify_session_token``; the signed owner id is returned.
``admin`` is a reserved owner id that may read every workspace's documents.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Cookie, Depends, HTTPException, status

from app.security.password import is_admin_owner
from app.security.session_store import resolve_session
from app.security.tokens import COOKIE_NAME


@dataclass(frozen=True)
class Owner:
    """Resolved tenant identity from a verified session cookie."""

    owner_id: str
    is_admin: bool

    @property
    def workspace_id(self) -> str:
        """Workspace scope used by document / chat queries."""
        return self.owner_id


def _owner_from_cookie(cookie_value: str | None) -> Owner | None:
    # Sync dependency: FastAPI runs it in a threadpool, so the DB lookup
    # inside ``resolve_session`` does not block the event loop.
    claims = resolve_session(cookie_value)
    if claims is None:
        return None
    return Owner(owner_id=claims.owner_id, is_admin=is_admin_owner(claims.owner_id))


def require_owner(
    app_session: str | None = Cookie(default=None, alias=COOKIE_NAME),
) -> Owner:
    """Dependency: raise 401 unless the cookie carries a verified owner."""
    owner = _owner_from_cookie(app_session)
    if owner is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return owner


def require_admin(owner: Owner = Depends(require_owner)) -> Owner:
    """Dependency: raise 403 unless the owner is the all-seeing admin."""
    if not owner.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return owner


def optional_owner(
    app_session: str | None = Cookie(default=None, alias=COOKIE_NAME),
) -> Owner | None:
    """Dependency: ``Owner`` if signed in, else ``None`` (no raise)."""
    return _owner_from_cookie(app_session)
