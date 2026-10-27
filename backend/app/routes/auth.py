"""Password gate: POST login / DELETE logout.

Port of ``~/source/scraper/main/app/api/auth/route.ts`` to FastAPI.

Flow:
  1. Rate-limit by client IP BEFORE any bcrypt work (bcrypt cost 12 ~= 250 ms
     CPU per compare).
  2. Resolve the plaintext password against ``APP_PASSWORD_HASH`` /
     ``APP_SECONDARY_HASHES`` to an owner id.
  3. On success, mint an HMAC v2 opaque session token and set it as a
     ``httpOnly`` cookie (``secure`` in production, ``sameSite=lax``).
  4. On failure, record it against the client bucket and return 401.
"""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.security import (
    COOKIE_NAME,
    check_login_rate,
    create_session_token,
    record_login_failure,
    record_login_success,
    resolve_owner_id,
)
from app.security.tokens import parse_session_token, session_max_age
from app.services.audit import audit_login, extract_client_ip

log = logging.getLogger(__name__)
router = APIRouter()

COOKIE_SAMESITE = "lax"


def _cookie_secure() -> bool:
    return get_settings().app_env == "production"


class LoginBody(BaseModel):
    password: Annotated[str, Field(min_length=1, max_length=4096)]


@router.post("")
async def login(body: LoginBody, request: Request, response: Response) -> dict:
    """Verify password, set session cookie, return ``{success: true}``."""
    client_ip = extract_client_ip(request)
    rate = check_login_rate(request)
    if not rate.allowed:
        log.warning("auth rate-limited %s - retry in %ss", rate.client_key, rate.retry_after_sec)
        audit_login(
            workspace_id=None,
            client_ip=client_ip,
            success=False,
            reason="rate_limited",
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many attempts. Try again later.",
            headers={"Retry-After": str(rate.retry_after_sec)},
        )

    settings = get_settings()
    if not settings.app_password_hash:
        log.error("APP_PASSWORD_HASH missing - auth not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication not configured",
        )
    if not settings.session_secret:
        log.error("SESSION_SECRET missing - refusing to mint tokens")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication not configured",
        )

    owner_id = resolve_owner_id(body.password)
    if not owner_id:
        record_login_failure(rate.client_key)
        audit_login(
            workspace_id=None,
            client_ip=client_ip,
            success=False,
            reason="invalid_password",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password",
        )

    record_login_success(rate.client_key)
    audit_login(
        workspace_id=owner_id,
        client_ip=client_ip,
        success=True,
    )
    token = create_session_token(owner_id)

    # Cookie Max-Age and the signed issued-at inside the token share one
    # lifetime, so browser expiry and server expiry happen together.
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=session_max_age(),
        httponly=True,
        secure=_cookie_secure(),
        samesite=COOKIE_SAMESITE,
        path="/",
    )
    return {"success": True}


@router.delete("")
async def logout(response: Response) -> dict:
    """Clear the session cookie.

    Attributes mirror ``set_cookie`` so the browser matches the same cookie
    (a ``Secure`` cookie is only reliably overwritten by a ``Secure`` delete).
    """
    response.delete_cookie(
        key=COOKIE_NAME,
        path="/",
        httponly=True,
        secure=_cookie_secure(),
        samesite=COOKIE_SAMESITE,
    )
    return {"success": True}


@router.get("/me")
async def me(request: Request) -> dict:
    """Return the current session's owner id and expiry, or 401.

    Convenience for the frontend to detect expired sessions without parsing
    cookies itself.
    """
    claims = parse_session_token(request.cookies.get(COOKIE_NAME))
    if claims is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return {
        "owner_id": claims.owner_id,
        "is_admin": claims.owner_id == "admin",
        "issued_at": claims.issued_at,
        "expires_at": claims.expires_at,
    }
