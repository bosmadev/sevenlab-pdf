"""ASGI middleware that gates ``/api/documents/*`` and ``/api/chat/*``.

Reads the signed ``app_session`` cookie, verifies the HMAC, and attaches
``request.state.owner_id`` + ``request.state.workspace_id`` for downstream
handlers. Requests without a valid session get a 401 before any body is read.

Deliberately route-prefix scoped: health, the login endpoint, OpenAPI docs
and the Next.js login page all stay reachable without a cookie.
"""
from __future__ import annotations

from typing import Iterable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.security.password import is_admin_owner
from app.security.tokens import COOKIE_NAME, verify_session_token

# Prefixes that require a verified session cookie.
PROTECTED_PREFIXES: tuple[str, ...] = (
    "/api/documents",
    "/api/chat",
)


def _is_protected(path: str, prefixes: Iterable[str] = PROTECTED_PREFIXES) -> bool:
    return any(path == p or path.startswith(f"{p}/") for p in prefixes)


class WorkspaceAuthMiddleware(BaseHTTPMiddleware):
    """Reject un-authenticated calls to document and chat routes."""

    def __init__(self, app, protected_prefixes: tuple[str, ...] = PROTECTED_PREFIXES) -> None:
        super().__init__(app)
        self._protected = protected_prefixes

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path

        # Always try to resolve the cookie so downstream code can rely on
        # ``request.state.workspace_id`` even on open routes.
        token = request.cookies.get(COOKIE_NAME)
        owner_id = verify_session_token(token)
        if owner_id:
            request.state.owner_id = owner_id
            request.state.workspace_id = owner_id
            request.state.is_admin = is_admin_owner(owner_id)
        else:
            request.state.owner_id = None
            request.state.workspace_id = None
            request.state.is_admin = False

        if _is_protected(path, self._protected) and not owner_id:
            return JSONResponse(
                {"error": "Not authenticated"},
                status_code=401,
            )

        return await call_next(request)
