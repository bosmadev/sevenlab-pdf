"""ASGI middleware that gates ``/api/documents/*`` and ``/api/chat/*``.

Reads the signed ``app_session`` cookie, verifies the HMAC, and attaches
``request.state.owner_id`` + ``request.state.workspace_id`` for downstream
handlers. Requests without a valid session get a 401 before any body is read.

Deliberately route-prefix scoped: health, the login endpoint, OpenAPI docs
and the Next.js login page all stay reachable without a cookie.
"""
from __future__ import annotations

from typing import Iterable

from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.security.password import is_admin_owner
from app.security.session_store import resolve_session
from app.security.tokens import COOKIE_NAME

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
        # HMAC + signed lifetime, then the ``sessions`` row (revocation).
        # The row lookup is a sync DB call, so keep it off the event loop.
        claims = await run_in_threadpool(resolve_session, token) if token else None
        owner_id = claims.owner_id if claims else None
        if owner_id:
            request.state.owner_id = owner_id
            request.state.workspace_id = owner_id
            request.state.is_admin = is_admin_owner(owner_id)
        else:
            request.state.owner_id = None
            request.state.workspace_id = None
            request.state.is_admin = False

        if _is_protected(path, self._protected) and not owner_id:
            # Same envelope as app.errors so the frontend has one shape to
            # parse; request_id is set by the outer RequestIdMiddleware.
            rid = getattr(request.state, "request_id", None) or "unknown"
            return JSONResponse(
                {
                    "detail": "Not authenticated",
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "Not authenticated. Please sign in again.",
                        "request_id": rid,
                    },
                },
                status_code=401,
            )

        return await call_next(request)
