"""Legacy import target for ``current_workspace_id`` used by document routes.

Resolves the signed ``app_session`` cookie to an owner id, which doubles as
the workspace scope for ingestion and chat. Prefer the newer dependency
``app.security.require_owner`` for new code.
"""
from __future__ import annotations

from fastapi import Request

from app.security import COOKIE_NAME, verify_session_token


async def current_workspace_id(request: Request) -> str | None:
    """Return the verified owner id, or ``None`` if the cookie is absent / forged."""
    # Middleware (``WorkspaceAuthMiddleware``) may have already resolved this.
    cached = getattr(request.state, "workspace_id", None)
    if cached:
        return str(cached)
    token = request.cookies.get(COOKIE_NAME)
    return verify_session_token(token)
