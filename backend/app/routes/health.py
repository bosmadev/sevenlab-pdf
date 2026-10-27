"""Health endpoints.

- `/api/health/status`   lightweight liveness, no DB touches.
- `/api/health/deep`     touches PostgreSQL (SELECT 1) and MongoDB Atlas
                         (admin.ping). Returns 503 if either is unreachable
                         so a probe can distinguish process-up vs serving.

Deep probe lives here already (T-003) rather than T-105 because T-003 added
both database wiring and a startup lifespan that pings Mongo — the only way
to verify both wires without turning a browser upload into the first probe
is a dedicated endpoint.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/health/status")
async def health() -> dict:
    return {"ok": True, "service": "sevenlab-pdf", "version": "0.1.0"}


def _check_postgres() -> dict:
    try:
        from sqlalchemy import text

        from app.db.postgres import engine

        with engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as exc:
        # This endpoint is unauthenticated: log the driver message (it can
        # carry hostnames), expose only the exception class to the probe.
        log.warning("health.postgres unreachable: %s: %s", type(exc).__name__, exc)
        return {"ok": False, "error": type(exc).__name__}


def _check_mongo() -> dict:
    try:
        from app.db.mongodb import ping

        res = ping()
    except Exception as exc:
        log.warning("health.mongodb unreachable: %s: %s", type(exc).__name__, exc)
        return {"ok": False, "error": type(exc).__name__}
    if not res.get("ok"):
        log.warning("health.mongodb unreachable: %s", res.get("error"))
        return {"ok": False, "error": "unreachable"}
    return res


@router.get("/api/health/deep")
async def deep_health() -> JSONResponse:
    pg = _check_postgres()
    mongo = _check_mongo()
    ok = bool(pg.get("ok")) and bool(mongo.get("ok"))
    body = {
        "ok": ok,
        "service": "sevenlab-pdf",
        "version": "0.1.0",
        "postgres": pg,
        "mongodb": mongo,
    }
    return JSONResponse(body, status_code=200 if ok else 503)
