"""FastAPI entrypoint. Binds 127.0.0.1 only; Caddy fronts TLS externally.

Startup lifespan (T-003):
    - Ping MongoDB Atlas once so a bad `MONGODB_URI` fails fast at boot
      rather than on the first upload. Non-fatal: we log a warning and keep
      serving so that routes which do not need Mongo (eg `/api/health`) are
      still reachable while the operator fixes credentials.
    - When `AUTO_INIT_SCHEMA=1`, run `Base.metadata.create_all` so that a
      fresh deployment comes up with the PostgreSQL schema already in
      place. Production deploys should leave this unset and run
      `python -m scripts.init_postgres` as an explicit step so DDL is not
      wedged into process startup.
    - When `RERANKER_WARMUP=1`, load the BGE cross-encoder
      (`BAAI/bge-reranker-v2-m3`, ~2.3 GB) in a background thread at boot.
      Without it the first chat question pays the full download + load,
      which on a fresh box is minutes, not seconds. Set it in production.
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.errors import RequestIdMiddleware, register_exception_handlers
from app.routes import (
    auth,
    chat,
    document_mgmt,
    documents,
    health,
    processing,
)
from app.security.middleware import WorkspaceAuthMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
log = logging.getLogger("sevenlab.main")

settings = get_settings()


def _warm_reranker() -> None:
    """Background-thread body: load the cross-encoder once, log the outcome."""
    try:
        from app.rag.reranker import preload

        name = preload()
        log.info("reranker warm: %s loaded", name)
    except Exception as exc:  # noqa: BLE001 — a cold reranker is slow, not fatal
        log.warning("reranker warmup failed, first query will load lazily: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.environ.get("RERANKER_WARMUP") == "1":
        threading.Thread(target=_warm_reranker, name="reranker-warmup", daemon=True).start()
        log.info("reranker warmup started in background (RERANKER_WARMUP=1)")

    if os.environ.get("AUTO_INIT_SCHEMA") == "1":
        try:
            from app.db.postgres import init_schema

            init_schema()
            log.info("postgres schema ensured (AUTO_INIT_SCHEMA=1)")
        except Exception as exc:
            log.warning("postgres schema init skipped: %s", exc)

    try:
        from app.db.mongodb import ping

        res = ping()
        if res.get("ok"):
            log.info("mongodb atlas reachable")
        else:
            log.warning("mongodb atlas not reachable: %s", res.get("error"))
    except Exception as exc:
        log.warning("mongodb atlas ping skipped: %s", exc)

    if os.environ.get("AUTO_INIT_VECTOR_INDEX", "1") == "1":
        try:
            from app.db.indices import ensure_all_indexes

            summary = ensure_all_indexes()
            log.info("mongodb atlas indexes ensured: %s", summary)
        except Exception as exc:
            log.warning("atlas index ensure skipped: %s", exc)

    yield


app = FastAPI(
    title="sevenlab-pdf",
    version="0.1.0",
    description="Internal PDF chat with hybrid RAG + OCR",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "https://sevenlab.bosma.dev"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# T-002: HMAC cookie auth for /api/documents/* and /api/chat/* (login + health open).
app.add_middleware(WorkspaceAuthMiddleware)

# Outermost: request-id correlation. Added last so it wraps every other
# middleware — the id is attached before auth runs, so a 401 still carries
# it, and streaming SSE responses are not buffered (pure-ASGI, not BaseHTTP).
app.add_middleware(RequestIdMiddleware)

# Global exception handlers for PDF-pipeline errors (validator / sanitizer /
# extractor / storage). Each maps to a stable JSON envelope with request_id.
register_exception_handlers(app)

app.include_router(health.router, tags=["health"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(document_mgmt.router, prefix="/api/documents", tags=["documents"])
# T-014: /api/documents/{id}/status + /retry (background processing poll surface)
app.include_router(processing.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
