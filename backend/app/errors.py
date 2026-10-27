"""Global exception handlers + request-id correlation middleware.

Every PDF-pipeline exception maps to a stable JSON envelope so the Next.js
frontend can switch on ``error.code`` instead of parsing prose. The envelope
also carries the request id, letting operators join a 5xx on the frontend
to the matching backend log line.
"""
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse

# Validator has zero heavy deps — always import it.
from app.services.pdf_validator import PdfValidationError

# Sanitizer (pikepdf), extractor (PyMuPDF) and storage (pymongo) each pull in
# a heavy native wheel. Import them lazily inside register_exception_handlers
# so this module stays importable in test envs without the full requirements
# set — the real deps are installed in the uvicorn container.

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


def _envelope(code: str, message: str, request_id: str) -> dict:
    """Stable error-envelope shape for the frontend."""
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        }
    }


def _request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    return rid if isinstance(rid, str) and rid else "unknown"


async def _validation_handler(request: Request, exc: PdfValidationError) -> JSONResponse:
    rid = _request_id(request)
    log.warning("pdf_validation_rejected code=%s rid=%s", exc.code, rid)
    return JSONResponse(
        status_code=400,
        content=_envelope(exc.code, str(exc), rid),
    )


async def _sanitization_handler(request: Request, exc: Exception) -> JSONResponse:
    rid = _request_id(request)
    log.warning("pdf_sanitization_failed rid=%s detail=%s", rid, exc)
    return JSONResponse(
        status_code=422,
        content=_envelope("PDF_SANITIZATION_FAILED", str(exc), rid),
    )


async def _extraction_handler(request: Request, exc: Exception) -> JSONResponse:
    rid = _request_id(request)
    log.warning("pdf_extraction_failed rid=%s detail=%s", rid, exc)
    return JSONResponse(
        status_code=422,
        content=_envelope("PDF_EXTRACTION_FAILED", str(exc), rid),
    )


async def _not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    rid = _request_id(request)
    return JSONResponse(
        status_code=404,
        content=_envelope("PDF_NOT_FOUND", str(exc), rid),
    )


async def _storage_handler(request: Request, exc: Exception) -> JSONResponse:
    rid = _request_id(request)
    # Never leak internal detail to the client — log it, respond generic.
    log.error("pdf_storage_failed rid=%s detail=%s", rid, exc)
    return JSONResponse(
        status_code=500,
        content=_envelope("STORAGE_UNAVAILABLE", "storage error", rid),
    )


def register_exception_handlers(app: "FastAPI") -> None:
    """Attach every pdf-pipeline exception handler available in this env.

    The validator handler is always wired. Sanitizer, extractor and storage
    handlers import their exception classes lazily and are skipped (with a
    warning) when the backing native dep is missing — so this module stays
    importable in minimal test envs without pikepdf / PyMuPDF / pymongo.
    """
    app.add_exception_handler(PdfValidationError, _validation_handler)

    try:
        from app.services.pdf_sanitizer import PdfSanitizationError

        app.add_exception_handler(PdfSanitizationError, _sanitization_handler)
    except Exception as exc:  # noqa: BLE001 — missing dep is non-fatal here
        log.warning("sanitization handler not wired: %s", exc)

    try:
        from app.services.pdf_extractor import PdfExtractionError

        app.add_exception_handler(PdfExtractionError, _extraction_handler)
    except Exception as exc:  # noqa: BLE001
        log.warning("extraction handler not wired: %s", exc)

    try:
        from app.services.gridfs_storage import PdfNotFoundError, StorageError

        # Subclass before parent so the 404 handler wins over the 500.
        app.add_exception_handler(PdfNotFoundError, _not_found_handler)
        app.add_exception_handler(StorageError, _storage_handler)
    except Exception as exc:  # noqa: BLE001
        log.warning("storage handlers not wired: %s", exc)


class RequestIdMiddleware:
    """Pure-ASGI middleware: attach a request id to every request + response.

    Honours an incoming ``X-Request-ID`` header (so the frontend can propagate
    its own correlation id) and falls back to a fresh UUID4 hex. The id is
    exposed on ``request.state.request_id`` for handlers and audit logging,
    and echoed back in the response ``X-Request-ID`` header.

    Written as pure ASGI (not ``BaseHTTPMiddleware``) so streaming responses
    for the chat SSE endpoint are not buffered.
    """

    _HEADER_KEY = REQUEST_ID_HEADER.lower().encode("latin-1")

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming: bytes | None = None
        for key, value in scope.get("headers") or ():
            if key == self._HEADER_KEY:
                incoming = value
                break
        rid = (
            incoming.decode("latin-1", errors="replace")[:64]
            if incoming
            else uuid.uuid4().hex
        )
        # Starlette's Request.state reads from scope["state"], so this makes
        # the id visible as request.state.request_id in every handler.
        scope.setdefault("state", {})
        scope["state"]["request_id"] = rid
        rid_bytes = rid.encode("latin-1", errors="replace")

        async def _send(message):
            if message["type"] == "http.response.start":
                headers_list = list(message.get("headers") or [])
                headers_list.append((self._HEADER_KEY, rid_bytes))
                message["headers"] = headers_list
            await send(message)

        await self.app(scope, receive, _send)
