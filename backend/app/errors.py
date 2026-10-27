"""Global exception handlers + request-id correlation middleware.

Every error leaving the API uses one JSON envelope so the Next.js frontend
can switch on ``error.code`` instead of parsing prose::

    {
      "detail": <FastAPI's classic key — string or dict, kept for old callers>,
      "error": {
        "code": "NOT_FOUND",
        "message": "document not found",
        "request_id": "3f9c..."
      }
    }

The envelope carries the request id, letting an operator join a 5xx seen on
the frontend to the matching line in ``backend/logs/error.log`` — every log
record is stamped with the same id via ``app.logging_config``.

Handler map (Starlette walks the exception MRO, so the most specific wins):

    PdfValidationError            400  code from the validator
    PdfSanitizationError          422  PDF_SANITIZATION_FAILED
    PdfExtractionError            422  PDF_EXTRACTION_FAILED
    PdfNotFoundError              404  PDF_NOT_FOUND
    StorageError                  500  STORAGE_UNAVAILABLE   (detail logged only)
    sqlalchemy OperationalError / pymongo ConnectionFailure
                                  503  DATABASE_UNAVAILABLE  (detail logged only)
    HTTPException                 as raised; code from a dict detail or status
    RequestValidationError        422  VALIDATION_ERROR, per-field list in detail
    Exception                     500  INTERNAL_ERROR        (traceback logged only)
"""
from __future__ import annotations

import logging
import uuid
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging_config import request_id_var

# Validator has zero heavy deps — always import it.
from app.services.pdf_validator import PdfValidationError

# Sanitizer (pikepdf), extractor (PyMuPDF), storage (pymongo) and the DB
# drivers each pull in a heavy native wheel. Import them lazily inside
# register_exception_handlers so this module stays importable in test envs
# without the full requirements set — the real deps live in the container.

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

_CODE_BY_STATUS: dict[int, str] = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMITED",
    500: "INTERNAL_ERROR",
    501: "NOT_IMPLEMENTED",
    503: "SERVICE_UNAVAILABLE",
}

# Operator-facing wording for the statuses a user can actually hit when the
# raising code gave no message of its own.
_MESSAGE_BY_STATUS: dict[int, str] = {
    401: "Not authenticated. Please sign in again.",
    403: "You do not have access to this resource.",
    404: "Not found.",
    405: "Method not allowed.",
    413: "The file is too large.",
    429: "Too many requests. Please wait and retry.",
    500: "Internal server error. Quote the request id when reporting it.",
    503: "Service temporarily unavailable. Please retry shortly.",
}

INTERNAL_ERROR_MESSAGE = _MESSAGE_BY_STATUS[500]
DB_UNAVAILABLE_MESSAGE = "Database temporarily unavailable. Please retry shortly."
STORAGE_ERROR_MESSAGE = "File storage is unavailable. Please retry shortly."


def _code_for_status(status_code: int) -> str:
    return _CODE_BY_STATUS.get(status_code, f"HTTP_{status_code}")


def _message_for_status(status_code: int) -> str:
    if status_code in _MESSAGE_BY_STATUS:
        return _MESSAGE_BY_STATUS[status_code]
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "Request failed."


def _envelope(code: str, message: str, request_id: str, *, detail: Any = None) -> dict:
    """Stable error-envelope shape for the frontend.

    ``detail`` mirrors FastAPI's default key so pre-envelope callers keep
    working; it defaults to the human message.
    """
    return {
        "detail": message if detail is None else detail,
        "error": {"code": code, "message": message, "request_id": request_id},
    }


def _request_id(request: Request) -> str:
    rid = getattr(request.state, "request_id", None)
    return rid if isinstance(rid, str) and rid else "unknown"


def _json(
    status_code: int,
    content: dict,
    request_id: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build the response, always echoing the request id header.

    The ``Exception`` handler's response is sent by Starlette's
    ``ServerErrorMiddleware`` *outside* ``RequestIdMiddleware``, so the header
    has to be set here for that path; the middleware skips it when present.
    """
    hdrs = dict(headers or {})
    hdrs.setdefault(REQUEST_ID_HEADER, request_id)
    return JSONResponse(status_code=status_code, content=content, headers=hdrs)


# ---------------------------------------------------------------------------
# PDF pipeline
# ---------------------------------------------------------------------------


async def _validation_handler(request: Request, exc: PdfValidationError) -> JSONResponse:
    rid = _request_id(request)
    log.warning("pdf_validation_rejected code=%s rid=%s", exc.code, rid)
    return _json(400, _envelope(exc.code, str(exc), rid), rid)


async def _sanitization_handler(request: Request, exc: Exception) -> JSONResponse:
    rid = _request_id(request)
    log.warning("pdf_sanitization_failed rid=%s detail=%s", rid, exc)
    return _json(422, _envelope("PDF_SANITIZATION_FAILED", str(exc), rid), rid)


async def _extraction_handler(request: Request, exc: Exception) -> JSONResponse:
    rid = _request_id(request)
    log.warning("pdf_extraction_failed rid=%s detail=%s", rid, exc)
    return _json(422, _envelope("PDF_EXTRACTION_FAILED", str(exc), rid), rid)


async def _not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    rid = _request_id(request)
    return _json(404, _envelope("PDF_NOT_FOUND", str(exc), rid), rid)


async def _storage_handler(request: Request, exc: Exception) -> JSONResponse:
    rid = _request_id(request)
    # Never leak internal detail to the client — log it, respond generic.
    log.error("pdf_storage_failed rid=%s detail=%s", rid, exc)
    return _json(500, _envelope("STORAGE_UNAVAILABLE", STORAGE_ERROR_MESSAGE, rid), rid)


# ---------------------------------------------------------------------------
# Infrastructure + framework
# ---------------------------------------------------------------------------


async def _db_unavailable_handler(request: Request, exc: Exception) -> JSONResponse:
    rid = _request_id(request)
    log.error(
        "database_unavailable rid=%s path=%s type=%s detail=%s",
        rid,
        request.url.path,
        type(exc).__name__,
        exc,
    )
    return _json(
        503,
        _envelope("DATABASE_UNAVAILABLE", DB_UNAVAILABLE_MESSAGE, rid),
        rid,
        headers={"Retry-After": "5"},
    )


async def _http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Normalise every ``HTTPException`` (routes, deps, unknown paths) to the envelope.

    A dict detail of the form ``{"code": ..., "reason": ...}`` (the upload
    route's convention) keeps its code; a plain string becomes the message
    and the code is derived from the status.
    """
    rid = _request_id(request)
    detail = exc.detail
    code: str | None = None
    message: str | None = None

    if isinstance(detail, dict):
        raw_code = detail.get("code")
        code = str(raw_code) if raw_code else None
        for key in ("reason", "message", "detail"):
            if detail.get(key):
                message = str(detail[key])
                break
    elif isinstance(detail, str) and detail.strip():
        message = detail

    code = code or _code_for_status(exc.status_code)
    message = message or _message_for_status(exc.status_code)

    if exc.status_code >= 500:
        log.error(
            "http_error status=%s code=%s rid=%s path=%s detail=%s",
            exc.status_code, code, rid, request.url.path, detail,
        )
    else:
        log.info(
            "http_error status=%s code=%s rid=%s path=%s",
            exc.status_code, code, rid, request.url.path,
        )

    return _json(
        exc.status_code,
        _envelope(code, message, rid, detail=detail),
        rid,
        headers=exc.headers,
    )


async def _request_validation_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422 with FastAPI's per-field list kept in ``detail`` plus a one-line summary."""
    rid = _request_id(request)
    errors = jsonable_encoder(exc.errors())
    parts: list[str] = []
    for err in errors[:5]:
        loc = [
            str(p)
            for p in err.get("loc", ())
            if p not in ("body", "query", "path", "header", "cookie")
        ]
        field = ".".join(loc)
        msg = str(err.get("msg", "invalid value"))
        parts.append(f"{field}: {msg}" if field else msg)
    message = "; ".join(parts) or "Invalid request."
    log.info(
        "request_validation_failed rid=%s path=%s errors=%d",
        rid, request.url.path, len(errors),
    )
    return _json(422, _envelope("VALIDATION_ERROR", message, rid, detail=errors), rid)


async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort: full traceback to the logs, nothing but the id to the client."""
    rid = _request_id(request)
    log.exception(
        "unhandled_exception rid=%s method=%s path=%s type=%s",
        rid, request.method, request.url.path, type(exc).__name__,
    )
    return _json(500, _envelope("INTERNAL_ERROR", INTERNAL_ERROR_MESSAGE, rid), rid)


def register_exception_handlers(app: "FastAPI") -> None:
    """Attach every exception handler available in this env.

    Framework handlers (HTTPException, validation, catch-all) and the
    validator handler are always wired. Sanitizer, extractor, storage and
    DB-driver handlers import their exception classes lazily and are skipped
    (with a warning) when the backing native dep is missing.
    """
    app.add_exception_handler(Exception, _unhandled_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _request_validation_handler)
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

    try:
        from sqlalchemy.exc import InterfaceError, OperationalError

        app.add_exception_handler(OperationalError, _db_unavailable_handler)
        app.add_exception_handler(InterfaceError, _db_unavailable_handler)
    except Exception as exc:  # noqa: BLE001
        log.warning("sqlalchemy handlers not wired: %s", exc)

    try:
        from pymongo.errors import ConnectionFailure

        app.add_exception_handler(ConnectionFailure, _db_unavailable_handler)
    except Exception as exc:  # noqa: BLE001
        log.warning("pymongo handler not wired: %s", exc)


class RequestIdMiddleware:
    """Pure-ASGI middleware: attach a request id to every request + response.

    Honours an incoming ``X-Request-ID`` header (so the frontend can propagate
    its own correlation id) and falls back to a fresh UUID4 hex. The id is
    exposed on ``request.state.request_id`` for handlers and audit logging,
    stored in the ``request_id_var`` contextvar so every log record carries
    it, and echoed back in the response ``X-Request-ID`` header.

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
                if not any(k == self._HEADER_KEY for k, _ in headers_list):
                    headers_list.append((self._HEADER_KEY, rid_bytes))
                message["headers"] = headers_list
            await send(message)

        token = request_id_var.set(rid)
        try:
            await self.app(scope, receive, _send)
        finally:
            request_id_var.reset(token)
