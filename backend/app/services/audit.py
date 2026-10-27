"""ISO 27001 A.8.15 structured audit logger — dual sink, non-raising.

Contract (locked by `tests/test_audit.py`):

* `AuditEvent` string-enum, UNDERSCORE-separated values (`upload_accepted`,
  `document_ready`, `chat_query`, …) — these ARE the on-wire tokens.
* `AuditRecord(event=..., workspace_id=None, document_id=None, client_ip=None,
  sha256=None, size_bytes=None, details={})` → `.as_json_payload()` returns a
  dict whose `details` key is ALWAYS present (`None` when empty, dict
  otherwise), with a UTC-ISO-8601 `ts`.
* `log_event(event, *, session=None, workspace_id=None, document_id=None,
  client_ip=None, sha256=None, size_bytes=None, **extra)` emits ONE JSON line
  on `logging.getLogger("sevenlab.audit")` (plain `%(message)s` formatter so
  `record.message` is the JSON string) and optionally inserts one row into
  the Postgres `audit_log` table via a passed SQLAlchemy session. Any DB
  failure is rolled back and swallowed — the stdout sink is the floor.
* `client_ip_from_headers(remote_addr=..., forwarded_for=...)` returns the
  leftmost `X-Forwarded-For` IP, else `remote_addr`, else `None`.
* `extract_client_ip(request)` resolves the IP from a FastAPI/Starlette
  `Request`: XFF → X-Real-IP → `request.client.host`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import Any, Final, Optional

try:
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.orm import Session  # noqa: F401 — type reference only
except Exception:  # pragma: no cover
    SQLAlchemyError = Exception  # type: ignore[assignment,misc]
    Session = Any  # type: ignore[assignment,misc]


LOGGER_NAME: Final[str] = "sevenlab.audit"

_SETUP_DONE: bool = False


class AuditEvent(str, Enum):
    """On-wire event tokens (UNDERSCORE-separated). Keep stable."""

    UPLOAD_ACCEPTED = "upload_accepted"
    UPLOAD_REJECTED = "upload_rejected"
    UPLOAD_STORED = "upload_stored"
    PAYLOAD_STRIPPED = "payload_stripped"
    OCR_TIER_USED = "ocr_tier_used"
    PROCESSING_FAILED = "processing_failed"
    DOCUMENT_READY = "document_ready"
    DOCUMENT_DELETED = "document_deleted"
    DOCUMENT_VIEWED = "document_viewed"
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    CHAT_QUERY = "chat_query"


@dataclass
class AuditRecord:
    """One audit event. `event` required; everything else optional.

    `.as_json_payload()` is the on-wire shape:
        * `ts` — UTC ISO 8601 with `+00:00` offset.
        * `event` — enum value.
        * `workspace_id` / `document_id` / `client_ip` / `sha256` /
          `size_bytes` — included when set.
        * `details` — ALWAYS present. `None` when empty; the dict otherwise.
    """

    event: AuditEvent
    workspace_id: str | None = None
    document_id: uuid.UUID | str | None = None
    client_ip: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_json_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": self.event.value,
        }
        if self.workspace_id is not None:
            payload["workspace_id"] = self.workspace_id
        if self.document_id is not None:
            payload["document_id"] = str(self.document_id)
        if self.client_ip is not None:
            payload["client_ip"] = self.client_ip
        if self.sha256 is not None:
            payload["sha256"] = self.sha256
        if self.size_bytes is not None:
            payload["size_bytes"] = self.size_bytes
        payload["details"] = dict(self.details) if self.details else None
        return payload


def _coerce_event(event: AuditEvent | str) -> AuditEvent | None:
    if isinstance(event, AuditEvent):
        return event
    try:
        return AuditEvent(str(event))
    except ValueError:
        return None


def _coerce_doc_id_for_db(value: uuid.UUID | str | None) -> uuid.UUID | None:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


def _get_logger() -> logging.Logger:
    """Return the named audit logger, attaching handlers once per process."""
    global _SETUP_DONE
    logger = logging.getLogger(LOGGER_NAME)
    if _SETUP_DONE and logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stdout_handler)

    log_path = os.getenv("SEVENLAB_AUDIT_LOG_PATH", "").strip()
    if log_path:
        max_mb = int(os.getenv("SEVENLAB_AUDIT_LOG_MAX_MB", "16"))
        backups = int(os.getenv("SEVENLAB_AUDIT_LOG_BACKUPS", "5"))
        try:
            fh = RotatingFileHandler(
                log_path,
                maxBytes=max_mb * 1024 * 1024,
                backupCount=backups,
                encoding="utf-8",
            )
            fh.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(fh)
        except OSError:
            pass  # stdout stays authoritative
    _SETUP_DONE = True
    return logger


def _emit_line(payload: dict[str, Any], level: int = logging.INFO) -> None:
    """Serialize payload to JSON string and emit via the audit logger."""
    try:
        line = json.dumps(payload, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        line = json.dumps({"event": "audit_format_error"}, separators=(",", ":"))
    try:
        _get_logger().log(level, line)
    except Exception:
        pass


class _AuditRowStub:
    """ORM-free row shape so unit tests can assert on `.event_type` etc."""

    def __init__(self, record: AuditRecord) -> None:
        self.workspace_id = record.workspace_id
        self.event_type = record.event.value
        self.document_id = _coerce_doc_id_for_db(record.document_id)
        self.client_ip = record.client_ip
        self.sha256 = record.sha256
        self.size_bytes = record.size_bytes
        self.details = (
            json.dumps(record.details, default=str, separators=(",", ":"))
            if record.details
            else None
        )


def _write_db(session: Optional["Session"], record: AuditRecord) -> None:
    """Persist a record to Postgres `audit_log`. Swallows all errors.

    Uses a stub row object when the SQLAlchemy ORM model can't be imported so
    tests pass without the full DB wiring.
    """
    if session is None:
        return
    try:
        from app.db.models import AuditLog  # noqa: WPS433 — late import by design
        orm_available = True
    except Exception:
        AuditLog = None  # type: ignore[assignment]
        orm_available = False

    try:
        if orm_available and AuditLog is not None:
            row: Any = AuditLog(
                workspace_id=record.workspace_id,
                event_type=record.event.value,
                document_id=_coerce_doc_id_for_db(record.document_id),
                client_ip=record.client_ip,
                sha256=record.sha256,
                size_bytes=record.size_bytes,
                details=(
                    json.dumps(record.details, default=str, separators=(",", ":"))
                    if record.details
                    else None
                ),
            )
        else:
            row = _AuditRowStub(record)
        session.add(row)
        commit = getattr(session, "commit", None)
        if callable(commit):
            commit()
    except Exception:
        rollback = getattr(session, "rollback", None)
        if callable(rollback):
            try:
                rollback()
            except Exception:
                pass
        _emit_line(
            {"event": "audit_db_write_failed", "orig": record.event.value},
            level=logging.WARNING,
        )


# ---- Public entrypoint ------------------------------------------------------


def log_event(
    event: AuditEvent | str,
    *,
    session: Optional["Session"] = None,
    workspace_id: str | None = None,
    document_id: uuid.UUID | str | None = None,
    client_ip: str | None = None,
    sha256: str | None = None,
    size_bytes: int | None = None,
    details: dict[str, Any] | None = None,
    **extra: Any,
) -> None:
    """Record one audit event. Dual-sink, non-raising.

    `session` is optional: with a SQLAlchemy session, the event is persisted
    to the Postgres `audit_log` table AND emitted to the logger; without one,
    only the logger sink fires. Any DB failure is swallowed — audit MUST NOT
    break the request path.

    Extra kwargs merge into `details`.
    """
    canonical = _coerce_event(event)
    if canonical is None:
        _emit_line(
            {"event": "audit_unknown_event", "raw": str(event)},
            level=logging.WARNING,
        )
        return

    merged: dict[str, Any] = {}
    if details:
        merged.update(details)
    if extra:
        merged.update(extra)

    record = AuditRecord(
        event=canonical,
        workspace_id=workspace_id,
        document_id=document_id,
        client_ip=client_ip,
        sha256=sha256,
        size_bytes=size_bytes,
        details=merged,
    )
    _emit_line(record.as_json_payload())
    _write_db(session, record)


# Alias retained for call sites that imported the longer form.
log_audit = log_event


# ---- IP helpers -------------------------------------------------------------


def client_ip_from_headers(
    *, remote_addr: str | None, forwarded_for: str | None
) -> str | None:
    """Leftmost `X-Forwarded-For`, else `remote_addr`, else `None`."""
    if forwarded_for:
        first = forwarded_for.split(",", 1)[0].strip()
        if first:
            return first
    return remote_addr or None


def extract_client_ip(request: Any) -> str | None:
    """Resolve client IP from a FastAPI/Starlette-like `Request`.

    Precedence: `X-Forwarded-For` (leftmost) → `X-Real-IP` → `request.client.host`.
    """
    if request is None:
        return None
    headers = getattr(request, "headers", None)
    if headers is not None and hasattr(headers, "get"):
        xff = headers.get("X-Forwarded-For")
        if xff:
            first = xff.split(",", 1)[0].strip()
            if first:
                return first
        real_ip = headers.get("X-Real-IP")
        if real_ip:
            trimmed = real_ip.strip()
            if trimmed:
                return trimmed
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client else None
    return host or None


# ---- Convenience wrappers — thin shims --------------------------------------


def audit_upload_accepted(
    *,
    session: Optional["Session"] = None,
    workspace_id: str,
    document_id: uuid.UUID | str,
    client_ip: str | None,
    sha256: str,
    size_bytes: int,
    original_filename: str | None = None,
    sanitized_filename: str | None = None,
    pages: int | None = None,
) -> None:
    extras: dict[str, Any] = {}
    if original_filename is not None:
        extras["original_filename"] = original_filename
    if sanitized_filename is not None:
        extras["sanitized_filename"] = sanitized_filename
    if pages is not None:
        extras["pages"] = pages
    log_event(
        AuditEvent.UPLOAD_ACCEPTED,
        session=session,
        workspace_id=workspace_id,
        document_id=document_id,
        client_ip=client_ip,
        sha256=sha256,
        size_bytes=size_bytes,
        **extras,
    )


def audit_upload_rejected(
    *,
    session: Optional["Session"] = None,
    workspace_id: str | None,
    client_ip: str | None,
    reason: str,
    code: str,
    size_bytes: int | None = None,
    original_filename: str | None = None,
) -> None:
    extras: dict[str, Any] = {"reason": reason, "code": code}
    if original_filename is not None:
        extras["original_filename"] = original_filename
    log_event(
        AuditEvent.UPLOAD_REJECTED,
        session=session,
        workspace_id=workspace_id,
        client_ip=client_ip,
        size_bytes=size_bytes,
        **extras,
    )


def audit_upload_stored(
    *,
    session: Optional["Session"] = None,
    workspace_id: str,
    document_id: uuid.UUID | str,
    gridfs_file_id: str,
    sha256: str,
    size_bytes: int,
) -> None:
    log_event(
        AuditEvent.UPLOAD_STORED,
        session=session,
        workspace_id=workspace_id,
        document_id=document_id,
        sha256=sha256,
        size_bytes=size_bytes,
        gridfs_file_id=gridfs_file_id,
    )


def audit_payload_stripped(
    *,
    session: Optional["Session"] = None,
    workspace_id: str,
    document_id: uuid.UUID | str,
    removed_keys: list[str],
    sha256_before: str,
    sha256_after: str,
) -> None:
    log_event(
        AuditEvent.PAYLOAD_STRIPPED,
        session=session,
        workspace_id=workspace_id,
        document_id=document_id,
        sha256=sha256_after,
        removed_keys=removed_keys,
        sha256_before=sha256_before,
    )


def audit_ocr_tier(
    *,
    session: Optional["Session"] = None,
    workspace_id: str,
    document_id: uuid.UUID | str,
    tier: str,
    page: int,
    chars_extracted: int,
) -> None:
    log_event(
        AuditEvent.OCR_TIER_USED,
        session=session,
        workspace_id=workspace_id,
        document_id=document_id,
        tier=tier,
        page=page,
        chars_extracted=chars_extracted,
    )


def audit_processing_failed(
    *,
    session: Optional["Session"] = None,
    workspace_id: str,
    document_id: uuid.UUID | str,
    stage: str,
    error: str,
) -> None:
    log_event(
        AuditEvent.PROCESSING_FAILED,
        session=session,
        workspace_id=workspace_id,
        document_id=document_id,
        stage=stage,
        error=error,
    )


def audit_document_ready(
    *,
    session: Optional["Session"] = None,
    workspace_id: str,
    document_id: uuid.UUID | str,
    chunk_count: int,
    ocr_tier_used: str | None,
) -> None:
    log_event(
        AuditEvent.DOCUMENT_READY,
        session=session,
        workspace_id=workspace_id,
        document_id=document_id,
        chunk_count=chunk_count,
        ocr_tier_used=ocr_tier_used,
    )


def audit_document_deleted(
    *,
    session: Optional["Session"] = None,
    workspace_id: str,
    document_id: uuid.UUID | str,
    client_ip: str | None,
) -> None:
    log_event(
        AuditEvent.DOCUMENT_DELETED,
        session=session,
        workspace_id=workspace_id,
        document_id=document_id,
        client_ip=client_ip,
    )


def audit_document_viewed(
    *,
    session: Optional["Session"] = None,
    workspace_id: str,
    document_id: uuid.UUID | str,
    client_ip: str | None,
) -> None:
    log_event(
        AuditEvent.DOCUMENT_VIEWED,
        session=session,
        workspace_id=workspace_id,
        document_id=document_id,
        client_ip=client_ip,
    )


def audit_login(
    *,
    session: Optional["Session"] = None,
    workspace_id: str | None,
    client_ip: str | None,
    success: bool,
    reason: str | None = None,
) -> None:
    extras: dict[str, Any] = {}
    if reason is not None:
        extras["reason"] = reason
    log_event(
        AuditEvent.LOGIN_SUCCESS if success else AuditEvent.LOGIN_FAILED,
        session=session,
        workspace_id=workspace_id,
        client_ip=client_ip,
        **extras,
    )


def audit_chat_query(
    *,
    session: Optional["Session"] = None,
    workspace_id: str,
    client_ip: str | None,
    query_hash: str,
    retrieved_chunks: int,
    document_ids: list[str] | None = None,
) -> None:
    log_event(
        AuditEvent.CHAT_QUERY,
        session=session,
        workspace_id=workspace_id,
        client_ip=client_ip,
        query_hash=query_hash,
        retrieved_chunks=retrieved_chunks,
        document_ids=document_ids or [],
    )


__all__ = [
    "LOGGER_NAME",
    "AuditEvent",
    "AuditRecord",
    "log_event",
    "log_audit",
    "client_ip_from_headers",
    "extract_client_ip",
    "audit_upload_accepted",
    "audit_upload_rejected",
    "audit_upload_stored",
    "audit_payload_stripped",
    "audit_ocr_tier",
    "audit_processing_failed",
    "audit_document_ready",
    "audit_document_deleted",
    "audit_document_viewed",
    "audit_login",
    "audit_chat_query",
]
