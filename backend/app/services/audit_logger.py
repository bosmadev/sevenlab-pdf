"""Compatibility shim — translates pre-rename call sites onto ``app.services.audit``.

The canonical implementation is ``app.services.audit`` (ISO 27001 A.8.15,
Postgres + stdout dual sink). An earlier draft of this module exposed
different keyword names; the shim lets callers that still use the old
names keep working while new code imports ``app.services.audit`` directly.

Keyword translations handled here:
    audit_upload_accepted — ``safe_filename`` → ``sanitized_filename``,
                             ``file_id`` → ``document_id``,
                             ``stripped_keys`` ignored (fire a separate
                             ``audit_payload_stripped`` instead).
    audit_upload_rejected — ``raw_filename`` → ``original_filename``.
"""
from __future__ import annotations

from typing import Any

from app.services.audit import (  # noqa: F401 — public re-exports
    AuditEvent,
    audit_chat_query,
    audit_document_deleted,
    audit_document_ready,
    audit_document_viewed,
    audit_login,
    audit_ocr_tier,
    audit_payload_stripped,
    audit_processing_failed,
    audit_upload_stored,
    extract_client_ip as client_ip_from_headers,
    log_audit as log_event,
)
from app.services import audit as _audit


def audit_upload_accepted(**kwargs: Any) -> None:
    """Legacy keyword surface for ``audit_upload_accepted``.

    Accepts the pre-rename names (``safe_filename``, ``file_id``,
    ``stripped_keys``) and forwards onto the canonical signature.
    """
    legacy = {
        "safe_filename": "sanitized_filename",
        "file_id": "document_id",
        "raw_filename": "original_filename",
    }
    for old, new in legacy.items():
        if old in kwargs and new not in kwargs:
            kwargs[new] = kwargs.pop(old)
        elif old in kwargs:
            kwargs.pop(old)
    # `stripped_keys` is a separate audit event — pop silently so the
    # upstream call does not raise `TypeError: unexpected kwarg`.
    kwargs.pop("stripped_keys", None)
    kwargs.setdefault("document_id", "unknown")
    _audit.audit_upload_accepted(**kwargs)


def audit_upload_rejected(**kwargs: Any) -> None:
    """Legacy keyword surface for ``audit_upload_rejected``.

    Translates ``raw_filename`` → ``original_filename``.
    """
    if "raw_filename" in kwargs and "original_filename" not in kwargs:
        kwargs["original_filename"] = kwargs.pop("raw_filename")
    elif "raw_filename" in kwargs:
        kwargs.pop("raw_filename")
    _audit.audit_upload_rejected(**kwargs)


__all__ = [
    "AuditEvent",
    "audit_chat_query",
    "audit_document_deleted",
    "audit_document_ready",
    "audit_document_viewed",
    "audit_login",
    "audit_ocr_tier",
    "audit_payload_stripped",
    "audit_processing_failed",
    "audit_upload_accepted",
    "audit_upload_rejected",
    "audit_upload_stored",
    "client_ip_from_headers",
    "log_event",
]
