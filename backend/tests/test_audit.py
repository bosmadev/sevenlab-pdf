"""Unit tests for the ISO 27001 A.8.15 audit logging service.

The audit module serialises each event as a one-line JSON string via
``logging.Formatter("%(message)s")`` on the ``sevenlab.audit`` logger. These
tests parse that string back to a dict and assert on the structured shape.
"""
from __future__ import annotations

import json
import logging
import re
import uuid

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.services import audit
from app.services.audit import (
    AuditEvent,
    AuditRecord,
    audit_login,
    audit_payload_stripped,
    audit_upload_accepted,
    audit_upload_rejected,
    client_ip_from_headers,
    extract_client_ip,
    log_event,
)


ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$")


@pytest.fixture(autouse=True)
def _reset_audit_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test rebuilds the audit logger with propagation on so caplog sees us.

    The audit module explicitly disables propagation so stdout emits exactly
    once in production; for the test we patch the setup helper to re-enable
    it after the module attaches its own handlers.
    """
    monkeypatch.setattr(audit, "_SETUP_DONE", False)
    logger = logging.getLogger(audit.LOGGER_NAME)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    orig_get_logger = audit._get_logger

    def _get_logger_with_propagation() -> logging.Logger:
        lg = orig_get_logger()
        lg.propagate = True
        return lg

    monkeypatch.setattr(audit, "_get_logger", _get_logger_with_propagation)


def _last(caplog: pytest.LogCaptureFixture) -> dict:
    assert caplog.records, "no audit log line captured"
    rendered = caplog.records[-1].message
    try:
        return json.loads(rendered)
    except json.JSONDecodeError as exc:  # pragma: no cover — informative on failure
        raise AssertionError(f"audit log line is not JSON: {rendered!r}") from exc


# ---- AuditRecord.as_json_payload -------------------------------------------


def test_record_has_utc_iso_timestamp() -> None:
    payload = AuditRecord(event=AuditEvent.UPLOAD_ACCEPTED).as_json_payload()
    assert ISO_UTC.match(payload["ts"]), f"ts not UTC ISO 8601: {payload['ts']}"
    assert payload["event"] == "upload_accepted"


def test_record_omits_none_fields_except_details() -> None:
    payload = AuditRecord(event=AuditEvent.UPLOAD_REJECTED).as_json_payload()
    for key in ("workspace_id", "document_id", "client_ip", "sha256", "size_bytes"):
        assert key not in payload
    # details is always present per contract — None when empty.
    assert "details" in payload
    assert payload["details"] is None


def test_record_includes_populated_fields() -> None:
    payload = AuditRecord(
        event=AuditEvent.UPLOAD_STORED,
        workspace_id="w-123",
        document_id="d-456",
        client_ip="203.0.113.9",
        sha256="deadbeef" * 8,
        size_bytes=12345,
        details={"pages": 42, "ocr_tier": "tesseract"},
    ).as_json_payload()
    assert payload["workspace_id"] == "w-123"
    assert payload["document_id"] == "d-456"
    assert payload["client_ip"] == "203.0.113.9"
    assert payload["sha256"].startswith("deadbeef")
    assert payload["size_bytes"] == 12345
    assert payload["details"]["pages"] == 42


def test_record_stringifies_uuid_doc_id() -> None:
    did = uuid.uuid4()
    payload = AuditRecord(
        event=AuditEvent.DOCUMENT_READY, document_id=did
    ).as_json_payload()
    assert payload["document_id"] == str(did)


# ---- log_event stdout sink -------------------------------------------------


def test_log_event_emits_json_line(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        log_event(
            AuditEvent.UPLOAD_ACCEPTED,
            workspace_id="w-9",
            client_ip="198.51.100.1",
            sha256="a" * 64,
            size_bytes=4096,
        )
    parsed = _last(caplog)
    assert parsed["event"] == "upload_accepted"
    assert parsed["workspace_id"] == "w-9"
    assert parsed["sha256"] == "a" * 64
    assert parsed["size_bytes"] == 4096


def test_log_event_accepts_string_event(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        log_event("login_success", workspace_id="w-1")
    assert _last(caplog)["event"] == "login_success"


def test_log_event_warns_on_unknown_event(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=audit.LOGGER_NAME):
        log_event("not_a_real_event", workspace_id="w-1")
    # Must still emit SOMETHING so the mistake is visible.
    assert _last(caplog)["event"] == "audit_unknown_event"


def test_log_event_folds_extras_into_details(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        log_event(
            AuditEvent.PROCESSING_FAILED,
            workspace_id="w",
            document_id=uuid.uuid4(),
            stage="chunking",
            error="list index out of range",
        )
    details = _last(caplog)["details"]
    assert details["stage"] == "chunking"
    assert details["error"] == "list index out of range"


# ---- log_event Postgres sink -----------------------------------------------


class _CapturingSession:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.commits = 0
        self.rollbacks = 0

    def add(self, row: object) -> None:
        self.added.append(row)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _BoomSession(_CapturingSession):
    def commit(self) -> None:
        raise SQLAlchemyError("db down")


def test_log_event_writes_to_db_session() -> None:
    sess = _CapturingSession()
    log_event(
        AuditEvent.UPLOAD_ACCEPTED,
        session=sess,
        workspace_id="w-x",
        sha256="b" * 64,
        size_bytes=100,
    )
    assert sess.commits == 1
    assert len(sess.added) == 1
    row = sess.added[0]
    assert row.event_type == "upload_accepted"
    assert row.workspace_id == "w-x"
    assert row.sha256 == "b" * 64


def test_log_event_never_raises_on_db_failure() -> None:
    sess = _BoomSession()
    log_event(
        AuditEvent.UPLOAD_REJECTED,
        session=sess,
        workspace_id="w-1",
        reason="MIME_REJECTED",
    )
    assert sess.rollbacks == 1


def test_log_event_coerces_uuid_doc_id() -> None:
    sess = _CapturingSession()
    did = uuid.uuid4()
    log_event(
        AuditEvent.DOCUMENT_READY,
        session=sess,
        workspace_id="w",
        document_id=did,
    )
    assert sess.added[0].document_id == did


def test_log_event_drops_invalid_doc_id_quietly() -> None:
    sess = _CapturingSession()
    log_event(
        AuditEvent.DOCUMENT_READY,
        session=sess,
        workspace_id="w",
        document_id="not-a-uuid",
    )
    assert sess.added[0].document_id is None


# ---- Convenience wrappers --------------------------------------------------


def test_upload_accepted_wrapper(caplog: pytest.LogCaptureFixture) -> None:
    sess = _CapturingSession()
    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        audit_upload_accepted(
            session=sess,
            workspace_id="w-7",
            document_id=uuid.uuid4(),
            client_ip="203.0.113.1",
            sha256="c" * 64,
            size_bytes=2048,
            original_filename="rep (2).pdf",
            sanitized_filename="rep_2.pdf",
            pages=10,
        )
    assert sess.commits == 1
    parsed = _last(caplog)
    assert parsed["event"] == "upload_accepted"
    assert parsed["details"]["original_filename"] == "rep (2).pdf"
    assert parsed["details"]["sanitized_filename"] == "rep_2.pdf"
    assert parsed["details"]["pages"] == 10


def test_upload_rejected_wrapper(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        audit_upload_rejected(
            workspace_id=None,
            client_ip="203.0.113.2",
            reason="file too big",
            code="SIZE_LIMIT",
            size_bytes=99_000_000,
        )
    parsed = _last(caplog)
    assert parsed["event"] == "upload_rejected"
    assert parsed["details"]["code"] == "SIZE_LIMIT"
    assert parsed["size_bytes"] == 99_000_000


def test_payload_stripped_wrapper(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        audit_payload_stripped(
            workspace_id="w-5",
            document_id=str(uuid.uuid4()),
            removed_keys=["/Root/JavaScript", "/Root/OpenAction"],
            sha256_before="aa",
            sha256_after="bb",
        )
    parsed = _last(caplog)
    assert parsed["event"] == "payload_stripped"
    assert parsed["sha256"] == "bb"
    assert parsed["details"]["sha256_before"] == "aa"
    assert "/Root/JavaScript" in parsed["details"]["removed_keys"]


def test_login_wrapper_success_and_failure(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=audit.LOGGER_NAME):
        audit_login(workspace_id="w", client_ip="1.1.1.1", success=True)
        audit_login(
            workspace_id=None,
            client_ip="1.1.1.1",
            success=False,
            reason="rate_limited",
        )
    events = [json.loads(r.message)["event"] for r in caplog.records[-2:]]
    assert events == ["login_success", "login_failed"]


# ---- client_ip_from_headers ------------------------------------------------


def test_client_ip_prefers_leftmost_xff() -> None:
    ip = client_ip_from_headers(
        remote_addr="127.0.0.1",
        forwarded_for="203.0.113.7, 70.41.3.18, 150.172.238.178",
    )
    assert ip == "203.0.113.7"


def test_client_ip_falls_back_to_remote_addr() -> None:
    ip = client_ip_from_headers(remote_addr="192.0.2.5", forwarded_for=None)
    assert ip == "192.0.2.5"


def test_client_ip_handles_empty_xff() -> None:
    ip = client_ip_from_headers(remote_addr="192.0.2.5", forwarded_for="  ,  ")
    assert ip == "192.0.2.5"


def test_client_ip_returns_none_when_nothing_known() -> None:
    assert client_ip_from_headers(remote_addr=None, forwarded_for=None) is None


# ---- extract_client_ip (request-aware helper) ------------------------------


class _FakeHeaders:
    def __init__(self, items: dict[str, str]) -> None:
        self._items = dict(items)

    def get(self, key: str, default: str | None = None) -> str | None:
        for k, v in self._items.items():
            if k.lower() == key.lower():
                return v
        return default


class _FakeClient:
    def __init__(self, host: str) -> None:
        self.host = host


class _FakeRequest:
    def __init__(self, headers: dict[str, str], host: str = "10.0.0.1") -> None:
        self.headers = _FakeHeaders(headers)
        self.client = _FakeClient(host)


def test_extract_client_ip_prefers_xff() -> None:
    req = _FakeRequest(
        {"X-Forwarded-For": "203.0.113.7, 70.41.3.18, 150.172.238.178"},
        host="127.0.0.1",
    )
    assert extract_client_ip(req) == "203.0.113.7"


def test_extract_client_ip_falls_back_to_real_ip() -> None:
    req = _FakeRequest({"X-Real-IP": "198.51.100.4"}, host="127.0.0.1")
    assert extract_client_ip(req) == "198.51.100.4"


def test_extract_client_ip_uses_socket_when_no_proxy() -> None:
    req = _FakeRequest({}, host="192.0.2.5")
    assert extract_client_ip(req) == "192.0.2.5"


def test_extract_client_ip_handles_none() -> None:
    assert extract_client_ip(None) is None
