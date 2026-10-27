"""app.logging_config — rotating files under backend/logs/, request-id stamping.

Needs nothing beyond the stdlib (python-json-logger is optional: the test
accepts either the JSON or the plain-text line format).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest


@pytest.fixture
def log_dir(tmp_path: Path):
    from app.logging_config import configure_logging, teardown_logging

    target = tmp_path / "logs"
    configured = configure_logging(log_dir=target, level="INFO")
    yield configured
    teardown_logging()


def _lines(path: Path) -> list[str]:
    for h in logging.getLogger().handlers:
        h.flush()
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_creates_dir_and_both_files(log_dir: Path) -> None:
    assert log_dir is not None and log_dir.is_dir()
    logging.getLogger("sevenlab.test").info("hello-info")
    logging.getLogger("sevenlab.test").warning("hello-warn")

    app_lines = _lines(log_dir / "app.log")
    err_lines = _lines(log_dir / "error.log")
    assert any("hello-info" in ln for ln in app_lines)
    assert any("hello-warn" in ln for ln in app_lines)
    assert any("hello-warn" in ln for ln in err_lines)
    assert not any("hello-info" in ln for ln in err_lines), "INFO must not reach error.log"


def test_lines_carry_request_id_from_contextvar(log_dir: Path) -> None:
    from app.logging_config import request_id_var

    token = request_id_var.set("rid-xyz-123")
    try:
        logging.getLogger("sevenlab.test").warning("tagged-line")
    finally:
        request_id_var.reset(token)

    line = next(ln for ln in _lines(log_dir / "error.log") if "tagged-line" in ln)
    assert "rid-xyz-123" in line
    # JSON format when python-json-logger is installed; plain text otherwise.
    if line.startswith("{"):
        rec = json.loads(line)
        assert rec["request_id"] == "rid-xyz-123"
        assert rec["level"] == "WARNING"


def test_configure_is_idempotent(log_dir: Path) -> None:
    from app.logging_config import configure_logging

    before = len(logging.getLogger().handlers)
    configure_logging(log_dir=log_dir)
    assert len(logging.getLogger().handlers) == before


def test_audit_log_path_defaults_into_dir(log_dir: Path, monkeypatch) -> None:
    import os

    from app.logging_config import configure_logging

    monkeypatch.delenv("SEVENLAB_AUDIT_LOG_PATH", raising=False)
    configure_logging(log_dir=log_dir)
    assert os.environ["SEVENLAB_AUDIT_LOG_PATH"] == str(log_dir / "audit.log")


def test_unwritable_dir_degrades_to_console_only(tmp_path: Path) -> None:
    from app.logging_config import configure_logging, teardown_logging

    blocker = tmp_path / "file-not-dir"
    blocker.write_text("x")
    try:
        assert configure_logging(log_dir=blocker / "logs") is None
        # Console handler still attached — nothing is silently dropped.
        assert any(
            getattr(h, "_sevenlab_handler", None) == "console"
            for h in logging.getLogger().handlers
        )
    finally:
        teardown_logging()
