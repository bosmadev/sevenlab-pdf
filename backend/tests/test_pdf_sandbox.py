"""Host-side PDF sandbox invoker tests (T-004 Docker sandbox, ISO 27001 A.8.7).

Covers ``app.services.pdf_sandbox`` — the canonical module. These tests do
NOT require Docker: they verify the docker argv isolation flags with a
patched subprocess, structured-failure propagation, and the inline-mode
fallback used in dev / CI.

If any `REQUIRED_DOCKER_FLAGS` entry disappears from the argv, this file
blocks the merge — the isolation is a security contract, not a
performance detail.
"""
from __future__ import annotations

import base64
import json
import subprocess

import pytest

from app.services import pdf_sandbox as sb


REQUIRED_DOCKER_FLAGS: tuple[str, ...] = (
    "--network=none",
    "--read-only",
    "--cap-drop",
    "--security-opt",
    "--pids-limit",
    "--memory",
    "--cpus",
    "-i",
    "--rm",
)


def _ok_payload() -> dict:
    return {
        "sanitizer": {
            "original_sha256": "a" * 64,
            "sanitized_sha256": "b" * 64,
            "stripped_keys": [],
            "pages": 2,
            "was_encrypted": False,
            "original_bytes": 10,
            "sanitized_bytes": 9,
        },
        "sanitized_pdf_b64": base64.b64encode(b"%PDF-1.4\n...").decode("ascii"),
        "extraction": {
            "page_count": 2,
            "truncated": False,
            "pages": [{"page_num": 1, "text": "hello", "bbox_image_regions": []}],
            "tables": [],
        },
    }


def _capture_argv(monkeypatch: pytest.MonkeyPatch, stdout: bytes) -> list[str]:
    captured: dict[str, list[str]] = {}

    def _fake_run(argv, **_kw):
        captured["argv"] = list(argv)
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout=stdout, stderr=b""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    sb._run_docker(b"%PDF-1.4\n...")
    return captured["argv"]


# ------------------------------------------------------------- isolation flags


def test_docker_argv_contains_every_isolation_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    argv = _capture_argv(monkeypatch, json.dumps(_ok_payload()).encode())
    for flag in REQUIRED_DOCKER_FLAGS:
        assert flag in argv, f"missing isolation flag: {flag}"
    assert argv[argv.index("--user") + 1] == "65534:65534"
    assert argv[0] == "docker"
    assert "run" in argv


def test_docker_argv_uses_configured_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PDF_SANDBOX_IMAGE", "sevenlab-pdf-worker:test-tag")
    argv = _capture_argv(monkeypatch, json.dumps(_ok_payload()).encode())
    assert argv[-1] == "sevenlab-pdf-worker:test-tag"


# --------------------------------------------------------- payload handling


def test_sandbox_process_rejects_empty_bytes() -> None:
    with pytest.raises(sb.SandboxError, match="empty payload"):
        sb.sandbox_process(b"")


def test_sandbox_process_docker_parses_ok_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PDF_SANDBOX_MODE", "docker")

    def _fake_run(argv, **_kw):
        return subprocess.CompletedProcess(
            args=argv,
            returncode=0,
            stdout=json.dumps(_ok_payload()).encode(),
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    result = sb.sandbox_process(b"%PDF-1.4\n...")
    assert result.mode == "docker"
    assert result.pages == 2
    assert result.original_sha256 == "a" * 64
    assert result.sanitized_pdf.startswith(b"%PDF-")


def test_sandbox_process_propagates_worker_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runner-side `{error, type}` MUST become a SandboxError the route can
    catch and audit-log — not an opaque crash.
    """
    monkeypatch.setenv("PDF_SANDBOX_MODE", "docker")
    err_payload = {
        "error": "encrypted PDFs are rejected",
        "type": "PdfSanitizationError",
    }

    def _fake_run(argv, **_kw):
        return subprocess.CompletedProcess(
            args=argv,
            returncode=2,
            stdout=json.dumps(err_payload).encode(),
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    with pytest.raises(sb.SandboxError) as excinfo:
        sb.sandbox_process(b"%PDF-1.4\n...")
    assert "PdfSanitizationError" in str(excinfo.value)
    assert "encrypted" in str(excinfo.value)


def test_sandbox_process_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PDF_SANDBOX_MODE", "docker")

    def _boom(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd=["docker"], timeout=1)

    monkeypatch.setattr(subprocess, "run", _boom)
    with pytest.raises(sb.SandboxError, match="timed out"):
        sb.sandbox_process(b"%PDF-1.4\n...")


def test_sandbox_process_raises_when_docker_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PDF_SANDBOX_MODE", "docker")

    def _no_docker(*_a, **_kw):
        raise FileNotFoundError(2, "No such file: docker")

    monkeypatch.setattr(subprocess, "run", _no_docker)
    with pytest.raises(sb.SandboxError, match="docker binary"):
        sb.sandbox_process(b"%PDF-1.4\n...")


def test_sandbox_process_raises_on_non_json_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PDF_SANDBOX_MODE", "docker")

    def _fake_run(argv, **_kw):
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout=b"<<not JSON>>", stderr=b""
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    with pytest.raises(sb.SandboxError, match="non-JSON"):
        sb.sandbox_process(b"%PDF-1.4\n...")


def test_sandbox_process_raises_on_rc_1(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only rc 0 (ok) and rc 2 (structured error) are valid from pdf_worker.
    Anything else (OOM-kill rc 137, docker rc 125) must raise.
    """
    monkeypatch.setenv("PDF_SANDBOX_MODE", "docker")

    def _fake_run(argv, **_kw):
        return subprocess.CompletedProcess(
            args=argv, returncode=137, stdout=b"", stderr=b"Killed"
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    with pytest.raises(sb.SandboxError, match="rc=137"):
        sb.sandbox_process(b"%PDF-1.4\n...")


# --------------------------------------------------------- inline fallback


def test_inline_mode_rejects_non_pdf_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PDF_SANDBOX_MODE", "inline")
    with pytest.raises(sb.SandboxError, match="sanitize failed"):
        sb.sandbox_process(b"this is not a pdf")


def test_inline_mode_round_trips_a_minimal_pdf(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real sanitizer+extractor to catch wiring regressions."""
    monkeypatch.setenv("PDF_SANDBOX_MODE", "inline")

    import io

    import pikepdf

    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    buf = io.BytesIO()
    pdf.save(buf)

    result = sb.sandbox_process(buf.getvalue())
    assert result.mode == "inline"
    assert result.pages == 1
    assert result.sanitized_pdf.startswith(b"%PDF-")
