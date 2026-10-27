"""Dockerized sandbox for PDF sanitizer + extractor (ISO 27001 A.8.7).

The sandbox contract is byte-level:
    input:  raw PDF bytes on stdin
    output: JSON on stdout with keys {sanitizer, extraction, sanitized_pdf_b64}
            or {error, type, trace} on failure (non-zero exit).

Why a container, not just a subprocess
--------------------------------------
PyMuPDF / pdfplumber / pikepdf execute parser code against attacker-chosen
input. The three hardening layers Docker provides that a bare subprocess
does not:
    --network=none      kills any exfiltration via embedded links / fonts
    --read-only + tmpfs makes dropped files vanish; nothing persists
    --pids-limit        caps the fork bomb class trivially

Mode selection via `PDF_SANDBOX_MODE`:
    docker  (default when docker is available)  Run the pdf-worker image.
    inline  (dev / CI without docker)           Run in-process. **NOT** safe
                                                 for untrusted uploads in
                                                 production.

Resource caps (overridable via env):
    PDF_SANDBOX_TIMEOUT_S       300   worker killed if it hangs
    PDF_SANDBOX_MEMORY          2g    docker --memory
    PDF_SANDBOX_CPUS            2     docker --cpus
    PDF_SANDBOX_IMAGE           sevenlab-pdf-worker:latest
    PDF_SANDBOX_PIDS_LIMIT      100
"""
from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


class SandboxError(Exception):
    """Raised when the sandboxed worker could not complete."""


@dataclass
class SandboxResult:
    sanitized_pdf: bytes
    sanitizer: dict[str, Any]
    extraction: dict[str, Any]
    stripped_keys: list[str] = field(default_factory=list)
    pages: int = 0
    original_sha256: str = ""
    sanitized_sha256: str = ""
    mode: str = "inline"


_IMAGE_DEFAULT = "sevenlab-pdf-worker:latest"
_TIMEOUT_DEFAULT = 300  # seconds
_MEMORY_DEFAULT = "2g"
_CPUS_DEFAULT = "2"
_PIDS_LIMIT_DEFAULT = "100"


def _env(name: str, default: str) -> str:
    val = os.getenv(name, "").strip()
    return val if val else default


def _mode() -> str:
    """Return `docker` or `inline`.

    Explicit env wins. Otherwise default to `docker` when the binary is
    on PATH, else `inline` with a loud warning.
    """
    requested = os.getenv("PDF_SANDBOX_MODE", "").strip().lower()
    if requested in ("docker", "inline"):
        return requested
    if shutil.which("docker"):
        return "docker"
    log.warning(
        "pdf_sandbox: docker not on PATH; falling back to inline mode. "
        "This is UNSAFE for untrusted uploads in production."
    )
    return "inline"


def _run_docker(pdf_bytes: bytes) -> dict[str, Any]:
    image = _env("PDF_SANDBOX_IMAGE", _IMAGE_DEFAULT)
    timeout_s = int(_env("PDF_SANDBOX_TIMEOUT_S", str(_TIMEOUT_DEFAULT)))
    memory = _env("PDF_SANDBOX_MEMORY", _MEMORY_DEFAULT)
    cpus = _env("PDF_SANDBOX_CPUS", _CPUS_DEFAULT)
    pids_limit = _env("PDF_SANDBOX_PIDS_LIMIT", _PIDS_LIMIT_DEFAULT)

    argv = [
        "docker", "run", "--rm", "-i",
        "--network=none",
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=512m",
        "--user", "65534:65534",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", pids_limit,
        "--memory", memory,
        "--memory-swap", memory,          # disable swap, matches memory
        "--cpus", cpus,
        "--stop-timeout", "5",
        image,
    ]
    log.info(
        "pdf_sandbox: launching docker image=%s mem=%s cpus=%s pids=%s timeout=%ss",
        image, memory, cpus, pids_limit, timeout_s,
    )
    try:
        proc = subprocess.run(
            argv,
            input=pdf_bytes,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise SandboxError(f"sandbox timed out after {timeout_s}s") from exc
    except FileNotFoundError as exc:
        raise SandboxError("docker binary not available") from exc

    if proc.returncode not in (0, 2):
        tail = proc.stderr[-400:].decode("utf-8", errors="replace") if proc.stderr else ""
        raise SandboxError(
            f"sandbox exited rc={proc.returncode}: {tail or '<no stderr>'}"
        )

    try:
        payload = json.loads(proc.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        head = proc.stdout[:200].decode("utf-8", errors="replace")
        raise SandboxError(f"sandbox returned non-JSON stdout: {head!r}") from exc

    if "error" in payload:
        raise SandboxError(
            f"{payload.get('type', 'WorkerError')}: {payload.get('error', 'unknown')}"
        )
    return payload


def _run_inline(pdf_bytes: bytes) -> dict[str, Any]:
    """Dev/CI fallback. Imports locally so docker-mode avoids the cost.

    Normalises sanitizer / extractor failures into SandboxError so the
    caller sees one error type regardless of the mode in use.
    """
    from dataclasses import asdict

    from app.services.pdf_extractor import PdfExtractionError, extract
    from app.services.pdf_sanitizer import PdfSanitizationError, sanitize_pdf

    try:
        sanitized_bytes, report = sanitize_pdf(pdf_bytes)
    except PdfSanitizationError as exc:
        raise SandboxError(f"sanitize failed: {exc}") from exc
    try:
        extraction = extract(sanitized_bytes)
    except PdfExtractionError as exc:
        raise SandboxError(f"extract failed: {exc}") from exc
    return {
        "sanitizer": {
            "original_sha256": report.original_sha256,
            "sanitized_sha256": report.sanitized_sha256,
            "stripped_keys": list(report.stripped_keys),
            "pages": report.pages,
            "was_encrypted": report.was_encrypted,
            "original_bytes": report.original_bytes,
            "sanitized_bytes": report.sanitized_bytes,
        },
        "sanitized_pdf_b64": base64.b64encode(sanitized_bytes).decode("ascii"),
        "extraction": {
            "page_count": extraction.page_count,
            "truncated": extraction.truncated,
            "pages": [asdict(p) for p in extraction.pages],
            "tables": [asdict(t) for t in extraction.tables],
        },
    }


def sandbox_process(pdf_bytes: bytes) -> SandboxResult:
    """Entry point used by the upload route (T-021).

    Returns the sanitized PDF plus structured extraction. The caller is
    responsible for persisting bytes to GridFS and chunks to the vector DB.
    """
    if not pdf_bytes:
        raise SandboxError("empty payload")

    mode = _mode()
    payload = _run_docker(pdf_bytes) if mode == "docker" else _run_inline(pdf_bytes)

    try:
        sanitized_b64 = payload["sanitized_pdf_b64"]
        sanitizer = payload["sanitizer"]
        extraction = payload["extraction"]
    except KeyError as exc:
        raise SandboxError(f"worker payload missing key: {exc}") from exc

    try:
        sanitized_bytes = base64.b64decode(sanitized_b64)
    except Exception as exc:  # noqa: BLE001
        raise SandboxError(f"worker returned invalid base64: {exc}") from exc

    return SandboxResult(
        sanitized_pdf=sanitized_bytes,
        sanitizer=sanitizer,
        extraction=extraction,
        stripped_keys=list(sanitizer.get("stripped_keys", [])),
        pages=int(sanitizer.get("pages") or 0),
        original_sha256=str(sanitizer.get("original_sha256") or ""),
        sanitized_sha256=str(sanitizer.get("sanitized_sha256") or ""),
        mode=mode,
    )
