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
                                                 production — refused when
                                                 APP_ENV=production.

Stages (`--stage` on the worker, `stage=` here):
    all       sanitize + extract in one container (default, round-trip tests)
    sanitize  upload path: strip active payloads, return the clean bytes
    extract   pipeline path: text/table extraction of already-sanitized bytes

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

    def to_extraction_result(self) -> Any:
        """Rebuild ``pdf_extractor.ExtractionResult`` from the worker's JSON.

        Lazy import: docker mode must not pay for PyMuPDF / pdfplumber at
        module load, and this path only needs the dataclasses.
        """
        from app.services.pdf_extractor import ExtractionResult, PageTable, PageText

        payload = self.extraction or {}
        pages = [
            PageText(
                page_num=int(p["page_num"]),
                text=str(p.get("text") or ""),
                bbox_image_regions=[
                    tuple(r) for r in (p.get("bbox_image_regions") or [])
                ],
            )
            for p in (payload.get("pages") or [])
        ]
        tables = [
            PageTable(page_num=int(t["page_num"]), rows=list(t.get("rows") or []))
            for t in (payload.get("tables") or [])
        ]
        return ExtractionResult(
            pages=pages,
            tables=tables,
            page_count=int(payload.get("page_count") or len(pages)),
            truncated=bool(payload.get("truncated", False)),
        )


_IMAGE_DEFAULT = "sevenlab-pdf-worker:latest"
_TIMEOUT_DEFAULT = 300  # seconds
_MEMORY_DEFAULT = "2g"
_CPUS_DEFAULT = "2"
_PIDS_LIMIT_DEFAULT = "100"


def _env(name: str, default: str) -> str:
    val = os.getenv(name, "").strip()
    return val if val else default


_STAGES: tuple[str, ...] = ("all", "sanitize", "extract")


def _is_production() -> bool:
    try:
        from app.config import get_settings

        return get_settings().app_env.strip().lower() == "production"
    except Exception:  # noqa: BLE001 — settings must never break the sandbox
        return os.getenv("APP_ENV", "").strip().lower() == "production"


def sandbox_mode() -> str:
    """Return `docker` or `inline` (no production guard — see ``_mode``).

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


def _mode() -> str:
    """Resolve the mode and fail closed: inline is refused in production.

    ``PDF_SANDBOX_ALLOW_INLINE_IN_PRODUCTION=1`` overrides — only for a box
    that cannot run Docker and whose uploaders are all trusted.
    """
    mode = sandbox_mode()
    if mode == "inline" and _is_production():
        if os.getenv("PDF_SANDBOX_ALLOW_INLINE_IN_PRODUCTION", "").strip() == "1":
            log.error("pdf_sandbox: inline mode in production (explicitly allowed by env)")
            return mode
        raise SandboxError(
            "PDF sandbox is in inline mode but APP_ENV=production; "
            "refusing to parse untrusted PDFs in-process (ISO 27001 A.8.7)"
        )
    return mode


def _run_docker(pdf_bytes: bytes, stage: str = "all") -> dict[str, Any]:
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
    if stage != "all":
        argv += ["--stage", stage]
    log.info(
        "pdf_sandbox: launching docker image=%s stage=%s mem=%s cpus=%s pids=%s timeout=%ss",
        image, stage, memory, cpus, pids_limit, timeout_s,
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


def _run_inline(pdf_bytes: bytes, stage: str = "all") -> dict[str, Any]:
    """Dev/CI fallback. Imports locally so docker-mode avoids the cost.

    Normalises sanitizer / extractor failures into SandboxError so the
    caller sees one error type regardless of the mode in use. Mirrors the
    stage contract of ``pdf_worker.py`` exactly so the two paths cannot drift.
    """
    from dataclasses import asdict

    payload: dict[str, Any] = {
        "sanitizer": None,
        "sanitized_pdf_b64": None,
        "extraction": None,
    }
    working = pdf_bytes

    if stage in ("all", "sanitize"):
        from app.services.pdf_sanitizer import PdfSanitizationError, sanitize_pdf

        try:
            working, report = sanitize_pdf(pdf_bytes)
        except PdfSanitizationError as exc:
            raise SandboxError(f"sanitize failed: {exc}") from exc
        payload["sanitizer"] = {
            "original_sha256": report.original_sha256,
            "sanitized_sha256": report.sanitized_sha256,
            "stripped_keys": list(report.stripped_keys),
            "pages": report.pages,
            "was_encrypted": report.was_encrypted,
            "original_bytes": report.original_bytes,
            "sanitized_bytes": report.sanitized_bytes,
        }
        payload["sanitized_pdf_b64"] = base64.b64encode(working).decode("ascii")

    if stage in ("all", "extract"):
        from app.services.pdf_extractor import PdfExtractionError, extract

        try:
            extraction = extract(working)
        except PdfExtractionError as exc:
            raise SandboxError(f"extract failed: {exc}") from exc
        payload["extraction"] = {
            "page_count": extraction.page_count,
            "truncated": extraction.truncated,
            "pages": [asdict(p) for p in extraction.pages],
            "tables": [asdict(t) for t in extraction.tables],
        }

    return payload


def sandbox_process(pdf_bytes: bytes, stage: str = "all") -> SandboxResult:
    """Run one sandbox stage over ``pdf_bytes`` and normalise the payload.

    ``stage="all"`` returns the sanitized PDF plus structured extraction.
    ``stage="sanitize"`` leaves ``extraction`` empty; ``stage="extract"``
    leaves ``sanitized_pdf`` empty. The caller is responsible for persisting
    bytes to GridFS and chunks to the vector DB.
    """
    if not pdf_bytes:
        raise SandboxError("empty payload")
    if stage not in _STAGES:
        raise SandboxError(f"unknown sandbox stage {stage!r}; expected one of {_STAGES}")

    mode = _mode()
    payload = (
        _run_docker(pdf_bytes, stage) if mode == "docker" else _run_inline(pdf_bytes, stage)
    )
    if not isinstance(payload, dict):
        raise SandboxError("worker payload is not a JSON object")

    sanitizer: dict[str, Any] = payload.get("sanitizer") or {}
    extraction: dict[str, Any] = payload.get("extraction") or {}
    sanitized_bytes = b""

    if stage in ("all", "sanitize"):
        sanitized_b64 = payload.get("sanitized_pdf_b64")
        if not sanitizer or sanitized_b64 is None:
            raise SandboxError("worker payload missing sanitizer output")
        try:
            sanitized_bytes = base64.b64decode(sanitized_b64)
        except Exception as exc:  # noqa: BLE001
            raise SandboxError(f"worker returned invalid base64: {exc}") from exc

    if stage in ("all", "extract") and not extraction:
        raise SandboxError("worker payload missing extraction output")

    return SandboxResult(
        sanitized_pdf=sanitized_bytes,
        sanitizer=sanitizer,
        extraction=extraction,
        stripped_keys=list(sanitizer.get("stripped_keys", [])),
        pages=int(sanitizer.get("pages") or extraction.get("page_count") or 0),
        original_sha256=str(sanitizer.get("original_sha256") or ""),
        sanitized_sha256=str(sanitizer.get("sanitized_sha256") or ""),
        mode=mode,
    )


def sandbox_sanitize(pdf_bytes: bytes) -> SandboxResult:
    """Upload path (``routes/documents.py``): strip active payloads, no extraction."""
    return sandbox_process(pdf_bytes, stage="sanitize")


def sandbox_extract(pdf_bytes: bytes) -> Any:
    """Pipeline path (``processing/pipeline.py``): extract already-sanitized bytes.

    Returns an ``app.services.pdf_extractor.ExtractionResult`` rebuilt from
    the worker's JSON so the chunking stage sees the same shape as an
    in-process ``extract()`` call.
    """
    result = sandbox_process(pdf_bytes, stage="extract")
    return _extraction_from_payload(result.extraction)


def _extraction_from_payload(data: dict[str, Any]) -> Any:
    from app.services.pdf_extractor import ExtractionResult, PageTable, PageText

    pages = [
        PageText(
            page_num=int(p.get("page_num") or 0),
            text=str(p.get("text") or ""),
            bbox_image_regions=[tuple(r) for r in (p.get("bbox_image_regions") or [])],
        )
        for p in (data.get("pages") or [])
    ]
    tables = [
        PageTable(page_num=int(t.get("page_num") or 0), rows=list(t.get("rows") or []))
        for t in (data.get("tables") or [])
    ]
    return ExtractionResult(
        pages=pages,
        tables=tables,
        page_count=int(data.get("page_count") or len(pages)),
        truncated=bool(data.get("truncated", False)),
    )
