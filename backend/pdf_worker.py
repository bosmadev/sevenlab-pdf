"""Isolated PDF worker — invoked inside the Docker sandbox.

Reads PDF bytes from stdin, runs sanitizer + extractor, writes a JSON
report to stdout. Any error terminates with a JSON `{"error": "..."}`
on stdout and non-zero exit, so the parent can surface the reason
without parsing stderr.

Called by `app.services.pdf_sandbox.sandbox_process`. Must stay small
and dependency-free beyond `app.services.pdf_sanitizer` / `pdf_extractor`,
because the sandbox image avoids installing anything else.
"""
from __future__ import annotations

import base64
import json
import sys
import traceback
from dataclasses import asdict
from typing import Any


def _read_stdin_bytes() -> bytes:
    buffer = sys.stdin.buffer.read()
    if not buffer:
        raise RuntimeError("empty stdin")
    return buffer


def _run(pdf_bytes: bytes) -> dict[str, Any]:
    from app.services.pdf_extractor import extract
    from app.services.pdf_sanitizer import sanitize_pdf

    sanitized_bytes, report = sanitize_pdf(pdf_bytes)
    extraction = extract(sanitized_bytes)

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


def main() -> int:
    try:
        pdf_bytes = _read_stdin_bytes()
        result = _run(pdf_bytes)
        sys.stdout.write(json.dumps(result))
        sys.stdout.flush()
        return 0
    except Exception as exc:  # noqa: BLE001 — this is the top-level trap
        payload = {
            "error": str(exc),
            "type": exc.__class__.__name__,
            "trace": traceback.format_exc(limit=4),
        }
        sys.stdout.write(json.dumps(payload))
        sys.stdout.flush()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
