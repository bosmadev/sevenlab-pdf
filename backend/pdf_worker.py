"""Isolated PDF worker — invoked inside the Docker sandbox.

Reads PDF bytes from stdin, runs the requested stage, writes a JSON report
to stdout. Any error terminates with a JSON `{"error": "..."}` on stdout
and exit code 2, so the parent can surface the reason without parsing
stderr.

Stages (``--stage``):
    all       sanitize + extract (default)
    sanitize  strip active payloads only — upload path
    extract   text/table extraction only — pipeline path, input is already
              sanitized bytes read back from GridFS

Keys absent for a stage are emitted as ``null`` so the host-side parser in
``app.services.pdf_sandbox`` can tell "not requested" from "missing".

Called by `app.services.pdf_sandbox.sandbox_process`. Must stay small
and dependency-free beyond `app.services.pdf_sanitizer` / `pdf_extractor`,
because the sandbox image avoids installing anything else.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import traceback
from dataclasses import asdict
from typing import Any

STAGES: tuple[str, ...] = ("all", "sanitize", "extract")


def _parse_stage(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(prog="pdf_worker", add_help=False)
    parser.add_argument("--stage", choices=STAGES, default="all")
    ns, _unknown = parser.parse_known_args(argv)
    return ns.stage


def _read_stdin_bytes() -> bytes:
    buffer = sys.stdin.buffer.read()
    if not buffer:
        raise RuntimeError("empty stdin")
    return buffer


def _run(pdf_bytes: bytes, stage: str = "all") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sanitizer": None,
        "sanitized_pdf_b64": None,
        "extraction": None,
    }
    working = pdf_bytes

    if stage in ("all", "sanitize"):
        from app.services.pdf_sanitizer import sanitize_pdf

        working, report = sanitize_pdf(pdf_bytes)
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
        from app.services.pdf_extractor import extract

        extraction = extract(working)
        payload["extraction"] = {
            "page_count": extraction.page_count,
            "truncated": extraction.truncated,
            "pages": [asdict(p) for p in extraction.pages],
            "tables": [asdict(t) for t in extraction.tables],
        }

    return payload


def main(argv: list[str] | None = None) -> int:
    try:
        stage = _parse_stage(sys.argv[1:] if argv is None else argv)
        pdf_bytes = _read_stdin_bytes()
        result = _run(pdf_bytes, stage)
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
