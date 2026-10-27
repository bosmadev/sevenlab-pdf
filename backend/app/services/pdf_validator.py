"""PDF upload validation — MIME, magic bytes, size, filename sanitization.

ISO 27001 A.8.29 input sanitization. OWASP file-upload best practices.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Final

MAX_PDF_SIZE_MB: Final[int] = int(os.getenv("MAX_PDF_SIZE_MB", "50"))
MAX_PDF_SIZE_BYTES: Final[int] = MAX_PDF_SIZE_MB * 1024 * 1024

ALLOWED_MIME: Final[frozenset[str]] = frozenset({"application/pdf"})
PDF_MAGIC: Final[bytes] = b"%PDF-"
PDF_MAGIC_SCAN_WINDOW: Final[int] = 1024

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")
_MULTI_DOT = re.compile(r"\.{2,}")
_DOUBLE_EXT = re.compile(r"\.pdf\.[A-Za-z0-9]{1,8}$", re.IGNORECASE)


class PdfValidationError(Exception):
    """Raised on any upload-validation failure. Message is user-safe."""

    def __init__(self, reason: str, code: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code


@dataclass(frozen=True)
class ValidatedUpload:
    safe_filename: str
    size_bytes: int
    mime: str


def _ensure(cond: bool, reason: str, code: str) -> None:
    if not cond:
        raise PdfValidationError(reason, code)


def sanitize_filename(raw: str) -> str:
    """Return a safe server-side filename.

    Strips path traversal, null bytes, double extensions (`.pdf.exe`),
    collapses `..`, caps length, forces `.pdf` suffix.
    """
    _ensure(bool(raw), "empty filename", "FILENAME_EMPTY")
    _ensure("\x00" not in raw, "null byte in filename", "FILENAME_NUL")
    name = os.path.basename(raw.replace("\\", "/"))
    _ensure(name not in ("", ".", ".."), "invalid filename", "FILENAME_INVALID")
    _ensure(
        not _DOUBLE_EXT.search(name),
        "double extension not allowed",
        "FILENAME_DOUBLE_EXT",
    )
    stem, dot, ext = name.rpartition(".")
    _ensure(
        bool(dot) and ext.lower() == "pdf",
        "filename must end with .pdf",
        "FILENAME_NOT_PDF",
    )
    stem = _MULTI_DOT.sub(".", stem).strip(". ")
    stem = _UNSAFE_NAME.sub("_", stem) or "document"
    stem = stem[:120]
    return f"{stem}.pdf"


def validate_size(size_bytes: int) -> None:
    _ensure(size_bytes > 0, "empty upload", "SIZE_EMPTY")
    _ensure(
        size_bytes <= MAX_PDF_SIZE_BYTES,
        f"file exceeds {MAX_PDF_SIZE_MB} MB limit",
        "SIZE_LIMIT",
    )


def validate_mime(declared_mime: str | None) -> str:
    mime = (declared_mime or "").split(";", 1)[0].strip().lower()
    _ensure(mime in ALLOWED_MIME, "only application/pdf accepted", "MIME_REJECTED")
    return mime


def validate_magic_bytes(head: bytes) -> None:
    """Verify `%PDF-` appears within first 1024 bytes (linearized PDFs).

    PDF spec allows up to 1024 bytes of leading junk before the header
    (ISO 32000-1 §7.5.2 note). Scan window, not offset-0-only.
    """
    _ensure(bool(head), "no bytes to inspect", "MAGIC_EMPTY")
    window = head[:PDF_MAGIC_SCAN_WINDOW]
    _ensure(PDF_MAGIC in window, "not a PDF (missing %PDF- header)", "MAGIC_MISSING")


def validate_upload(
    *, raw_filename: str, declared_mime: str | None, size_bytes: int, head_bytes: bytes
) -> ValidatedUpload:
    """One-shot pre-processing validation. Raises PdfValidationError on reject."""
    mime = validate_mime(declared_mime)
    validate_size(size_bytes)
    validate_magic_bytes(head_bytes)
    safe = sanitize_filename(raw_filename)
    return ValidatedUpload(safe_filename=safe, size_bytes=size_bytes, mime=mime)
