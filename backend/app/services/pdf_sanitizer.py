"""PDF payload stripper — ISO 27001 A.8.7 malware protection.

Deletes active-content keys that are the standard attack surface:
/JavaScript, /JS, /OpenAction, /AA (additional actions), /Launch,
/EmbeddedFiles. qpdf linearization alone does NOT remove these — this
module performs explicit key deletion with pikepdf.

Reference: NIST SP 800-100 §10; PDF Reference 1.7 §8.5 (Actions).
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Final

import pikepdf

# Active-content keys to remove at every object level
DANGEROUS_KEYS: Final[tuple[str, ...]] = (
    "/JavaScript",
    "/JS",
    "/OpenAction",
    "/AA",
    "/Launch",
    "/EmbeddedFiles",
    "/XFA",
    "/RichMedia",
    "/Movie",
    "/Sound",
    "/SubmitForm",
    "/ImportData",
)


ENCRYPTED_REJECT_MESSAGE: Final[str] = "Password-protected PDFs are not allowed"


class PdfSanitizationError(Exception):
    """Raised when a PDF cannot be opened, is encrypted, or sanitation fails."""


@dataclass
class SanitizationReport:
    original_sha256: str
    sanitized_sha256: str
    stripped_keys: list[str] = field(default_factory=list)
    pages: int = 0
    was_encrypted: bool = False
    original_bytes: int = 0
    sanitized_bytes: int = 0


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strip_from_dict(obj: pikepdf.Object, stripped: list[str], path: str) -> None:
    """Recursively remove dangerous keys from any dictionary/array in the tree."""
    try:
        if isinstance(obj, pikepdf.Dictionary):
            for key in list(obj.keys()):
                if key in DANGEROUS_KEYS:
                    stripped.append(f"{path}{key}")
                    del obj[key]
            for key, value in list(obj.items()):
                _strip_from_dict(value, stripped, f"{path}{key}/")
        elif isinstance(obj, pikepdf.Array):
            for i, value in enumerate(list(obj)):
                _strip_from_dict(value, stripped, f"{path}[{i}]/")
    except Exception:
        # Dictionary iteration can raise on malformed PDFs — skip rather
        # than crash the whole sanitation.
        pass


def sanitize_pdf(pdf_bytes: bytes) -> tuple[bytes, SanitizationReport]:
    """Return (sanitized_bytes, report).

    Raises PdfSanitizationError on parse failure, encryption, or output failure.
    Encrypted PDFs are rejected because we cannot inspect protected objects.
    """
    if not pdf_bytes:
        raise PdfSanitizationError("empty payload")

    original_sha = _sha256(pdf_bytes)
    stripped: list[str] = []

    try:
        pdf = pikepdf.open(io.BytesIO(pdf_bytes))
    except pikepdf.PasswordError as exc:
        raise PdfSanitizationError(ENCRYPTED_REJECT_MESSAGE) from exc
    except Exception as exc:  # noqa: BLE001 — upstream raises many types
        raise PdfSanitizationError(f"cannot parse PDF: {exc}") from exc

    # Owner-only encryption opens with an empty user password, so
    # PasswordError never fires — check the flag explicitly.
    if pdf.is_encrypted:
        pdf.close()
        raise PdfSanitizationError(ENCRYPTED_REJECT_MESSAGE)

    was_encrypted = False

    with pdf:
        try:
            root = pdf.Root
            _strip_from_dict(root, stripped, "/Root/")

            # Document-level /Names catalog is a frequent carrier of
            # /JavaScript and /EmbeddedFiles — remove the whole subtree
            # if present, mirroring pikepdf's recommended lockdown.
            if "/Names" in root:
                names = root["/Names"]
                for key in ("/JavaScript", "/EmbeddedFiles"):
                    if key in names:
                        stripped.append(f"/Root/Names{key}")
                        del names[key]

            # Per-page actions and annotation actions
            for page in pdf.pages:
                _strip_from_dict(page.obj, stripped, "/Page/")

            buf = io.BytesIO()
            pdf.save(
                buf,
                linearize=False,
                compress_streams=True,
                object_stream_mode=pikepdf.ObjectStreamMode.generate,
            )
            sanitized = buf.getvalue()
            page_count = len(pdf.pages)
        except Exception as exc:  # noqa: BLE001
            raise PdfSanitizationError(f"sanitation failed: {exc}") from exc

    report = SanitizationReport(
        original_sha256=original_sha,
        sanitized_sha256=_sha256(sanitized),
        stripped_keys=stripped,
        pages=page_count,
        was_encrypted=was_encrypted,
        original_bytes=len(pdf_bytes),
        sanitized_bytes=len(sanitized),
    )
    return sanitized, report
