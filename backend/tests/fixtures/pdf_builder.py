"""Build small, inline PDF fixtures (no external files on disk).

The sanitizer and validator must be exercised against PDFs that actually
contain the active-content keys the stripper is supposed to remove. These
builders use pikepdf to synthesise one-page PDFs with those keys attached,
so the test harness is hermetic and reproducible on CI.
"""
from __future__ import annotations

import io

import pikepdf


def _one_page_pdf() -> pikepdf.Pdf:
    """Return an open one-page PDF with a trivial media box."""
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(72, 72))  # 1in x 1in
    return pdf


def make_clean_pdf() -> bytes:
    """Return a valid, benign one-page PDF with no active content."""
    pdf = _one_page_pdf()
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def make_malicious_pdf() -> bytes:
    """Return a one-page PDF carrying every key the sanitizer must strip.

    Attaches an ``/OpenAction`` running JavaScript, a page-level ``/AA``
    additional-action, ``/Launch`` on the catalog, an ``/EmbeddedFiles``
    entry under ``/Names``, and a top-level ``/JavaScript`` names tree.
    These are the standard PDF-malware carriers (NIST SP 800-100 §10).
    """
    pdf = _one_page_pdf()

    # 1. /OpenAction with embedded JavaScript at document load.
    js_action = pikepdf.Dictionary(
        Type=pikepdf.Name("/Action"),
        S=pikepdf.Name("/JavaScript"),
        JS=pikepdf.String("app.alert('owned');"),
    )
    pdf.Root["/OpenAction"] = js_action

    # 2. /Launch action targeting a local binary.
    launch_action = pikepdf.Dictionary(
        Type=pikepdf.Name("/Action"),
        S=pikepdf.Name("/Launch"),
        F=pikepdf.String("calc.exe"),
    )
    pdf.Root["/Launch"] = launch_action

    # 3. /Names catalog with /JavaScript and /EmbeddedFiles subtrees.
    pdf.Root["/Names"] = pikepdf.Dictionary(
        JavaScript=pikepdf.Dictionary(
            Names=pikepdf.Array([
                pikepdf.String("payload"),
                pikepdf.Dictionary(
                    Type=pikepdf.Name("/Action"),
                    S=pikepdf.Name("/JavaScript"),
                    JS=pikepdf.String("app.alert('names-tree');"),
                ),
            ])
        ),
        EmbeddedFiles=pikepdf.Dictionary(
            Names=pikepdf.Array([pikepdf.String("evil.exe"), pikepdf.String("file-ref")])
        ),
    )

    # 4. Per-page additional-action (/AA) with /O (page open) trigger.
    page = pdf.pages[0]
    page.obj["/AA"] = pikepdf.Dictionary(
        O=pikepdf.Dictionary(
            Type=pikepdf.Name("/Action"),
            S=pikepdf.Name("/JavaScript"),
            JS=pikepdf.String("app.alert('aa-trigger');"),
        )
    )

    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue()


def make_encrypted_pdf(password: str = "secret") -> bytes:
    """Return a password-protected PDF (the sanitizer must reject this)."""
    pdf = _one_page_pdf()
    buf = io.BytesIO()
    pdf.save(
        buf,
        encryption=pikepdf.Encryption(owner=password, user=password, R=4),
    )
    return buf.getvalue()


def make_pdf_with_leading_junk(junk_bytes: int = 512) -> bytes:
    """A valid PDF with ``junk_bytes`` of garbage before the ``%PDF-`` header.

    ISO 32000-1 §7.5.2 allows up to 1024 bytes of leading junk — the validator
    must still accept these when the magic is within the scan window.
    """
    base = make_clean_pdf()
    return (b"X" * junk_bytes) + base


def make_pdf_truncated_header(trim_to: int = 3) -> bytes:
    """Return bytes missing most of the ``%PDF-`` header (validator must reject)."""
    return b"%PD"[:trim_to]


__all__ = [
    "make_clean_pdf",
    "make_malicious_pdf",
    "make_encrypted_pdf",
    "make_pdf_with_leading_junk",
    "make_pdf_truncated_header",
]
