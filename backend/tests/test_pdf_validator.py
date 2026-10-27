"""Tests for app.services.pdf_validator (T-004, T-015).

Covers MIME + magic-bytes + size + filename-sanitization behaviour. Each
failure mode gets its own test with the explicit error code asserted, so a
drift in error codes is caught before it leaks into audit logs.
"""
from __future__ import annotations

import pytest

from app.services.pdf_validator import (
    MAX_PDF_SIZE_BYTES,
    PdfValidationError,
    sanitize_filename,
    validate_magic_bytes,
    validate_mime,
    validate_size,
    validate_upload,
)
from tests.fixtures.pdf_builder import (
    make_clean_pdf,
    make_pdf_truncated_header,
    make_pdf_with_leading_junk,
)


class TestSanitizeFilename:
    def test_normal_filename_passes_through(self):
        assert sanitize_filename("report.pdf") == "report.pdf"

    def test_strips_path_traversal(self):
        assert sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
        assert sanitize_filename("..\\..\\system.pdf") == "system.pdf"

    def test_rejects_double_extension(self):
        with pytest.raises(PdfValidationError) as exc:
            sanitize_filename("payload.pdf.exe")
        assert exc.value.code == "FILENAME_DOUBLE_EXT"

    def test_rejects_null_byte(self):
        with pytest.raises(PdfValidationError) as exc:
            sanitize_filename("report\x00.pdf")
        assert exc.value.code == "FILENAME_NUL"

    def test_rejects_empty(self):
        with pytest.raises(PdfValidationError) as exc:
            sanitize_filename("")
        assert exc.value.code == "FILENAME_EMPTY"

    def test_rejects_non_pdf_extension(self):
        with pytest.raises(PdfValidationError) as exc:
            sanitize_filename("doc.docx")
        assert exc.value.code == "FILENAME_NOT_PDF"

    def test_accepts_case_insensitive_pdf(self):
        assert sanitize_filename("REPORT.PDF").endswith(".pdf")

    def test_sanitizes_unsafe_chars(self):
        # The sanitizer substitutes everything outside [A-Za-z0-9._-] with _.
        safe = sanitize_filename("my report; drop table.pdf")
        assert safe.endswith(".pdf")
        assert ";" not in safe and " " not in safe

    def test_caps_stem_length(self):
        very_long = ("A" * 500) + ".pdf"
        safe = sanitize_filename(very_long)
        # 120-char stem cap + ".pdf"
        assert len(safe) <= 124


class TestValidateSize:
    def test_zero_rejected(self):
        with pytest.raises(PdfValidationError) as exc:
            validate_size(0)
        assert exc.value.code == "SIZE_EMPTY"

    def test_at_limit_accepted(self):
        validate_size(MAX_PDF_SIZE_BYTES)  # no raise

    def test_over_limit_rejected(self):
        with pytest.raises(PdfValidationError) as exc:
            validate_size(MAX_PDF_SIZE_BYTES + 1)
        assert exc.value.code == "SIZE_LIMIT"


class TestValidateMime:
    def test_application_pdf_accepted(self):
        assert validate_mime("application/pdf") == "application/pdf"

    def test_charset_parameter_stripped(self):
        assert validate_mime("application/pdf; charset=binary") == "application/pdf"

    def test_octet_stream_rejected(self):
        with pytest.raises(PdfValidationError) as exc:
            validate_mime("application/octet-stream")
        assert exc.value.code == "MIME_REJECTED"

    def test_html_rejected(self):
        with pytest.raises(PdfValidationError) as exc:
            validate_mime("text/html")
        assert exc.value.code == "MIME_REJECTED"

    def test_missing_rejected(self):
        with pytest.raises(PdfValidationError) as exc:
            validate_mime(None)
        assert exc.value.code == "MIME_REJECTED"


class TestValidateMagicBytes:
    def test_clean_pdf_passes(self):
        validate_magic_bytes(make_clean_pdf())

    def test_truncated_header_rejected(self):
        with pytest.raises(PdfValidationError) as exc:
            validate_magic_bytes(make_pdf_truncated_header())
        assert exc.value.code == "MAGIC_MISSING"

    def test_html_rejected(self):
        with pytest.raises(PdfValidationError) as exc:
            validate_magic_bytes(b"<html><body>not a pdf</body></html>")
        assert exc.value.code == "MAGIC_MISSING"

    def test_leading_junk_under_1kb_accepted(self):
        # ISO 32000-1 §7.5.2 allows up to 1024 bytes of leading junk.
        validate_magic_bytes(make_pdf_with_leading_junk(512))

    def test_leading_junk_beyond_window_rejected(self):
        # 1500 bytes of junk pushes %PDF- outside the 1024-byte scan window.
        head = (b"X" * 1500) + b"%PDF-1.4\n"
        with pytest.raises(PdfValidationError) as exc:
            validate_magic_bytes(head[:1024])
        assert exc.value.code == "MAGIC_MISSING"

    def test_empty_head_rejected(self):
        with pytest.raises(PdfValidationError) as exc:
            validate_magic_bytes(b"")
        assert exc.value.code == "MAGIC_EMPTY"


class TestValidateUpload:
    def test_clean_upload_passes(self):
        head = make_clean_pdf()
        result = validate_upload(
            raw_filename="report.pdf",
            declared_mime="application/pdf",
            size_bytes=len(head),
            head_bytes=head[:1024],
        )
        assert result.safe_filename == "report.pdf"
        assert result.mime == "application/pdf"

    def test_rejected_on_wrong_mime(self):
        head = make_clean_pdf()
        with pytest.raises(PdfValidationError):
            validate_upload(
                raw_filename="r.pdf",
                declared_mime="text/html",
                size_bytes=len(head),
                head_bytes=head[:1024],
            )
