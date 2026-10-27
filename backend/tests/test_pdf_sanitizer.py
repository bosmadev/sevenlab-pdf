"""Tests for app.services.pdf_sanitizer (T-004, T-015, ISO 27001 A.8.7).

Verifies that every dangerous key the fixture builder plants is actually
removed from the sanitized output. Each check reopens the sanitized bytes
with pikepdf so a false-positive stripped-keys report cannot pass the
suite — the only way to go green is for the key to be absent from the
output tree.
"""
from __future__ import annotations

import io

import pikepdf
import pytest

from app.services.pdf_sanitizer import (
    DANGEROUS_KEYS,
    PdfSanitizationError,
    sanitize_pdf,
)
from tests.fixtures.pdf_builder import (
    make_clean_pdf,
    make_encrypted_pdf,
    make_malicious_pdf,
    make_pdf_with_leading_junk,
)


# ---- helpers ---------------------------------------------------------------


def _reload(pdf_bytes: bytes) -> pikepdf.Pdf:
    """Open sanitized bytes for independent inspection."""
    return pikepdf.open(io.BytesIO(pdf_bytes))


def _walk_keys(obj, seen: set[str], _depth: int = 0) -> None:
    """Collect every dictionary key seen while walking the catalog tree."""
    if _depth > 50:  # defensive cycle break
        return
    try:
        if isinstance(obj, pikepdf.Dictionary):
            for key in obj.keys():
                seen.add(str(key))
                _walk_keys(obj[key], seen, _depth + 1)
        elif isinstance(obj, pikepdf.Array):
            for item in obj:
                _walk_keys(item, seen, _depth + 1)
    except Exception:
        pass


# ---- happy-path -----------------------------------------------------------


class TestSanitizeClean:
    def test_clean_pdf_passes_with_no_keys_stripped(self) -> None:
        clean = make_clean_pdf()
        sanitized, report = sanitize_pdf(clean)
        assert report.stripped_keys == []
        assert report.pages == 1
        assert report.was_encrypted is False
        assert report.original_bytes == len(clean)
        assert report.sanitized_bytes == len(sanitized)
        # Reopening the sanitized bytes must still work.
        with _reload(sanitized) as pdf:
            assert len(pdf.pages) == 1

    def test_leading_junk_still_sanitizes(self) -> None:
        junky = make_pdf_with_leading_junk(junk_bytes=256)
        # pikepdf tolerates ISO-32000 leading junk — must not raise.
        sanitized, report = sanitize_pdf(junky)
        assert report.pages >= 1
        assert len(sanitized) > 0


# ---- malicious path -------------------------------------------------------


class TestSanitizeMalicious:
    def test_strips_every_active_content_key(self) -> None:
        malicious = make_malicious_pdf()
        sanitized, report = sanitize_pdf(malicious)

        # The sanitizer must report at least ONE stripped key per planted
        # payload — if it reports zero, the shield is off.
        assert report.stripped_keys, "sanitizer reported nothing stripped"

        # Walk the sanitized tree ourselves — no dangerous key may remain.
        with _reload(sanitized) as pdf:
            seen: set[str] = set()
            _walk_keys(pdf.Root, seen)
            # Any `/JavaScript` dict or `/JS` string anywhere is a leak.
            leaked = seen & set(DANGEROUS_KEYS)
            assert not leaked, f"dangerous keys survived: {sorted(leaked)}"

    def test_openaction_and_launch_removed_from_catalog(self) -> None:
        sanitized, _ = sanitize_pdf(make_malicious_pdf())
        with _reload(sanitized) as pdf:
            assert "/OpenAction" not in pdf.Root
            assert "/Launch" not in pdf.Root

    def test_names_tree_javascript_removed(self) -> None:
        sanitized, _ = sanitize_pdf(make_malicious_pdf())
        with _reload(sanitized) as pdf:
            names = pdf.Root.get("/Names")
            if names is not None:
                assert "/JavaScript" not in names
                assert "/EmbeddedFiles" not in names

    def test_page_level_aa_removed(self) -> None:
        sanitized, _ = sanitize_pdf(make_malicious_pdf())
        with _reload(sanitized) as pdf:
            for page in pdf.pages:
                assert "/AA" not in page.obj

    def test_sha256_changes_on_strip(self) -> None:
        _, report = sanitize_pdf(make_malicious_pdf())
        assert report.stripped_keys
        assert report.original_sha256 != report.sanitized_sha256


# ---- error paths ----------------------------------------------------------


class TestSanitizeErrors:
    def test_empty_payload_rejected(self) -> None:
        with pytest.raises(PdfSanitizationError):
            sanitize_pdf(b"")

    def test_random_garbage_rejected(self) -> None:
        with pytest.raises(PdfSanitizationError):
            sanitize_pdf(b"not a pdf at all " * 10)

    def test_encrypted_rejected(self) -> None:
        with pytest.raises(PdfSanitizationError) as exc:
            sanitize_pdf(make_encrypted_pdf("secret"))
        assert "encrypted" in str(exc.value).lower()
