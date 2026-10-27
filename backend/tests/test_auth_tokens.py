"""Tests for app.security.tokens + app.security.password (T-002, T-015).

Session-cookie hygiene (OWASP ASVS 3.4) and bcrypt password resolution are
the Phase-1 auth surface. These tests do not touch the HTTP layer — they
exercise the signing + resolution primitives directly, which keeps them
fast and makes it obvious when a forged or truncated token slips through.
"""
from __future__ import annotations

import os
import hashlib
from unittest.mock import patch

import bcrypt
import pytest

from app.config import get_settings
from app.security.password import (
    ADMIN_OWNER_ID,
    is_valid_owner_id,
    parse_secondary_entries,
    resolve_owner_id,
)
from app.security.tokens import (
    TOKEN_VERSION,
    create_session_token,
    verify_session_token,
)


# ----- fixtures ---------------------------------------------------------------

@pytest.fixture
def test_secret(monkeypatch):
    """Install a known SESSION_SECRET so the signer has something to sign with."""
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-32bytes-xxxxx")
    get_settings.cache_clear()
    yield "test-session-secret-32bytes-xxxxx"
    get_settings.cache_clear()


@pytest.fixture
def test_password_hash(monkeypatch):
    """Install a bcrypt hash of 'sevenlab' as the primary password."""
    password = "sevenlab"
    hash_bytes = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=4))
    monkeypatch.setenv("APP_PASSWORD_HASH", hash_bytes.decode())
    get_settings.cache_clear()
    yield password
    get_settings.cache_clear()


# ----- is_valid_owner_id ------------------------------------------------------

class TestOwnerIdValidation:
    def test_admin_is_valid(self):
        assert is_valid_owner_id("admin")

    def test_charset_allows_alnum_dash_underscore(self):
        assert is_valid_owner_id("u_abc123")
        assert is_valid_owner_id("tenant-7")

    def test_rejects_dot(self):
        # Dots would break the "." split used by the token format.
        assert not is_valid_owner_id("admin.evil")

    def test_rejects_empty(self):
        assert not is_valid_owner_id("")

    def test_rejects_uppercase(self):
        assert not is_valid_owner_id("Admin")

    def test_caps_length_at_32(self):
        assert is_valid_owner_id("a" * 32)
        assert not is_valid_owner_id("a" * 33)


# ----- tokens -----------------------------------------------------------------

class TestSessionTokens:
    def test_round_trip_recovers_owner(self, test_secret):
        token = create_session_token("admin")
        assert verify_session_token(token) == "admin"

    def test_token_has_four_segments(self, test_secret):
        token = create_session_token("admin")
        assert len(token.split(".")) == 4

    def test_prefix_is_current_version(self, test_secret):
        assert create_session_token("admin").startswith(f"{TOKEN_VERSION}.")

    def test_tampered_signature_rejected(self, test_secret):
        token = create_session_token("admin")
        head, _, _ = token.rpartition(".")
        forged = f"{head}.{'0' * 64}"
        assert verify_session_token(forged) is None

    def test_tampered_owner_rejected(self, test_secret):
        # Swap "admin" -> "attacker" while keeping the original signature.
        token = create_session_token("admin")
        v, _owner, nonce, sig = token.split(".")
        forged = f"{v}.attacker.{nonce}.{sig}"
        assert verify_session_token(forged) is None

    def test_v1_format_rejected(self, test_secret):
        # Three-segment token (legacy v1) must not upgrade silently.
        assert verify_session_token("payload.nonce.hmac") is None

    def test_none_token_rejected(self, test_secret):
        assert verify_session_token(None) is None
        assert verify_session_token("") is None

    def test_random_garbage_rejected(self, test_secret):
        assert verify_session_token("just random text") is None

    def test_signing_without_secret_fails_closed(self, monkeypatch):
        """A deployment with no SESSION_SECRET must refuse to mint tokens."""
        monkeypatch.setenv("SESSION_SECRET", "")
        get_settings.cache_clear()
        try:
            with pytest.raises(RuntimeError):
                create_session_token("admin")
        finally:
            get_settings.cache_clear()

    def test_invalid_owner_id_refused_at_signing(self, test_secret):
        with pytest.raises(ValueError):
            create_session_token("admin.evil")  # dot disallowed

    def test_two_tokens_for_same_owner_differ(self, test_secret):
        """Nonces must make repeated tokens non-identical (replay resistance)."""
        a = create_session_token("admin")
        b = create_session_token("admin")
        assert a != b


# ----- password resolution ----------------------------------------------------

class TestResolveOwnerId:
    def test_correct_primary_password_resolves_admin(self, test_password_hash):
        assert resolve_owner_id(test_password_hash) == ADMIN_OWNER_ID

    def test_wrong_password_returns_none(self, test_password_hash):
        assert resolve_owner_id("wrong") is None

    def test_empty_password_returns_none(self, test_password_hash):
        assert resolve_owner_id("") is None

    def test_no_hash_configured_returns_none(self, monkeypatch):
        monkeypatch.setenv("APP_PASSWORD_HASH", "")
        get_settings.cache_clear()
        try:
            assert resolve_owner_id("anything") is None
        finally:
            get_settings.cache_clear()

    def test_malformed_hash_does_not_crash(self, monkeypatch):
        monkeypatch.setenv("APP_PASSWORD_HASH", "$2b$garbage")
        get_settings.cache_clear()
        try:
            # Must return None, never raise.
            assert resolve_owner_id("sevenlab") is None
        finally:
            get_settings.cache_clear()


class TestSecondaryEntries:
    def test_empty_returns_no_entries(self):
        assert parse_secondary_entries("") == []
        assert parse_secondary_entries(None) == []

    def test_label_colon_hash_parses(self):
        raw = "danny:$2b$12$aaaaaaaaaaaaaaaaaaaaaa.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        entries = parse_secondary_entries(raw)
        assert len(entries) == 1
        assert entries[0].owner_id == "danny"

    def test_bare_hash_derives_owner_id(self):
        raw = "$2b$12$aaaaaaaaaaaaaaaaaaaaaa.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        entries = parse_secondary_entries(raw)
        assert len(entries) == 1
        assert entries[0].owner_id.startswith("u_")

    def test_label_admin_rejected_to_prevent_collision(self):
        raw = "admin:$2b$12$aaaaaaaaaaaaaaaaaaaaaa.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        entries = parse_secondary_entries(raw)
        # Falls back to hash-derived id, which never equals "admin".
        assert len(entries) == 1
        assert entries[0].owner_id != ADMIN_OWNER_ID

    def test_malformed_entry_skipped_not_raised(self):
        raw = "not-a-hash,$2b$12$aaaaaaaaaaaaaaaaaaaaaa.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        entries = parse_secondary_entries(raw)
        assert len(entries) == 1
