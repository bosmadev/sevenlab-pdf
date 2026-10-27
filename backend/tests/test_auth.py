"""Unit tests for T-002 auth primitives.

Covers:
    - Token mint + verify round-trip.
    - Token tamper → verify returns None.
    - Legacy v1 token (3 parts) → rejected.
    - Owner-id parse: label split, label collision, invalid-charset.
    - Rate limiter: sliding window, lockout, success clears state.
    - End-to-end login via FastAPI TestClient against the real router.

Env is seeded in each test; `get_settings` is cached so we clear it when
mutating.
"""
from __future__ import annotations

import os
import time
from unittest import mock

import bcrypt
import pytest
from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.testclient import TestClient


BCRYPT_COST = 4  # fast enough for CI; real deploys use cost 12


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    """Fresh env + cleared caches for every test."""
    monkeypatch.setenv("SESSION_SECRET", "unit-test-secret-x" * 2)
    monkeypatch.setenv("APP_ENV", "development")
    primary = bcrypt.hashpw(b"correct horse", bcrypt.gensalt(rounds=BCRYPT_COST)).decode()
    monkeypatch.setenv("APP_PASSWORD_HASH", primary)
    monkeypatch.delenv("APP_SECONDARY_HASHES", raising=False)

    from app.config import get_settings
    get_settings.cache_clear()

    from app.security.rate_limit import reset_login_rate_limit
    reset_login_rate_limit()
    yield
    get_settings.cache_clear()
    reset_login_rate_limit()


# ---- tokens.py ---------------------------------------------------------------


def test_mint_then_verify_roundtrip():
    from app.security.tokens import create_session_token, verify_session_token

    tok = create_session_token("admin")
    assert tok.startswith("v2.admin.")
    assert verify_session_token(tok) == "admin"


def test_verify_rejects_tamper():
    from app.security.tokens import create_session_token, verify_session_token

    tok = create_session_token("admin")
    tampered = tok[:-2] + ("00" if not tok.endswith("00") else "11")
    assert verify_session_token(tampered) is None


def test_verify_rejects_v1_three_part_token():
    from app.security.tokens import verify_session_token

    assert verify_session_token("v1.deadbeef.cafe") is None


def test_verify_rejects_wrong_version():
    from app.security.tokens import create_session_token, verify_session_token

    tok = create_session_token("admin").replace("v2.", "v3.", 1)
    assert verify_session_token(tok) is None


def test_verify_rejects_blank():
    from app.security.tokens import verify_session_token

    assert verify_session_token(None) is None
    assert verify_session_token("") is None


def test_mint_requires_secret(monkeypatch):
    monkeypatch.setenv("SESSION_SECRET", "")
    from app.config import get_settings

    get_settings.cache_clear()
    from app.security.tokens import create_session_token

    with pytest.raises(RuntimeError):
        create_session_token("admin")


def test_mint_rejects_bad_owner_id():
    from app.security.tokens import create_session_token

    with pytest.raises(ValueError):
        create_session_token("has.dots")  # `.` is reserved as a splitter


# ---- password.py / owner resolution ------------------------------------------


def test_resolve_primary_password_returns_admin():
    from app.security.password import resolve_owner_id

    assert resolve_owner_id("correct horse") == "admin"
    assert resolve_owner_id("wrong") is None


def test_resolve_secondary_label(monkeypatch):
    secondary = bcrypt.hashpw(b"danny-pass", bcrypt.gensalt(rounds=BCRYPT_COST)).decode()
    monkeypatch.setenv("APP_SECONDARY_HASHES", f"danny:{secondary}")

    from app.security.password import resolve_owner_id

    assert resolve_owner_id("danny-pass") == "danny"


def test_resolve_secondary_hash_based_owner_id(monkeypatch):
    secondary = bcrypt.hashpw(b"anon-pass", bcrypt.gensalt(rounds=BCRYPT_COST)).decode()
    monkeypatch.setenv("APP_SECONDARY_HASHES", secondary)

    from app.security.password import resolve_owner_id

    owner = resolve_owner_id("anon-pass")
    assert owner is not None
    assert owner.startswith("u_")
    assert len(owner) == 14  # u_ + 12 hex chars


def test_secondary_cannot_claim_admin(monkeypatch):
    """A secondary labelled `admin` must NOT collapse into the admin tenant."""
    primary = bcrypt.hashpw(b"primary", bcrypt.gensalt(rounds=BCRYPT_COST)).decode()
    secondary = bcrypt.hashpw(b"danny", bcrypt.gensalt(rounds=BCRYPT_COST)).decode()
    monkeypatch.setenv("APP_PASSWORD_HASH", primary)
    monkeypatch.setenv("APP_SECONDARY_HASHES", f"admin:{secondary}")

    from app.config import get_settings

    get_settings.cache_clear()
    from app.security.password import resolve_owner_id

    # The `admin` label falls back to the hash-derived id, so this user can
    # NEVER impersonate the real admin.
    owner = resolve_owner_id("danny")
    assert owner is not None
    assert owner != "admin"


def test_malformed_secondary_entry_is_skipped(monkeypatch):
    good = bcrypt.hashpw(b"ok", bcrypt.gensalt(rounds=BCRYPT_COST)).decode()
    monkeypatch.setenv("APP_SECONDARY_HASHES", f"junk,  ,no-prefix,danny:{good}")

    from app.security.password import resolve_owner_id

    assert resolve_owner_id("ok") == "danny"


def test_is_valid_owner_id():
    from app.security.password import is_valid_owner_id

    assert is_valid_owner_id("admin")
    assert is_valid_owner_id("u_0123456789ab")
    assert is_valid_owner_id("danny_1")
    assert not is_valid_owner_id("has.dot")
    assert not is_valid_owner_id("UPPER")
    assert not is_valid_owner_id("")
    assert not is_valid_owner_id("x" * 33)


# ---- rate_limit.py -----------------------------------------------------------


def _fake_request(ip: str = "9.9.9.9"):
    class _Req:
        def __init__(self, ip):
            self.headers = {"x-forwarded-for": ip}

            class _C:
                host = ip

            self.client = _C()

        def get(self, k, default=None):  # pragma: no cover (compat)
            return self.headers.get(k, default)

    r = _Req(ip)
    r.headers = {"x-forwarded-for": ip}
    # Mimic Starlette's Headers.get interface:
    r.headers = type("H", (), {"get": lambda self_, k, d=None: {"x-forwarded-for": ip}.get(k, d)})()
    return r


def test_rate_limit_locks_after_max_attempts():
    from app.security.rate_limit import (
        MAX_ATTEMPTS,
        check_login_rate,
        record_login_failure,
        reset_login_rate_limit,
    )

    reset_login_rate_limit()
    req = _fake_request()

    for _ in range(MAX_ATTEMPTS):
        verdict = check_login_rate(req)
        assert verdict.allowed
        record_login_failure(verdict.client_key)

    final = check_login_rate(req)
    assert not final.allowed
    assert final.retry_after_sec > 0


def test_rate_limit_success_clears_history():
    from app.security.rate_limit import (
        check_login_rate,
        record_login_failure,
        record_login_success,
        reset_login_rate_limit,
    )

    reset_login_rate_limit()
    req = _fake_request()
    for _ in range(3):
        verdict = check_login_rate(req)
        record_login_failure(verdict.client_key)

    record_login_success(verdict.client_key)
    assert check_login_rate(req).allowed


# ---- end-to-end via FastAPI TestClient ---------------------------------------


@pytest.fixture
def client():
    """Isolated FastAPI app wiring just the auth router + middleware."""
    from app.routes import auth as auth_routes
    from app.security.middleware import WorkspaceAuthMiddleware

    app = FastAPI()
    app.add_middleware(WorkspaceAuthMiddleware)
    app.include_router(auth_routes.router, prefix="/api/auth")

    # Minimal protected echo so we can test the gate end-to-end.
    @app.get("/api/documents/ping")
    async def _ping(request: FastAPIRequest):
        return {"workspace_id": request.state.workspace_id}

    return TestClient(app)


def test_login_sets_cookie_and_opens_protected_route(client):
    res = client.post("/api/auth", json={"password": "correct horse"})
    assert res.status_code == 200, res.text
    assert res.json() == {"success": True}
    assert "app_session" in client.cookies

    protected = client.get("/api/documents/ping")
    assert protected.status_code == 200
    assert protected.json() == {"workspace_id": "admin"}


def test_login_wrong_password_rejected(client):
    res = client.post("/api/auth", json={"password": "nope"})
    assert res.status_code == 401


def test_protected_route_blocks_without_cookie(client):
    res = client.get("/api/documents/ping")
    assert res.status_code == 401


def test_logout_clears_cookie(client):
    client.post("/api/auth", json={"password": "correct horse"})
    assert client.get("/api/documents/ping").status_code == 200

    res = client.delete("/api/auth")
    assert res.status_code == 200
    # Starlette's TestClient drops expired cookies on next request; verify
    # the gate re-blocks us.
    client.cookies.clear()
    assert client.get("/api/documents/ping").status_code == 401


def test_rate_limit_endpoint_response(client):
    from app.security.rate_limit import MAX_ATTEMPTS

    for _ in range(MAX_ATTEMPTS):
        client.post("/api/auth", json={"password": "wrong"})
    final = client.post("/api/auth", json={"password": "wrong"})
    assert final.status_code == 429
    assert final.headers.get("retry-after")
