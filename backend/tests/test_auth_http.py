"""HTTP-level auth coverage beyond ``test_auth.py`` (T-002 / T-015).

``test_auth.py`` proves login works. This file pins the contract the
frontend and the other routers depend on:

* ``GET /api/auth/me``: 401 → identity, ``is_admin`` only for the primary
  password, secondary tenants resolve to their label.
* Cookie hygiene: ``HttpOnly``, ``SameSite=lax``, ``Path=/``, 7-day
  ``Max-Age``, ``Secure`` only when ``APP_ENV=production``.
* Fail-closed: 500 (not 401) when the hash or secret is unset, so a
  misconfigured deploy is loud instead of silently rejecting everyone.
* Body validation: empty / missing / oversize password → 422.
* ``require_admin`` 403, ``optional_owner`` None, middleware prefix
  boundaries, 401 body shape, ``request.state`` population on open routes.
* Rate limiter: lockout expiry, sliding window, per-client isolation,
  client-key precedence, bucket pruning.
"""
from __future__ import annotations

import time
from typing import Any

import bcrypt
import pytest
from fastapi import Depends, FastAPI, Request as FastAPIRequest
from fastapi.testclient import TestClient
from starlette.requests import Request as StarletteRequest

from app.config import get_settings
from app.security import rate_limit
from app.security import tokens as tokens_mod
from app.security.deps import Owner, optional_owner, require_admin, require_owner
from app.security.middleware import WorkspaceAuthMiddleware, _is_protected
from app.security.rate_limit import (
    LOCKOUT_SEC,
    MAX_ATTEMPTS,
    WINDOW_SEC,
    check_login_rate,
    client_key_for,
    record_login_failure,
    reset_login_rate_limit,
)
from app.security.tokens import (
    CLOCK_SKEW_SEC,
    COOKIE_NAME,
    SESSION_MAX_AGE,
    create_session_token,
    parse_session_token,
    session_max_age,
    verify_session_token,
)

BCRYPT_COST = 4
PRIMARY_PW = "correct horse"
DANNY_PW = "danny-pass"


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    monkeypatch.setenv("SESSION_SECRET", "auth-http-secret-" * 2)
    monkeypatch.setenv("APP_ENV", "development")
    primary = bcrypt.hashpw(PRIMARY_PW.encode(), bcrypt.gensalt(rounds=BCRYPT_COST)).decode()
    danny = bcrypt.hashpw(DANNY_PW.encode(), bcrypt.gensalt(rounds=BCRYPT_COST)).decode()
    monkeypatch.setenv("APP_PASSWORD_HASH", primary)
    monkeypatch.setenv("APP_SECONDARY_HASHES", f"danny:{danny}")
    get_settings.cache_clear()
    reset_login_rate_limit()
    yield
    get_settings.cache_clear()
    reset_login_rate_limit()


def _build_app() -> FastAPI:
    from app.routes import auth as auth_routes

    app = FastAPI()
    app.add_middleware(WorkspaceAuthMiddleware)
    app.include_router(auth_routes.router, prefix="/api/auth")

    @app.get("/api/documents/ping")
    async def _ping(request: FastAPIRequest):
        return {"workspace_id": request.state.workspace_id, "is_admin": request.state.is_admin}

    @app.get("/api/documentsx/ping")
    async def _ping_sibling():
        return {"open": True}

    @app.get("/api/admin-only")
    async def _admin_only(owner: Owner = Depends(require_admin)):
        return {"owner_id": owner.owner_id}

    @app.get("/api/owner-required")
    async def _owner_required(owner: Owner = Depends(require_owner)):
        return {"owner_id": owner.owner_id}

    @app.get("/open/state")
    async def _open_state(request: FastAPIRequest, owner: Owner | None = Depends(optional_owner)):
        return {
            "state_owner": request.state.owner_id,
            "state_admin": request.state.is_admin,
            "dep_owner": owner.owner_id if owner else None,
        }

    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_build_app())


def _login(client: TestClient, password: str = PRIMARY_PW):
    return client.post("/api/auth", json={"password": password})


# ---- /me ---------------------------------------------------------------------


def test_me_requires_cookie(client):
    r = client.get("/api/auth/me")
    assert r.status_code == 401
    assert r.json()["detail"] == "Not authenticated"


def _assert_me(body: dict, owner_id: str, is_admin: bool) -> None:
    assert body["owner_id"] == owner_id
    assert body["is_admin"] is is_admin
    assert body["expires_at"] - body["issued_at"] == session_max_age()
    assert body["issued_at"] <= int(time.time()) <= body["expires_at"]


def test_me_returns_admin_identity_after_primary_login(client):
    assert _login(client).status_code == 200
    r = client.get("/api/auth/me")
    assert r.status_code == 200
    _assert_me(r.json(), "admin", True)


def test_me_secondary_tenant_is_not_admin(client):
    assert _login(client, DANNY_PW).status_code == 200
    r = client.get("/api/auth/me")
    assert r.status_code == 200
    _assert_me(r.json(), "danny", False)


def test_me_rejects_forged_cookie(client):
    client.cookies.set(COOKIE_NAME, "v2.admin.deadbeef." + "0" * 64)
    assert client.get("/api/auth/me").status_code == 401


# ---- cookie hygiene ----------------------------------------------------------


def _set_cookie_header(res) -> str:
    raw = [v for k, v in res.headers.multi_items() if k.lower() == "set-cookie"]
    assert raw, "login must emit a Set-Cookie header"
    header = next(h for h in raw if h.startswith(f"{COOKIE_NAME}="))
    return header.lower()


def test_login_cookie_attributes_in_development(client):
    res = _login(client)
    assert res.status_code == 200
    header = _set_cookie_header(res)
    assert "httponly" in header
    assert "samesite=lax" in header
    assert "path=/" in header
    assert f"max-age={session_max_age()}" in header
    assert session_max_age() == SESSION_MAX_AGE == 7 * 24 * 3600
    assert "secure" not in header


def test_cookie_max_age_follows_configured_session_lifetime(monkeypatch):
    monkeypatch.setenv("SESSION_MAX_AGE_SECONDS", "120")
    get_settings.cache_clear()
    client = TestClient(_build_app())
    res = _login(client)
    assert res.status_code == 200
    assert "max-age=120" in _set_cookie_header(res)
    me = client.get("/api/auth/me").json()
    assert me["expires_at"] - me["issued_at"] == 120


def test_login_cookie_is_secure_in_production(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    get_settings.cache_clear()
    client = TestClient(_build_app())
    res = _login(client)
    assert res.status_code == 200
    assert "secure" in _set_cookie_header(res)


def test_failed_login_sets_no_session_cookie(client):
    res = _login(client, "nope")
    assert res.status_code == 401
    assert COOKIE_NAME not in client.cookies
    assert not any(k.lower() == "set-cookie" for k, _ in res.headers.multi_items())


def test_logout_response_expires_cookie(client):
    _login(client)
    res = client.delete("/api/auth")
    assert res.status_code == 200
    header = _set_cookie_header(res)
    assert "max-age=0" in header or "expires=" in header


# ---- fail-closed configuration ----------------------------------------------


def test_login_500_when_password_hash_missing(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD_HASH", "")
    get_settings.cache_clear()
    client = TestClient(_build_app())
    res = _login(client)
    assert res.status_code == 500
    assert res.json()["detail"] == "Authentication not configured"
    assert COOKIE_NAME not in client.cookies


def test_login_500_when_session_secret_missing(monkeypatch):
    monkeypatch.setenv("SESSION_SECRET", "")
    get_settings.cache_clear()
    client = TestClient(_build_app())
    res = _login(client)
    assert res.status_code == 500
    assert COOKIE_NAME not in client.cookies


def test_verify_fails_closed_without_secret(monkeypatch):
    client = TestClient(_build_app())
    _login(client)
    assert client.get("/api/documents/ping").status_code == 200
    monkeypatch.setenv("SESSION_SECRET", "")
    get_settings.cache_clear()
    assert client.get("/api/documents/ping").status_code == 401


# ---- body validation ---------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"password": ""},
        {},
        {"password": "x" * 4097},
        {"password": None},
    ],
)
def test_login_body_validation_422(client, body):
    res = client.post("/api/auth", json=body)
    assert res.status_code == 422


def test_login_wrong_password_counts_toward_lockout(client):
    for _ in range(MAX_ATTEMPTS - 1):
        assert _login(client, "nope").status_code == 401
    assert _login(client).status_code == 200
    assert _login(client, "nope").status_code == 401


# ---- dependencies ------------------------------------------------------------


def test_require_admin_403_for_tenant_200_for_admin(client):
    _login(client, DANNY_PW)
    r = client.get("/api/admin-only")
    assert r.status_code == 403
    assert r.json()["detail"] == "Admin privileges required"

    client.cookies.clear()
    _login(client)
    r = client.get("/api/admin-only")
    assert r.status_code == 200
    assert r.json() == {"owner_id": "admin"}


def test_require_owner_outside_protected_prefix_still_gates(client):
    assert client.get("/api/owner-required").status_code == 401
    _login(client, DANNY_PW)
    r = client.get("/api/owner-required")
    assert r.status_code == 200
    assert r.json() == {"owner_id": "danny"}


def test_optional_owner_and_state_on_open_route(client):
    r = client.get("/open/state")
    assert r.status_code == 200
    assert r.json() == {"state_owner": None, "state_admin": False, "dep_owner": None}

    _login(client)
    r = client.get("/open/state")
    assert r.json() == {"state_owner": "admin", "state_admin": True, "dep_owner": "admin"}


# ---- middleware --------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "protected"),
    [
        ("/api/documents", True),
        ("/api/documents/", True),
        ("/api/documents/abc/status", True),
        ("/api/chat", True),
        ("/api/chat/ask/stream", True),
        ("/api/documentsx", False),
        ("/api/documentsx/ping", False),
        ("/api/chatter", False),
        ("/api/auth", False),
        ("/api/auth/me", False),
        ("/api/health/status", False),
        ("/", False),
    ],
)
def test_is_protected_prefix_boundaries(path, protected):
    assert _is_protected(path) is protected


def test_middleware_prefix_boundary_over_http(client):
    assert client.get("/api/documentsx/ping").status_code == 200
    assert client.get("/api/documents/ping").status_code == 401


def test_middleware_401_body_shape(client):
    r = client.get("/api/documents/ping")
    assert r.status_code == 401
    body = r.json()
    assert body["detail"] == "Not authenticated"
    assert body["error"]["code"] == "UNAUTHORIZED"
    assert body["error"]["message"]
    # This harness has no RequestIdMiddleware, so the id falls back to "unknown".
    assert body["error"]["request_id"] == "unknown"


def test_middleware_populates_state_for_tenant(client):
    _login(client, DANNY_PW)
    r = client.get("/api/documents/ping")
    assert r.status_code == 200
    assert r.json() == {"workspace_id": "danny", "is_admin": False}


def test_middleware_rejects_token_signed_with_other_secret(monkeypatch, client):
    monkeypatch.setenv("SESSION_SECRET", "a-completely-different-secret-value")
    get_settings.cache_clear()
    foreign = create_session_token("admin")
    monkeypatch.setenv("SESSION_SECRET", "auth-http-secret-" * 2)
    get_settings.cache_clear()
    client.cookies.set(COOKIE_NAME, foreign)
    assert client.get("/api/documents/ping").status_code == 401


# ---- rate limiter ------------------------------------------------------------


class _Clock:
    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def time(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(rate_limit, "time", c)
    return c


def _scope(headers: dict[str, str] | None = None, client: tuple[str, int] | None = ("9.9.9.9", 1234)) -> dict[str, Any]:
    return {
        "type": "http",
        "method": "GET",
        "path": "/",
        "query_string": b"",
        "scheme": "http",
        "server": ("testserver", 80),
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": client,
        "state": {},
    }


def _req(headers: dict[str, str] | None = None, client=("9.9.9.9", 1234)) -> StarletteRequest:
    return StarletteRequest(_scope(headers, client))


def test_lockout_expires_after_lockout_window(clock):
    req = _req({"x-forwarded-for": "1.1.1.1"})
    for _ in range(MAX_ATTEMPTS):
        record_login_failure(check_login_rate(req).client_key)
    locked = check_login_rate(req)
    assert not locked.allowed
    assert locked.retry_after_sec == pytest.approx(LOCKOUT_SEC, abs=1)

    clock.advance(LOCKOUT_SEC - 5)
    still = check_login_rate(req)
    assert not still.allowed
    assert 1 <= still.retry_after_sec <= 6

    clock.advance(6)
    assert check_login_rate(req).allowed


def test_failures_slide_out_of_window(clock):
    req = _req({"x-forwarded-for": "2.2.2.2"})
    for _ in range(MAX_ATTEMPTS - 1):
        record_login_failure(check_login_rate(req).client_key)
    assert check_login_rate(req).allowed

    clock.advance(WINDOW_SEC + 1)
    record_login_failure(check_login_rate(req).client_key)
    assert check_login_rate(req).allowed


def test_rate_limit_isolated_per_client(client):
    for _ in range(MAX_ATTEMPTS):
        client.post("/api/auth", json={"password": "nope"}, headers={"x-forwarded-for": "10.0.0.1"})
    locked = client.post("/api/auth", json={"password": "nope"}, headers={"x-forwarded-for": "10.0.0.1"})
    assert locked.status_code == 429

    other = client.post("/api/auth", json={"password": PRIMARY_PW}, headers={"x-forwarded-for": "10.0.0.2"})
    assert other.status_code == 200


def test_locked_client_cannot_login_even_with_correct_password(client):
    for _ in range(MAX_ATTEMPTS):
        client.post("/api/auth", json={"password": "nope"})
    res = _login(client)
    assert res.status_code == 429
    assert int(res.headers["retry-after"]) >= 1
    assert COOKIE_NAME not in client.cookies


def test_client_key_precedence():
    assert client_key_for(_req({"x-forwarded-for": "1.2.3.4, 5.6.7.8"})) == "1.2.3.4"
    assert client_key_for(_req({"x-forwarded-for": " , 5.6.7.8", "x-real-ip": "7.7.7.7"})) == "7.7.7.7"
    assert client_key_for(_req({"x-real-ip": " 8.8.8.8 "})) == "8.8.8.8"
    assert client_key_for(_req()) == "9.9.9.9"
    assert client_key_for(_req(client=None)) == "unknown"


def test_prune_evicts_expired_buckets_when_over_cap(clock, monkeypatch):
    monkeypatch.setattr(rate_limit, "MAX_TRACKED", 2)
    for ip in ("3.3.3.1", "3.3.3.2", "3.3.3.3"):
        record_login_failure(ip)
    assert len(rate_limit._buckets) == 3

    clock.advance(WINDOW_SEC + 1)
    check_login_rate(_req({"x-forwarded-for": "3.3.3.9"}))
    assert rate_limit._buckets == {}


def test_prune_keeps_locked_buckets(clock, monkeypatch):
    monkeypatch.setattr(rate_limit, "MAX_TRACKED", 1)
    for _ in range(MAX_ATTEMPTS):
        record_login_failure("4.4.4.4")
    record_login_failure("4.4.4.5")
    clock.advance(WINDOW_SEC + 1)
    check_login_rate(_req({"x-forwarded-for": "4.4.4.9"}))
    assert set(rate_limit._buckets) == {"4.4.4.4"}
    assert not check_login_rate(_req({"x-forwarded-for": "4.4.4.4"})).allowed


# ---- token lifetime ----------------------------------------------------------


@pytest.fixture
def token_clock(monkeypatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(tokens_mod, "time", c)
    return c


def test_token_expires_after_session_lifetime(token_clock):
    tok = create_session_token("admin")
    claims = parse_session_token(tok)
    assert claims is not None
    assert claims.issued_at == int(token_clock.now)
    assert claims.expires_at == claims.issued_at + session_max_age()

    token_clock.advance(session_max_age() - 1)
    assert verify_session_token(tok) == "admin"
    token_clock.advance(1)
    assert verify_session_token(tok) is None


def test_expired_cookie_is_rejected_over_http(token_clock):
    client = TestClient(_build_app())
    _login(client)
    assert client.get("/api/documents/ping").status_code == 200
    token_clock.advance(session_max_age() + 1)
    assert client.get("/api/documents/ping").status_code == 401
    assert client.get("/api/auth/me").status_code == 401


def test_token_from_the_future_is_rejected_beyond_skew(token_clock):
    tok = create_session_token("admin")
    token_clock.advance(-(CLOCK_SKEW_SEC - 1))
    assert verify_session_token(tok) == "admin"
    token_clock.advance(-2)
    assert verify_session_token(tok) is None


def test_signed_token_with_malformed_random_segment_is_rejected():
    secret = tokens_mod._secret_bytes()
    body = "v2.admin.tooshort"
    forged = f"{body}.{tokens_mod._sign(body, secret)}"
    assert verify_session_token(forged) is None

    body = "v2.admin." + "zz" * 28
    forged = f"{body}.{tokens_mod._sign(body, secret)}"
    assert verify_session_token(forged) is None


def test_two_logins_mint_distinct_tokens():
    a = TestClient(_build_app())
    b = TestClient(_build_app())
    _login(a)
    _login(b)
    assert a.cookies[COOKIE_NAME] != b.cookies[COOKIE_NAME]


# ---- legacy session helper ---------------------------------------------------


@pytest.mark.asyncio
async def test_current_workspace_id_prefers_state_then_cookie():
    from app.auth.session import current_workspace_id

    cached = _req()
    cached.state.workspace_id = "from-state"
    assert await current_workspace_id(cached) == "from-state"

    token = create_session_token("danny")
    with_cookie = _req({"cookie": f"{COOKIE_NAME}={token}"})
    assert await current_workspace_id(with_cookie) == "danny"

    forged = _req({"cookie": f"{COOKIE_NAME}=v2.danny.nonce.bad"})
    assert await current_workspace_id(forged) is None

    assert await current_workspace_id(_req()) is None
