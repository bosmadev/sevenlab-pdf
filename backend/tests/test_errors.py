"""Smoke tests for app.errors — request-id middleware + exception envelope.

These tests only need fastapi + httpx (the DB-outage case importorskips
sqlalchemy). They do not touch MongoDB, Postgres or any heavy PDF deps, so
they run in CI without the full requirements set.
"""
from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel


class _Body(BaseModel):
    question: str


def _build_app() -> FastAPI:
    from app.errors import RequestIdMiddleware, register_exception_handlers
    from app.services.pdf_validator import PdfValidationError

    app = FastAPI()
    app.add_middleware(RequestIdMiddleware)
    register_exception_handlers(app)

    @app.get("/echo")
    def echo(request: Request) -> dict:
        return {"rid": request.state.request_id}

    @app.get("/boom")
    def boom() -> dict:
        raise PdfValidationError("not a pdf", "MAGIC_MISSING")

    @app.get("/http-str")
    def http_str() -> dict:
        raise HTTPException(status_code=404, detail="document not found")

    @app.get("/http-dict")
    def http_dict() -> dict:
        raise HTTPException(
            status_code=413,
            detail={"code": "SIZE_LIMIT", "reason": "file exceeds 50 MB limit"},
            headers={"Retry-After": "7"},
        )

    @app.get("/http-bare")
    def http_bare() -> dict:
        raise HTTPException(status_code=403)

    @app.post("/validate")
    def validate(body: _Body) -> dict:
        return {"ok": body.question}

    @app.get("/crash")
    def crash() -> dict:
        raise RuntimeError("secret internal detail")

    @app.get("/db-down")
    def db_down() -> dict:
        from sqlalchemy.exc import OperationalError

        raise OperationalError(
            "SELECT 1", {}, Exception("connection refused host=db.internal")
        )

    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_build_app())


@pytest.fixture
def lenient_client() -> TestClient:
    """Does not re-raise server exceptions, so the 500 envelope is observable."""
    return TestClient(_build_app(), raise_server_exceptions=False)


# --- request id -------------------------------------------------------------


def test_request_id_generated_when_missing(client: TestClient) -> None:
    resp = client.get("/echo")
    assert resp.status_code == 200, resp.text
    rid = resp.json()["rid"]
    assert rid and len(rid) >= 16
    assert resp.headers.get("X-Request-ID") == rid


def test_request_id_honours_incoming_header(client: TestClient) -> None:
    resp = client.get("/echo", headers={"X-Request-ID": "abc-123"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["rid"] == "abc-123"
    assert resp.headers.get("X-Request-ID") == "abc-123"


def test_incoming_header_is_length_capped(client: TestClient) -> None:
    long = "x" * 500
    resp = client.get("/echo", headers={"X-Request-ID": long})
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["rid"]) <= 64


def test_request_id_header_not_duplicated_on_error(client: TestClient) -> None:
    """Handlers set the header themselves; the middleware must not add a second."""
    resp = client.get("/http-str", headers={"X-Request-ID": "once"})
    assert resp.status_code == 404
    assert resp.headers.get_list("x-request-id") == ["once"]


def test_request_id_contextvar_is_reset_after_request(client: TestClient) -> None:
    from app.logging_config import request_id_var

    client.get("/echo", headers={"X-Request-ID": "ctx-1"})
    assert request_id_var.get() == "-"


# --- pdf pipeline -----------------------------------------------------------


def test_validation_error_envelope(client: TestClient) -> None:
    resp = client.get("/boom")
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["error"]["code"] == "MAGIC_MISSING"
    assert body["error"]["message"] == "not a pdf"
    assert body["error"]["request_id"] == resp.headers["X-Request-ID"]
    assert body["detail"] == "not a pdf"


# --- HTTPException ----------------------------------------------------------


def test_http_exception_string_detail(client: TestClient) -> None:
    resp = client.get("/http-str")
    assert resp.status_code == 404
    body = resp.json()
    assert body["detail"] == "document not found"
    assert body["error"]["code"] == "NOT_FOUND"
    assert body["error"]["message"] == "document not found"
    assert body["error"]["request_id"] == resp.headers["X-Request-ID"]


def test_http_exception_dict_detail_keeps_code_and_headers(client: TestClient) -> None:
    resp = client.get("/http-dict")
    assert resp.status_code == 413
    body = resp.json()
    assert body["detail"] == {"code": "SIZE_LIMIT", "reason": "file exceeds 50 MB limit"}
    assert body["error"]["code"] == "SIZE_LIMIT"
    assert body["error"]["message"] == "file exceeds 50 MB limit"
    assert resp.headers["Retry-After"] == "7"


def test_http_exception_without_detail_gets_status_code(client: TestClient) -> None:
    resp = client.get("/http-bare")
    assert resp.status_code == 403
    body = resp.json()
    assert body["error"]["code"] == "FORBIDDEN"
    assert body["error"]["message"]


def test_unknown_route_is_enveloped_404(client: TestClient) -> None:
    resp = client.get("/no-such-route")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "NOT_FOUND"
    assert body["error"]["request_id"] == resp.headers["X-Request-ID"]


# --- request validation -----------------------------------------------------


def test_request_validation_envelope(client: TestClient) -> None:
    resp = client.post("/validate", json={})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert "question" in body["error"]["message"]
    # FastAPI's per-field list is preserved for clients that parse it.
    assert isinstance(body["detail"], list) and body["detail"]
    assert body["detail"][0]["loc"][-1] == "question"


# --- 5xx --------------------------------------------------------------------


def test_unhandled_exception_hides_detail_and_logs_it(
    lenient_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger="app.errors"):
        resp = lenient_client.get("/crash", headers={"X-Request-ID": "crash-1"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert body["error"]["request_id"] == "crash-1"
    assert resp.headers.get_list("x-request-id") == ["crash-1"]
    assert "secret internal detail" not in resp.text
    # ...but the operator can find it by request id in error.log.
    assert "unhandled_exception" in caplog.text
    assert "crash-1" in caplog.text
    assert "secret internal detail" in caplog.text


def test_db_outage_maps_to_503(client: TestClient) -> None:
    pytest.importorskip("sqlalchemy")
    resp = client.get("/db-down")
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "DATABASE_UNAVAILABLE"
    assert "db.internal" not in resp.text
    assert resp.headers["Retry-After"] == "5"
