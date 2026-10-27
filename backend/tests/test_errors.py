"""Smoke tests for app.errors — request-id middleware + exception envelope.

These tests only need fastapi + httpx. They do not touch MongoDB, Postgres
or any heavy PDF deps, so they run in CI without the full requirements set.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
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

    return TestClient(app)


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


def test_validation_error_envelope(client: TestClient) -> None:
    resp = client.get("/boom")
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["error"]["code"] == "MAGIC_MISSING"
    assert body["error"]["message"] == "not a pdf"
    assert body["error"]["request_id"] == resp.headers["X-Request-ID"]


def test_incoming_header_is_length_capped(client: TestClient) -> None:
    long = "x" * 500
    resp = client.get("/echo", headers={"X-Request-ID": long})
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["rid"]) <= 64
