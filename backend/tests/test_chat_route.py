"""Route tests for POST /api/chat/ask/stream (T-012).

The pipeline internals are covered by ``test_rag_pipeline.py``; this file
focuses on the FastAPI wiring:

* 401 when no session cookie is present (``require_owner``)
* 401 when the cookie is forged / HMAC mismatch
* 400 when the body is missing or ``question`` is blank/whitespace
* 200 + ``text/event-stream`` when a valid cookie is sent, and the
  workspace used by the pipeline is the signed owner id — NOT a value
  smuggled through the body.

No network: ``answer_stream`` is monkey-patched to a deterministic SSE
emitter so the test exercises only route plumbing.
"""
from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("SESSION_SECRET", "x" * 64)
os.environ.setdefault("AZURE_AI_API_KEY", "test-key")

from fastapi import FastAPI  # noqa: E402

from app.routes import chat as chat_route  # noqa: E402
from app.security.tokens import COOKIE_NAME, create_session_token  # noqa: E402


def _build_app() -> FastAPI:
    """Minimal app that mounts ONLY the chat router.

    Avoids ``app.main`` so these tests do not break when sibling routers
    (document_mgmt, errors, lifespan hooks) are still landing — the
    chat-route contract is what T-012 owns.
    """
    app = FastAPI()
    app.include_router(chat_route.router, prefix="/api/chat")
    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_build_app())


@pytest.fixture
def captured_workspace(monkeypatch):
    """Record the ``workspace_id`` the pipeline is called with."""
    seen: dict = {}

    async def _fake_answer_stream(workspace_id: str, question: str, doc_id=None):
        seen["workspace_id"] = workspace_id
        seen["question"] = question
        seen["doc_id"] = doc_id
        yield f'data: {json.dumps({"type": "citations", "citations": []})}\n\n'
        yield f'data: {json.dumps({"type": "token", "text": "hi"})}\n\n'
        yield f'data: {json.dumps({"type": "done"})}\n\n'

    monkeypatch.setattr(chat_route, "answer_stream", _fake_answer_stream)
    return seen


def _auth_cookie(owner_id: str = "alice") -> dict[str, str]:
    return {COOKIE_NAME: create_session_token(owner_id)}


def test_rejects_missing_cookie(client: TestClient):
    r = client.post("/api/chat/ask/stream", json={"question": "hi?"})
    assert r.status_code == 401


def test_rejects_forged_cookie(client: TestClient):
    r = client.post(
        "/api/chat/ask/stream",
        json={"question": "hi?"},
        cookies={COOKIE_NAME: "v2.alice.deadbeef.0000"},
    )
    assert r.status_code == 401


def test_rejects_blank_question(client: TestClient, captured_workspace):
    r = client.post(
        "/api/chat/ask/stream",
        json={"question": "   "},
        cookies=_auth_cookie(),
    )
    # Pydantic Field(min_length=1) runs before the strip() in the handler, so
    # a 3-space string reaches the handler and triggers the 400 branch.
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_rejects_missing_question_field(client: TestClient):
    r = client.post(
        "/api/chat/ask/stream",
        json={},
        cookies=_auth_cookie(),
    )
    assert r.status_code == 422  # pydantic validation


def test_authenticated_request_streams_sse(client: TestClient, captured_workspace):
    r = client.post(
        "/api/chat/ask/stream",
        json={"question": "Why is the sky blue?"},
        cookies=_auth_cookie("alice"),
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers.get("x-accel-buffering") == "no"

    body = r.text
    frames = [chunk for chunk in body.split("\n\n") if chunk.strip()]
    assert len(frames) == 3
    types = [json.loads(f[len("data: ") :])["type"] for f in frames]
    assert types == ["citations", "token", "done"]


def test_workspace_comes_from_cookie_not_body(client: TestClient, captured_workspace):
    """Attempt to query another tenant's workspace via the body — must fail.

    AskRequest has no ``workspace_id`` field, so pydantic silently drops the
    smuggled key. The handler always uses ``owner.workspace_id`` from the
    verified cookie. This test fixes that contract in stone.
    """
    r = client.post(
        "/api/chat/ask/stream",
        json={"question": "secret?", "workspace_id": "bob"},
        cookies=_auth_cookie("alice"),
    )
    assert r.status_code == 200
    assert captured_workspace["workspace_id"] == "alice"
    assert captured_workspace["question"] == "secret?"


def test_doc_id_passed_through_from_body(client: TestClient, captured_workspace):
    r = client.post(
        "/api/chat/ask/stream",
        json={"question": "summary?", "doc_id": "doc-123"},
        cookies=_auth_cookie("alice"),
    )
    assert r.status_code == 200
    assert captured_workspace["doc_id"] == "doc-123"
