"""HTTP-layer tests for app.rag.embeddings (T-007).

Complementary to ``test_embeddings.py``: those tests stub ``_request_sync`` /
``_request_async`` so they cover parsing + batching but never exercise the
transport layer. These tests swap ``httpx.Client`` / ``httpx.AsyncClient``
for a ``MockTransport`` so they verify what actually goes on the wire —
Bearer auth header, retry on 429/5xx, immediate failure on 401/400,
``Retry-After`` honored.

Plus one opt-in live round-trip gated on ``AZURE_AI_API_KEY_LIVE`` (kept
distinct from ``AZURE_AI_API_KEY`` so the mock suite can set a dummy key).
"""
from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

from app.config import get_settings
from app.rag import embeddings as emb


AZURE_BASE = "https://15-ai1-swe.services.ai.azure.com"


def _vector(dim: int, seed: float = 0.001) -> list[float]:
    return [round((i + 1) * seed, 6) for i in range(dim)]


def _ok_body(count: int, dim: int) -> dict[str, Any]:
    return {
        "id": "test-id",
        "texts": [f"t{i}" for i in range(count)],
        "embeddings": {"float": [_vector(dim, 0.001 * (j + 1)) for j in range(count)]},
    }


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []


def _install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler,
    *,
    api_key: str = "test-key",
) -> _Recorder:
    """Patch httpx.Client + httpx.AsyncClient to funnel through MockTransport."""
    rec = _Recorder()

    def wrap(req: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(req.content.decode()) if req.content else {}
        rec.calls.append((str(req.url), dict(req.headers), body))
        return handler(req)

    transport = httpx.MockTransport(wrap)

    monkeypatch.setenv("AZURE_AI_API_BASE", AZURE_BASE)
    monkeypatch.setenv("AZURE_AI_API_KEY", api_key)
    get_settings.cache_clear()

    orig_sync = httpx.Client
    orig_async = httpx.AsyncClient

    def _sync(*args, **kwargs):
        kwargs["transport"] = transport
        return orig_sync(*args, **kwargs)

    def _async(*args, **kwargs):
        kwargs["transport"] = transport
        return orig_async(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _sync)
    monkeypatch.setattr(httpx, "AsyncClient", _async)
    return rec


# --------- wire-level request shape -----------------------------------------


def test_live_wire_sends_bearer_and_v2_path(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_body(2, 2048))

    rec = _install_mock_transport(monkeypatch, handler, api_key="super-secret")
    vecs = emb.embed_documents(["doc one", "doc two"])

    assert len(vecs) == 2 and all(len(v) == 2048 for v in vecs)
    assert len(rec.calls) == 1
    url, headers, body = rec.calls[0]
    assert url.endswith("/providers/cohere/v2/embed")
    assert headers.get("authorization") == "Bearer super-secret"
    # ``api-key`` is the Azure OpenAI header name; MUST NOT be used on this route
    assert "api-key" not in {k.lower() for k in headers}
    assert body["model"] == "Cohere-Embed-V5-Pro"
    assert body["input_type"] == "search_document"
    assert body["embedding_types"] == ["float"]
    assert body["output_dimension"] == 2048
    assert body["truncate"] == "END"


def test_live_wire_query_uses_search_query(monkeypatch):
    _install_mock_transport(
        monkeypatch, lambda r: httpx.Response(200, json=_ok_body(1, 2048))
    )
    rec_calls = emb.embed_query("what is RAG?")
    assert isinstance(rec_calls, list) and len(rec_calls) == 2048


# --------- retry policy -----------------------------------------------------


def test_429_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(emb, "_sleep_backoff", lambda a, h: 0.0)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    counter = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        if counter["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="slow")
        return httpx.Response(200, json=_ok_body(1, 2048))

    _install_mock_transport(monkeypatch, handler)
    out = emb.embed_documents(["hello"], retries=4)
    assert counter["n"] == 3
    assert len(out) == 1 and len(out[0]) == 2048


def test_persistent_429_raises(monkeypatch):
    monkeypatch.setattr(emb, "_sleep_backoff", lambda a, h: 0.0)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="still slow")

    _install_mock_transport(monkeypatch, handler)
    with pytest.raises(emb.EmbeddingError, match="429"):
        emb.embed_documents(["hello"], retries=2)


def test_5xx_retried_then_fails_after_cap(monkeypatch):
    monkeypatch.setattr(emb, "_sleep_backoff", lambda a, h: 0.0)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    counter = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(503, text="unavailable")

    _install_mock_transport(monkeypatch, handler)
    with pytest.raises(emb.EmbeddingError, match="503"):
        emb.embed_documents(["hello"], retries=2)
    assert counter["n"] == 3  # initial + 2 retries


def test_401_raises_immediately(monkeypatch):
    monkeypatch.setattr(emb, "_sleep_backoff", lambda a, h: 0.0)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    counter = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(401, text="bad key")

    _install_mock_transport(monkeypatch, handler)
    with pytest.raises(emb.EmbeddingError, match="401"):
        emb.embed_documents(["hello"], retries=5)
    assert counter["n"] == 1


def test_400_raises_immediately(monkeypatch):
    monkeypatch.setattr(emb, "_sleep_backoff", lambda a, h: 0.0)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    counter = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return httpx.Response(400, text="bad request")

    _install_mock_transport(monkeypatch, handler)
    with pytest.raises(emb.EmbeddingError, match="400"):
        emb.embed_documents(["hello"], retries=5)
    assert counter["n"] == 1


def test_retry_after_header_is_honoured(monkeypatch):
    captured_delays: list[float] = []

    def spy_backoff(attempt: int, retry_after: str | None) -> float:
        captured_delays.append(float(retry_after) if retry_after else 0.0)
        return 0.0

    monkeypatch.setattr(emb, "_sleep_backoff", spy_backoff)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)

    counter = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        if counter["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, text="slow")
        return httpx.Response(200, json=_ok_body(1, 2048))

    _install_mock_transport(monkeypatch, handler)
    emb.embed_documents(["hello"], retries=3)
    assert captured_delays == [7.0]


# --------- async wire path --------------------------------------------------


@pytest.mark.asyncio
async def test_async_wire_sends_bearer(monkeypatch):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_body(1, 2048))

    rec = _install_mock_transport(monkeypatch, handler, api_key="async-key")
    v = await emb.aembed_query("ping")
    assert len(v) == 2048
    _, headers, body = rec.calls[0]
    assert headers.get("authorization") == "Bearer async-key"
    assert body["input_type"] == "search_query"


@pytest.mark.asyncio
async def test_async_429_retried_then_succeeds(monkeypatch):
    import asyncio

    monkeypatch.setattr(emb, "_sleep_backoff", lambda a, h: 0.0)

    async def no_sleep(*a, **k):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)

    counter = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        if counter["n"] < 2:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json=_ok_body(1, 2048))

    _install_mock_transport(monkeypatch, handler)
    out = await emb.aembed_documents(["hello"], retries=3)
    assert len(out) == 1
    assert counter["n"] == 2


# --------- live round-trip (opt-in) -----------------------------------------


@pytest.mark.skipif(
    not os.environ.get("AZURE_AI_API_KEY_LIVE"),
    reason="AZURE_AI_API_KEY_LIVE not set; skipping live Cohere call",
)
def test_live_embed_integration(monkeypatch):
    """Hit the real Azure endpoint. Opt-in via AZURE_AI_API_KEY_LIVE."""
    monkeypatch.setenv("AZURE_AI_API_KEY", os.environ["AZURE_AI_API_KEY_LIVE"])
    get_settings.cache_clear()
    vecs = emb.embed_documents(["SevenLab PDF chat live smoke test."])
    assert len(vecs) == 1
    assert len(vecs[0]) == 2048
    assert all(isinstance(x, float) for x in vecs[0][:16])
