"""Unit tests for app.rag.embeddings (T-007).

These tests never hit Azure. They stub the ``httpx`` call via
``monkeypatch`` on the private sync/async request helpers so we exercise
the parsing + batching logic deterministically.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("AZURE_AI_API_KEY", "test-key")
os.environ.setdefault("AZURE_AI_API_BASE", "https://example.invalid")

from app.rag import embeddings


def _vec(dim: int = 2048, fill: float = 0.1) -> list[float]:
    return [fill] * dim


def test_endpoint_uses_v2_path() -> None:
    assert embeddings._endpoint().endswith("/providers/cohere/v2/embed")


def test_headers_rejects_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("AZURE_AI_API_KEY", "")
    with pytest.raises(embeddings.EmbeddingError):
        embeddings._headers()
    get_settings.cache_clear()


def test_validate_dim_accepts_known() -> None:
    for d in embeddings.ALLOWED_DIMENSIONS:
        assert embeddings._validate_dim(d) == d


def test_validate_dim_rejects_unknown() -> None:
    with pytest.raises(embeddings.EmbeddingError):
        embeddings._validate_dim(1000)


def test_coerce_texts_rejects_empty_and_blank() -> None:
    with pytest.raises(embeddings.EmbeddingError):
        embeddings._coerce_texts([])
    with pytest.raises(embeddings.EmbeddingError):
        embeddings._coerce_texts(["  "])
    with pytest.raises(embeddings.EmbeddingError):
        embeddings._coerce_texts(["ok", ""])


def test_coerce_texts_rejects_non_str() -> None:
    with pytest.raises(embeddings.EmbeddingError):
        embeddings._coerce_texts(["ok", 42])  # type: ignore[list-item]


def test_batches_splits_at_max() -> None:
    items = [f"t{i}" for i in range(embeddings.MAX_BATCH * 2 + 5)]
    chunks = list(embeddings._batches(items))
    assert len(chunks) == 3
    assert len(chunks[0]) == embeddings.MAX_BATCH
    assert len(chunks[1]) == embeddings.MAX_BATCH
    assert len(chunks[2]) == 5


def test_extract_floats_v2_shape() -> None:
    body = {"embeddings": {"float": [[0.0, 1.0], [2.0, 3.0]]}}
    vecs = embeddings._extract_floats(body)
    assert vecs == [[0.0, 1.0], [2.0, 3.0]]


def test_extract_floats_v1_fallback() -> None:
    body = {"embeddings": [[0.0, 1.0]]}
    vecs = embeddings._extract_floats(body)
    assert vecs == [[0.0, 1.0]]


def test_extract_floats_rejects_unexpected() -> None:
    with pytest.raises(embeddings.EmbeddingError):
        embeddings._extract_floats({"wat": 1})


def test_embed_documents_batches_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """100 texts should yield 2 upstream calls (96+4) and 100 vectors in order."""
    calls: list[list[str]] = []

    def fake_request(payload: dict[str, Any], *, timeout: float, retries: int) -> dict:
        texts = payload["texts"]
        calls.append(list(texts))
        return {"embeddings": {"float": [_vec() for _ in texts]}}

    monkeypatch.setattr(embeddings, "_request_sync", fake_request)
    inputs = [f"chunk {i}" for i in range(100)]
    out = embeddings.embed_documents(inputs)
    assert len(out) == 100
    assert all(len(v) == 2048 for v in out)
    assert len(calls) == 2
    assert len(calls[0]) == embeddings.MAX_BATCH
    assert len(calls[1]) == 4
    assert calls[0][0] == "chunk 0" and calls[1][-1] == "chunk 99"


def test_embed_query_sends_search_query_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_request(payload: dict[str, Any], *, timeout: float, retries: int) -> dict:
        captured.update(payload)
        return {"embeddings": {"float": [_vec()]}}

    monkeypatch.setattr(embeddings, "_request_sync", fake_request)
    vec = embeddings.embed_query("what is RAG")
    assert len(vec) == 2048
    assert captured["input_type"] == "search_query"
    assert captured["model"] == "Cohere-Embed-V5-Pro"
    assert captured["output_dimension"] == 2048
    assert captured["embedding_types"] == ["float"]
    assert captured["texts"] == ["what is RAG"]


def test_dim_mismatch_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(payload: dict[str, Any], *, timeout: float, retries: int) -> dict:
        return {"embeddings": {"float": [_vec(dim=10)]}}

    monkeypatch.setattr(embeddings, "_request_sync", fake_request)
    with pytest.raises(embeddings.EmbeddingError):
        embeddings.embed_query("hi")


def test_count_mismatch_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(payload: dict[str, Any], *, timeout: float, retries: int) -> dict:
        return {"embeddings": {"float": [_vec(), _vec()]}}  # sent 1, got 2

    monkeypatch.setattr(embeddings, "_request_sync", fake_request)
    with pytest.raises(embeddings.EmbeddingError):
        embeddings.embed_query("hi")


def test_embed_batch_cap_enforced() -> None:
    with pytest.raises(embeddings.EmbeddingError):
        embeddings.embed_batch(
            [f"t{i}" for i in range(embeddings.MAX_BATCH + 1)],
            "search_document",
        )


@pytest.mark.asyncio
async def test_aembed_query(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_async_request(
        payload: dict[str, Any], *, timeout: float, retries: int
    ) -> dict:
        return {"embeddings": {"float": [_vec()]}}

    monkeypatch.setattr(embeddings, "_request_async", fake_async_request)
    v = await embeddings.aembed_query("hello")
    assert len(v) == 2048


@pytest.mark.asyncio
async def test_aembed_documents_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    async def fake_async_request(
        payload: dict[str, Any], *, timeout: float, retries: int
    ) -> dict:
        n = len(payload["texts"])
        calls.append(n)
        return {"embeddings": {"float": [_vec() for _ in range(n)]}}

    monkeypatch.setattr(embeddings, "_request_async", fake_async_request)
    out = await embeddings.aembed_documents([f"t{i}" for i in range(97)])
    assert len(out) == 97
    assert calls == [embeddings.MAX_BATCH, 1]


@pytest.mark.asyncio
async def test_aembed_query_accepts_dimension_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retriever (T-010) calls `await embed(text, dimension=…)`; the alias
    must forward to `output_dimension` without TypeError."""
    captured: dict[str, Any] = {}

    async def fake_async_request(
        payload: dict[str, Any], *, timeout: float, retries: int
    ) -> dict:
        captured.update(payload)
        return {"embeddings": {"float": [_vec(dim=1024)]}}

    monkeypatch.setattr(embeddings, "_request_async", fake_async_request)
    v = await embeddings.aembed_query("hi", dimension=1024)
    assert len(v) == 1024
    assert captured["output_dimension"] == 1024


@pytest.mark.asyncio
async def test_embed_query_async_matches_retriever_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`embed_query_async(text, *, dimension=…)` is the drop-in retriever
    default; verify the signature + plumbing round-trip."""

    async def fake_async_request(
        payload: dict[str, Any], *, timeout: float, retries: int
    ) -> dict:
        assert payload["input_type"] == "search_query"
        assert payload["output_dimension"] == 1536
        return {"embeddings": {"float": [_vec(dim=1536)]}}

    monkeypatch.setattr(embeddings, "_request_async", fake_async_request)
    v = await embeddings.embed_query_async("what is RAG?", dimension=1536)
    assert len(v) == 1536
