"""Reranker warmup + model resolution — no model download.

The 2.3 GB BGE cross-encoder is never loaded here; the singleton's loader
is monkey-patched so these tests verify only the plumbing that ``app.main``
relies on when ``RERANKER_WARMUP=1``.
"""
from __future__ import annotations

import pytest

from app.rag import reranker


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch):
    monkeypatch.setattr(reranker._CrossEncoderSingleton, "_instance", None)
    yield
    monkeypatch.setattr(reranker._CrossEncoderSingleton, "_instance", None)


def test_resolve_model_name_precedence(monkeypatch):
    monkeypatch.delenv("RERANKER_MODEL", raising=False)
    assert reranker.resolve_model_name() == reranker.DEFAULT_RERANKER_MODEL
    monkeypatch.setenv("RERANKER_MODEL", "org/tiny-reranker")
    assert reranker.resolve_model_name() == "org/tiny-reranker"
    assert reranker.resolve_model_name("explicit/model") == "explicit/model"


def test_preload_loads_singleton_once(monkeypatch):
    calls: list[dict] = []

    class _FakeSingleton:
        pass

    def fake_get(cls, *, model_name, device, max_length):
        calls.append({"model_name": model_name, "device": device, "max_length": max_length})
        reranker._CrossEncoderSingleton._instance = _FakeSingleton()  # type: ignore[assignment]
        return reranker._CrossEncoderSingleton._instance

    monkeypatch.setattr(
        reranker._CrossEncoderSingleton, "get", classmethod(fake_get)
    )
    monkeypatch.delenv("RERANKER_MODEL", raising=False)

    assert reranker.is_loaded() is False
    name = reranker.preload(device="cpu")
    assert name == reranker.DEFAULT_RERANKER_MODEL
    assert reranker.is_loaded() is True
    assert calls == [
        {
            "model_name": reranker.DEFAULT_RERANKER_MODEL,
            "device": "cpu",
            "max_length": reranker.DEFAULT_RERANKER_MAX_LEN,
        }
    ]


def test_rerank_sorts_desc_and_truncates(monkeypatch):
    class _FakeEncoder:
        def score(self, pairs, batch_size):
            # Score = length of candidate text, so ordering is deterministic.
            return [float(len(text)) for _, text in pairs]

    monkeypatch.setattr(
        reranker._CrossEncoderSingleton,
        "get",
        classmethod(lambda cls, **kw: _FakeEncoder()),
    )
    cands = [
        reranker.RerankCandidate(id="short", text="ab"),
        reranker.RerankCandidate(id="long", text="abcdefgh"),
        reranker.RerankCandidate(id="mid", text="abcde"),
    ]
    out = reranker.rerank("q", cands, top_k=2)
    assert [h.id for h in out] == ["long", "mid"]
    assert out[0].score == 8.0


def test_rerank_empty_candidates_returns_empty():
    assert reranker.rerank("q", []) == []


def test_rerank_rejects_non_positive_top_k():
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("q", [reranker.RerankCandidate(id="a", text="x")], top_k=0)
