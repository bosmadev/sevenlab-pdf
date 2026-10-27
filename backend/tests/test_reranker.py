"""Tests for ``app.rag.reranker.rerank`` (T-010 stage 5).

``test_retriever.py`` swaps the whole reranker for a fake. These tests run
the real ``rerank`` function with only the cross-encoder *model* stubbed, so
the sort / truncate / pair-building / model-resolution logic is exercised
without downloading the 2 GB BGE weights.
"""
from __future__ import annotations

import pytest

from app.rag import reranker as rr
from app.rag.reranker import RerankCandidate, RerankedHit, rerank


class _StubEncoder:
    """Scores each candidate text via a lookup table; records every call."""

    def __init__(self, scores: dict[str, float] | None = None, *, override: list[float] | None = None):
        self.scores = scores or {}
        self.override = override
        self.calls: list[tuple[list[tuple[str, str]], int]] = []

    def score(self, pairs, batch_size):
        self.calls.append((list(pairs), batch_size))
        if self.override is not None:
            return list(self.override)
        return [self.scores.get(text, 0.0) for _, text in pairs]


@pytest.fixture
def encoder(monkeypatch):
    """Install a stub singleton and capture the kwargs ``get`` was called with."""
    stub = _StubEncoder()
    seen: dict = {}

    def fake_get(cls, **kwargs):
        seen.update(kwargs)
        return stub

    monkeypatch.setattr(rr._CrossEncoderSingleton, "get", classmethod(fake_get))
    stub.get_kwargs = seen  # type: ignore[attr-defined]
    return stub


def _cands(*texts: str) -> list[RerankCandidate]:
    return [RerankCandidate(id=f"c{i}", text=t, payload={"i": i}) for i, t in enumerate(texts)]


def test_empty_candidates_short_circuit_without_loading_model(monkeypatch):
    def boom(cls, **kw):
        raise AssertionError("model must not load for empty input")

    monkeypatch.setattr(rr._CrossEncoderSingleton, "get", classmethod(boom))
    assert rerank("q", []) == []
    assert rerank("q", iter(())) == []


def test_non_positive_top_k_rejected(encoder):
    with pytest.raises(ValueError, match="top_k"):
        rerank("q", _cands("a"), top_k=0)
    with pytest.raises(ValueError):
        rerank("q", _cands("a"), top_k=-3)


def test_sorts_by_score_desc_and_truncates(encoder):
    encoder.scores = {"low": 0.1, "mid": 0.5, "high": 0.9, "top": 0.95}
    out = rerank("q", _cands("low", "mid", "high", "top"), top_k=2)
    assert [h.text for h in out] == ["top", "high"]
    assert [h.score for h in out] == [0.95, 0.9]
    assert all(isinstance(h, RerankedHit) for h in out)


def test_payload_and_id_round_trip(encoder):
    encoder.scores = {"a": 1.0}
    cands = [RerankCandidate(id="chunk-42", text="a", payload={"page_num": 7, "doc_id": "d"})]
    out = rerank("q", cands)
    assert out[0].id == "chunk-42"
    assert out[0].payload == {"page_num": 7, "doc_id": "d"}


def test_ties_break_on_id_ascending(encoder):
    cands = [
        RerankCandidate(id="z", text="same"),
        RerankCandidate(id="a", text="same"),
        RerankCandidate(id="m", text="same"),
    ]
    encoder.scores = {"same": 0.5}
    assert [h.id for h in rerank("q", cands)] == ["a", "m", "z"]


def test_pairs_are_query_text_in_candidate_order(encoder):
    rerank("why sky blue", _cands("b", "a", "c"), batch_size=7)
    pairs, batch = encoder.calls[0]
    assert pairs == [("why sky blue", "b"), ("why sky blue", "a"), ("why sky blue", "c")]
    assert batch == 7


def test_top_k_larger_than_pool_returns_all(encoder):
    encoder.scores = {"a": 0.3, "b": 0.2}
    assert len(rerank("q", _cands("a", "b"), top_k=50)) == 2


def test_accepts_generator_input(encoder):
    encoder.scores = {"x": 0.4, "y": 0.6}
    gen = (c for c in _cands("x", "y"))
    assert [h.text for h in rerank("q", gen)] == ["y", "x"]


def test_model_resolution_default_env_and_explicit(encoder, monkeypatch):
    monkeypatch.delenv("RERANKER_MODEL", raising=False)
    rerank("q", _cands("a"))
    assert encoder.get_kwargs["model_name"] == rr.DEFAULT_RERANKER_MODEL
    assert encoder.get_kwargs["device"] is None
    assert encoder.get_kwargs["max_length"] == rr.DEFAULT_RERANKER_MAX_LEN

    monkeypatch.setenv("RERANKER_MODEL", "org/tiny-reranker")
    rerank("q", _cands("a"))
    assert encoder.get_kwargs["model_name"] == "org/tiny-reranker"

    rerank("q", _cands("a"), model_name="explicit/model", device="cpu", max_length=128)
    assert encoder.get_kwargs == {"model_name": "explicit/model", "device": "cpu", "max_length": 128}


def test_score_length_mismatch_is_loud(encoder):
    encoder.override = [0.1]
    with pytest.raises(ValueError):
        rerank("q", _cands("a", "b"))


def test_singleton_reuses_loaded_instance(monkeypatch):
    sentinel = _StubEncoder({"a": 0.7})
    monkeypatch.setattr(rr._CrossEncoderSingleton, "_instance", sentinel)
    assert rr._CrossEncoderSingleton.get() is sentinel
    assert rr._CrossEncoderSingleton.get(model_name="ignored/after-first-load") is sentinel
    out = rerank("q", _cands("a"))
    assert out[0].score == pytest.approx(0.7)


def test_singleton_score_empty_pairs_skips_model():
    bare = object.__new__(rr._CrossEncoderSingleton)
    assert bare.score([], batch_size=32) == []


def test_singleton_score_casts_numpy_like_to_float():
    class _Model:
        def predict(self, pairs, batch_size, show_progress_bar, convert_to_numpy):
            assert show_progress_bar is False and convert_to_numpy is True
            return [1, 2.5]

    enc = object.__new__(rr._CrossEncoderSingleton)
    enc.model = _Model()
    out = enc.score([("q", "a"), ("q", "b")], batch_size=4)
    assert out == [1.0, 2.5]
    assert all(type(s) is float for s in out)
