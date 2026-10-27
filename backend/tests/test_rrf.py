"""Unit tests for Reciprocal Rank Fusion — T-010."""
from __future__ import annotations

import pytest

from app.rag.rrf import FusedHit, RankedHit, reciprocal_rank_fusion, rankings_from_payloads


def _hits(ids):
    return [RankedHit(id=i, payload={"_id": i, "x": i}) for i in ids]


def test_single_ranker_preserves_order():
    out = reciprocal_rank_fusion({"vector": _hits(["a", "b", "c"])}, k=60)
    assert [h.id for h in out] == ["a", "b", "c"]
    # Score decreases with rank
    assert out[0].score > out[1].score > out[2].score


def test_two_rankers_agreement_wins():
    # 'b' is top in both, so it should outrank 'a' which is top in only one.
    out = reciprocal_rank_fusion(
        {
            "vector": _hits(["b", "a", "c"]),
            "bm25": _hits(["b", "d", "a"]),
        },
        k=60,
    )
    assert out[0].id == "b"
    assert set(out[0].sources) == {"vector", "bm25"}


def test_rrf_formula_exact():
    # Hand-computed values at k=60.
    out = reciprocal_rank_fusion(
        {
            "vector": _hits(["x", "y"]),
            "bm25": _hits(["y", "x"]),
        },
        k=60,
    )
    # Both appear at ranks (1,2) across the two rankers, so scores are equal.
    assert out[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert out[1].score == pytest.approx(1 / 61 + 1 / 62)


def test_top_k_truncates():
    out = reciprocal_rank_fusion(
        {"vector": _hits(["a", "b", "c", "d"])},
        k=60,
        top_k=2,
    )
    assert len(out) == 2
    assert [h.id for h in out] == ["a", "b"]


def test_rejects_zero_k():
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion({"v": _hits(["a"])}, k=0)


def test_payload_preserved_and_prefers_richer():
    rich = RankedHit(id="a", payload={"_id": "a", "chunk_text": "hello"})
    bare = RankedHit(id="a", payload=None)
    out = reciprocal_rank_fusion({"v": [rich], "b": [bare]}, k=60)
    assert out[0].payload is not None
    assert out[0].payload["chunk_text"] == "hello"


def test_rankings_from_payloads_adapter():
    docs = [{"_id": "1", "chunk_text": "hi"}, {"_id": "2", "chunk_text": "bye"}]
    wrapped = rankings_from_payloads({"vector": docs})
    assert len(wrapped["vector"]) == 2
    assert wrapped["vector"][0].id == "1"
    assert wrapped["vector"][0].payload["chunk_text"] == "hi"


def test_adapter_missing_id_field_raises():
    with pytest.raises(KeyError, match="_id"):
        rankings_from_payloads({"vector": [{"text": "no id"}]})
