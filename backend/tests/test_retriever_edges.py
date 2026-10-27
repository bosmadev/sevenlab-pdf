"""Hybrid retriever edge cases (T-010) not covered by ``test_retriever.py``.

Argument plumbing into each stage, the fused-pool cap, double-failure,
embedding failure propagation, RRF score carry-over, the legacy
``document_id`` fallback, citation rendering and the T-012 dict adapter.
"""
from __future__ import annotations

import pytest

from app.config import get_settings
from app.rag import bm25 as _bm25_mod
from app.rag import reranker as _reranker_mod
from app.rag import retriever as retriever_mod
from app.rag.retriever import (
    DEFAULT_BM25_TOP_K,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_VECTOR_TOP_K,
    HybridRetriever,
    RetrievedChunk,
    _normalise_chunk,
    chunk_to_dict,
    get_default_retriever,
)


def _vec(cid: str, text: str = "t", doc_id: str = "d1", page: int | None = 1) -> dict:
    return {
        "_id": cid,
        "doc_id": doc_id,
        "workspace_id": "ws1",
        "chunk_text": text,
        "page_num": page,
        "vector_score": 0.5,
    }


def _bm(cid: str, text: str = "t", doc_id: str = "d1", page: int = 1) -> _bm25_mod.BM25Hit:
    return _bm25_mod.BM25Hit(
        chunk_id=cid, doc_id=doc_id, workspace_id="ws1",
        chunk_text=text, page_num=page, score=3.0,
    )


async def _embed(text, *, dimension=None):
    return [0.0] * (dimension or 8)


def _identity_rerank(query, candidates, *, top_k=8, model_name=None, device=None,
                     batch_size=32, max_length=512):
    cands = list(candidates)
    return [
        _reranker_mod.RerankedHit(id=c.id, text=c.text, score=float(len(cands) - i), payload=c.payload)
        for i, c in enumerate(cands)
    ][:top_k]


# ---- pure helpers ------------------------------------------------------------


def test_normalise_chunk_prefers_doc_id_then_legacy_document_id():
    assert _normalise_chunk({"doc_id": "new", "document_id": "old"})["doc_id"] == "new"
    assert _normalise_chunk({"document_id": "old"})["doc_id"] == "old"
    assert _normalise_chunk({})["doc_id"] == ""


def test_normalise_chunk_maps_bm25_raw_score_and_stringifies_ids():
    from bson import ObjectId

    oid = ObjectId()
    out = _normalise_chunk({"_id": oid, "score": 4.2, "page_num": 3})
    assert out["_id"] == str(oid)
    assert out["bm25_score"] == 4.2
    assert out["vector_score"] is None
    assert out["page_num"] == 3
    assert _normalise_chunk({"bm25_score": 1.5, "score": 9.9})["bm25_score"] == 1.5


def test_citation_with_and_without_page():
    base = dict(chunk_id="c", doc_id="d9", workspace_id="w", chunk_text="x",
                rerank_score=1.0, rrf_score=0.1, vector_score=None, bm25_score=None,
                sources=("vector",))
    assert RetrievedChunk(page_num=4, **base).citation() == "[doc d9 p.4]"
    assert RetrievedChunk(page_num=None, **base).citation() == "[doc d9]"


def test_chunk_to_dict_exposes_rerank_score_as_headline():
    chunk = RetrievedChunk(
        chunk_id="c1", doc_id="d1", workspace_id="ws", chunk_text="hello",
        page_num=2, rerank_score=0.8, rrf_score=0.03, vector_score=0.9,
        bm25_score=None, sources=("vector", "bm25"),
    )
    d = chunk_to_dict(chunk)
    assert d["score"] == 0.8 == d["rerank_score"]
    assert d["sources"] == ["vector", "bm25"]
    assert set(d) == {
        "chunk_id", "doc_id", "workspace_id", "chunk_text", "page_num", "score",
        "rerank_score", "rrf_score", "vector_score", "bm25_score", "sources",
    }


def test_default_retriever_is_a_singleton_with_plan_defaults():
    a = get_default_retriever()
    assert a is get_default_retriever()
    assert (a.vector_top_k, a.bm25_top_k, a.rrf_k, a.rerank_top_k, a.fused_pool_k) == (20, 20, 60, 8, 40)
    assert (DEFAULT_VECTOR_TOP_K, DEFAULT_BM25_TOP_K, DEFAULT_RERANK_TOP_K) == (20, 20, 8)


# ---- stage plumbing ----------------------------------------------------------


@pytest.mark.asyncio
async def test_each_stage_receives_the_configured_arguments(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "16")
    get_settings.cache_clear()
    seen: dict = {}

    async def embed(text, *, dimension=None):
        seen["embed"] = (text, dimension)
        return [0.25] * dimension

    def vector(workspace, qvec, top_k, doc_id):
        seen["vector"] = (workspace, len(qvec), top_k, doc_id)
        return [_vec("c1")]

    def bm25(workspace, query, top_k):
        seen["bm25"] = (workspace, query, top_k)
        return [_bm("c2")]

    def rerank(query, candidates, *, top_k, model_name, device, batch_size=32, max_length=512):
        seen["rerank"] = (query, [c.id for c in candidates], top_k, model_name, device)
        return _identity_rerank(query, candidates, top_k=top_k)

    try:
        r = HybridRetriever(
            vector_top_k=7, bm25_top_k=5, rerank_top_k=3,
            reranker_model="tiny/model", reranker_device="cpu",
            embed_query_fn=embed, vector_search_fn=vector, bm25_search_fn=bm25, rerank_fn=rerank,
        )
        out = await r.search("ws1", "  padded query  ", doc_id="d1")
    finally:
        get_settings.cache_clear()

    assert seen["embed"] == ("padded query", 16)
    assert seen["vector"] == ("ws1", 16, 7, "d1")
    assert seen["bm25"] == ("ws1", "padded query", 5)
    assert seen["rerank"] == ("padded query", ["c1", "c2"], 3, "tiny/model", "cpu")
    assert len(out) == 2


@pytest.mark.asyncio
async def test_fused_pool_cap_limits_reranker_input():
    handed: list[int] = []

    def rerank(query, candidates, *, top_k, model_name=None, device=None, **_):
        cands = list(candidates)
        handed.append(len(cands))
        return _identity_rerank(query, cands, top_k=top_k)

    r = HybridRetriever(
        fused_pool_k=5, rerank_top_k=50,
        embed_query_fn=_embed,
        vector_search_fn=lambda *a: [_vec(f"v{i}") for i in range(30)],
        bm25_search_fn=lambda *a: [_bm(f"b{i}") for i in range(30)],
        rerank_fn=rerank,
    )
    out = await r.search("ws1", "q")
    assert handed == [5]
    assert len(out) == 5


@pytest.mark.asyncio
async def test_both_retrievers_failing_returns_empty_without_reranking():
    def boom(*a):
        raise RuntimeError("down")

    called = {"rerank": False}

    def rerank(*a, **k):
        called["rerank"] = True
        return []

    r = HybridRetriever(embed_query_fn=_embed, vector_search_fn=boom, bm25_search_fn=boom, rerank_fn=rerank)
    assert await r.search("ws1", "q") == []
    assert called["rerank"] is False


@pytest.mark.asyncio
async def test_embedding_failure_propagates():
    async def embed(text, *, dimension=None):
        raise RuntimeError("cohere 503")

    r = HybridRetriever(
        embed_query_fn=embed,
        vector_search_fn=lambda *a: [_vec("c1")],
        bm25_search_fn=lambda *a: [],
        rerank_fn=_identity_rerank,
    )
    with pytest.raises(RuntimeError, match="cohere"):
        await r.search("ws1", "q")


@pytest.mark.asyncio
async def test_rrf_score_rewards_agreement_between_retrievers():
    r = HybridRetriever(
        rrf_k=60,
        embed_query_fn=_embed,
        vector_search_fn=lambda *a: [_vec("both"), _vec("vec-only")],
        bm25_search_fn=lambda *a: [_bm("both"), _bm("bm-only")],
        rerank_fn=_identity_rerank,
    )
    out = {h.chunk_id: h for h in await r.search("ws1", "q")}
    assert out["both"].rrf_score == pytest.approx(2 / 61)
    assert out["vec-only"].rrf_score == pytest.approx(1 / 62)
    assert out["bm-only"].rrf_score == pytest.approx(1 / 62)
    assert set(out["both"].sources) == {"vector", "bm25"}
    assert out["both"].vector_score == 0.5
    assert out["both"].bm25_score == 3.0
    assert out["vec-only"].bm25_score is None
    assert out["bm-only"].vector_score is None


@pytest.mark.asyncio
async def test_doc_filter_with_no_surviving_bm25_hits_still_returns_vector():
    r = HybridRetriever(
        embed_query_fn=_embed,
        vector_search_fn=lambda ws, q, k, doc_id: [_vec("c1", doc_id=doc_id)],
        bm25_search_fn=lambda *a: [_bm("c9", doc_id="other")],
        rerank_fn=_identity_rerank,
    )
    out = await r.search("ws1", "q", doc_id="d7")
    assert [h.chunk_id for h in out] == ["c1"]
    assert out[0].doc_id == "d7"


@pytest.mark.asyncio
async def test_legacy_document_id_vector_rows_keep_their_doc_link():
    legacy = {"_id": "c1", "document_id": "legacy-doc", "workspace_id": "ws1",
              "chunk_text": "old row", "page_num": 2, "vector_score": 0.4}
    r = HybridRetriever(
        embed_query_fn=_embed,
        vector_search_fn=lambda *a: [legacy],
        bm25_search_fn=lambda *a: [],
        rerank_fn=_identity_rerank,
    )
    out = await r.search("ws1", "q")
    assert out[0].doc_id == "legacy-doc"
    assert out[0].page_num == 2


@pytest.mark.asyncio
async def test_hybrid_search_top_k_default_and_override(monkeypatch):
    def vector(*a):
        return [_vec(f"c{i}", text=f"chunk {i}") for i in range(12)]

    fake = HybridRetriever(
        embed_query_fn=_embed, vector_search_fn=vector,
        bm25_search_fn=lambda *a: [], rerank_fn=_identity_rerank,
    )
    monkeypatch.setattr(retriever_mod, "_DEFAULT_RETRIEVER", fake)
    assert len(await retriever_mod.hybrid_search("ws1", "q")) == DEFAULT_RERANK_TOP_K
    assert len(await retriever_mod.hybrid_search("ws1", "q", top_k=2)) == 2
