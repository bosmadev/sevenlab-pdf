"""Hybrid retriever orchestration tests — T-010.

The real embedder, Mongo and BGE model are mocked. The point of these
tests is to verify the pipeline order, fallback behaviour, score
plumbing and workspace scoping — not re-test the stages they call.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.rag import bm25 as _bm25_mod
from app.rag import reranker as _reranker_mod
from app.rag.retriever import HybridRetriever


def _mk_vec_hit(cid: str, doc_id: str, text: str, page: int, score: float = 0.9) -> dict:
    return {
        "_id": cid,
        "doc_id": doc_id,  # vector writer schema
        "workspace_id": "ws1",
        "chunk_text": text,
        "page_num": page,
        "vector_score": score,
    }


def _mk_bm25_hit(cid: str, doc_id: str, text: str, page: int, score: float = 7.3) -> _bm25_mod.BM25Hit:
    return _bm25_mod.BM25Hit(
        chunk_id=cid,
        doc_id=doc_id,  # schema aligned on doc_id post task #27
        workspace_id="ws1",
        chunk_text=text,
        page_num=page,
        score=score,
    )


def _fake_rerank(top_k_val: int):
    """Fake cross-encoder — scores by text length descending, no model load."""

    def rerank(query, candidates, *, top_k=8, model_name=None, device=None,
               batch_size=32, max_length=512):
        hits = [
            _reranker_mod.RerankedHit(
                id=c.id,
                text=c.text,
                score=float(len(c.text)),
                payload=c.payload,
            )
            for c in candidates
        ]
        hits.sort(key=lambda h: (-h.score, h.id))
        return hits[:top_k]

    return rerank


@pytest.mark.asyncio
async def test_hybrid_pipeline_happy_path():
    async def fake_embed(text, *, dimension=None):
        return [0.1] * (dimension or 2048)

    def fake_vector(workspace, qvec, top_k, doc_id):
        assert workspace == "ws1"
        assert len(qvec) == 2048
        return [
            _mk_vec_hit("c1", "d1", "the mitochondrion is the powerhouse of the cell", 2),
            _mk_vec_hit("c2", "d1", "chlorophyll fluorescence", 5),
        ]

    def fake_bm25(workspace, query, top_k, doc_id=None):
        return [
            _mk_bm25_hit("c2", "d1", "chlorophyll fluorescence", 5),
            _mk_bm25_hit("c3", "d1", "osmosis across semi-permeable membranes", 7),
        ]

    r = HybridRetriever(
        embed_query_fn=fake_embed,
        vector_search_fn=fake_vector,
        bm25_search_fn=fake_bm25,
        rerank_fn=_fake_rerank(8),
    )
    out = await r.search("ws1", "powerhouse of the cell")
    assert len(out) == 3
    # Longest text wins the fake reranker: c1
    assert out[0].chunk_id == "c1"
    # c2 appeared in both retrievers — sources must include both
    c2 = next(h for h in out if h.chunk_id == "c2")
    assert set(c2.sources) == {"vector", "bm25"}
    # Score plumbing: c2 carries both vector and bm25 scores forward
    assert c2.vector_score == pytest.approx(0.9)
    assert c2.bm25_score == pytest.approx(7.3)
    # Both writers now agree on doc_id (post task #27 alignment);
    # retriever must round-trip it unchanged.
    assert all(h.doc_id == "d1" for h in out)


@pytest.mark.asyncio
async def test_hybrid_search_module_returns_dicts_for_t012():
    """``hybrid_search`` returns dicts shaped for ``app.rag.pipeline``."""
    import app.rag.retriever as retriever

    async def fake_embed(text, *, dimension=None):
        return [0.1] * 2048

    def fake_vector(workspace, qvec, top_k, doc_id):
        return [_mk_vec_hit("c1", "d1", "example chunk text", 2)]

    def fake_bm25(workspace, query, top_k, doc_id=None):
        return []

    original = retriever._DEFAULT_RETRIEVER
    retriever._DEFAULT_RETRIEVER = HybridRetriever(
        embed_query_fn=fake_embed,
        vector_search_fn=fake_vector,
        bm25_search_fn=fake_bm25,
        rerank_fn=_fake_rerank(8),
    )
    try:
        out = await retriever.hybrid_search("ws1", "question", top_k=5)
        assert isinstance(out, list) and len(out) == 1
        d = out[0]
        # T-012 pipeline keys
        for key in ("chunk_id", "doc_id", "chunk_text", "page_num", "score"):
            assert key in d, f"missing T-012 key {key!r}"
        assert d["doc_id"] == "d1"
        # ``score`` must be the rerank score — not RRF, not vector
        assert d["score"] == d["rerank_score"]
    finally:
        retriever._DEFAULT_RETRIEVER = original


@pytest.mark.asyncio
async def test_pipeline_degrades_when_bm25_fails():
    async def fake_embed(text, *, dimension=None):
        return [0.1] * 2048

    def fake_vector(workspace, qvec, top_k, doc_id):
        return [_mk_vec_hit("c1", "d1", "vector-only answer", 1)]

    def fake_bm25(workspace, query, top_k, doc_id=None):
        raise RuntimeError("bm25_index missing")

    r = HybridRetriever(
        embed_query_fn=fake_embed,
        vector_search_fn=fake_vector,
        bm25_search_fn=fake_bm25,
        rerank_fn=_fake_rerank(8),
    )
    out = await r.search("ws1", "any query")
    assert [h.chunk_id for h in out] == ["c1"]
    assert out[0].sources == ("vector",)


@pytest.mark.asyncio
async def test_pipeline_degrades_when_vector_fails():
    async def fake_embed(text, *, dimension=None):
        return [0.1] * 2048

    def fake_vector(workspace, qvec, top_k, doc_id):
        raise RuntimeError("mongo down")

    def fake_bm25(workspace, query, top_k, doc_id=None):
        return [_mk_bm25_hit("c9", "d9", "sparse-only answer", 3)]

    r = HybridRetriever(
        embed_query_fn=fake_embed,
        vector_search_fn=fake_vector,
        bm25_search_fn=fake_bm25,
        rerank_fn=_fake_rerank(8),
    )
    out = await r.search("ws1", "any query")
    assert [h.chunk_id for h in out] == ["c9"]
    assert out[0].sources == ("bm25",)


@pytest.mark.asyncio
async def test_pipeline_empty_both_returns_empty():
    async def fake_embed(text, *, dimension=None):
        return [0.1] * 2048

    r = HybridRetriever(
        embed_query_fn=fake_embed,
        vector_search_fn=lambda *a, **kw: [],
        bm25_search_fn=lambda *a, **kw: [],
        rerank_fn=_fake_rerank(8),
    )
    out = await r.search("ws1", "orphan query")
    assert out == []


@pytest.mark.asyncio
async def test_document_filter_restricts_bm25_payloads():
    async def fake_embed(text, *, dimension=None):
        return [0.1] * 2048

    seen_bm25_doc: dict[str, Any] = {}

    def fake_vector(workspace, qvec, top_k, doc_id):
        # Vector respects doc_id directly.
        return [_mk_vec_hit("c1", "d1", "doc-1 chunk", 2)] if doc_id == "d1" else []

    def fake_bm25(workspace, query, top_k, doc_id=None):
        # Simulate a BM25 backend that IGNORES the filter — the retriever's
        # post-filter must still drop the sibling document's chunk.
        seen_bm25_doc["doc_id"] = doc_id
        return [
            _mk_bm25_hit("c2", "d1", "doc-1 keyword hit", 3),
            _mk_bm25_hit("c9", "d9", "doc-9 keyword hit, must be dropped", 1),
        ]

    r = HybridRetriever(
        embed_query_fn=fake_embed,
        vector_search_fn=fake_vector,
        bm25_search_fn=fake_bm25,
        rerank_fn=_fake_rerank(8),
    )
    out = await r.search("ws1", "doc specific query", doc_id="d1")
    ids = {h.chunk_id for h in out}
    assert ids == {"c1", "c2"}, f"doc filter failed: {ids}"
    # The filter is forwarded server-side too, not only applied after the fact.
    assert seen_bm25_doc["doc_id"] == "d1"


@pytest.mark.asyncio
async def test_rejects_empty_query():
    r = HybridRetriever(
        embed_query_fn=lambda *a, **kw: [],
        vector_search_fn=lambda *a, **kw: [],
        bm25_search_fn=lambda *a, **kw: [],
        rerank_fn=_fake_rerank(8),
    )
    with pytest.raises(ValueError):
        await r.search("ws1", "   ")


@pytest.mark.asyncio
async def test_rejects_empty_workspace():
    r = HybridRetriever(
        embed_query_fn=lambda *a, **kw: [],
        vector_search_fn=lambda *a, **kw: [],
        bm25_search_fn=lambda *a, **kw: [],
        rerank_fn=_fake_rerank(8),
    )
    with pytest.raises(ValueError, match="workspace"):
        await r.search("", "query")


@pytest.mark.asyncio
async def test_top_k_override_respected():
    async def fake_embed(text, *, dimension=None):
        return [0.1] * 2048

    def fake_vector(workspace, qvec, top_k, doc_id):
        return [_mk_vec_hit(f"c{i}", "d1", f"chunk {i}" * (10 - i), i) for i in range(10)]

    def fake_bm25(workspace, query, top_k, doc_id=None):
        return []

    r = HybridRetriever(
        embed_query_fn=fake_embed,
        vector_search_fn=fake_vector,
        bm25_search_fn=fake_bm25,
        rerank_fn=_fake_rerank(8),
    )
    out = await r.search("ws1", "q", top_k=3)
    assert len(out) == 3


# ---------------------------------------------------------------------------
# Soft-deleted document guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_soft_deleted_documents_dropped_from_both_retrievers():
    async def fake_embed(text, *, dimension=None):
        return [0.1] * 2048

    def fake_vector(workspace, qvec, top_k, doc_id):
        return [
            _mk_vec_hit("c1", "d1", "live document chunk", 1),
            _mk_vec_hit("c9", "d9", "deleted document chunk, must vanish", 1),
        ]

    def fake_bm25(workspace, query, top_k, doc_id=None):
        return [
            _mk_bm25_hit("c2", "d1", "live keyword hit", 2),
            _mk_bm25_hit("c8", "d9", "deleted keyword hit, must vanish too", 2),
        ]

    seen: dict[str, Any] = {}

    def fake_deleted(workspace, doc_ids):
        seen["workspace"] = workspace
        seen["doc_ids"] = set(doc_ids)
        return {"d9"}

    r = HybridRetriever(
        embed_query_fn=fake_embed,
        vector_search_fn=fake_vector,
        bm25_search_fn=fake_bm25,
        rerank_fn=_fake_rerank(8),
        deleted_doc_ids_fn=fake_deleted,
    )
    out = await r.search("ws1", "anything")
    assert {h.chunk_id for h in out} == {"c1", "c2"}
    # The lookup is one call carrying the union of both retrievers' doc ids.
    assert seen == {"workspace": "ws1", "doc_ids": {"d1", "d9"}}


@pytest.mark.asyncio
async def test_all_hits_soft_deleted_returns_empty():
    async def fake_embed(text, *, dimension=None):
        return [0.1] * 2048

    r = HybridRetriever(
        embed_query_fn=fake_embed,
        vector_search_fn=lambda *a, **kw: [_mk_vec_hit("c9", "d9", "gone", 1)],
        bm25_search_fn=lambda *a, **kw: [_mk_bm25_hit("c8", "d9", "gone too", 1)],
        rerank_fn=_fake_rerank(8),
        deleted_doc_ids_fn=lambda ws, ids: {"d9"},
    )
    assert await r.search("ws1", "anything") == []


@pytest.mark.asyncio
async def test_deleted_guard_outage_fails_open():
    async def fake_embed(text, *, dimension=None):
        return [0.1] * 2048

    def fake_deleted(workspace, doc_ids):
        raise RuntimeError("postgres down")

    r = HybridRetriever(
        embed_query_fn=fake_embed,
        vector_search_fn=lambda *a, **kw: [_mk_vec_hit("c1", "d1", "still served", 1)],
        bm25_search_fn=lambda *a, **kw: [],
        rerank_fn=_fake_rerank(8),
        deleted_doc_ids_fn=fake_deleted,
    )
    out = await r.search("ws1", "anything")
    assert [h.chunk_id for h in out] == ["c1"]


def test_deleted_doc_ids_reads_postgres_deleted_at():
    """The default guard queries ``documents.deleted_at`` — SQLite stands in."""
    import uuid
    from datetime import datetime, timezone

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.db.models import Document, User
    from app.db.postgres import Base
    from app.rag.retriever import deleted_doc_ids

    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, future=True)

    live_id = uuid.uuid4()
    gone_id = uuid.uuid4()
    with SessionLocal() as db:
        user = User(id=uuid.uuid4(), workspace_id="alice", password_hash="x")
        db.add(user)
        db.flush()
        for did, deleted in ((live_id, False), (gone_id, True)):
            db.add(
                Document(
                    id=did,
                    user_id=user.id,
                    workspace_id="alice",
                    original_filename="paper.pdf",
                    sanitized_filename="paper.pdf",
                    gridfs_file_id="0" * 24,
                    sha256="a" * 64,
                    size_bytes=1,
                    status="deleted" if deleted else "ready",
                    deleted_at=datetime.now(timezone.utc) if deleted else None,
                )
            )
        db.commit()

    found = deleted_doc_ids(
        "alice",
        [str(live_id), str(gone_id), "not-a-uuid"],
        session_factory=SessionLocal,
    )
    assert found == {str(gone_id)}
    assert deleted_doc_ids("alice", [], session_factory=SessionLocal) == set()
