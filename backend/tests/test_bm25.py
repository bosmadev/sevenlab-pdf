"""Unit tests for app.rag.bm25 — pipeline shape and ranked-hit decoding.

These tests stub `pdfs_collection()` so no live Atlas or network is required.
They verify:
    1. Pipeline carries the correct index, text clause, workspace filter, limit.
    2. Pipeline rejects empty query, empty workspace_id, non-positive top_k.
    3. top_k over MAX_TOP_K is clamped, not raised.
    4. `bm25_search` decodes aggregation output into ordered `BM25Hit`s.
    5. Score falls back to 0.0 when a document is missing the meta field.
"""

from __future__ import annotations

import pytest

from app.rag import bm25 as bm25_mod
from app.rag.bm25 import (
    BM25_INDEX_NAME,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    BM25Hit,
    bm25_search,
    build_bm25_pipeline,
)


# ---------- build_bm25_pipeline ---------------------------------------------


def test_pipeline_shape_defaults():
    p = build_bm25_pipeline("ws-1", "quarterly revenue")

    assert len(p) == 3
    search_stage = p[0]["$search"]
    assert search_stage["index"] == BM25_INDEX_NAME

    must = search_stage["compound"]["must"]
    assert must == [{"text": {"query": "quarterly revenue", "path": "chunk_text"}}]

    filt = search_stage["compound"]["filter"]
    assert filt == [{"equals": {"path": "workspace_id", "value": "ws-1"}}]

    assert p[1] == {"$limit": DEFAULT_TOP_K}
    project = p[2]["$project"]
    assert project["score"] == {"$meta": "searchScore"}
    assert project["chunk_text"] == 1


def test_pipeline_strips_whitespace_from_query():
    p = build_bm25_pipeline("ws-1", "   padded query   ")
    assert p[0]["$search"]["compound"]["must"][0]["text"]["query"] == "padded query"


def test_pipeline_rejects_empty_query():
    with pytest.raises(ValueError):
        build_bm25_pipeline("ws-1", "")
    with pytest.raises(ValueError):
        build_bm25_pipeline("ws-1", "   ")


def test_pipeline_rejects_empty_workspace():
    with pytest.raises(ValueError):
        build_bm25_pipeline("", "query")


def test_pipeline_rejects_non_positive_top_k():
    with pytest.raises(ValueError):
        build_bm25_pipeline("ws-1", "query", top_k=0)
    with pytest.raises(ValueError):
        build_bm25_pipeline("ws-1", "query", top_k=-5)


def test_pipeline_clamps_top_k_to_max():
    p = build_bm25_pipeline("ws-1", "query", top_k=MAX_TOP_K * 10)
    assert p[1] == {"$limit": MAX_TOP_K}


def test_pipeline_accepts_custom_index_name():
    p = build_bm25_pipeline("ws-1", "query", index_name="alt_bm25")
    assert p[0]["$search"]["index"] == "alt_bm25"


# ---------- bm25_search decoding --------------------------------------------


class _FakeCollection:
    """Minimal stand-in for pymongo.Collection.aggregate()."""

    def __init__(self, docs: list[dict]):
        self._docs = docs
        self.last_pipeline: list[dict] | None = None

    def aggregate(self, pipeline):
        self.last_pipeline = pipeline
        return iter(self._docs)


def _install_fake(monkeypatch, docs):
    fake = _FakeCollection(docs)
    monkeypatch.setattr(bm25_mod, "get_pdfs_collection", lambda: fake)
    return fake


def test_bm25_search_decodes_hits_in_order(monkeypatch):
    docs = [
        {
            "_id": "chunk-1",
            "doc_id": "doc-A",
            "workspace_id": "ws-1",
            "chunk_text": "Q3 revenue beat guidance",
            "page_num": 7,
            "score": 12.5,
        },
        {
            "_id": "chunk-2",
            "doc_id": "doc-B",
            "workspace_id": "ws-1",
            "chunk_text": "Fiscal 2026 outlook",
            "page_num": 3,
            "score": 8.1,
        },
    ]
    fake = _install_fake(monkeypatch, docs)

    hits = bm25_search("ws-1", "revenue", top_k=5)

    assert fake.last_pipeline is not None
    assert fake.last_pipeline[1] == {"$limit": 5}

    assert len(hits) == 2
    assert all(isinstance(h, BM25Hit) for h in hits)
    assert [h.chunk_id for h in hits] == ["chunk-1", "chunk-2"]
    assert hits[0].score == pytest.approx(12.5)
    assert hits[0].page_num == 7
    assert hits[1].chunk_text == "Fiscal 2026 outlook"


def test_bm25_search_defaults_missing_score_and_page(monkeypatch):
    _install_fake(
        monkeypatch,
        [
            {
                "_id": "chunk-x",
                "doc_id": "doc-X",
                "workspace_id": "ws-1",
                "chunk_text": "no meta",
            }
        ],
    )
    hits = bm25_search("ws-1", "no meta")
    assert len(hits) == 1
    assert hits[0].score == 0.0
    assert hits[0].page_num is None


def test_bm25_search_empty_cursor(monkeypatch):
    _install_fake(monkeypatch, [])
    assert bm25_search("ws-1", "nothing matches") == []


def test_bm25hit_to_pipeline_chunk_shape():
    hit = BM25Hit(
        chunk_id="chunk-1",
        doc_id="doc-A",
        workspace_id="ws-1",
        chunk_text="Q3 revenue beat guidance",
        page_num=7,
        score=12.5,
    )
    chunk = hit.to_pipeline_chunk()
    assert chunk == {
        "doc_id": "doc-A",
        "chunk_id": "chunk-1",
        "chunk_text": "Q3 revenue beat guidance",
        "page_num": 7,
        "score": 12.5,
        "source": "bm25",
    }
    assert "workspace_id" not in chunk  # retrieval-time only, never leaked downstream
