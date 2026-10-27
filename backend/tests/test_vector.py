"""Tests for app.db.vector (T-008 — MongoDB Atlas Vector Search data-plane).

Pure-logic tests only: no network to Atlas. The collection boundary is
monkey-patched so we validate upsert operations and query pipeline shapes
without a live cluster. Atlas index creation is covered separately in
tests for `app.db.indices`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.db import vector as v


class _FakeCollection:
    """Minimal pymongo Collection stand-in capturing calls for assertions."""

    def __init__(self) -> None:
        self.bulk_calls: list[list[Any]] = []
        self.deleted: list[dict[str, Any]] = []
        self.aggregates: list[list[dict[str, Any]]] = []

    def bulk_write(self, operations, ordered=True):
        self.bulk_calls.append(operations)
        result = MagicMock()
        result.upserted_ids = {i: f"id-{i}" for i in range(len(operations))}
        result.modified_count = 0
        return result

    def delete_many(self, query):
        self.deleted.append(query)
        result = MagicMock()
        result.deleted_count = 3
        return result

    def aggregate(self, pipeline):
        self.aggregates.append(pipeline)
        return iter(
            [
                {
                    "_id": "doc1:0",
                    "workspace_id": "ws-A",
                    "doc_id": "doc1",
                    "chunk_index": 0,
                    "chunk_text": "hello",
                    "page_num": 1,
                    "vector_score": 0.97,
                }
            ]
        )

    def count_documents(self, query):
        return 42 if query else 0


@pytest.fixture
def fake_coll(monkeypatch: pytest.MonkeyPatch) -> _FakeCollection:
    coll = _FakeCollection()
    monkeypatch.setattr(v, "_collection", lambda: coll)
    return coll


def test_upsert_chunks_validates_dimensions(fake_coll: _FakeCollection) -> None:
    with pytest.raises(ValueError, match="embedding has 3 dims"):
        v.upsert_chunks(
            workspace_id="ws-A",
            doc_id="doc1",
            chunks=[{"chunk_index": 0, "chunk_text": "x", "embedding": [0.1, 0.2, 0.3]}],
        )


def test_upsert_chunks_requires_workspace_id(fake_coll: _FakeCollection) -> None:
    with pytest.raises(ValueError, match="workspace_id is required"):
        v.upsert_chunks(workspace_id="", doc_id="doc1", chunks=[])


def test_upsert_chunks_requires_doc_id(fake_coll: _FakeCollection) -> None:
    with pytest.raises(ValueError, match="doc_id is required"):
        v.upsert_chunks(workspace_id="ws-A", doc_id="", chunks=[])


def test_upsert_chunks_rejects_missing_fields(fake_coll: _FakeCollection) -> None:
    with pytest.raises(ValueError, match="requires chunk_index"):
        v.upsert_chunks(
            workspace_id="ws-A",
            doc_id="doc1",
            chunks=[{"chunk_text": "x", "embedding": [0.0] * 2048}],
        )


def test_upsert_chunks_empty_iterable_is_noop(fake_coll: _FakeCollection) -> None:
    assert v.upsert_chunks(workspace_id="ws-A", doc_id="doc1", chunks=[]) == 0
    assert fake_coll.bulk_calls == []


def test_upsert_chunks_builds_correct_upsert_ops(fake_coll: _FakeCollection) -> None:
    emb = [0.0] * 2048
    chunks = [
        {"chunk_index": 0, "chunk_text": "first", "embedding": emb, "page_num": 1},
        {"chunk_index": 1, "chunk_text": "second", "embedding": emb, "page_num": 2,
         "ocr_tier": "tesseract"},
    ]
    rows = v.upsert_chunks(workspace_id="ws-A", doc_id="doc1", chunks=chunks)
    assert rows == 2  # upserted_ids has 2 entries

    assert len(fake_coll.bulk_calls) == 1
    ops = fake_coll.bulk_calls[0]
    assert len(ops) == 2

    op0 = ops[0]
    filter_doc = op0._filter
    update_doc = op0._doc
    assert filter_doc == {"_id": "doc1:0"}
    assert op0._upsert is True
    set_fields = update_doc["$set"]
    assert set_fields[v.WORKSPACE_FIELD] == "ws-A"
    assert set_fields[v.DOC_ID_FIELD] == "doc1"
    assert set_fields[v.CHUNK_INDEX_FIELD] == 0
    assert set_fields["chunk_text"] == "first"
    assert set_fields["page_num"] == 1
    assert len(set_fields[v.VECTOR_FIELD]) == 2048
    assert "created_at" in update_doc["$setOnInsert"]

    # extra metadata passes through
    assert ops[1]._doc["$set"]["ocr_tier"] == "tesseract"


def test_vector_search_builds_pipeline_with_workspace_filter(fake_coll: _FakeCollection) -> None:
    query_emb = [0.0] * 2048
    results = v.vector_search(workspace_id="ws-A", query_embedding=query_emb, top_k=8)

    assert len(results) == 1
    assert results[0]["vector_score"] == 0.97

    pipeline = fake_coll.aggregates[0]
    stage = pipeline[0]["$vectorSearch"]
    assert stage["index"] == v.VECTOR_INDEX_NAME
    assert stage["path"] == v.VECTOR_FIELD
    assert stage["limit"] == 8
    assert stage["numCandidates"] == 150  # max(8*10, 150)
    assert stage["filter"] == {v.WORKSPACE_FIELD: {"$eq": "ws-A"}}


def test_vector_search_num_candidates_scales_with_top_k(fake_coll: _FakeCollection) -> None:
    v.vector_search(workspace_id="ws-A", query_embedding=[0.0] * 2048, top_k=50)
    stage = fake_coll.aggregates[0][0]["$vectorSearch"]
    assert stage["numCandidates"] == 500  # max(50*10, 150)


def test_vector_search_num_candidates_override(fake_coll: _FakeCollection) -> None:
    v.vector_search(
        workspace_id="ws-A",
        query_embedding=[0.0] * 2048,
        top_k=20,
        num_candidates=1000,
    )
    stage = fake_coll.aggregates[0][0]["$vectorSearch"]
    assert stage["numCandidates"] == 1000


def test_vector_search_optional_doc_filter(fake_coll: _FakeCollection) -> None:
    v.vector_search(
        workspace_id="ws-A",
        query_embedding=[0.0] * 2048,
        top_k=20,
        doc_id="doc1",
    )
    stage = fake_coll.aggregates[0][0]["$vectorSearch"]
    assert stage["filter"] == {
        v.WORKSPACE_FIELD: {"$eq": "ws-A"},
        v.DOC_ID_FIELD: {"$eq": "doc1"},
    }


def test_vector_search_rejects_wrong_dim(fake_coll: _FakeCollection) -> None:
    with pytest.raises(ValueError, match="query_embedding has 10 dims"):
        v.vector_search(workspace_id="ws-A", query_embedding=[0.0] * 10, top_k=8)


def test_delete_document_chunks(fake_coll: _FakeCollection) -> None:
    n = v.delete_document_chunks("ws-A", "doc1")
    assert n == 3
    assert fake_coll.deleted == [{v.WORKSPACE_FIELD: "ws-A", v.DOC_ID_FIELD: "doc1"}]


def test_count_chunks_scoped(fake_coll: _FakeCollection) -> None:
    assert v.count_chunks("ws-A") == 42
    assert v.count_chunks("ws-A", doc_id="doc1") == 42
