"""Live end-to-end tests against MongoDB Atlas.

Skipped unless MONGODB_URI is exported — these touch the real cluster
`sevenlab.bsbpajx.mongodb.net` and exercise the two search indexes created
by `app.db.indices`:

    - vector_index  (2048-dim, cosine, workspace_id + doc_id filters)
    - bm25_index    (lucene.standard, workspace_id + doc_id tokens)

Chunk field is `doc_id` (task #27 reconciliation, see app.db.vector.DOC_ID_FIELD).

Isolation: every write is tagged with workspace_id="_t003_integration" and
cleaned up in the fixture teardown. The regular pdfs data is never touched.

Run:
    MONGODB_URI='mongodb+srv://...' pytest tests/test_db_live_atlas.py -v
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("MONGODB_URI"),
    reason="live Atlas tests require MONGODB_URI",
)

TEST_WORKSPACE = "_t003_integration"
VECTOR_DIM = 2048


@pytest.fixture(scope="module")
def pdfs_collection():
    from app.db.mongodb import get_pdfs_collection

    coll = get_pdfs_collection()
    # Pre-clean any leftover rows from a prior failed run.
    coll.delete_many({"workspace_id": TEST_WORKSPACE})
    yield coll
    coll.delete_many({"workspace_id": TEST_WORKSPACE})


def _chunk(doc_id: str, chunk_index: int, text: str, embedding: list[float]) -> dict:
    return {
        "workspace_id": TEST_WORKSPACE,
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "chunk_text": text,
        "page_num": 1,
        "embedding": embedding,
        "created_at": datetime.now(timezone.utc),
    }


def test_atlas_ping() -> None:
    from app.db.mongodb import ping

    assert ping().get("ok") is True


def test_search_indexes_present_and_queryable(pdfs_collection) -> None:
    from app.db.indices import SEARCH_INDEX_NAME, VECTOR_INDEX_NAME

    names = {i.get("name"): i for i in pdfs_collection.list_search_indexes()}
    assert VECTOR_INDEX_NAME in names, f"missing {VECTOR_INDEX_NAME}; have {list(names)}"
    assert SEARCH_INDEX_NAME in names, f"missing {SEARCH_INDEX_NAME}; have {list(names)}"
    for name in (VECTOR_INDEX_NAME, SEARCH_INDEX_NAME):
        assert names[name].get("status") == "READY", (
            f"{name} not READY: {names[name].get('status')}"
        )
        assert names[name].get("queryable") is True, (
            f"{name} not queryable: {names[name]}"
        )


def test_vector_search_round_trip(pdfs_collection) -> None:
    """Insert a chunk, run $vectorSearch, confirm it comes back ranked first."""
    from app.db.indices import VECTOR_INDEX_NAME

    doc_id = str(uuid.uuid4())
    # Deterministic embedding: a single hot dimension, rest zero.
    hot_vec = [0.0] * VECTOR_DIM
    hot_vec[0] = 1.0
    cold_vec = [0.0] * VECTOR_DIM
    cold_vec[1] = 1.0

    pdfs_collection.insert_many(
        [
            _chunk(doc_id, 0, "the hot chunk about revenue growth", hot_vec),
            _chunk(doc_id, 1, "the cold chunk about something else", cold_vec),
        ]
    )

    # $vectorSearch is near-realtime after insert; brief backoff for propagation.
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": hot_vec,
                "numCandidates": 10,
                "limit": 5,
                "filter": {"workspace_id": TEST_WORKSPACE},
            }
        },
        {
            "$project": {
                "_id": 0,
                "chunk_index": 1,
                "chunk_text": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    hits: list = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not hits:
        hits = list(pdfs_collection.aggregate(pipeline))
        if hits:
            break
        time.sleep(2)

    assert hits, "vector search returned no hits within 30s"
    assert hits[0]["chunk_index"] == 0, f"expected hot chunk first; got {hits}"
    assert hits[0]["score"] > hits[1]["score"] if len(hits) > 1 else True


def test_bm25_search_round_trip(pdfs_collection) -> None:
    """Insert chunks, run Atlas $search, confirm BM25 scoring matches."""
    from app.db.indices import SEARCH_INDEX_NAME

    doc_id = str(uuid.uuid4())
    zero = [0.0] * VECTOR_DIM
    pdfs_collection.insert_many(
        [
            _chunk(doc_id, 0, "quarterly revenue report with specific figures", zero),
            _chunk(doc_id, 1, "unrelated marketing newsletter template", zero),
        ]
    )

    # workspace_id is a token field — must use `equals`, not `text`.
    # Matches the live caller shape in app.rag.bm25.build_bm25_pipeline.
    pipeline = [
        {
            "$search": {
                "index": SEARCH_INDEX_NAME,
                "compound": {
                    "must": [
                        {"text": {"query": "revenue", "path": "chunk_text"}},
                    ],
                    "filter": [
                        {"equals": {"path": "workspace_id", "value": TEST_WORKSPACE}},
                    ],
                },
            }
        },
        {"$limit": 5},
        {
            "$project": {
                "_id": 0,
                "chunk_index": 1,
                "chunk_text": 1,
                "score": {"$meta": "searchScore"},
            }
        },
    ]

    hits: list = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not hits:
        hits = list(pdfs_collection.aggregate(pipeline))
        if hits:
            break
        time.sleep(2)

    assert hits, "BM25 search returned no hits within 30s"
    assert hits[0]["chunk_index"] == 0, f"expected revenue chunk first; got {hits}"


def test_gridfs_round_trip() -> None:
    """Store + retrieve a tiny PDF via GridFS."""
    from app.db.mongodb import get_gridfs

    fs = get_gridfs()
    payload = b"%PDF-1.4\n%EOF\n" + b"\x00" * 128
    file_id = fs.put(
        payload,
        filename=f"_t003_test_{uuid.uuid4().hex}.pdf",
        contentType="application/pdf",
        metadata={"workspace_id": TEST_WORKSPACE},
    )
    try:
        got = fs.get(file_id).read()
        assert got == payload
        assert fs.exists(file_id)
    finally:
        fs.delete(file_id)
        assert not fs.exists(file_id)


def test_regular_indexes_present(pdfs_collection) -> None:
    names = {i["name"] for i in pdfs_collection.list_indexes()}
    for expected in (
        "ix_workspace_doc_chunk",
        "ix_workspace_created",
        "ix_doc",
    ):
        assert expected in names, f"missing regular index {expected}; have {names}"


def test_chats_and_gridfs_indexes_present() -> None:
    """`ensure_all_indexes` also covers `chats` and the GridFS bucket."""
    from app.db.indices import (
        CHATS_INDEX_NAME,
        GRIDFS_BUCKET,
        GRIDFS_WORKSPACE_INDEX_NAME,
        ensure_chats_indexes,
        ensure_gridfs_indexes,
    )
    from app.db.mongodb import get_chats_collection, get_mongo_db

    ensure_chats_indexes()
    ensure_gridfs_indexes()

    chat_names = {i["name"] for i in get_chats_collection().list_indexes()}
    assert CHATS_INDEX_NAME in chat_names, chat_names

    db = get_mongo_db()
    file_names = {i["name"] for i in db[f"{GRIDFS_BUCKET}.files"].list_indexes()}
    assert "filename_1_uploadDate_1" in file_names, file_names
    assert GRIDFS_WORKSPACE_INDEX_NAME in file_names, file_names
    chunk_names = {i["name"] for i in db[f"{GRIDFS_BUCKET}.chunks"].list_indexes()}
    assert "files_id_1_n_1" in chunk_names, chunk_names
