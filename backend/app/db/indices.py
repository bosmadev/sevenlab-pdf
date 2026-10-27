"""MongoDB Atlas index management (idempotent).

Covers T-003, T-008, T-009:
    1. Regular indexes on `pdfs` (workspace_id, doc_id, created_at).
    2. Atlas Vector Search index — 2048-dim Cohere V5-Pro vectors, cosine.
    3. Atlas $search text index — BM25 scoring on chunk_text.
    4. `chats` history index (workspace_id, document_id, created_at).
    5. GridFS `pdf_files` bucket indexes — the two pymongo defaults plus a
       (metadata.workspace_id, uploadDate) index for per-workspace listing.

Canonical chunk field name is `doc_id` (task #27 reconciliation). This matches
`app.db.vector.DOC_ID_FIELD` — the writer — plus `app.rag.chunker.Chunk.doc_id`,
`app.rag.rrf` and `app.rag.pipeline` SSE citations. The Postgres column keeps
its own name `document_id` (`app.db.models.AuditLog.document_id`); Postgres and
the Mongo chunk field are intentionally separate names.

M0 cap: 3 Atlas Search indexes per cluster. We stay at 2 search indexes
(vector + bm25) + regular indexes (which don't count against the search cap).

Run via:  python -m app.db.indices
"""

from __future__ import annotations

import logging
import time

from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import OperationFailure

from app.db.mongodb import get_chats_collection, get_mongo_db, get_pdfs_collection

log = logging.getLogger(__name__)

VECTOR_INDEX_NAME = "vector_index"
# Match app.rag.bm25.BM25_INDEX_NAME — a rename here breaks T-009 queries.
SEARCH_INDEX_NAME = "bm25_index"
VECTOR_DIM = 2048  # Cohere Embed V5-Pro default (configurable: 256/512/768/1024/1536/2048)
# Shared with app.services.chat_history._ensure_index — one name, two callers.
CHATS_INDEX_NAME = "chats_wks_doc_time"
# Must equal the `collection=` passed to GridFS() in app.db.mongodb.get_gridfs.
GRIDFS_BUCKET = "pdf_files"
GRIDFS_WORKSPACE_INDEX_NAME = "ix_files_workspace_upload"


def ensure_regular_indexes(coll: Collection | None = None) -> list[str]:
    """Create non-search indexes (free, don't count against M0 cap of 3)."""
    if coll is None:
        coll = get_pdfs_collection()
    created = []
    created.append(
        coll.create_index(
            [("workspace_id", 1), ("doc_id", 1), ("chunk_index", 1)],
            name="ix_workspace_doc_chunk",
            background=True,
        )
    )
    created.append(
        coll.create_index(
            [("workspace_id", 1), ("created_at", -1)],
            name="ix_workspace_created",
            background=True,
        )
    )
    created.append(
        coll.create_index(
            [("doc_id", 1)],
            name="ix_doc",
            background=True,
        )
    )
    return created


def ensure_chats_indexes(coll: Collection | None = None) -> list[str]:
    """Chronological-read index for `chats` (see app.services.chat_history)."""
    if coll is None:
        coll = get_chats_collection()
    return [
        coll.create_index(
            [("workspace_id", 1), ("document_id", 1), ("created_at", 1)],
            name=CHATS_INDEX_NAME,
            background=True,
        )
    ]


def ensure_gridfs_indexes(db: Database | None = None) -> list[str]:
    """Indexes on the `pdf_files` GridFS bucket.

    pymongo only creates its two default indexes lazily on the first `put()`
    into an EMPTY bucket, so a bucket restored from a dump or created by hand
    never gets them. The default key specs are repeated here WITHOUT custom
    names so they match what pymongo would create (same keys + different name
    is an IndexOptionsConflict on the server). The third index backs
    `app.services.gridfs_storage.list_pdfs`, which filters on
    `metadata.workspace_id` and sorts on `uploadDate` descending.
    """
    if db is None:
        db = get_mongo_db()
    files = db[f"{GRIDFS_BUCKET}.files"]
    chunks = db[f"{GRIDFS_BUCKET}.chunks"]
    created = []
    created.append(files.create_index([("filename", 1), ("uploadDate", 1)]))
    created.append(chunks.create_index([("files_id", 1), ("n", 1)], unique=True))
    created.append(
        files.create_index(
            [("metadata.workspace_id", 1), ("uploadDate", -1)],
            name=GRIDFS_WORKSPACE_INDEX_NAME,
            background=True,
        )
    )
    return created


def _search_index_exists(coll: Collection, name: str) -> bool:
    try:
        existing = list(coll.list_search_indexes())
    except OperationFailure as e:
        # listSearchIndexes requires Atlas Search tier
        log.warning("list_search_indexes failed (not on Atlas Search?): %s", e)
        return False
    return any(idx.get("name") == name for idx in existing)


def ensure_vector_index(coll: Collection | None = None, dim: int = VECTOR_DIM) -> str:
    """Create Atlas Vector Search index on `embedding` field (2048-dim, cosine).

    Idempotent: skips if an index of the same name already exists.
    """
    if coll is None:
        coll = get_pdfs_collection()
    if _search_index_exists(coll, VECTOR_INDEX_NAME):
        log.info("vector index %r already exists — skipping", VECTOR_INDEX_NAME)
        return VECTOR_INDEX_NAME

    definition = {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": dim,
                "similarity": "cosine",
            },
            {
                "type": "filter",
                "path": "workspace_id",
            },
            {
                "type": "filter",
                "path": "doc_id",
            },
        ]
    }
    coll.create_search_index(
        {
            "name": VECTOR_INDEX_NAME,
            "type": "vectorSearch",
            "definition": definition,
        }
    )
    log.info("created vector index %r (%d-dim, cosine)", VECTOR_INDEX_NAME, dim)
    return VECTOR_INDEX_NAME


def ensure_search_index(coll: Collection | None = None) -> str:
    """Create Atlas $search text index on chunk_text (BM25 scoring built-in).

    Idempotent: skips if an index of the same name already exists.
    """
    if coll is None:
        coll = get_pdfs_collection()
    if _search_index_exists(coll, SEARCH_INDEX_NAME):
        log.info("search index %r already exists — skipping", SEARCH_INDEX_NAME)
        return SEARCH_INDEX_NAME

    definition = {
        "mappings": {
            "dynamic": False,
            "fields": {
                "chunk_text": {
                    "type": "string",
                    "analyzer": "lucene.standard",
                },
                "workspace_id": {
                    "type": "token",
                },
                "doc_id": {
                    "type": "token",
                },
            },
        }
    }
    coll.create_search_index(
        {
            "name": SEARCH_INDEX_NAME,
            "type": "search",
            "definition": definition,
        }
    )
    log.info("created text search index %r", SEARCH_INDEX_NAME)
    return SEARCH_INDEX_NAME


def wait_for_index_ready(coll: Collection, name: str, timeout_s: int = 300) -> bool:
    """Poll until the named search index reports status READY (or timeout)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            indexes = list(coll.list_search_indexes(name))
        except OperationFailure:
            return False
        for idx in indexes:
            if idx.get("name") == name and idx.get("status") == "READY":
                return True
        time.sleep(5)
    return False


def ensure_all_indexes(wait: bool = False) -> dict:
    """Create every index needed for the pdfs collection. Returns a summary."""
    coll = get_pdfs_collection()
    regular = ensure_regular_indexes(coll)
    vector = ensure_vector_index(coll)
    text = ensure_search_index(coll)
    chats = ensure_chats_indexes()
    gridfs = ensure_gridfs_indexes()
    summary = {
        "regular_indexes": regular,
        "vector_index": vector,
        "search_index": text,
        "chats_indexes": chats,
        "gridfs_indexes": gridfs,
    }
    if wait:
        summary["vector_ready"] = wait_for_index_ready(coll, vector)
        summary["search_ready"] = wait_for_index_ready(coll, text)
    return summary


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        result = ensure_all_indexes(wait="--wait" in sys.argv)
        print(json.dumps(result, default=str, indent=2))
    except Exception as e:
        log.exception("index creation failed")
        print(json.dumps({"ok": False, "error": str(e)}), file=sys.stderr)
        sys.exit(1)
