"""MongoDB Atlas client + database handle.

Env vars:
    MONGODB_URI — Atlas SRV connection string
    MONGODB_DB_NAME — default sevenlab_pdf

The client is lazily built, pooled (max 50, min 5) and shared process-wide.
Collections used by this module:
    pdfs        — one document per chunk: {workspace_id, doc_id, chunk_text,
                  page_num, embedding (2048-dim), created_at, ...}

Atlas search indexes (created separately via create_indexes.py, not here):
    vector_index   — $vectorSearch on `embedding`
    bm25_index     — $search text index on `chunk_text` with workspace_id filter
"""

from __future__ import annotations

import os
from functools import lru_cache

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


_PDFS_COLLECTION = "pdfs"


@lru_cache(maxsize=1)
def mongo_client() -> MongoClient:
    uri = os.environ.get("MONGODB_URI", "").strip()
    if not uri:
        raise RuntimeError("MONGODB_URI is missing — set it in backend/.env")
    return MongoClient(
        uri,
        maxPoolSize=50,
        minPoolSize=5,
        serverSelectionTimeoutMS=5000,
        retryWrites=True,
    )


def mongo_db() -> Database:
    name = os.environ.get("MONGODB_DB_NAME", "sevenlab_pdf").strip() or "sevenlab_pdf"
    return mongo_client()[name]


def pdfs_collection() -> Collection:
    return mongo_db()[_PDFS_COLLECTION]
