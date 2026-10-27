"""MongoDB Atlas connection + GridFS + `pdfs` collection.

Env vars:
    MONGODB_URI — Atlas connection string (mongodb+srv://...)
    MONGODB_DB_NAME — default "sevenlab_pdf"

Design (SEVENLAB-PDFCHAT.md § MongoDB Atlas Integration Guide):
    - SINGLE `pdfs` collection with workspace_id filter (M0: 3 index limit —
      per-user collections would exceed this).
    - Vector embeddings: 2048-dim (Cohere Embed V5-Pro default), cosine.
    - GridFS stores original sanitized PDF bytes (AES-256 at rest on Atlas).
    - Pattern reference: ~/source/scraper/main/lib/mongodb.ts.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

from gridfs import GridFS
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.server_api import ServerApi

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_DEFAULT_DB = "sevenlab_pdf"
_PDFS = "pdfs"
_CHATS = "chats"

_lock = threading.Lock()
_client: MongoClient | None = None


def _build_client() -> MongoClient:
    uri = os.environ.get("MONGODB_URI")
    if not uri or not uri.startswith(("mongodb://", "mongodb+srv://")):
        raise RuntimeError(
            "MONGODB_URI is missing or invalid — expected mongodb:// or mongodb+srv://"
        )
    return MongoClient(
        uri,
        maxPoolSize=50,
        minPoolSize=5,
        maxIdleTimeMS=30_000,
        serverSelectionTimeoutMS=5_000,
        connectTimeoutMS=5_000,
        retryWrites=True,
        server_api=ServerApi("1"),
        appname="sevenlab-pdfchat",
    )


def get_mongo_client() -> MongoClient:
    """Return a process-wide MongoClient (connection pool is managed internally)."""
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = _build_client()
    return _client


def get_mongo_db() -> Database:
    name = os.environ.get("MONGODB_DB_NAME", _DEFAULT_DB)
    return get_mongo_client()[name]


def get_pdfs_collection() -> Collection:
    """The single collection holding chunks + embeddings.

    Document shape (per chunk) — field names must match app.db.vector and
    app.rag.chunker (DOC_ID_FIELD = "doc_id"); renaming here is a silent break:
        {
          "_id": ObjectId,
          "workspace_id": str,          # for isolation + $search filter
          "doc_id": str,                # Postgres documents.id (UUID str)
          "chunk_index": int,
          "chunk_text": str,
          "page_num": int | None,
          "embedding": [float] * 2048,  # Cohere V5-Pro
          "metadata": { ... },
          "created_at": datetime,
        }
    """
    return get_mongo_db()[_PDFS]


def get_chats_collection() -> Collection:
    """Chat message history (per document)."""
    return get_mongo_db()[_CHATS]


def get_gridfs() -> GridFS:
    """GridFS bucket for original PDF bytes."""
    return GridFS(get_mongo_db(), collection="pdf_files")


def ping() -> dict:
    """Health check — returns {ok: True} if Atlas is reachable."""
    try:
        res = get_mongo_client().admin.command("ping")
        return {"ok": True, "server": res.get("ok") == 1.0}
    except Exception as e:
        return {"ok": False, "error": str(e)}


class MongoDB:
    """Namespace shim for callers that prefer a class-style accessor."""

    client = staticmethod(get_mongo_client)
    db = staticmethod(get_mongo_db)
    pdfs = staticmethod(get_pdfs_collection)
    chats = staticmethod(get_chats_collection)
    gridfs = staticmethod(get_gridfs)
    ping = staticmethod(ping)
