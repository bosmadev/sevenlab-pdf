"""MongoDB Atlas Vector Search — T-008 data-plane for the `pdfs` collection.

Index creation lives in `app.db.indices` (shared with T-003 and T-009). This
module owns only the DATA plane: upsert, query, delete, count.

Chunk document shape (plan §Vector Storage):
    {
      "_id":          "<doc_id>:<chunk_index>",
      "workspace_id": str,       # per-user isolation (M0 single-collection pattern)
      "doc_id":       str,       # parent document (GridFS file id as hex string)
      "chunk_index":  int,       # 0-based within the document
      "chunk_text":   str,       # original chunk text (BM25 target too, T-009)
      "page_num":     int|None,  # 1-based page number from parser
      "embedding":    list[float],  # 2048 floats, Cohere Embed V5-Pro
      "created_at":   datetime,  # UTC
      "updated_at":   datetime,  # UTC
    }
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from pymongo import UpdateOne
from pymongo.collection import Collection

from app.config import get_settings

logger = logging.getLogger(__name__)

# Index + field names kept in lock-step with app/db/indices.py.
VECTOR_INDEX_NAME = "vector_index"
VECTOR_FIELD = "embedding"
WORKSPACE_FIELD = "workspace_id"
DOC_ID_FIELD = "doc_id"
CHUNK_INDEX_FIELD = "chunk_index"


def _collection() -> Collection[Mapping[str, Any]]:
    """Resolve the shared `pdfs` collection via `app.db.mongodb`.

    Falls back to building a direct pymongo client from settings if the sibling
    module is somehow unavailable — covers test environments and keeps vector
    operations working while db-impl modules settle.
    """
    try:
        from app.db.mongodb import get_pdfs_collection  # type: ignore

        return get_pdfs_collection()
    except ImportError:
        from pymongo import MongoClient

        settings = get_settings()
        if not settings.mongodb_uri:
            raise RuntimeError(
                "MONGODB_URI is missing — set it in backend/.env"
            ) from None
        client: MongoClient[Mapping[str, Any]] = MongoClient(
            settings.mongodb_uri,
            maxPoolSize=50,
            minPoolSize=5,
            serverSelectionTimeoutMS=5000,
        )
        return client[settings.mongodb_db_name]["pdfs"]


def _chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}:{chunk_index}"


def upsert_chunks(
    workspace_id: str,
    doc_id: str,
    chunks: Iterable[Mapping[str, Any]],
) -> int:
    """Upsert chunk documents for a single parent PDF.

    Each `chunks` item must carry at least:
        chunk_index: int
        chunk_text:  str
        embedding:   list[float]   # length == settings.embedding_dimensions
    Optional: page_num (int); any extra keys pass through to the stored doc.

    Returns the number of upserted + modified rows (bulk result sum).
    """
    if not workspace_id:
        raise ValueError("workspace_id is required for per-tenant isolation")
    if not doc_id:
        raise ValueError("doc_id is required")

    settings = get_settings()
    expected_dim = settings.embedding_dimensions
    now = datetime.now(timezone.utc)

    operations: list[UpdateOne] = []
    for chunk in chunks:
        chunk_index = chunk.get("chunk_index")
        embedding = chunk.get("embedding")
        chunk_text = chunk.get("chunk_text")
        if chunk_index is None or embedding is None or chunk_text is None:
            raise ValueError(
                "chunk requires chunk_index, chunk_text, embedding"
            )
        if len(embedding) != expected_dim:
            raise ValueError(
                f"embedding has {len(embedding)} dims, expected {expected_dim}"
            )

        doc: dict[str, Any] = {
            WORKSPACE_FIELD: workspace_id,
            DOC_ID_FIELD: doc_id,
            CHUNK_INDEX_FIELD: int(chunk_index),
            "chunk_text": chunk_text,
            VECTOR_FIELD: list(embedding),
            "page_num": chunk.get("page_num"),
            "updated_at": now,
        }
        for key, value in chunk.items():
            if key not in doc and key not in {"chunk_index", "embedding", "chunk_text"}:
                doc[key] = value

        operations.append(
            UpdateOne(
                {"_id": _chunk_id(doc_id, int(chunk_index))},
                {
                    "$set": doc,
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
        )

    if not operations:
        return 0

    result = _collection().bulk_write(operations, ordered=False)
    upserted = len(result.upserted_ids or {})
    modified = result.modified_count
    logger.info(
        "upsert_chunks workspace=%s doc=%s inserted=%d modified=%d",
        workspace_id,
        doc_id,
        upserted,
        modified,
    )
    return upserted + modified


def delete_document_chunks(workspace_id: str, doc_id: str) -> int:
    """Remove all chunks for a document (used by T-013 soft delete / cleanup)."""
    result = _collection().delete_many(
        {WORKSPACE_FIELD: workspace_id, DOC_ID_FIELD: doc_id}
    )
    return result.deleted_count


def vector_search(
    workspace_id: str,
    query_embedding: list[float],
    top_k: int = 20,
    doc_id: str | None = None,
    num_candidates: int | None = None,
) -> list[dict[str, Any]]:
    """Run `$vectorSearch` scoped to a workspace.

    `num_candidates` defaults to `max(top_k * 10, 150)` per Atlas guidance:
    recall climbs sharply up to 10× top_k and flattens after. Caller should
    override for recall-sensitive queries.

    Returns a list of chunk documents with an added `vector_score` field
    (`$meta: "vectorSearchScore"`, cosine similarity ∈ [0, 1]).
    """
    settings = get_settings()
    if len(query_embedding) != settings.embedding_dimensions:
        raise ValueError(
            f"query_embedding has {len(query_embedding)} dims, "
            f"expected {settings.embedding_dimensions}"
        )

    filter_clause: dict[str, Any] = {WORKSPACE_FIELD: {"$eq": workspace_id}}
    if doc_id is not None:
        filter_clause[DOC_ID_FIELD] = {"$eq": doc_id}

    pipeline: list[dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": VECTOR_FIELD,
                "queryVector": query_embedding,
                "numCandidates": num_candidates or max(top_k * 10, 150),
                "limit": top_k,
                "filter": filter_clause,
            }
        },
        {
            "$project": {
                "_id": 1,
                WORKSPACE_FIELD: 1,
                DOC_ID_FIELD: 1,
                CHUNK_INDEX_FIELD: 1,
                "chunk_text": 1,
                "page_num": 1,
                "created_at": 1,
                "vector_score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(_collection().aggregate(pipeline))


def count_chunks(workspace_id: str, doc_id: str | None = None) -> int:
    """Return chunk count for a workspace, optionally scoped to one document."""
    query: dict[str, Any] = {WORKSPACE_FIELD: workspace_id}
    if doc_id is not None:
        query[DOC_ID_FIELD] = doc_id
    return _collection().count_documents(query)


__all__ = [
    "VECTOR_INDEX_NAME",
    "VECTOR_FIELD",
    "WORKSPACE_FIELD",
    "DOC_ID_FIELD",
    "CHUNK_INDEX_FIELD",
    "count_chunks",
    "delete_document_chunks",
    "upsert_chunks",
    "vector_search",
]
