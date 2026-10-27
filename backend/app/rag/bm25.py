"""BM25 keyword search over PDF chunks using MongoDB Atlas `$search`.

Why Atlas native `$search` (T-009 design note):
    An earlier draft used `rank_bm25` + pickle to persist the index, which
    is a known RCE vector (pickle deserialises arbitrary code). Atlas
    `$search` computes BM25 server-side on a managed Lucene index, so no
    model file is ever shipped or loaded. Scoring, tokenisation, and
    stopword filtering live in the Atlas index definition.

Expected search index — defined once in `app.db.indices.ensure_search_index`
(create via `python -m scripts.create_search_indexes`, name=bm25_index):
    {
      "mappings": {
        "dynamic": false,
        "fields": {
          "chunk_text":   { "type": "string", "analyzer": "lucene.standard" },
          "workspace_id": { "type": "token" },
          "doc_id":       { "type": "token" }
        }
      }
    }

Usage:
    hits = bm25_search(workspace_id, "quarterly revenue", top_k=20)
    for h in hits:
        print(h.chunk_id, h.score, h.chunk_text[:80])

Chunk schema (see app.db.vector.DOC_ID_FIELD and app.rag.chunker.Chunk):
    doc_id, workspace_id, chunk_text, page_num, embedding (2048), ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.db.mongodb import get_pdfs_collection


BM25_INDEX_NAME = "bm25_index"
DEFAULT_TOP_K = 20
MAX_TOP_K = 100


@dataclass(frozen=True)
class BM25Hit:
    """One ranked chunk returned by Atlas `$search` BM25 scoring."""

    chunk_id: str
    doc_id: str
    workspace_id: str
    chunk_text: str
    page_num: int | None
    score: float

    def to_pipeline_chunk(self) -> dict[str, Any]:
        """Return the chunk shape expected by `app.rag.pipeline._format_context`.

        T-012's SSE pipeline keys on `doc_id` and `chunk_id` — the same
        field names the storage layer (`app.db.vector`) uses, so this
        method is a pure projection rather than a rename.
        """
        return {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "chunk_text": self.chunk_text,
            "page_num": self.page_num,
            "score": self.score,
            "source": "bm25",
        }


def _coerce_top_k(top_k: int) -> int:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return min(top_k, MAX_TOP_K)


def _coerce_query(query: str) -> str:
    q = (query or "").strip()
    if not q:
        raise ValueError("query must be a non-empty string")
    return q


def build_bm25_pipeline(
    workspace_id: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    index_name: str = BM25_INDEX_NAME,
    *,
    doc_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return the aggregation pipeline for a workspace-scoped BM25 search.

    The `compound` stage pairs a text match on `chunk_text` with a
    `filter` on `workspace_id` (and on `doc_id` when the caller scopes
    the search to one PDF). A filter clause contributes zero score, so
    BM25 ranking is driven purely by the text match while isolation stays
    strict. Filtering server-side matters for per-document chat: a
    workspace-wide top-20 followed by a client-side doc filter can return
    zero hits for a document that has plenty of keyword matches.
    """

    if not workspace_id:
        raise ValueError("workspace_id must be a non-empty string")
    q = _coerce_query(query)
    k = _coerce_top_k(top_k)

    filters: list[dict[str, Any]] = [
        {"equals": {"path": "workspace_id", "value": workspace_id}}
    ]
    if doc_id:
        filters.append({"equals": {"path": "doc_id", "value": doc_id}})

    return [
        {
            "$search": {
                "index": index_name,
                "compound": {
                    "must": [
                        {
                            "text": {
                                "query": q,
                                "path": "chunk_text",
                            }
                        }
                    ],
                    "filter": filters,
                },
                "scoreDetails": False,
            }
        },
        {"$limit": k},
        {
            "$project": {
                "_id": 1,
                "doc_id": 1,
                "workspace_id": 1,
                "chunk_text": 1,
                "page_num": 1,
                "score": {"$meta": "searchScore"},
            }
        },
    ]


def bm25_search(
    workspace_id: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    index_name: str = BM25_INDEX_NAME,
    doc_id: str | None = None,
) -> list[BM25Hit]:
    """Run BM25 search on the `pdfs` collection and return ranked chunks.

    Returned list is ordered by descending BM25 score, length ≤ top_k.
    ``doc_id`` narrows the search to one PDF inside the workspace.
    Raises `pymongo.errors.OperationFailure` if the `$search` stage is
    rejected — typically because `bm25_index` does not exist yet.
    """

    pipeline = build_bm25_pipeline(
        workspace_id, query, top_k, index_name, doc_id=doc_id
    )
    coll = get_pdfs_collection()
    cursor = coll.aggregate(pipeline)
    hits: list[BM25Hit] = []
    for doc in cursor:
        hits.append(
            BM25Hit(
                chunk_id=str(doc["_id"]),
                doc_id=str(doc.get("doc_id", "")),
                workspace_id=str(doc.get("workspace_id", "")),
                chunk_text=str(doc.get("chunk_text", "")),
                page_num=doc.get("page_num"),
                score=float(doc.get("score", 0.0)),
            )
        )
    return hits
