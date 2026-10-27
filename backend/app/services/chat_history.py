"""Chat message history store (per document), backed by MongoDB.

Collection ``chats`` (per ``app.db.mongodb.get_chats_collection``), shape::

    {
      "_id": ObjectId,
      "workspace_id": str,
      "document_id": str,      # Postgres documents.id (UUID str)
      "role": "user" | "assistant" | "system",
      "content": str,
      "citations": [{"page_num": int, "chunk_id": str, "chunk_index": int}],
      "metadata": { ... },
      "created_at": datetime,
    }

An ``(workspace_id, document_id, created_at)`` index is created on first
use so chronological reads stay cheap even once chats grow large.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from app.db.indices import CHATS_INDEX_NAME
from app.db.mongodb import get_chats_collection

log = logging.getLogger(__name__)

_INDEXED = False
_INDEX_LOCK = threading.Lock()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_index() -> None:
    """Create the (workspace, document, created_at) index once per process."""
    global _INDEXED
    if _INDEXED:
        return
    with _INDEX_LOCK:
        if _INDEXED:
            return
        try:
            get_chats_collection().create_index(
                [
                    ("workspace_id", ASCENDING),
                    ("document_id", ASCENDING),
                    ("created_at", ASCENDING),
                ],
                name=CHATS_INDEX_NAME,
                background=True,
            )
        except Exception as exc:  # pragma: no cover — logged, not fatal
            log.warning("chats index create failed: %s", exc)
        _INDEXED = True


def _normalize(doc: dict[str, Any]) -> dict[str, Any]:
    """Shape a Mongo row for the Pydantic ``ChatMessage`` response."""
    citations = doc.get("citations") or []
    if not isinstance(citations, list):
        citations = []
    return {
        "id": str(doc["_id"]),
        "document_id": str(doc["document_id"]),
        "role": doc.get("role", "user"),
        "content": doc.get("content", ""),
        "citations": citations,
        "created_at": doc.get("created_at") or _utcnow(),
    }


def list_messages(
    *,
    workspace_id: str,
    document_id: str,
    limit: int = 200,
    offset: int = 0,
    is_admin: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Return chronological chat history for one document."""
    _ensure_index()
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))

    col = get_chats_collection()
    query: dict[str, Any] = {"document_id": str(document_id)}
    if not is_admin:
        query["workspace_id"] = workspace_id

    total = col.count_documents(query)
    cursor = (
        col.find(query)
        .sort("created_at", ASCENDING)
        .skip(offset)
        .limit(limit)
    )
    return [_normalize(d) for d in cursor], int(total)


def append_message(
    *,
    workspace_id: str,
    document_id: str,
    role: str,
    content: str,
    citations: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Insert one chat message and return it in response shape."""
    _ensure_index()
    now = _utcnow()
    row: dict[str, Any] = {
        "workspace_id": workspace_id,
        "document_id": str(document_id),
        "role": role,
        "content": content,
        "citations": citations or [],
        "metadata": metadata or {},
        "created_at": now,
    }
    res = get_chats_collection().insert_one(row)
    row["_id"] = res.inserted_id
    return _normalize(row)


def delete_history(*, workspace_id: str, document_id: str) -> int:
    """Remove every chat message for a document; returns the delete count."""
    res = get_chats_collection().delete_many(
        {"workspace_id": workspace_id, "document_id": str(document_id)}
    )
    return int(res.deleted_count)


def latest_message(
    *,
    workspace_id: str,
    document_id: str,
    is_admin: bool = False,
) -> dict[str, Any] | None:
    """Peek the most recent message (used by document summary rows)."""
    _ensure_index()
    query: dict[str, Any] = {"document_id": str(document_id)}
    if not is_admin:
        query["workspace_id"] = workspace_id
    doc = get_chats_collection().find_one(query, sort=[("created_at", DESCENDING)])
    return _normalize(doc) if doc else None


__all__ = [
    "append_message",
    "delete_history",
    "latest_message",
    "list_messages",
]


# Re-export for callers that only need the ObjectId import plumbing.
_ = ObjectId  # keep bson import active for tests that patch it
