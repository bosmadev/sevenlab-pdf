"""Document + chat-history response schemas (T-013)."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    """Row shape for ``GET /api/documents/``."""

    id: uuid.UUID
    original_filename: str
    size_bytes: int
    page_count: int | None = None
    status: str
    chunk_count: int
    created_at: datetime
    processed_at: datetime | None = None


class DocumentDetail(DocumentSummary):
    """Full metadata for ``GET /api/documents/{id}``."""

    sanitized_filename: str
    sha256: str
    mime_type: str
    status_message: str | None = None
    ocr_tier_used: str | None = None


class DocumentDeleteResponse(BaseModel):
    """Soft-delete acknowledgement."""

    id: uuid.UUID
    deleted_at: datetime
    status: Literal["soft_deleted"] = "soft_deleted"


class ChatCitation(BaseModel):
    """Stored citation. ``page_num``/``chunk_id``/``chunk_index`` are the
    original T-013 fields; the rest mirror the RAG pipeline's live SSE
    citation so a reloaded chat renders the same source list."""

    n: int | None = None
    doc_id: str | None = None
    page_num: int | None = None
    chunk_id: str | None = None
    chunk_index: int | None = None
    filename: str | None = None
    score: float | None = None
    preview: str | None = None


class ChatMessage(BaseModel):
    """One stored chat message."""

    id: str = Field(..., description="Mongo _id as hex string")
    document_id: uuid.UUID
    role: Literal["user", "assistant", "system"]
    content: str
    citations: list[ChatCitation] = Field(default_factory=list)
    created_at: datetime


class ChatMessageCreate(BaseModel):
    """Append a chat message to a document's history."""

    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, max_length=32_768)
    citations: list[ChatCitation] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class ChatHistoryResponse(BaseModel):
    document_id: uuid.UUID
    messages: list[ChatMessage]
    total: int


# ---------------------------------------------------------------------------
# T-004 upload schemas (owner: pdf-impl-1)
# ---------------------------------------------------------------------------


class UploadRejected(BaseModel):
    code: str
    reason: str


class UploadAccepted(BaseModel):
    document_id: str = Field(..., description="GridFS file id")
    server_filename: str
    sha256: str
    size_bytes: int
    pages: int
    stripped_keys: list[str]
    status: str = "accepted"
