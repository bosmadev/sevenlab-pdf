"""MongoDB GridFS wrapper for sanitized PDF persistence.

Server-side UUID filenames (never trust client). Workspace isolation via
metadata filter. Encryption at rest is Atlas default (AES-256).

Bucket name is `pdf_files` and is shared with `app.db.mongodb.get_gridfs()`.
Routing all GridFS access through that helper keeps put/get/delete/list
pointed at the same `pdf_files.files` + `pdf_files.chunks` collections —
mixing bucket names silently splits a document across two buckets.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from bson import ObjectId
from bson.errors import InvalidId
from gridfs.errors import NoFile

from app.db.mongodb import get_gridfs, get_mongo_db

# GridFS bucket collection prefix (see get_gridfs: collection="pdf_files")
_FILES_COLL = "pdf_files.files"


class StorageError(Exception):
    """Raised on any GridFS put/get/delete failure."""


# Public alias — exception handlers register against this name.
PdfStorageError = StorageError


class PdfNotFoundError(StorageError):
    """Missing file_id OR workspace mismatch — same error on purpose
    (do not leak existence across workspaces)."""


@dataclass
class StoredPdf:
    file_id: str
    server_filename: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class PdfRecord:
    """Metadata snapshot of a GridFS entry (listing / lookup result)."""

    file_id: str
    server_filename: str
    original_filename: str
    size_bytes: int
    sha256: str
    workspace_id: str
    uploaded_at: datetime
    pages: Optional[int] = None


def _to_oid(file_id: str) -> ObjectId:
    try:
        return ObjectId(file_id)
    except (InvalidId, TypeError) as exc:
        raise PdfNotFoundError("invalid file_id") from exc


def store_pdf(
    *,
    pdf_bytes: bytes,
    workspace_id: str,
    original_filename: str,
    sha256: str,
    extra_metadata: dict[str, Any] | None = None,
) -> StoredPdf:
    """Put sanitized PDF into GridFS. Returns file-id + server filename."""
    if not workspace_id:
        raise StorageError("workspace_id required")
    if not pdf_bytes:
        raise StorageError("empty payload")

    server_filename = f"{uuid4().hex}.pdf"
    metadata: dict[str, Any] = {
        "workspace_id": workspace_id,
        "original_filename": original_filename,
        "sha256": sha256,
        "uploaded_at": datetime.now(timezone.utc),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    try:
        file_id = get_gridfs().put(
            pdf_bytes,
            filename=server_filename,
            contentType="application/pdf",
            metadata=metadata,
        )
    except Exception as exc:  # noqa: BLE001 — pymongo raises many types
        raise StorageError(f"GridFS put failed: {exc}") from exc
    return StoredPdf(
        file_id=str(file_id),
        server_filename=server_filename,
        sha256=sha256,
        size_bytes=len(pdf_bytes),
    )


def _load_grid_out(file_id: str, workspace_id: str):
    oid = _to_oid(file_id)
    try:
        grid_out = get_gridfs().get(oid)
    except NoFile as exc:
        raise PdfNotFoundError("pdf not found") from exc
    except Exception as exc:  # noqa: BLE001
        raise StorageError(f"GridFS get failed: {exc}") from exc

    meta = grid_out.metadata or {}
    if meta.get("workspace_id") != workspace_id:
        raise PdfNotFoundError("pdf not found")
    return grid_out


def read_pdf(file_id: str, *, workspace_id: str) -> bytes:
    """Retrieve PDF bytes. Raises PdfNotFoundError on missing/mismatch."""
    grid_out = _load_grid_out(file_id, workspace_id)
    try:
        return grid_out.read()
    finally:
        grid_out.close()


def read_pdf_metadata(file_id: str, *, workspace_id: str) -> PdfRecord:
    """Metadata-only lookup — no payload bytes transferred."""
    grid_out = _load_grid_out(file_id, workspace_id)
    meta = grid_out.metadata or {}
    try:
        return PdfRecord(
            file_id=str(grid_out._id),
            server_filename=grid_out.filename,
            original_filename=meta.get("original_filename") or grid_out.filename,
            size_bytes=grid_out.length,
            sha256=meta.get("sha256", ""),
            workspace_id=meta.get("workspace_id", workspace_id),
            uploaded_at=meta.get("uploaded_at") or grid_out.upload_date,
            pages=meta.get("pages"),
        )
    finally:
        grid_out.close()


def delete_pdf(file_id: str, *, workspace_id: str) -> None:
    """Delete a stored PDF. Raises PdfNotFoundError on missing/mismatch."""
    oid = _to_oid(file_id)
    grid_out = _load_grid_out(file_id, workspace_id)
    grid_out.close()
    try:
        get_gridfs().delete(oid)
    except Exception as exc:  # noqa: BLE001
        raise StorageError(f"GridFS delete failed: {exc}") from exc


def list_pdfs(*, workspace_id: str, limit: int = 100) -> list[PdfRecord]:
    """List workspace PDFs newest-first (capped at `limit`, max 500)."""
    if not workspace_id:
        raise StorageError("workspace_id required")
    limit = max(1, min(int(limit), 500))

    coll = get_mongo_db()[_FILES_COLL]
    cursor = (
        coll.find({"metadata.workspace_id": workspace_id})
        .sort("uploadDate", -1)
        .limit(limit)
    )
    out: list[PdfRecord] = []
    for row in cursor:
        meta = row.get("metadata") or {}
        out.append(
            PdfRecord(
                file_id=str(row["_id"]),
                server_filename=row.get("filename", ""),
                original_filename=meta.get("original_filename")
                or row.get("filename", ""),
                size_bytes=int(row.get("length", 0)),
                sha256=meta.get("sha256", ""),
                workspace_id=meta.get("workspace_id", workspace_id),
                uploaded_at=meta.get("uploaded_at") or row.get("uploadDate"),
                pages=meta.get("pages"),
            )
        )
    return out
