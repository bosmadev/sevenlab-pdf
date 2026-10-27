"""Document endpoints — upload (T-004), listing + delete (T-013).

``POST /api/documents/upload`` is the T-004 wiring: it chains the already-
landed building blocks owned by the other impl agents:

    validate_upload (pdf-impl-1)
      → sandbox_process (docker --network=none worker: sanitize + extract, A.8.7)
        → store_pdf (support-impl-3, GridFS)
          → Document row (SQLAlchemy, this file)
            → audit_upload_* (support-impl-10)
              → BackgroundTasks.add_task(_schedule_processing)  (hands off to T-014)

Auth is enforced by the ``require_owner`` dependency — anonymous callers
get HTTP 401 before the file bytes are even read.

Listing / retrieval / delete are 501 stubs; doc-mgmt-impl-1 owns T-013 and
will fill them in without touching the upload path.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from app.db.models import Document, User
from app.db.postgres import get_db
from app.security import Owner, require_owner
from app.services.audit_logger import audit_upload_accepted, audit_upload_rejected
from app.services.gridfs_storage import StorageError, store_pdf
from app.services.pdf_sandbox import SandboxError, sandbox_process
from app.services.pdf_validator import (
    MAX_PDF_SIZE_BYTES,
    MAX_PDF_SIZE_MB,
    PDF_MAGIC_SCAN_WINDOW,
    PdfValidationError,
    validate_upload,
)

log = logging.getLogger(__name__)
router = APIRouter()

# One 64 KiB read per iteration — bounded buffer keeps memory predictable
# even when a client lies about Content-Length.
_READ_CHUNK = 64 * 1024


def _client_ip(request: Request) -> str:
    """Trust only the left-most XFF entry (Caddy is the single proxy hop)."""
    xff = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if xff:
        return xff
    return request.client.host if request.client else "unknown"


def _get_or_create_user(db: Session, owner_id: str) -> User:
    """Resolve the SQL User row for a session owner.

    Auth-impl-1 (T-002) owns provisioning. Until that lands we create a
    placeholder row keyed on the owner_id so uploads don't 500 on first use.
    """
    user = db.query(User).filter(User.workspace_id == owner_id).one_or_none()
    if user is not None:
        return user
    user = User(workspace_id=owner_id, password_hash="")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


async def _schedule_processing(doc_id: str) -> None:
    """Hand the doc id to the T-014 processor if it's already wired.

    ``async`` on purpose: Starlette runs a *sync* background task in its
    threadpool where no event loop is running, which sent ``enqueue`` down
    its no-loop path and the pipeline never started after an upload.
    """
    try:
        from app.processing import enqueue  # imported lazy — bg-proc-impl-1 owns it
    except Exception as exc:  # noqa: BLE001 — module may not be ready yet
        log.info("processing pipeline not ready for %s: %s", doc_id, exc)
        return
    try:
        enqueue(doc_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("enqueue(%s) failed: %s", doc_id, exc)


async def _read_bounded(
    file: UploadFile,
    *,
    workspace_id: str,
    client_ip: str,
    raw_filename: str,
) -> bytes:
    """Stream the upload into memory, rejecting as soon as the cap is passed."""
    buf = bytearray()
    while True:
        chunk = await file.read(_READ_CHUNK)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > MAX_PDF_SIZE_BYTES:
            reason = f"file exceeds {MAX_PDF_SIZE_MB} MB limit"
            audit_upload_rejected(
                workspace_id=workspace_id,
                client_ip=client_ip,
                reason=reason,
                code="SIZE_LIMIT",
                raw_filename=raw_filename,
                size_bytes=len(buf),
            )
            # Starlette renamed the status constant; use the numeric literal
            # so we work on both pre- and post-rename versions.
            raise HTTPException(
                status_code=413,
                detail={"code": "SIZE_LIMIT", "reason": reason},
            )
    return bytes(buf)


def _reject(
    *,
    code: str,
    reason: str,
    http_status: int,
    workspace_id: str,
    client_ip: str,
    raw_filename: str,
    size_bytes: int | None,
) -> HTTPException:
    audit_upload_rejected(
        workspace_id=workspace_id,
        client_ip=client_ip,
        reason=reason,
        code=code,
        raw_filename=raw_filename,
        size_bytes=size_bytes,
    )
    return HTTPException(
        status_code=http_status,
        detail={"code": code, "reason": reason},
    )


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload(
    request: Request,
    background_tasks: BackgroundTasks,
    owner: Annotated[Owner, Depends(require_owner)],
    db: Annotated[Session, Depends(get_db)],
    file: Annotated[UploadFile, File(...)],
) -> dict[str, Any]:
    """Accept one PDF, validate + sanitize + persist, enqueue background pipeline."""
    client_ip = _client_ip(request)
    workspace_id = owner.owner_id
    raw_filename = file.filename or ""

    pdf_bytes = await _read_bounded(
        file,
        workspace_id=workspace_id,
        client_ip=client_ip,
        raw_filename=raw_filename,
    )

    try:
        validated = validate_upload(
            raw_filename=raw_filename,
            declared_mime=file.content_type,
            size_bytes=len(pdf_bytes),
            head_bytes=pdf_bytes[:PDF_MAGIC_SCAN_WINDOW],
        )
    except PdfValidationError as exc:
        raise _reject(
            code=exc.code,
            reason=exc.reason,
            http_status=status.HTTP_400_BAD_REQUEST,
            workspace_id=workspace_id,
            client_ip=client_ip,
            raw_filename=raw_filename,
            size_bytes=len(pdf_bytes),
        ) from exc

    # Attacker bytes never meet parser code in this process: sandbox_process
    # runs sanitizer + extractor in the docker worker (PDF_SANDBOX_MODE=docker)
    # and only falls back to in-process for dev/CI (PDF_SANDBOX_MODE=inline).
    # It blocks on the subprocess, so keep it off the event loop.
    try:
        report = await asyncio.to_thread(sandbox_process, pdf_bytes)
    except SandboxError as exc:
        raise _reject(
            code="SANITIZE_FAILED",
            reason=str(exc),
            http_status=status.HTTP_400_BAD_REQUEST,
            workspace_id=workspace_id,
            client_ip=client_ip,
            raw_filename=raw_filename,
            size_bytes=len(pdf_bytes),
        ) from exc
    sanitized_bytes = report.sanitized_pdf
    log.info(
        "upload sanitized via pdf_sandbox mode=%s pages=%d stripped=%d",
        report.mode,
        report.pages,
        len(report.stripped_keys),
    )

    user = _get_or_create_user(db, workspace_id)

    try:
        stored = store_pdf(
            pdf_bytes=sanitized_bytes,
            workspace_id=workspace_id,
            original_filename=validated.safe_filename,
            sha256=report.sanitized_sha256,
            extra_metadata={
                "original_sha256": report.original_sha256,
                "stripped_keys": report.stripped_keys,
                "pages": report.pages,
            },
        )
    except StorageError as exc:
        raise _reject(
            code="STORAGE_ERROR",
            reason="storage unavailable",
            http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
            workspace_id=workspace_id,
            client_ip=client_ip,
            raw_filename=raw_filename,
            size_bytes=len(sanitized_bytes),
        ) from exc

    doc = Document(
        user_id=user.id,
        workspace_id=workspace_id,
        original_filename=(raw_filename or validated.safe_filename)[:512],
        sanitized_filename=validated.safe_filename,
        gridfs_file_id=stored.file_id,
        sha256=report.sanitized_sha256,
        size_bytes=stored.size_bytes,
        mime_type=validated.mime,
        page_count=report.pages,
        status="pending",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    audit_upload_accepted(
        workspace_id=workspace_id,
        client_ip=client_ip,
        sha256=report.sanitized_sha256,
        size_bytes=stored.size_bytes,
        safe_filename=validated.safe_filename,
        pages=report.pages,
        stripped_keys=report.stripped_keys,
        file_id=stored.file_id,
    )

    background_tasks.add_task(_schedule_processing, str(doc.id))

    return {
        "id": str(doc.id),
        "status": doc.status,
        "sanitized_filename": doc.sanitized_filename,
        "sha256": doc.sha256,
        "size_bytes": doc.size_bytes,
        "page_count": doc.page_count,
        "stripped_keys": report.stripped_keys,
        "mime_type": doc.mime_type,
    }


# T-013 list / get / delete routes live in ``app.routes.document_mgmt`` — the
# earlier 501 stubs here collided with that real router (duplicate operation
# ids on the OpenAPI schema) so they are intentionally gone.
