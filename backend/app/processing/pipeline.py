"""Async document ingestion pipeline (T-014 — Phase 1, zero infrastructure).

Phase 1 keeps the queue in-process via ``asyncio.create_task``. The pipeline
walks every document from ``pending`` through ``parsing → chunking →
embedding`` and finishes at ``ready`` (or ``error``). All transitions are
persisted on ``documents.status`` so the frontend can poll
``GET /api/documents/{id}/status`` without any shared memory.

Phase 2 (T-101) will swap ``_schedule_task`` for Celery enqueueing without
touching the stage functions.

Stage dependencies are late-bound through ``importlib`` so the pipeline file
itself has no import-time dependency on T-004 / T-007 / T-008 landing: if a
sibling module is missing, the stage flips the document to ``error`` with a
readable message instead of exploding the worker.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import AuditLog, Document
from app.db.postgres import SessionLocal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status state machine
# ---------------------------------------------------------------------------

STATUS_PENDING = "pending"
STATUS_PARSING = "parsing"
STATUS_CHUNKING = "chunking"
STATUS_EMBEDDING = "embedding"
STATUS_READY = "ready"
STATUS_ERROR = "error"

STATUSES: tuple[str, ...] = (
    STATUS_PENDING,
    STATUS_PARSING,
    STATUS_CHUNKING,
    STATUS_EMBEDDING,
    STATUS_READY,
    STATUS_ERROR,
)

# Rough percentage milestones surfaced to the frontend. Keep in sync with
# STATUSES order so the UI can render a progress bar off the string alone.
_PROGRESS_PCT: dict[str, int] = {
    STATUS_PENDING: 0,
    STATUS_PARSING: 20,
    STATUS_CHUNKING: 50,
    STATUS_EMBEDDING: 75,
    STATUS_READY: 100,
    STATUS_ERROR: 0,
}


class PipelineError(RuntimeError):
    """Any failure the pipeline will persist to documents.status_message."""


class PipelineStageError(PipelineError):
    """A single stage failed. ``stage`` identifies which one."""

    def __init__(self, stage: str, message: str):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message


# ---------------------------------------------------------------------------
# Retry policy — exponential backoff, cap at 3 attempts per stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    initial_delay_s: float = 1.0
    backoff_multiplier: float = 2.0

    def delay_for(self, attempt_index: int) -> float:
        """attempt_index is 0-based (0 == before first retry)."""
        return self.initial_delay_s * (self.backoff_multiplier ** attempt_index)


DEFAULT_POLICY = RetryPolicy()

# A short, deterministic set of retryable error substrings. Everything else is
# treated as fatal — a KeyError from a bug should not loop three times.
_RETRYABLE_HINTS: tuple[str, ...] = (
    "429",
    "rate limit",
    "timeout",
    "temporarily",
    "connection reset",
    "connection aborted",
    "server disconnected",
    "service unavailable",
    "503",
    "504",
)


def _is_retryable(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(hint in text for hint in _RETRYABLE_HINTS)


# ---------------------------------------------------------------------------
# Status persistence helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _set_status(
    db: Session,
    doc_id: uuid.UUID,
    *,
    status: str,
    message: str | None = None,
    chunk_count: int | None = None,
    page_count: int | None = None,
    ocr_tier_used: str | None = None,
    mark_processed: bool = False,
) -> None:
    """Persist a single status transition. Caller owns the session lifecycle."""
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}; expected one of {STATUSES}")

    doc = db.get(Document, doc_id)
    if doc is None:
        logger.warning("set_status: document %s not found", doc_id)
        return

    doc.status = status
    doc.status_message = message
    if chunk_count is not None:
        doc.chunk_count = chunk_count
    if page_count is not None:
        doc.page_count = page_count
    if ocr_tier_used is not None:
        doc.ocr_tier_used = ocr_tier_used
    if mark_processed:
        doc.processed_at = _utcnow()
    db.add(doc)
    db.commit()


def _audit(
    db: Session,
    *,
    workspace_id: str | None,
    doc_id: uuid.UUID,
    event_type: str,
    details: str | None = None,
) -> None:
    row = AuditLog(
        workspace_id=workspace_id,
        document_id=doc_id,
        event_type=event_type,
        details=details,
    )
    db.add(row)
    db.commit()


def _load_doc(db: Session, doc_id: uuid.UUID) -> Document | None:
    stmt = select(Document).where(Document.id == doc_id)
    return db.execute(stmt).scalar_one_or_none()


# ---------------------------------------------------------------------------
# Late-bound stage adapters — tolerate T-004/T-007/T-008 landing after T-014
# ---------------------------------------------------------------------------


def _import_attr(dotted_path: str, attr: str) -> Any:
    """``importlib.import_module(dotted_path).<attr>`` or raise PipelineStageError."""
    try:
        module = importlib.import_module(dotted_path)
    except ImportError as exc:  # noqa: BLE001
        raise PipelineStageError(
            dotted_path, f"module not installed yet: {exc}"
        ) from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:  # noqa: BLE001
        raise PipelineStageError(
            dotted_path, f"missing attribute {attr!r}: {exc}"
        ) from exc


def _load_pdf_bytes(doc: Document) -> bytes:
    """Pull the sanitized PDF out of GridFS. Stage: parsing.

    Routed through ``gridfs_storage.read_pdf`` so the hex ``gridfs_file_id``
    is coerced to an ObjectId (a bare ``GridFS.get(str)`` never matches) and
    the workspace filter is enforced on the read.
    """
    read_pdf = _import_attr("app.services.gridfs_storage", "read_pdf")
    try:
        return read_pdf(doc.gridfs_file_id, workspace_id=doc.workspace_id)
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError("parsing", f"gridfs read failed: {exc}") from exc


def _extract(pdf_bytes: bytes) -> Any:
    """Run PyMuPDF + pdfplumber extraction inside the sandbox. Stage: parsing.

    The GridFS copy is already sanitized, but PyMuPDF / pdfplumber still run
    parser code against it, so it goes through the same docker worker as the
    upload (A.8.7). Inline mode (dev/CI) keeps the old in-process behaviour.
    """
    sandbox_process = _import_attr("app.services.pdf_sandbox", "sandbox_process")
    try:
        return sandbox_process(pdf_bytes).to_extraction_result()
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError("parsing", f"extract failed: {exc}") from exc


def _ocr_enrich(pdf_bytes: bytes, extraction: Any) -> str | None:
    """Run the 3-tier OCR bridge over every page in place. Stage: parsing.

    Rewrites ``extraction.pages[*].text`` with OCR-enriched text and returns
    the dominant tier (``proximity`` / ``tesseract`` / ``azure_cv``), or
    ``None`` when no OCR fired. Best-effort: any failure logs and keeps the
    native text instead of failing the document.
    """
    pages = list(getattr(extraction, "pages", []) or [])
    if not pages:
        return None
    try:
        enrich = _import_attr("app.rag.vision", "enrich_text_with_ocr")
    except PipelineStageError as exc:
        logger.warning("ocr skipped: %s", exc)
        return None
    settings = get_settings()
    try:
        enriched, dominant = enrich(
            pdf_bytes, pages, enable_azure=bool(getattr(settings, "ocr_enable_azure", True))
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ocr enrichment failed, keeping native text: %s", exc)
        return None
    by_page = {num: text for num, text in enriched}
    for page in pages:
        new_text = by_page.get(getattr(page, "page_num", None))
        if new_text is None:
            continue
        try:
            page.text = new_text
        except Exception:  # noqa: BLE001 — frozen stub pages in tests
            pass
    return dominant


def _chunker_pages(extraction: Any) -> list[Any]:
    """Collapse an ExtractionResult into chunker pages with tables merged.

    ``pdf_extractor.to_chunker_pages`` appends each page's pdfplumber tables
    as pipe-delimited rows. Extraction objects without a ``tables`` attribute
    (unit-test stubs) fall through to the bare page list.
    """
    if getattr(extraction, "tables", None) is None:
        return list(getattr(extraction, "pages", []) or [])
    to_pages = _import_attr("app.services.pdf_extractor", "to_chunker_pages")
    try:
        return list(to_pages(extraction))
    except Exception as exc:  # noqa: BLE001
        # Fail closed: a silent text-only fallback would drop tables from the index again (T-004).
        raise PipelineStageError("chunking", f"table merge failed: {exc}") from exc


def _chunk(pages: list[Any], doc_id: str) -> list[Any]:
    """Split per-page text into overlapping chunks. Stage: chunking."""
    chunker_mod = importlib.import_module("app.rag.chunker")
    chunk_pages = getattr(chunker_mod, "chunk_pages", None)
    page_cls = getattr(chunker_mod, "Page", None)
    if chunk_pages is None or page_cls is None:
        raise PipelineStageError(
            "chunking", "app.rag.chunker missing chunk_pages/Page"
        )
    try:
        # ExtractionResult.pages carry (page_num, text, bbox_image_regions).
        # chunker.Page only needs page_num + text.
        chunker_pages = [page_cls(page_num=p.page_num, text=p.text) for p in pages]
        return chunk_pages(chunker_pages, doc_id=doc_id)
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError("chunking", f"chunk failed: {exc}") from exc


def _embed_and_store(
    workspace_id: str, doc_id: str, chunks: list[Any]
) -> int:
    """Embed every chunk via Cohere and upsert into Atlas. Stage: embedding."""
    # Prefer the T-007 embedder (app.rag.embeddings.embed_documents) and
    # fall back to the Cohere SDK directly if that module is still absent.
    embedder = None
    for mod_name in ("app.rag.embeddings", "app.rag.embedder"):
        try:
            embedder = importlib.import_module(mod_name)
            break
        except ImportError:
            continue

    settings = get_settings()
    texts: list[str] = [c.chunk_text for c in chunks]

    vectors: list[list[float]]
    try:
        if embedder is not None and hasattr(embedder, "embed_documents"):
            vectors = embedder.embed_documents(texts)
        else:
            vectors = _embed_via_sdk(texts, settings)
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError("embedding", f"embed failed: {exc}") from exc

    if len(vectors) != len(chunks):
        raise PipelineStageError(
            "embedding",
            f"embedding count {len(vectors)} != chunk count {len(chunks)}",
        )

    upsert_chunks = _import_attr("app.db.vector", "upsert_chunks")
    delete_chunks = _import_attr("app.db.vector", "delete_document_chunks")
    try:
        # Clear any chunks left by an earlier (failed or shorter) run so a
        # retry can never leave stale rows past the new chunk count.
        delete_chunks(workspace_id, doc_id)
        payload = [
            {
                "chunk_index": c.chunk_index,
                "chunk_text": c.chunk_text,
                "embedding": vec,
                "page_num": c.page_num,
            }
            for c, vec in zip(chunks, vectors)
        ]
        return upsert_chunks(workspace_id, doc_id, payload)
    except Exception as exc:  # noqa: BLE001
        raise PipelineStageError("embedding", f"upsert failed: {exc}") from exc


def _embed_via_sdk(texts: list[str], settings: Any) -> list[list[float]]:
    """Fallback: call Cohere ClientV2 directly on the Azure endpoint.

    Batched at the Cohere limit (96 texts per request).
    """
    try:
        import cohere  # type: ignore[import-untyped]
    except ImportError as exc:  # noqa: BLE001
        raise PipelineStageError(
            "embedding", f"cohere SDK not installed: {exc}"
        ) from exc

    base_url = f"{settings.azure_ai_api_base.rstrip('/')}/providers/cohere"
    client = cohere.ClientV2(api_key=settings.azure_ai_api_key, base_url=base_url)

    out: list[list[float]] = []
    batch_size = 96
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embed(
            texts=batch,
            model=settings.embedding_model,
            input_type="search_document",
            embedding_types=["float"],
            output_dimension=settings.embedding_dimensions,
        )
        # Pydantic alias — Cohere SDK exposes the float list as `float_`.
        floats = getattr(response.embeddings, "float_", None)
        if floats is None:
            floats = getattr(response.embeddings, "float", None)
        if floats is None:
            raise PipelineStageError(
                "embedding", "cohere response missing float embeddings"
            )
        out.extend(list(v) for v in floats)
    return out


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


@dataclass
class _StageSpec:
    name: str
    status_before: str
    func: Callable[[Session, Document, dict[str, Any]], None]


def _stage_parsing(db: Session, doc: Document, state: dict[str, Any]) -> None:
    _set_status(db, doc.id, status=STATUS_PARSING)
    pdf_bytes = _load_pdf_bytes(doc)
    result = _extract(pdf_bytes)
    state["extraction"] = result
    state["page_count"] = getattr(result, "page_count", None)
    state["ocr_tier_used"] = _ocr_enrich(pdf_bytes, result)


def _stage_chunking(db: Session, doc: Document, state: dict[str, Any]) -> None:
    _set_status(
        db,
        doc.id,
        status=STATUS_CHUNKING,
        page_count=state.get("page_count"),
        ocr_tier_used=state.get("ocr_tier_used"),
    )
    extraction = state["extraction"]
    chunks = _chunk(_chunker_pages(extraction), doc_id=str(doc.id))
    state["chunks"] = chunks


def _stage_embedding(db: Session, doc: Document, state: dict[str, Any]) -> None:
    _set_status(db, doc.id, status=STATUS_EMBEDDING)
    chunks = state["chunks"]
    if not chunks:
        state["chunk_count"] = 0
        return
    written = _embed_and_store(doc.workspace_id, str(doc.id), chunks)
    state["chunk_count"] = written


_STAGES: tuple[_StageSpec, ...] = (
    _StageSpec("parsing", STATUS_PARSING, _stage_parsing),
    _StageSpec("chunking", STATUS_CHUNKING, _stage_chunking),
    _StageSpec("embedding", STATUS_EMBEDDING, _stage_embedding),
)


def _execute_stage(
    db: Session,
    doc: Document,
    stage: _StageSpec,
    state: dict[str, Any],
    policy: RetryPolicy,
) -> None:
    last_exc: BaseException | None = None
    for attempt in range(policy.attempts):
        try:
            stage.func(db, doc, state)
            return
        except PipelineStageError as exc:
            last_exc = exc
            if attempt == policy.attempts - 1 or not _is_retryable(exc):
                raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == policy.attempts - 1 or not _is_retryable(exc):
                raise PipelineStageError(stage.name, str(exc)) from exc
        delay = policy.delay_for(attempt)
        logger.warning(
            "stage %s attempt %d/%d failed, retrying in %.1fs: %s",
            stage.name,
            attempt + 1,
            policy.attempts,
            delay,
            last_exc,
        )
        time.sleep(delay)
    # Unreachable — the loop either returns or raises.


def run_pipeline(doc_id: uuid.UUID | str, *, policy: RetryPolicy | None = None) -> str:
    """Walk a document through every stage synchronously.

    Returns the final status (``ready`` or ``error``). Safe to call from a
    plain thread, a Celery worker (Phase 2), or ``asyncio.to_thread``.
    """
    if isinstance(doc_id, str):
        doc_id = uuid.UUID(doc_id)
    policy = policy or DEFAULT_POLICY

    db = SessionLocal()
    final_status = STATUS_ERROR
    try:
        doc = _load_doc(db, doc_id)
        if doc is None:
            logger.error("run_pipeline: document %s not found", doc_id)
            return STATUS_ERROR

        _audit(
            db,
            workspace_id=doc.workspace_id,
            doc_id=doc.id,
            event_type="pipeline.start",
            details=f"status_at_start={doc.status}",
        )

        state: dict[str, Any] = {}
        try:
            for stage in _STAGES:
                _execute_stage(db, doc, stage, state, policy)

            _set_status(
                db,
                doc.id,
                status=STATUS_READY,
                chunk_count=state.get("chunk_count", 0),
                page_count=state.get("page_count"),
                ocr_tier_used=state.get("ocr_tier_used"),
                mark_processed=True,
            )
            _audit(
                db,
                workspace_id=doc.workspace_id,
                doc_id=doc.id,
                event_type="pipeline.ready",
                details=(
                    f"chunks={state.get('chunk_count', 0)} "
                    f"ocr_tier={state.get('ocr_tier_used') or 'none'}"
                ),
            )
            final_status = STATUS_READY
        except PipelineStageError as exc:
            logger.exception("pipeline failed for %s", doc_id)
            _set_status(
                db,
                doc.id,
                status=STATUS_ERROR,
                message=str(exc),
            )
            _audit(
                db,
                workspace_id=doc.workspace_id,
                doc_id=doc.id,
                event_type="pipeline.error",
                details=f"stage={exc.stage} msg={exc.message}",
            )
            final_status = STATUS_ERROR
    finally:
        db.close()
    return final_status


# ---------------------------------------------------------------------------
# Async scheduling — Phase 1
# ---------------------------------------------------------------------------


def _schedule_task(doc_id: uuid.UUID) -> asyncio.Task | None:
    """Submit a document to the running asyncio loop as a background task.

    Returns the Task when a loop is running. Without a running loop (CLI
    scripts, a sync Starlette background task running in its threadpool)
    the pipeline is started on a daemon thread and ``None`` is returned.
    The caller may ignore the return value — the pipeline writes its own
    status to Postgres so the frontend never depends on the Task object.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        thread = threading.Thread(
            target=run_pipeline,
            args=(doc_id,),
            name=f"pipeline:{doc_id}",
            daemon=True,
        )
        thread.start()
        return None

    async def _runner() -> None:
        await asyncio.to_thread(run_pipeline, doc_id)

    task = loop.create_task(_runner(), name=f"pipeline:{doc_id}")
    # Hold a strong reference so Python does not GC the Task mid-run. The
    # module-level set is cleared on completion by a one-shot callback.
    _ACTIVE_TASKS.add(task)
    task.add_done_callback(_ACTIVE_TASKS.discard)
    return task


_ACTIVE_TASKS: set[asyncio.Task] = set()


def enqueue(doc_id: uuid.UUID | str) -> None:
    """Schedule a document for processing. Non-blocking.

    If an asyncio loop is running (FastAPI handlers always run inside one),
    the pipeline runs via ``asyncio.to_thread``. Outside a loop it runs on a
    daemon thread instead, so no caller silently drops a document.
    """
    if isinstance(doc_id, str):
        doc_id = uuid.UUID(doc_id)
    _schedule_task(doc_id)


# ---------------------------------------------------------------------------
# Status read API — consumed by the /status endpoint
# ---------------------------------------------------------------------------


@dataclass
class StatusReport:
    doc_id: str
    status: str
    message: str | None
    progress_pct: int
    chunk_count: int
    page_count: int | None
    ocr_tier_used: str | None
    created_at: str | None
    processed_at: str | None
    is_terminal: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "status": self.status,
            "message": self.message,
            "progress_pct": self.progress_pct,
            "chunk_count": self.chunk_count,
            "page_count": self.page_count,
            "ocr_tier_used": self.ocr_tier_used,
            "created_at": self.created_at,
            "processed_at": self.processed_at,
            "is_terminal": self.is_terminal,
        }


def _progress_pct(status: str) -> int:
    return _PROGRESS_PCT.get(status, 0)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def get_status(doc_id: uuid.UUID | str) -> StatusReport | None:
    """Snapshot a document's current pipeline state. ``None`` when not found."""
    if isinstance(doc_id, str):
        doc_id = uuid.UUID(doc_id)

    db = SessionLocal()
    try:
        doc = _load_doc(db, doc_id)
        if doc is None:
            return None
        return StatusReport(
            doc_id=str(doc.id),
            status=doc.status,
            message=doc.status_message,
            progress_pct=_progress_pct(doc.status),
            chunk_count=doc.chunk_count,
            page_count=doc.page_count,
            ocr_tier_used=doc.ocr_tier_used,
            created_at=_iso(doc.created_at),
            processed_at=_iso(doc.processed_at),
            is_terminal=doc.status in (STATUS_READY, STATUS_ERROR),
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Manual retry — called from the /retry endpoint
# ---------------------------------------------------------------------------


def retry(doc_id: uuid.UUID | str) -> bool:
    """Flip an ``error`` document back to ``pending`` and re-enqueue it.

    Returns True when the retry was scheduled, False when the document is
    missing or still in a terminal non-error state (``ready``).
    """
    if isinstance(doc_id, str):
        doc_id = uuid.UUID(doc_id)
    db = SessionLocal()
    try:
        doc = _load_doc(db, doc_id)
        if doc is None:
            return False
        if doc.status == STATUS_READY:
            return False
        _set_status(db, doc.id, status=STATUS_PENDING, message=None)
        _audit(
            db,
            workspace_id=doc.workspace_id,
            doc_id=doc.id,
            event_type="pipeline.retry",
        )
    finally:
        db.close()
    enqueue(doc_id)
    return True
