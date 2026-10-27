"""Background processing — async ingestion state machine (T-014).

Public API:
    enqueue(doc_id)      — schedule a document for background processing
    get_status(doc_id)   — read the current pipeline state for a document
    run_pipeline(doc_id) — synchronous driver used by tests + Celery (Phase 2)

State machine (persisted in documents.status):
    pending → parsing → chunking → embedding → ready
                                             → error (any stage may land here)
"""
from app.processing.pipeline import (  # noqa: F401
    PipelineError,
    PipelineStageError,
    STATUSES,
    STATUS_ERROR,
    STATUS_PARSING,
    STATUS_CHUNKING,
    STATUS_EMBEDDING,
    STATUS_PENDING,
    STATUS_READY,
    enqueue,
    get_status,
    run_pipeline,
)
