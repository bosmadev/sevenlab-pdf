"""Cross-encoder reranking stage for T-010.

Uses ``BAAI/bge-reranker-v2-m3`` — a multilingual cross-encoder that scores
each (query, candidate) pair jointly. Unlike dual-encoder (bi-encoder)
retrieval, the cross-encoder reads both texts at once, so it catches
subtle entailment and term-sense alignment that cosine similarity misses.

Measured in the BGE technical report: v2-m3 lifts top-K precision by
15-20% over vector-only on English+multilingual retrieval benchmarks, in
line with the plan's claimed `+15-20%` headroom.

Lazy singleton: the model (~2.3 GB fp32, ~1.1 GB fp16) is loaded on first
call so FastAPI boot stays cheap. CPU inference is ~200ms for 40 pairs
of ~300-char chunks on a modern x86; GPU with ``device="cuda"`` is 10x.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from threading import Lock
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_BATCH = 32
DEFAULT_RERANKER_MAX_LEN = 512


@dataclass(frozen=True)
class RerankCandidate:
    """Input: one candidate from the fused retriever stage."""

    id: str
    text: str
    payload: dict | None = None


@dataclass(frozen=True)
class RerankedHit:
    """Output: candidate with the cross-encoder's relevance score."""

    id: str
    text: str
    score: float
    payload: dict | None = None


class _CrossEncoderSingleton:
    """Thread-safe lazy-loaded cross-encoder.

    Kept as a module-local class rather than a free function so the model
    survives across requests (loading is the slow part, scoring is fast).
    """

    _lock = Lock()
    _instance: "_CrossEncoderSingleton | None" = None

    def __init__(self, model_name: str, device: str | None, max_length: int) -> None:
        # Deferred import: sentence-transformers pulls in torch + transformers,
        # ~1 GB on disk. Importing at module scope would make unit tests of
        # pure code (rrf.py) pay that cost. Caller does the import only when
        # the first query actually reranks.
        from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

        logger.info("Loading cross-encoder %s (device=%s)", model_name, device or "auto")
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )
        self.model_name = model_name

    @classmethod
    def get(
        cls,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
        max_length: int = DEFAULT_RERANKER_MAX_LEN,
    ) -> "_CrossEncoderSingleton":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(model_name, device, max_length)
        return cls._instance

    def score(self, pairs: Sequence[tuple[str, str]], batch_size: int) -> list[float]:
        if not pairs:
            return []
        scores = self.model.predict(
            list(pairs),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]


def rerank(
    query: str,
    candidates: Iterable[RerankCandidate],
    *,
    top_k: int = 8,
    model_name: str | None = None,
    device: str | None = None,
    batch_size: int = DEFAULT_RERANKER_BATCH,
    max_length: int = DEFAULT_RERANKER_MAX_LEN,
) -> list[RerankedHit]:
    """Score every candidate against ``query`` and return the top ``top_k``.

    Args:
        query: user question.
        candidates: iterable of :class:`RerankCandidate`, usually ~40 fused
            hits from the vector+BM25 RRF stage.
        top_k: how many to return. Plan default is 8 (fits LLM context well
            after we add page citations).
        model_name: override the default BGE model (unit tests mock this
            with a tiny model to skip the 2 GB download).
        device: ``"cpu"``, ``"cuda"``, ``"mps"``, or ``None`` to auto-pick.
        batch_size, max_length: cross-encoder forward-pass sizing.

    Returns:
        List of :class:`RerankedHit` sorted by descending score, length
        ``min(top_k, len(candidates))``. Empty list if no candidates.
    """
    cand_list = list(candidates)
    if not cand_list:
        return []
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    resolved_model = model_name or os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
    encoder = _CrossEncoderSingleton.get(
        model_name=resolved_model,
        device=device,
        max_length=max_length,
    )

    pairs = [(query, c.text) for c in cand_list]
    scores = encoder.score(pairs, batch_size=batch_size)

    scored = [
        RerankedHit(
            id=c.id,
            text=c.text,
            score=s,
            payload=c.payload,
        )
        for c, s in zip(cand_list, scores, strict=True)
    ]
    # Score desc; stable tie-break on id for determinism in tests.
    scored.sort(key=lambda h: (-h.score, h.id))
    return scored[:top_k]


__all__ = [
    "RerankCandidate",
    "RerankedHit",
    "rerank",
    "DEFAULT_RERANKER_MODEL",
]
