"""Hybrid retrieval pipeline — T-010.

Orchestrates the four stages that turn a user question into the top-K
grounded chunks the LLM will cite:

    1. Query embedding   -> ``app.rag.embeddings.embed_query``
    2. Dense retrieval   -> ``app.db.vector.vector_search``       (top-20)
    3. Sparse retrieval  -> ``app.rag.bm25.bm25_search``          (top-20)
    4. Rank fusion       -> ``app.rag.rrf.reciprocal_rank_fusion``
    5. Cross-encoder     -> ``app.rag.reranker.rerank``           (top-8)

The two retrievers run concurrently (dense is HTTP → Mongo, sparse is
pure Mongo), fused by rank position with RRF so disagreeing score scales
never need to be calibrated. The BGE cross-encoder then re-reads each
(query, chunk) pair jointly, which lifts top-K precision by 15-20% over
vector-only in the BGE-v2-m3 benchmarks.

Schema notes (post task #27, 2026-10-27):
    Both the vector writer (``app.db.vector`` — ``DOC_ID_FIELD = "doc_id"``)
    and the BM25 writer (``app.rag.bm25.BM25Hit.doc_id``) now agree on
    ``doc_id`` as the Postgres link. :func:`_normalise_chunk` still accepts
    a ``document_id`` key as a read-only fallback, but no live writer
    emits it — the fallback is unreachable in dev and only exists so a
    pre-#27 ingest replay (if one ever surfaces) does not silently drop
    the doc link.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from app.config import get_settings
from app.rag import bm25 as _bm25_mod
from app.rag import embeddings as _embed_mod
from app.rag import reranker as _reranker_mod
from app.rag import rrf as _rrf_mod
from app.db import vector as _vector_mod

logger = logging.getLogger(__name__)


# ---- Tuning constants (overridable per-call) --------------------------------

DEFAULT_VECTOR_TOP_K = 20     # dense retriever recall budget
DEFAULT_BM25_TOP_K = 20       # sparse retriever recall budget
DEFAULT_RRF_K = 60            # paper default — do not tune per query
DEFAULT_RERANK_TOP_K = 8      # final LLM-context budget
DEFAULT_FUSED_POOL_K = 40     # how many fused hits to send to the reranker


# ---- Public result type -----------------------------------------------------


@dataclass(frozen=True)
class RetrievedChunk:
    """One final hit after fusion + reranking, ready for the RAG prompt."""

    chunk_id: str
    doc_id: str
    workspace_id: str
    chunk_text: str
    page_num: int | None
    rerank_score: float
    rrf_score: float
    vector_score: float | None   # from $vectorSearch meta if the hit came through dense
    bm25_score: float | None     # from $search meta if the hit came through sparse
    sources: tuple[str, ...]     # ("vector",), ("bm25",) or ("vector", "bm25")

    def citation(self) -> str:
        """Short human-readable citation for prompt inclusion."""
        if self.page_num is None:
            return f"[doc {self.doc_id}]"
        return f"[doc {self.doc_id} p.{self.page_num}]"


# ---- Internal helpers -------------------------------------------------------


def _normalise_chunk(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce a mongo chunk doc into a stable shape across retrievers.

    Both the vector writer (``app.db.vector``) and the BM25 writer
    (``app.rag.bm25``) now agree on ``doc_id`` as the Postgres link
    (per task #27 schema alignment). ``document_id`` is accepted as a
    read-only fallback for legacy rows during the migration window.
    """
    doc_id = doc.get("doc_id") or doc.get("document_id") or ""
    return {
        "_id": str(doc.get("_id", "")),
        "doc_id": str(doc_id),
        "workspace_id": str(doc.get("workspace_id", "")),
        "chunk_text": str(doc.get("chunk_text", "")),
        "page_num": doc.get("page_num"),
        "vector_score": doc.get("vector_score"),
        "bm25_score": doc.get("bm25_score") or doc.get("score"),
    }


def _vector_hits_to_payloads(
    hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [_normalise_chunk(h) for h in hits]


def _bm25_hits_to_payloads(
    hits: list[_bm25_mod.BM25Hit],
) -> list[dict[str, Any]]:
    return [
        _normalise_chunk(
            {
                "_id": h.chunk_id,
                "doc_id": h.doc_id,
                "workspace_id": h.workspace_id,
                "chunk_text": h.chunk_text,
                "page_num": h.page_num,
                "score": h.score,  # BM25 raw score
            }
        )
        for h in hits
    ]


# ---- Public entrypoint ------------------------------------------------------


@dataclass
class HybridRetriever:
    """Thin configurable wrapper so routes and background workers share one
    instance of the knobs, and so unit tests can inject fakes.

    Default behaviour matches the plan verbatim:
        top-20 dense + top-20 sparse → RRF k=60 → BGE rerank top-8.

    The two retrievers run inside :func:`asyncio.to_thread` so the loop
    is never blocked on pymongo's sync driver.
    """

    vector_top_k: int = DEFAULT_VECTOR_TOP_K
    bm25_top_k: int = DEFAULT_BM25_TOP_K
    rrf_k: int = DEFAULT_RRF_K
    rerank_top_k: int = DEFAULT_RERANK_TOP_K
    fused_pool_k: int = DEFAULT_FUSED_POOL_K
    reranker_model: str | None = None
    reranker_device: str | None = None
    # Optional injection points — unit tests swap these for fakes.
    embed_query_fn: Any = None          # async (text, dimension=None) -> list[float]
    vector_search_fn: Any = None        # sync (workspace, qvec, top_k, doc_id?) -> list[dict]
    bm25_search_fn: Any = None          # sync (workspace, q, top_k, doc_id=None) -> list[BM25Hit]
    rerank_fn: Any = None               # sync (query, candidates, top_k, model, device) -> list

    def _embed(self):
        # embed_query_async matches our (text, *, dimension=None) async contract
        # (shipped by support-impl-15 in embeddings.py). The sync embed_query
        # has kwarg output_dimension= and returns list[float], not awaitable.
        return self.embed_query_fn or _embed_mod.embed_query_async

    def _vector(self):
        return self.vector_search_fn or _vector_mod.vector_search

    def _bm25(self):
        return self.bm25_search_fn or _bm25_mod.bm25_search

    def _rerank(self):
        return self.rerank_fn or _reranker_mod.rerank

    async def search(
        self,
        workspace_id: str,
        query: str,
        *,
        doc_id: str | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Run the full hybrid+rerank pipeline for one user question.

        Args:
            workspace_id: hard isolation key — passed to both retrievers.
            query: user question, free-form text.
            doc_id: optional filter to one PDF (for per-doc chat).
            top_k: override :attr:`rerank_top_k` for this one call.

        Returns:
            Up to ``top_k`` :class:`RetrievedChunk` sorted by cross-encoder
            score. Empty list if neither retriever found any candidates
            (empty workspace, bad query, both indexes unbuilt, etc.).
        """
        if not workspace_id:
            raise ValueError("workspace_id required (hybrid search is workspace-scoped)")
        q = (query or "").strip()
        if not q:
            raise ValueError("query must be a non-empty string")

        final_top_k = top_k if top_k is not None else self.rerank_top_k
        settings = get_settings()

        # Stage 1 + 2a + 2b: embed query, launch both retrievers concurrently.
        embed = self._embed()
        query_vec = await embed(q, dimension=settings.embedding_dimensions)

        vec_task = asyncio.to_thread(
            self._vector(),
            workspace_id,
            query_vec,
            self.vector_top_k,
            doc_id,
        )
        bm25_task = asyncio.to_thread(
            self._bm25(),
            workspace_id,
            q,
            self.bm25_top_k,
            doc_id=doc_id,
        )
        vec_hits_raw, bm25_hits_raw = await asyncio.gather(
            vec_task, bm25_task, return_exceptions=True
        )

        vec_payloads: list[dict[str, Any]] = []
        bm25_payloads: list[dict[str, Any]] = []

        if isinstance(vec_hits_raw, BaseException):
            logger.warning("vector_search failed, degrading to BM25-only: %s", vec_hits_raw)
        else:
            vec_payloads = _vector_hits_to_payloads(vec_hits_raw)

        if isinstance(bm25_hits_raw, BaseException):
            logger.warning("bm25_search failed, degrading to vector-only: %s", bm25_hits_raw)
        else:
            bm25_payloads = _bm25_hits_to_payloads(bm25_hits_raw)

        # Both retrievers filter on doc_id server-side. This post-filter is
        # defence in depth for injected search functions that ignore the
        # kwarg, so a per-document chat can never leak sibling documents.
        if doc_id is not None:
            bm25_payloads = [p for p in bm25_payloads if p["doc_id"] == doc_id]

        if not vec_payloads and not bm25_payloads:
            return []

        # Stage 3: Reciprocal Rank Fusion.
        fused = _rrf_mod.reciprocal_rank_fusion(
            {
                "vector": [
                    _rrf_mod.RankedHit(id=p["_id"], payload=p) for p in vec_payloads
                ],
                "bm25": [
                    _rrf_mod.RankedHit(id=p["_id"], payload=p) for p in bm25_payloads
                ],
            },
            k=self.rrf_k,
            top_k=self.fused_pool_k,
        )
        if not fused:
            return []

        # Stage 4: Cross-encoder reranking of the fused pool.
        rerank = self._rerank()
        candidates = [
            _reranker_mod.RerankCandidate(
                id=f.id,
                text=(f.payload or {}).get("chunk_text", ""),
                payload=dict(f.payload) if f.payload else None,
            )
            for f in fused
        ]
        reranked = await asyncio.to_thread(
            rerank,
            q,
            candidates,
            top_k=final_top_k,
            model_name=self.reranker_model,
            device=self.reranker_device,
        )

        # Build the final return structure, carrying every score forward.
        fused_by_id = {f.id: f for f in fused}
        out: list[RetrievedChunk] = []
        for hit in reranked:
            fused_entry = fused_by_id.get(hit.id)
            payload = hit.payload or {}
            out.append(
                RetrievedChunk(
                    chunk_id=hit.id,
                    doc_id=str(payload.get("doc_id", "")),
                    workspace_id=str(payload.get("workspace_id", workspace_id)),
                    chunk_text=hit.text,
                    page_num=payload.get("page_num"),
                    rerank_score=hit.score,
                    rrf_score=fused_entry.score if fused_entry else 0.0,
                    vector_score=payload.get("vector_score"),
                    bm25_score=payload.get("bm25_score"),
                    sources=fused_entry.sources if fused_entry else (),
                )
            )
        return out


# Module-level default instance so callers can just `from retriever import search`.
_DEFAULT_RETRIEVER: HybridRetriever | None = None


def get_default_retriever() -> HybridRetriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = HybridRetriever()
    return _DEFAULT_RETRIEVER


async def hybrid_search(
    workspace_id: str,
    query: str,
    *,
    doc_id: str | None = None,
    top_k: int = DEFAULT_RERANK_TOP_K,
) -> list[dict[str, Any]]:
    """Convenience function — returns plain dicts for T-012's SSE pipeline.

    Each dict carries the keys ``app.rag.pipeline._format_context`` reads:
    ``chunk_id``, ``doc_id``, ``chunk_text``, ``page_num``, ``score``
    (the final rerank score — not RRF or vector), plus provenance in
    ``vector_score``, ``bm25_score``, ``rrf_score``, ``sources``.
    """
    hits = await get_default_retriever().search(
        workspace_id,
        query,
        doc_id=doc_id,
        top_k=top_k,
    )
    return [chunk_to_dict(h) for h in hits]


def chunk_to_dict(chunk: RetrievedChunk) -> dict[str, Any]:
    """Flatten a :class:`RetrievedChunk` to the dict shape T-012 expects."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "workspace_id": chunk.workspace_id,
        "chunk_text": chunk.chunk_text,
        "page_num": chunk.page_num,
        "score": chunk.rerank_score,  # headline score for citation ranking
        "rerank_score": chunk.rerank_score,
        "rrf_score": chunk.rrf_score,
        "vector_score": chunk.vector_score,
        "bm25_score": chunk.bm25_score,
        "sources": list(chunk.sources),
    }


__all__ = [
    "RetrievedChunk",
    "HybridRetriever",
    "hybrid_search",
    "chunk_to_dict",
    "get_default_retriever",
    "DEFAULT_VECTOR_TOP_K",
    "DEFAULT_BM25_TOP_K",
    "DEFAULT_RRF_K",
    "DEFAULT_RERANK_TOP_K",
    "DEFAULT_FUSED_POOL_K",
]
