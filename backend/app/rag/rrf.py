"""Reciprocal Rank Fusion for merging vector + BM25 results (T-010).

Reference: Cormack, Clarke, Buettcher (2009) "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods".

Formula:
    rrf_score(d) = sum_over_rankers( 1 / (k + rank(d, ranker)) )

where ``rank`` is 1-based. The constant ``k`` (default 60, per the paper and
Elastic/Weaviate defaults) damps the contribution of top-ranked items so a
hit that appears mid-pack across every ranker can still beat a single list's
runaway winner — exactly the behaviour we want when fusing semantic
(vector) and lexical (BM25) retrievers that disagree on term overlap.

The function is deliberately framework-free: it fuses opaque ``id`` strings
and returns the sorted fused list. Rank-position only; raw scores from each
ranker are ignored on purpose — mixing cosine-sim with BM25 is the thing
RRF was invented to avoid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RankedHit:
    """One retriever result, pre-fusion. ``id`` must be stable across lists."""

    id: str
    payload: Mapping | None = None  # original chunk doc, preserved untouched


@dataclass(frozen=True)
class FusedHit:
    """Post-RRF item: the id, its fused score, and the richest payload seen."""

    id: str
    score: float
    payload: Mapping | None
    sources: tuple[str, ...]  # which rankers contributed


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RankedHit]],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> list[FusedHit]:
    """Merge several ranked lists into one by RRF.

    Args:
        rankings: ``{ranker_name: [RankedHit, ...]}``. Order inside each list
            is the ranker's own ranking (position 0 = rank 1 = best).
        k: RRF damping constant. 60 is the paper default and the one every
            major hybrid-search engine ships with — do not tune it per
            query, tune the retriever stage instead.
        top_k: if given, truncate the fused result to this many hits.

    Returns:
        Fused list, sorted by descending RRF score. Ties are broken by the
        first-seen ranker + original rank (deterministic across calls).

    Raises:
        ValueError: if ``k`` <= 0 (RRF requires a positive damping constant).
    """
    if k <= 0:
        raise ValueError("RRF k must be positive (paper default: 60)")

    agg: dict[str, dict] = {}
    for ranker_name, hits in rankings.items():
        for rank_pos, hit in enumerate(hits, start=1):
            contribution = 1.0 / (k + rank_pos)
            bucket = agg.get(hit.id)
            if bucket is None:
                agg[hit.id] = {
                    "score": contribution,
                    "payload": hit.payload,
                    "sources": [ranker_name],
                    # Preserve first-seen rank for stable tie-break.
                    "first_rank": rank_pos,
                }
            else:
                bucket["score"] += contribution
                bucket["sources"].append(ranker_name)
                # Prefer the richest payload seen (vector hits usually
                # carry the chunk doc with metadata; BM25 may be lighter).
                if bucket["payload"] is None and hit.payload is not None:
                    bucket["payload"] = hit.payload

    fused = [
        FusedHit(
            id=doc_id,
            score=state["score"],
            payload=state["payload"],
            sources=tuple(state["sources"]),
        )
        for doc_id, state in agg.items()
    ]
    # Primary: fused score desc. Tie-break: fewer sources last (prefer multi-
    # retriever agreement). Final tie-break: id asc for determinism.
    fused.sort(key=lambda f: (-f.score, -len(f.sources), f.id))

    if top_k is not None:
        fused = fused[:top_k]
    return fused


def rankings_from_payloads(
    rankings: Mapping[str, Iterable[Mapping]],
    *,
    id_field: str = "_id",
) -> dict[str, list[RankedHit]]:
    """Adapter: wrap raw MongoDB docs into :class:`RankedHit` by ``id_field``.

    Caller supplies ``{ "vector": [doc, doc, ...], "bm25": [doc, ...] }`` where
    each ``doc`` is the aggregation stage output. Ids are stringified so the
    RRF dict key stays hashable even for ``ObjectId``.
    """
    out: dict[str, list[RankedHit]] = {}
    for name, docs in rankings.items():
        wrapped: list[RankedHit] = []
        for doc in docs:
            if id_field not in doc:
                raise KeyError(f"ranker {name!r}: doc missing id field {id_field!r}")
            wrapped.append(RankedHit(id=str(doc[id_field]), payload=doc))
        out[name] = wrapped
    return out


__all__ = [
    "RankedHit",
    "FusedHit",
    "reciprocal_rank_fusion",
    "rankings_from_payloads",
]
