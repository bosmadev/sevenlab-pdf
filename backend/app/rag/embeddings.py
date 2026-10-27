"""Cohere Embed V5-Pro client (T-007).

Produces 2048-dim float vectors against Azure's hosted deployment:

    POST {AZURE_AI_API_BASE}/providers/cohere/v2/embed

Why a thin REST client and not the ``cohere`` SDK:
    The SDK's ClientV2 silently falls back to Cohere's public API when the
    Azure base URL path shape does not match its expectations, which has
    bitten this project before — the key logs with the string
    ``AZURE_AI_API_KEY``, so a mis-routed request leaks the key to a vendor
    that never saw it. One ``httpx`` call with an explicit URL cannot
    accidentally reach a second host.

Public API:
    embed_documents(texts) -> list[list[float]]        # input_type=search_document
    embed_query(text)      -> list[float]              # input_type=search_query
    async aembed_documents / aembed_query              # same, non-blocking

Both return 2048-dim vectors by default (overridable via ``output_dimension``)
to match the MongoDB Atlas ``vector_index`` created by T-008. The batch helpers
chunk inputs to the Azure limit of 96 texts per request and retry 429/5xx with
jittered exponential backoff.

References:
    - Plan integration guide: SEVENLAB-PDFCHAT.md § Cohere Embed V5 Pro
    - Azure endpoint path confirmed in T-007 doc: `/providers/cohere/v2/embed`
    - 128k context window; `truncate=END` is server-side default
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

InputType = Literal["search_document", "search_query", "classification", "clustering"]

MAX_BATCH = 96
"""Azure's hard cap on ``texts`` per /embed call."""

ALLOWED_DIMENSIONS = frozenset({256, 512, 768, 1024, 1536, 2048})

_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class EmbeddingError(RuntimeError):
    """Raised when Cohere V5-Pro cannot return a usable embedding."""


def _endpoint() -> str:
    base = get_settings().azure_ai_api_base.rstrip("/")
    return f"{base}/providers/cohere/v2/embed"


def _headers() -> dict[str, str]:
    key = get_settings().azure_ai_api_key
    if not key:
        raise EmbeddingError(
            "AZURE_AI_API_KEY missing — set it in backend/.env (never commit)."
        )
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _validate_dim(dim: int) -> int:
    if dim not in ALLOWED_DIMENSIONS:
        raise EmbeddingError(
            f"output_dimension={dim} not in {sorted(ALLOWED_DIMENSIONS)}"
        )
    return dim


def _coerce_texts(texts: Sequence[str] | Iterable[str]) -> list[str]:
    out: list[str] = []
    for t in texts:
        if not isinstance(t, str):
            raise EmbeddingError(f"text must be str, got {type(t).__name__}")
        s = t.strip()
        if not s:
            raise EmbeddingError("empty text cannot be embedded")
        out.append(s)
    if not out:
        raise EmbeddingError("texts must contain at least one non-empty string")
    return out


def _batches(items: list[str], size: int = MAX_BATCH) -> Iterable[list[str]]:
    size = min(size, MAX_BATCH)
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _payload(
    texts: list[str],
    input_type: InputType,
    output_dimension: int,
    model: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "texts": texts,
        "input_type": input_type,
        "embedding_types": ["float"],
        "output_dimension": output_dimension,
        "truncate": "END",
    }


def _extract_floats(resp_json: dict[str, Any]) -> list[list[float]]:
    """Pull the 2048-dim ``float`` arrays out of a V2 ``/embed`` response.

    V2 schema (confirmed on Azure 2026-10-27):
        { "embeddings": { "float": [[...], [...]] }, "id": "...", ... }

    V1 payloads landed as a top-level list; we accept both so a stray V1 route
    does not silently drop vectors.
    """
    embeddings = resp_json.get("embeddings")
    if isinstance(embeddings, dict):
        floats = embeddings.get("float")
        if isinstance(floats, list):
            return [list(map(float, v)) for v in floats]
    if isinstance(embeddings, list):
        return [list(map(float, v)) for v in embeddings]
    raise EmbeddingError(
        f"unexpected /embed response shape: keys={list(resp_json)[:5]}"
    )


def _sleep_backoff(attempt: int, retry_after_header: str | None) -> float:
    if retry_after_header:
        try:
            return max(0.1, float(retry_after_header))
        except ValueError:
            pass
    base = min(8.0, 0.5 * (2**attempt))
    return base + random.uniform(0, 0.25)


def _request_sync(payload: dict[str, Any], *, timeout: float, retries: int) -> dict:
    url = _endpoint()
    headers = _headers()
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, headers=headers, json=payload)
            if r.status_code == 200:
                return r.json()
            if r.status_code in _RETRYABLE_STATUS and attempt < retries:
                delay = _sleep_backoff(attempt, r.headers.get("Retry-After"))
                logger.warning(
                    "cohere embed %d, retry %d/%d in %.2fs",
                    r.status_code,
                    attempt + 1,
                    retries,
                    delay,
                )
                time.sleep(delay)
                continue
            body = r.text[:500]
            raise EmbeddingError(f"HTTP {r.status_code}: {body}")
        except httpx.HTTPError as e:
            last_err = e
            if attempt >= retries:
                break
            delay = _sleep_backoff(attempt, None)
            logger.warning(
                "cohere embed transport error, retry %d/%d in %.2fs: %s",
                attempt + 1,
                retries,
                delay,
                e,
            )
            time.sleep(delay)
    raise EmbeddingError(f"transport failure after {retries + 1} attempts: {last_err}")


async def _request_async(
    payload: dict[str, Any], *, timeout: float, retries: int
) -> dict:
    import asyncio

    url = _endpoint()
    headers = _headers()
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, headers=headers, json=payload)
            if r.status_code == 200:
                return r.json()
            if r.status_code in _RETRYABLE_STATUS and attempt < retries:
                delay = _sleep_backoff(attempt, r.headers.get("Retry-After"))
                logger.warning(
                    "cohere embed %d, retry %d/%d in %.2fs (async)",
                    r.status_code,
                    attempt + 1,
                    retries,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            body = r.text[:500]
            raise EmbeddingError(f"HTTP {r.status_code}: {body}")
        except httpx.HTTPError as e:
            last_err = e
            if attempt >= retries:
                break
            delay = _sleep_backoff(attempt, None)
            logger.warning(
                "cohere embed transport error, retry %d/%d in %.2fs (async): %s",
                attempt + 1,
                retries,
                delay,
                e,
            )
            await asyncio.sleep(delay)
    raise EmbeddingError(f"transport failure after {retries + 1} attempts: {last_err}")


def _resolve(output_dimension: int | None, model: str | None) -> tuple[int, str]:
    s = get_settings()
    dim = _validate_dim(output_dimension if output_dimension is not None else s.embedding_dimensions)
    name = model or s.embedding_model
    return dim, name


def embed_batch(
    texts: Sequence[str],
    input_type: InputType,
    *,
    output_dimension: int | None = None,
    model: str | None = None,
    timeout: float = 60.0,
    retries: int = 3,
) -> list[list[float]]:
    """Embed up to ``MAX_BATCH`` texts in one call. Lower-level than
    ``embed_documents`` / ``embed_query`` — callers already batching should
    use this; everyone else should use the ``_documents`` / ``_query`` helpers
    which chunk automatically.
    """
    coerced = _coerce_texts(texts)
    if len(coerced) > MAX_BATCH:
        raise EmbeddingError(
            f"embed_batch cap is {MAX_BATCH} texts per call; got {len(coerced)}"
        )
    dim, name = _resolve(output_dimension, model)
    payload = _payload(coerced, input_type, dim, name)
    data = _request_sync(payload, timeout=timeout, retries=retries)
    vecs = _extract_floats(data)
    if len(vecs) != len(coerced):
        raise EmbeddingError(
            f"vector count mismatch: sent {len(coerced)}, got {len(vecs)}"
        )
    for v in vecs:
        if len(v) != dim:
            raise EmbeddingError(
                f"vector dim mismatch: expected {dim}, got {len(v)}"
            )
    return vecs


def embed_documents(
    texts: Sequence[str],
    *,
    output_dimension: int | None = None,
    model: str | None = None,
    timeout: float = 60.0,
    retries: int = 3,
) -> list[list[float]]:
    """Embed chunks for indexing (``input_type=search_document``).

    Transparently batches at ``MAX_BATCH=96`` so a caller can hand in the
    whole chunk list for a PDF; order is preserved across batches.
    """
    coerced = _coerce_texts(texts)
    out: list[list[float]] = []
    for batch in _batches(coerced):
        vecs = embed_batch(
            batch,
            "search_document",
            output_dimension=output_dimension,
            model=model,
            timeout=timeout,
            retries=retries,
        )
        out.extend(vecs)
    return out


def embed_query(
    text: str,
    *,
    output_dimension: int | None = None,
    model: str | None = None,
    timeout: float = 30.0,
    retries: int = 3,
) -> list[float]:
    """Embed a single search query (``input_type=search_query``)."""
    vecs = embed_batch(
        [text],
        "search_query",
        output_dimension=output_dimension,
        model=model,
        timeout=timeout,
        retries=retries,
    )
    return vecs[0]


async def aembed_documents(
    texts: Sequence[str],
    *,
    output_dimension: int | None = None,
    model: str | None = None,
    timeout: float = 60.0,
    retries: int = 3,
) -> list[list[float]]:
    """Async variant of :func:`embed_documents` using ``httpx.AsyncClient``."""
    coerced = _coerce_texts(texts)
    dim, name = _resolve(output_dimension, model)
    out: list[list[float]] = []
    for batch in _batches(coerced):
        payload = _payload(batch, "search_document", dim, name)
        data = await _request_async(payload, timeout=timeout, retries=retries)
        vecs = _extract_floats(data)
        if len(vecs) != len(batch):
            raise EmbeddingError(
                f"vector count mismatch: sent {len(batch)}, got {len(vecs)}"
            )
        for v in vecs:
            if len(v) != dim:
                raise EmbeddingError(
                    f"vector dim mismatch: expected {dim}, got {len(v)}"
                )
        out.extend(vecs)
    return out


async def aembed_query(
    text: str,
    *,
    output_dimension: int | None = None,
    dimension: int | None = None,
    model: str | None = None,
    timeout: float = 30.0,
    retries: int = 3,
) -> list[float]:
    """Async variant of :func:`embed_query`.

    Accepts ``dimension=`` as an alias for ``output_dimension=`` so the T-010
    hybrid retriever can keep its documented convention
    (``async embed(text, *, dimension=None)``) without a param-name mismatch.
    """
    if output_dimension is None and dimension is not None:
        output_dimension = dimension
    dim, name = _resolve(output_dimension, model)
    payload = _payload(_coerce_texts([text]), "search_query", dim, name)
    data = await _request_async(payload, timeout=timeout, retries=retries)
    vecs = _extract_floats(data)
    if not vecs or len(vecs[0]) != dim:
        raise EmbeddingError(
            f"query embedding dim mismatch: expected {dim}, got {len(vecs[0]) if vecs else 0}"
        )
    return vecs[0]


# ---------------------------------------------------------------------------
# Retriever compat shim
# ---------------------------------------------------------------------------
#
# ``app.rag.retriever`` resolves its default query embedder as
# ``_embed_mod.embed_query`` and then ``await``\ s it with ``dimension=…``
# (plan: T-010 hybrid search). The T-007 ``embed_query`` is intentionally
# sync and uses ``output_dimension=``, so the production default would raise
# a TypeError. Expose an async, retriever-shaped alias so the default path
# just works; tests continue to pass their own fakes through
# ``embed_query_fn``.


async def embed_query_async(
    text: str,
    *,
    dimension: int | None = None,
    output_dimension: int | None = None,
    model: str | None = None,
    timeout: float = 30.0,
    retries: int = 3,
) -> list[float]:
    """Async query embedder shaped for :class:`app.rag.retriever.RAGRetriever`.

    Signature matches the retriever's documented injection contract
    ``async (text, *, dimension=None) -> list[float]`` so swapping the
    retriever default from ``embed_query`` to ``embed_query_async`` is a
    one-line change.
    """
    return await aembed_query(
        text,
        dimension=dimension,
        output_dimension=output_dimension,
        model=model,
        timeout=timeout,
        retries=retries,
    )


__all__ = [
    "EmbeddingError",
    "InputType",
    "MAX_BATCH",
    "ALLOWED_DIMENSIONS",
    "embed_batch",
    "embed_documents",
    "embed_query",
    "aembed_documents",
    "aembed_query",
    "embed_query_async",
]
