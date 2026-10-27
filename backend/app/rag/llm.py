"""Azure OpenAI chat completion client with SSE token streaming (T-012).

Uses the Azure AI endpoint already provisioned for Cohere embeddings:
    POST {AZURE_AI_API_BASE}/openai/deployments/{deployment}/chat/completions
         ?api-version={LLM_API_VERSION}

Headers:
    api-key: <AZURE_AI_API_KEY>
    Authorization: Bearer <AZURE_AI_API_KEY>   # Azure accepts either; send both
    Content-Type: application/json

Streaming: `stream=true` returns SSE lines `data: {...}\n\n`, terminator `data: [DONE]`.
Each delta chunk is yielded as the raw text token (never the control frame).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# Retry budget for retriable upstream errors (429 / 502 / 503 / 504).
# Each attempt waits base * 2^(n-1) seconds, jittered ±25%, capped at `cap`.
_RETRY_ATTEMPTS = 3
_RETRY_BASE_SECONDS = 1.0
_RETRY_CAP_SECONDS = 8.0
_RETRIABLE_STATUSES = {429, 502, 503, 504}


class LLMError(RuntimeError):
    """Raised on any upstream LLM failure the caller should surface to SSE."""


def _backoff(attempt: int, retry_after: float | None) -> float:
    """Backoff for `attempt` (1-based). Honours `Retry-After` when sent."""
    if retry_after is not None and retry_after > 0:
        return min(retry_after, _RETRY_CAP_SECONDS)
    base = _RETRY_BASE_SECONDS * (2 ** (attempt - 1))
    jitter = random.uniform(0.75, 1.25)
    return min(base * jitter, _RETRY_CAP_SECONDS)


def _parse_retry_after(h: str | None) -> float | None:
    if not h:
        return None
    try:
        return float(h)
    except ValueError:
        return None


def _endpoint() -> str:
    s = get_settings()
    base = s.azure_ai_api_base.rstrip("/")
    return (
        f"{base}/openai/deployments/{s.llm_deployment}"
        f"/chat/completions?api-version={s.llm_api_version}"
    )


def _headers() -> dict[str, str]:
    key = get_settings().azure_ai_api_key
    if not key:
        raise LLMError("AZURE_AI_API_KEY missing — set it in backend/.env")
    return {
        "api-key": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }


def _payload(messages: list[dict[str, str]]) -> dict:
    s = get_settings()
    return {
        "messages": messages,
        "max_tokens": s.llm_max_tokens,
        "temperature": s.llm_temperature,
        "stream": True,
    }


async def _open_stream(
    client: httpx.AsyncClient, url: str, payload: dict
) -> httpx.Response:
    """Open the upstream stream with retry on 429/5xx. Returns the response
    once the status is 200. Raises LLMError on terminal failure. The caller
    must enter the response as an async context manager."""
    last_err: str = ""
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        req = client.build_request("POST", url, headers=_headers(), json=payload)
        resp = await client.send(req, stream=True)
        if resp.status_code == 200:
            return resp
        body = (await resp.aread()).decode(errors="replace")[:500]
        await resp.aclose()
        last_err = f"HTTP {resp.status_code}: {body}"
        if resp.status_code not in _RETRIABLE_STATUSES or attempt == _RETRY_ATTEMPTS:
            raise LLMError(f"LLM {last_err}")
        wait = _backoff(attempt, _parse_retry_after(resp.headers.get("Retry-After")))
        logger.warning(
            "llm: retriable %s on attempt %d/%d, sleeping %.2fs",
            resp.status_code, attempt, _RETRY_ATTEMPTS, wait,
        )
        await asyncio.sleep(wait)
    raise LLMError(f"LLM {last_err}")


async def stream_chat(messages: list[dict[str, str]]) -> AsyncIterator[str]:
    """Yield one text delta per iteration until the model signals [DONE].

    Retries 429 / 502 / 503 / 504 up to 3 times with exponential backoff +
    `Retry-After` honouring. Mid-stream disconnects do NOT retry — we have
    already emitted tokens, so a resumed stream would duplicate. Caller
    translates terminal failures into an SSE `error` frame.
    """
    url = _endpoint()
    timeout = httpx.Timeout(get_settings().llm_timeout_seconds, connect=10.0)
    payload = _payload(messages)

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await _open_stream(client, url, payload)
        try:
            async for raw_line in resp.aiter_lines():
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[5:].lstrip()
                if data == "[DONE]":
                    return
                try:
                    frame = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("llm stream: non-JSON frame skipped: %r", data[:120])
                    continue
                choices = frame.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                token = delta.get("content")
                if token:
                    yield token
        finally:
            await resp.aclose()
