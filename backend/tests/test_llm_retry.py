"""Unit tests for the LLM retry policy (T-012 hardening).

Uses httpx MockTransport to simulate Azure OpenAI returning 429 → 429 → 200,
plus a terminal 400 that must NOT be retried. No real network.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Iterator

import httpx
import pytest

os.environ.setdefault("AZURE_AI_API_KEY", "test-key")

from app.rag import llm


SSE_BODY = (
    'data: {"choices":[{"delta":{"content":"Hi "}}]}\n\n'
    'data: {"choices":[{"delta":{"content":"there."}}]}\n\n'
    'data: [DONE]\n\n'
)


_REAL_ASYNC_CLIENT = httpx.AsyncClient  # capture before any monkeypatching


def _mock_client(responses: Iterator[httpx.Response]) -> httpx.AsyncClient:
    """Build an httpx.AsyncClient driven by a canned response sequence."""
    def handler(request: httpx.Request) -> httpx.Response:
        return next(responses)
    return _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler))


async def _collect(it: AsyncIterator[str]) -> list[str]:
    return [t async for t in it]


@pytest.mark.asyncio
async def test_retries_on_429_then_succeeds(monkeypatch):
    """429 twice, then 200 streaming body — stream_chat yields the tokens."""
    sleeps: list[float] = []

    async def no_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(llm.asyncio, "sleep", no_sleep)

    responses = iter([
        httpx.Response(429, headers={"Retry-After": "0"}, text="rate limited"),
        httpx.Response(429, headers={"Retry-After": "0"}, text="still limited"),
        httpx.Response(200, text=SSE_BODY, headers={"content-type": "text/event-stream"}),
    ])

    def fake_async_client(**kwargs):  # noqa: ARG001
        return _mock_client(responses)

    monkeypatch.setattr(llm.httpx, "AsyncClient", fake_async_client)

    tokens = await _collect(llm.stream_chat([{"role": "user", "content": "hi"}]))
    assert tokens == ["Hi ", "there."]
    assert len(sleeps) == 2  # slept twice between the 3 attempts


@pytest.mark.asyncio
async def test_400_is_not_retried(monkeypatch):
    """A 400 Bad Request is terminal — stream_chat raises LLMError immediately."""
    sleeps: list[float] = []

    async def no_sleep(s):
        sleeps.append(s)

    monkeypatch.setattr(llm.asyncio, "sleep", no_sleep)

    responses = iter([
        httpx.Response(400, text='{"error":{"message":"bad prompt"}}'),
    ])

    def fake_async_client(**kwargs):  # noqa: ARG001
        return _mock_client(responses)

    monkeypatch.setattr(llm.httpx, "AsyncClient", fake_async_client)

    with pytest.raises(llm.LLMError) as exc:
        await _collect(llm.stream_chat([{"role": "user", "content": "hi"}]))
    assert "400" in str(exc.value)
    assert sleeps == []  # never slept — not retried


@pytest.mark.asyncio
async def test_retry_exhaustion_raises(monkeypatch):
    """Three consecutive 503s exhaust the retry budget and raise LLMError."""
    async def no_sleep(s):  # noqa: ARG001
        pass
    monkeypatch.setattr(llm.asyncio, "sleep", no_sleep)

    responses = iter([
        httpx.Response(503, text="down"),
        httpx.Response(503, text="down"),
        httpx.Response(503, text="down"),
    ])
    monkeypatch.setattr(llm.httpx, "AsyncClient", lambda **k: _mock_client(responses))

    with pytest.raises(llm.LLMError) as exc:
        await _collect(llm.stream_chat([{"role": "user", "content": "hi"}]))
    assert "503" in str(exc.value)


def test_backoff_honours_retry_after():
    assert llm._backoff(1, 2.5) == 2.5
    # cap at 8.0
    assert llm._backoff(1, 1000.0) == 8.0


def test_backoff_grows_without_retry_after():
    a = llm._backoff(1, None)
    b = llm._backoff(3, None)
    assert 0.75 <= a <= 1.25
    assert 3.0 <= b <= 5.0  # 4 * [0.75..1.25]
