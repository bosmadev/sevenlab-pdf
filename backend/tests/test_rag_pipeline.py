"""Unit tests for T-012 RAG pipeline — no network calls.

Covers:
    * answer_stream yields citations frame, token frames, done frame
    * citations frame is sent BEFORE any token (UI needs them to render links)
    * empty question returns a single error frame
    * missing hybrid_search module surfaces as an error frame, not a traceback
    * context truncation respects rag_max_context_chars
    * SSE frames are well-formed `data: {...}\\n\\n`
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import AsyncIterator

import pytest

from app.rag import pipeline


def _parse(frames: list[str]) -> list[dict]:
    out = []
    for f in frames:
        assert f.startswith("data: ")
        assert f.endswith("\n\n")
        out.append(json.loads(f[6:-2]))
    return out


@pytest.fixture
def fake_hybrid(monkeypatch):
    """Install a stub `app.rag.retriever.hybrid_search` for the test."""
    def install(chunks):
        mod = types.ModuleType("app.rag.retriever")

        def hybrid_search(workspace_id, query, top_k, doc_id=None):  # noqa: ARG001
            return chunks[:top_k]

        mod.hybrid_search = hybrid_search  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "app.rag.retriever", mod)

    return install


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace stream_chat with a deterministic token generator."""
    def install(tokens, raise_exc=None):
        async def _stream(messages):  # noqa: ARG001
            if raise_exc is not None:
                raise raise_exc
            for t in tokens:
                yield t

        monkeypatch.setattr(pipeline, "stream_chat", _stream)

    return install


async def _collect(it: AsyncIterator[str]) -> list[str]:
    return [frame async for frame in it]


@pytest.mark.asyncio
async def test_happy_path_streams_citations_then_tokens_then_done(fake_hybrid, fake_llm):
    fake_hybrid([
        {
            "doc_id": "d1",
            "chunk_id": "c1",
            "chunk_text": "The sky is blue because of Rayleigh scattering.",
            "page_num": 3,
            "filename": "physics.pdf",
            "score": 0.92,
        },
    ])
    fake_llm(["Blue ", "sky ", "[1]."])

    frames = _parse(await _collect(pipeline.answer_stream("ws1", "Why is the sky blue?")))

    assert frames[0]["type"] == "citations"
    assert len(frames[0]["citations"]) == 1
    assert frames[0]["citations"][0]["n"] == 1
    assert frames[0]["citations"][0]["page"] == 3

    tokens = [f for f in frames if f["type"] == "token"]
    assert [t["text"] for t in tokens] == ["Blue ", "sky ", "[1]."]

    assert frames[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_empty_question_returns_single_error_frame(fake_hybrid, fake_llm):
    fake_hybrid([])
    fake_llm([])

    frames = _parse(await _collect(pipeline.answer_stream("ws1", "   ")))
    assert len(frames) == 1
    assert frames[0]["type"] == "error"
    assert "empty" in frames[0]["message"].lower()


@pytest.mark.asyncio
async def test_missing_retriever_surfaces_as_error_frame(monkeypatch, fake_llm):
    fake_llm(["should not stream"])
    # Make sure retriever is NOT installed
    sys.modules.pop("app.rag.retriever", None)

    # Simulate ImportError by shadowing with a module missing the symbol
    bad = types.ModuleType("app.rag.retriever")
    monkeypatch.setitem(sys.modules, "app.rag.retriever", bad)

    frames = _parse(await _collect(pipeline.answer_stream("ws1", "any?")))
    # No tokens — error frame instead
    assert all(f["type"] != "token" for f in frames)
    assert frames[-1]["type"] == "error"


@pytest.mark.asyncio
async def test_context_truncation_drops_tail_chunks(fake_hybrid, fake_llm, monkeypatch):
    big = "x" * 5000
    fake_hybrid([
        {"doc_id": "d1", "chunk_id": f"c{i}", "chunk_text": big,
         "page_num": i, "filename": "big.pdf", "score": 0.5}
        for i in range(10)
    ])
    fake_llm(["ok"])
    # Force a small cap
    from app.config import get_settings
    get_settings.cache_clear()  # type: ignore[attr-defined]
    monkeypatch.setenv("RAG_MAX_CONTEXT_CHARS", "6000")

    frames = _parse(await _collect(pipeline.answer_stream("ws1", "q?")))
    cits = frames[0]["citations"]
    # 5000 chars + header -> at most 1 fits in 6000 cap
    assert len(cits) <= 2


@pytest.mark.asyncio
async def test_citation_filename_resolved_from_postgres(fake_hybrid, fake_llm, monkeypatch):
    """Chunks carry only doc_id; the citation must show the uploaded filename."""
    doc_uuid = "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"
    fake_hybrid([
        {"doc_id": doc_uuid, "chunk_id": "c1", "chunk_text": "ctx", "page_num": 4, "score": 0.5},
    ])
    fake_llm(["ok"])
    seen: dict = {}

    def fake_lookup(doc_ids):
        seen["ids"] = list(doc_ids)
        return {doc_uuid: "Q3-report.pdf"}

    monkeypatch.setattr(pipeline, "_lookup_filenames", fake_lookup)

    frames = _parse(await _collect(pipeline.answer_stream("ws1", "q?")))
    assert seen["ids"] == [doc_uuid]
    assert frames[0]["citations"][0]["filename"] == "Q3-report.pdf"


@pytest.mark.asyncio
async def test_citation_filename_lookup_failure_degrades_to_document(fake_hybrid, fake_llm, monkeypatch):
    doc_uuid = "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"
    fake_hybrid([
        {"doc_id": doc_uuid, "chunk_id": "c1", "chunk_text": "ctx", "page_num": 1, "score": 0.5},
    ])
    fake_llm(["ok"])

    def boom(doc_ids):
        raise RuntimeError("postgres down")

    monkeypatch.setattr(pipeline, "_lookup_filenames", boom)

    frames = _parse(await _collect(pipeline.answer_stream("ws1", "q?")))
    assert frames[0]["citations"][0]["filename"] == "document"
    assert frames[-1]["type"] == "done"


def test_lookup_filenames_skips_non_uuid_ids_without_touching_postgres(monkeypatch):
    """Legacy / test doc ids like "d1" must short-circuit before SessionLocal."""
    import app.db.postgres as pg

    def must_not_be_called():
        raise AssertionError("SessionLocal must not be touched for non-UUID ids")

    monkeypatch.setattr(pg, "SessionLocal", must_not_be_called)
    assert pipeline._lookup_filenames(["d1", "", "not-a-uuid"]) == {}


@pytest.mark.asyncio
async def test_doc_id_is_forwarded_to_retriever(fake_llm, monkeypatch):
    seen: dict = {}
    mod = types.ModuleType("app.rag.retriever")

    def hybrid_search(workspace_id, query, top_k, doc_id=None):  # noqa: ARG001
        seen["doc_id"] = doc_id
        return []

    mod.hybrid_search = hybrid_search  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "app.rag.retriever", mod)
    fake_llm(["ok"])

    await _collect(pipeline.answer_stream("ws1", "q?", doc_id="doc-42"))
    assert seen["doc_id"] == "doc-42"


@pytest.mark.asyncio
async def test_llm_error_becomes_error_frame_not_exception(fake_hybrid, fake_llm):
    fake_hybrid([
        {"doc_id": "d", "chunk_id": "c", "chunk_text": "ctx", "page_num": 1,
         "filename": "f.pdf", "score": 0.1},
    ])
    fake_llm([], raise_exc=pipeline.LLMError("boom"))

    frames = _parse(await _collect(pipeline.answer_stream("ws1", "q?")))
    assert frames[0]["type"] == "citations"
    assert frames[-1]["type"] == "error"
    assert "boom" in frames[-1]["message"]
