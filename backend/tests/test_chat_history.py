"""T-013 chat-history tests — monkeypatch the Mongo collection.

Live Atlas is not required: ``get_chats_collection`` is swapped for a
dict-backed stub that honours the subset of the API ``chat_history`` uses.
"""
from __future__ import annotations

import threading
from bson import ObjectId

import pytest

from app.services import chat_history


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def sort(self, *_, **__):
        return self

    def skip(self, n):
        self._rows = self._rows[n:]
        return self

    def limit(self, n):
        self._rows = self._rows[:n]
        return self

    def __iter__(self):
        return iter(self._rows)


class _FakeCollection:
    """Minimal pymongo-compatible stub for chat_history tests."""

    def __init__(self):
        self._rows: list[dict] = []
        self._lock = threading.Lock()

    def _match(self, row: dict, query: dict) -> bool:
        return all(row.get(k) == v for k, v in query.items())

    def create_index(self, *_, **__):
        return "noop"

    def insert_one(self, row):
        with self._lock:
            row["_id"] = ObjectId()
            self._rows.append(dict(row))
        return type("R", (), {"inserted_id": row["_id"]})

    def count_documents(self, query):
        return sum(1 for r in self._rows if self._match(r, query))

    def find(self, query):
        matched = [r for r in self._rows if self._match(r, query)]
        matched.sort(key=lambda r: r.get("created_at"))
        return _FakeCursor(matched)

    def find_one(self, query, sort=None):
        matched = [r for r in self._rows if self._match(r, query)]
        if not matched:
            return None
        if sort:
            key, direction = sort[0]
            matched.sort(key=lambda r: r.get(key), reverse=direction == -1)
        return matched[0]

    def delete_many(self, query):
        with self._lock:
            before = len(self._rows)
            self._rows = [r for r in self._rows if not self._match(r, query)]
            return type("R", (), {"deleted_count": before - len(self._rows)})


@pytest.fixture(autouse=True)
def fake_mongo(monkeypatch):
    """Swap the live chats collection for a dict-backed stub."""
    col = _FakeCollection()
    monkeypatch.setattr(chat_history, "get_chats_collection", lambda: col)
    chat_history._INDEXED = False
    yield col


def test_append_and_list_roundtrip(fake_mongo):
    chat_history.append_message(
        workspace_id="alice",
        document_id="doc-1",
        role="user",
        content="hello",
    )
    chat_history.append_message(
        workspace_id="alice",
        document_id="doc-1",
        role="assistant",
        content="hi there",
        citations=[{"page_num": 1, "chunk_id": "c1", "chunk_index": 0}],
    )

    rows, total = chat_history.list_messages(
        workspace_id="alice", document_id="doc-1"
    )
    assert total == 2
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[1]["citations"][0]["page_num"] == 1


def test_list_isolates_by_workspace(fake_mongo):
    chat_history.append_message(
        workspace_id="alice", document_id="doc-1", role="user", content="a"
    )
    chat_history.append_message(
        workspace_id="bob", document_id="doc-1", role="user", content="b"
    )

    rows, total = chat_history.list_messages(
        workspace_id="alice", document_id="doc-1"
    )
    assert total == 1
    assert rows[0]["content"] == "a"


def test_admin_sees_every_workspace(fake_mongo):
    chat_history.append_message(
        workspace_id="alice", document_id="doc-x", role="user", content="a"
    )
    chat_history.append_message(
        workspace_id="bob", document_id="doc-x", role="user", content="b"
    )
    rows, total = chat_history.list_messages(
        workspace_id="admin", document_id="doc-x", is_admin=True
    )
    assert total == 2


def test_delete_history_wipes_only_target_doc(fake_mongo):
    chat_history.append_message(
        workspace_id="alice", document_id="doc-1", role="user", content="a"
    )
    chat_history.append_message(
        workspace_id="alice", document_id="doc-2", role="user", content="b"
    )

    removed = chat_history.delete_history(workspace_id="alice", document_id="doc-1")
    assert removed == 1
    rows, total = chat_history.list_messages(
        workspace_id="alice", document_id="doc-2"
    )
    assert total == 1


def test_latest_message_returns_newest(fake_mongo):
    import time

    chat_history.append_message(
        workspace_id="alice", document_id="doc-1", role="user", content="first"
    )
    time.sleep(0.001)
    chat_history.append_message(
        workspace_id="alice", document_id="doc-1", role="assistant", content="latest"
    )

    latest = chat_history.latest_message(workspace_id="alice", document_id="doc-1")
    assert latest is not None
    assert latest["content"] == "latest"
