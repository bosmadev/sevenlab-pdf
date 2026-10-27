"""Dict-backed stand-in for the Mongo ``chats`` collection.

Implements only the pymongo surface ``app.services.chat_history`` touches:
``create_index``, ``insert_one``, ``count_documents``, ``find`` (with the
``sort/skip/limit`` cursor chain), ``find_one`` and ``delete_many``.
"""
from __future__ import annotations

from typing import Any

from bson import ObjectId


class _Cursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def sort(self, *_: Any, **__: Any) -> "_Cursor":
        return self

    def skip(self, n: int) -> "_Cursor":
        self._rows = self._rows[n:]
        return self

    def limit(self, n: int) -> "_Cursor":
        self._rows = self._rows[:n]
        return self

    def __iter__(self):
        return iter(self._rows)


class FakeChatsCollection:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    @staticmethod
    def _match(row: dict[str, Any], query: dict[str, Any]) -> bool:
        return all(row.get(k) == v for k, v in query.items())

    def create_index(self, *_: Any, **__: Any) -> None:
        return None

    def insert_one(self, row: dict[str, Any]):
        row["_id"] = ObjectId()
        self.rows.append(dict(row))
        return type("R", (), {"inserted_id": row["_id"]})

    def count_documents(self, query: dict[str, Any]) -> int:
        return sum(1 for r in self.rows if self._match(r, query))

    def find(self, query: dict[str, Any]) -> _Cursor:
        matched = [r for r in self.rows if self._match(r, query)]
        matched.sort(key=lambda r: r.get("created_at"))
        return _Cursor(matched)

    def find_one(self, query: dict[str, Any], sort: Any = None):
        matched = [r for r in self.rows if self._match(r, query)]
        return matched[0] if matched else None

    def delete_many(self, query: dict[str, Any]):
        before = len(self.rows)
        self.rows = [r for r in self.rows if not self._match(r, query)]
        return type("R", (), {"deleted_count": before - len(self.rows)})


__all__ = ["FakeChatsCollection"]
