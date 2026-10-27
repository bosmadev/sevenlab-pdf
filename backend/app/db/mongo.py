"""Legacy import shim — the real MongoDB client lives in `app.db.mongodb`.

Early T-003 drafts put a second MongoClient builder here. That was a duplicate
source of truth: a divergent pool size or timeout tweak in one file silently
left the other half of the app on different settings. Everything now routes
through `mongodb.py`; this module exists only so `from app.db.mongo import X`
keeps resolving for callers (notably `app.rag.bm25` and the index creator)
that were authored against the earlier name.

Prefer the full-name module in new code:
    from app.db.mongodb import get_pdfs_collection, get_mongo_db
"""

from __future__ import annotations

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from app.db.mongodb import (
    get_mongo_client,
    get_mongo_db,
    get_pdfs_collection,
)


def mongo_client() -> MongoClient:
    """Deprecated alias for `app.db.mongodb.get_mongo_client`."""
    return get_mongo_client()


def mongo_db() -> Database:
    """Deprecated alias for `app.db.mongodb.get_mongo_db`."""
    return get_mongo_db()


def pdfs_collection() -> Collection:
    """Deprecated alias for `app.db.mongodb.get_pdfs_collection`."""
    return get_pdfs_collection()


__all__ = ["mongo_client", "mongo_db", "pdfs_collection"]
