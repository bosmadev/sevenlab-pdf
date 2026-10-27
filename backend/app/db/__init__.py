"""Database layer: PostgreSQL (metadata) + MongoDB Atlas (PDFs + vectors)."""

from app.db.mongodb import (
    MongoDB,
    get_gridfs,
    get_mongo_client,
    get_mongo_db,
    get_pdfs_collection,
)
from app.db.postgres import Base, SessionLocal, engine, get_db
from app.db.models import Document, SessionToken, User

__all__ = [
    "Base",
    "Document",
    "MongoDB",
    "SessionLocal",
    "SessionToken",
    "User",
    "engine",
    "get_db",
    "get_gridfs",
    "get_mongo_client",
    "get_mongo_db",
    "get_pdfs_collection",
]
