"""Database layer: PostgreSQL (metadata) + MongoDB Atlas (PDFs + vectors).

Submodules are imported on demand — importing this package does NOT
pull in SQLAlchemy or pymongo. This matters because the embedding/parsing
subsystems share only the Mongo client and should not force a Postgres
driver install when running MongoDB-only scripts (e.g. `python -m app.db.indices`).

Use the explicit submodule paths:
    from app.db.postgres  import Base, SessionLocal, engine, get_db
    from app.db.models    import User, Document, SessionToken, AuditLog
    from app.db.mongodb   import MongoDB, get_pdfs_collection, get_gridfs
    from app.db.indices   import ensure_all_indexes
"""
