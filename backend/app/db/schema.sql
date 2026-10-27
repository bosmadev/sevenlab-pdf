-- SevenLab PDF Chat — PostgreSQL schema (reference / bootstrap).
-- SQLAlchemy Base.metadata.create_all() is the source of truth; this file
-- mirrors it for `psql` smoke-tests and ops.
--
-- Apply:  psql "$POSTGRESQL_URI" -f backend/app/db/schema.sql

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id    VARCHAR(64) UNIQUE NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at   TIMESTAMPTZ,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS ix_users_workspace_id ON users (workspace_id);

CREATE TABLE IF NOT EXISTS documents (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id        VARCHAR(64) NOT NULL,
    original_filename   VARCHAR(512) NOT NULL,
    sanitized_filename  VARCHAR(512) NOT NULL,
    gridfs_file_id      VARCHAR(64) NOT NULL,
    sha256              VARCHAR(64) NOT NULL,
    size_bytes          BIGINT NOT NULL,
    mime_type           VARCHAR(128) NOT NULL DEFAULT 'application/pdf',
    page_count          INTEGER,
    status              VARCHAR(32) NOT NULL DEFAULT 'pending',
    status_message      TEXT,
    chunk_count         INTEGER NOT NULL DEFAULT 0,
    ocr_tier_used       VARCHAR(32),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at        TIMESTAMPTZ,
    deleted_at          TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_documents_workspace_id       ON documents (workspace_id);
CREATE INDEX IF NOT EXISTS ix_documents_sha256             ON documents (sha256);
CREATE INDEX IF NOT EXISTS ix_documents_workspace_status   ON documents (workspace_id, status);
CREATE INDEX IF NOT EXISTS ix_documents_workspace_created  ON documents (workspace_id, created_at);

CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id    VARCHAR(64) NOT NULL,
    token_hash      VARCHAR(128) UNIQUE NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ NOT NULL,
    revoked_at      TIMESTAMPTZ,
    client_ip       VARCHAR(64),
    user_agent      VARCHAR(512)
);

CREATE INDEX IF NOT EXISTS ix_sessions_workspace_id ON sessions (workspace_id);
CREATE INDEX IF NOT EXISTS ix_sessions_token_hash   ON sessions (token_hash);
CREATE INDEX IF NOT EXISTS ix_sessions_expires_at   ON sessions (expires_at);

CREATE TABLE IF NOT EXISTS audit_log (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workspace_id    VARCHAR(64),
    event_type      VARCHAR(64) NOT NULL,
    document_id     UUID,
    client_ip       VARCHAR(64),
    sha256          VARCHAR(64),
    size_bytes      BIGINT,
    details         TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_audit_log_workspace_id ON audit_log (workspace_id);
CREATE INDEX IF NOT EXISTS ix_audit_log_event_type   ON audit_log (event_type);
CREATE INDEX IF NOT EXISTS ix_audit_log_document_id  ON audit_log (document_id);
CREATE INDEX IF NOT EXISTS ix_audit_log_created_at   ON audit_log (created_at);
