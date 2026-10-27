-- SevenLab PDF Chat — PostgreSQL schema (reference / psql smoke-test only).
--
-- SOURCE OF TRUTH: backend/alembic/versions/*.py (apply with
-- `cd backend && alembic upgrade head`, or `python -m scripts.init_postgres`).
-- This file mirrors revision 0001 for operators who want to eyeball the DDL
-- or bootstrap a scratch database with plain psql. If you apply it by hand,
-- the final INSERT stamps `alembic_version` so a later `alembic upgrade head`
-- treats the database as already at 0001 instead of re-creating tables.
--
-- Apply:  psql "$POSTGRESQL_URI" -f backend/app/db/schema.sql

CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY,
    workspace_id    VARCHAR(64) NOT NULL,
    password_hash   VARCHAR(255) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL,
    last_login_at   TIMESTAMPTZ,
    is_active       BOOLEAN NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_users_workspace_id ON users (workspace_id);

CREATE TABLE IF NOT EXISTS documents (
    id                  UUID PRIMARY KEY,
    user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id        VARCHAR(64) NOT NULL,
    original_filename   VARCHAR(512) NOT NULL,
    sanitized_filename  VARCHAR(512) NOT NULL,
    gridfs_file_id      VARCHAR(64) NOT NULL,
    sha256              VARCHAR(64) NOT NULL,
    size_bytes          BIGINT NOT NULL,
    mime_type           VARCHAR(128) NOT NULL,
    page_count          INTEGER,
    status              VARCHAR(32) NOT NULL,
    status_message      TEXT,
    chunk_count         INTEGER NOT NULL,
    ocr_tier_used       VARCHAR(32),
    created_at          TIMESTAMPTZ NOT NULL,
    processed_at        TIMESTAMPTZ,
    deleted_at          TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_documents_workspace_id       ON documents (workspace_id);
CREATE INDEX IF NOT EXISTS ix_documents_sha256             ON documents (sha256);
CREATE INDEX IF NOT EXISTS ix_documents_workspace_status   ON documents (workspace_id, status);
CREATE INDEX IF NOT EXISTS ix_documents_workspace_created  ON documents (workspace_id, created_at);

CREATE TABLE IF NOT EXISTS sessions (
    id              UUID PRIMARY KEY,
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id    VARCHAR(64) NOT NULL,
    token_hash      VARCHAR(128) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL,
    expires_at      TIMESTAMPTZ NOT NULL,
    revoked_at      TIMESTAMPTZ,
    client_ip       VARCHAR(64),
    user_agent      VARCHAR(512)
);

CREATE INDEX        IF NOT EXISTS ix_sessions_workspace_id ON sessions (workspace_id);
CREATE UNIQUE INDEX IF NOT EXISTS ix_sessions_token_hash   ON sessions (token_hash);
CREATE INDEX        IF NOT EXISTS ix_sessions_expires_at   ON sessions (expires_at);

CREATE TABLE IF NOT EXISTS audit_log (
    id              UUID PRIMARY KEY,
    workspace_id    VARCHAR(64),
    event_type      VARCHAR(64) NOT NULL,
    document_id     UUID,
    client_ip       VARCHAR(64),
    sha256          VARCHAR(64),
    size_bytes      BIGINT,
    details         TEXT,
    created_at      TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_audit_log_workspace_id ON audit_log (workspace_id);
CREATE INDEX IF NOT EXISTS ix_audit_log_event_type   ON audit_log (event_type);
CREATE INDEX IF NOT EXISTS ix_audit_log_document_id  ON audit_log (document_id);
CREATE INDEX IF NOT EXISTS ix_audit_log_created_at   ON audit_log (created_at);

-- Alembic bookkeeping: stamp this hand-applied schema as revision 0001.
CREATE TABLE IF NOT EXISTS alembic_version (
    version_num VARCHAR(32) NOT NULL,
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);
INSERT INTO alembic_version (version_num)
SELECT '0001'
WHERE NOT EXISTS (SELECT 1 FROM alembic_version);
