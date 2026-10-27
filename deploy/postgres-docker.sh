#!/usr/bin/env bash
#
# postgres-docker.sh — start (or restart) the local PostgreSQL 16 the backend
# needs for users / documents / sessions / audit_log (app/db/models.py).
#
# Why a container: this box has no system PostgreSQL (checked 2026-10-27:
# nothing listening on 5432, no postgresql.service, no psql). The only
# Postgres present is Nextcloud AIO's private container, which must not be
# shared. A dedicated container bound to 127.0.0.1 costs $0 and keeps data in
# a named Docker volume that survives `docker rm`.
#
# Usage:
#   POSTGRES_PASSWORD='...' bash deploy/postgres-docker.sh
# Then put the matching URI in /opt/sevenlab/backend/.env:
#   POSTGRESQL_URI=postgresql+psycopg://sevenlab:<password>@127.0.0.1:5432/sevenlab_pdf
# and create the schema once:
#   cd /opt/sevenlab/backend && .venv/bin/python -m scripts.init_postgres
#
# Idempotent: re-running starts the existing container; it never recreates
# the volume. Override NAME/VOLUME/PORT/IMAGE/DB/USER via PG_* env vars.

set -euo pipefail

NAME="${PG_CONTAINER:-sevenlab-postgres}"
VOLUME="${PG_VOLUME:-sevenlab-postgres-data}"
PORT="${PG_PORT:-5432}"
IMAGE="${PG_IMAGE:-postgres:16-alpine}"
DB="${PG_DB:-sevenlab_pdf}"
PG_ROLE="${PG_USER:-sevenlab}"

: "${POSTGRES_PASSWORD:?set POSTGRES_PASSWORD to the password used in backend/.env POSTGRESQL_URI}"

command -v docker >/dev/null 2>&1 || { echo "missing: docker" >&2; exit 1; }

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo "container ${NAME} already exists — starting it"
    docker start "${NAME}" >/dev/null
else
    docker volume create "${VOLUME}" >/dev/null
    docker run -d --name "${NAME}" \
        --restart unless-stopped \
        -p "127.0.0.1:${PORT}:5432" \
        -e POSTGRES_DB="${DB}" \
        -e POSTGRES_USER="${PG_ROLE}" \
        -e POSTGRES_PASSWORD="${POSTGRES_PASSWORD}" \
        -v "${VOLUME}:/var/lib/postgresql/data" \
        --memory 1g --pids-limit 200 \
        "${IMAGE}" >/dev/null
    echo "started ${NAME} (${IMAGE}) on 127.0.0.1:${PORT}, data in volume ${VOLUME}"
fi

for _ in $(seq 1 30); do
    if docker exec "${NAME}" pg_isready -U "${PG_ROLE}" -d "${DB}" >/dev/null 2>&1; then
        echo "postgres ready: postgresql+psycopg://${PG_ROLE}:***@127.0.0.1:${PORT}/${DB}"
        exit 0
    fi
    sleep 1
done

echo "postgres did not become ready in 30s — inspect with: docker logs ${NAME}" >&2
exit 1
