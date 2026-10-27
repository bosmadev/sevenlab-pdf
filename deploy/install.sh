#!/usr/bin/env bash
#
# install.sh — one-shot deployment of sevenlab-pdf to /opt/sevenlab.
#
# What it does:
#   1. rsync the git tree (minus node_modules/.venv/.next) into /opt/sevenlab/.
#   2. (Re)build the Python venv and install backend requirements.
#   3. npm ci + npm run build in the frontend.
#   4. Install the two systemd units and the Caddy snippet.
#   5. systemctl daemon-reload, enable + restart both services, reload Caddy.
#
# Idempotent: safe to re-run after a `git pull` on the box.
# Needs: sudo for the /opt/sevenlab writes, /etc/systemd and /etc/caddy copies.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="${REPO_ROOT}/deploy"
INSTALL_ROOT="${INSTALL_ROOT:-/opt/sevenlab}"
SERVICE_USER="${SERVICE_USER:-dennis}"
SERVICE_GROUP="${SERVICE_GROUP:-dennis}"

say() { printf '\n== %s ==\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1"; exit 1; }; }

need rsync
need python3
need npm
need systemctl

if [[ ! -d "${DEPLOY_DIR}" ]]; then
    echo "deploy dir not found: ${DEPLOY_DIR}" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# 1. Sync source tree
# -----------------------------------------------------------------------------
say "Syncing ${REPO_ROOT} -> ${INSTALL_ROOT}"
sudo mkdir -p "${INSTALL_ROOT}"
sudo chown "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_ROOT}"

# Exclude build/VCS cruft. Preserve the server-side .env files (never clobber).
sudo -u "${SERVICE_USER}" rsync -a --delete \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='node_modules/' \
    --exclude='.next/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    --exclude='*.pyc' \
    --exclude='backend/.env' \
    --exclude='frontend/.env' \
    --exclude='backend/logs/' \
    "${REPO_ROOT}/" "${INSTALL_ROOT}/"

# Backend logs dir — service unit has it under ReadWritePaths.
sudo -u "${SERVICE_USER}" mkdir -p "${INSTALL_ROOT}/backend/logs"

# .env must exist and be 0600; refuse to continue if missing.
for f in backend/.env frontend/.env; do
    if [[ ! -f "${INSTALL_ROOT}/${f}" ]]; then
        echo "Missing ${INSTALL_ROOT}/${f}" >&2
        echo "Create it from ${f}.sample before re-running install.sh." >&2
        exit 1
    fi
    sudo chmod 600 "${INSTALL_ROOT}/${f}"
    sudo chown "${SERVICE_USER}:${SERVICE_GROUP}" "${INSTALL_ROOT}/${f}"
done

# -----------------------------------------------------------------------------
# 2. Backend venv
# -----------------------------------------------------------------------------
say "Rebuilding backend venv"
cd "${INSTALL_ROOT}/backend"
if [[ ! -d .venv ]]; then
    sudo -u "${SERVICE_USER}" python3 -m venv .venv
fi
sudo -u "${SERVICE_USER}" .venv/bin/pip install --upgrade pip wheel >/dev/null
# CPU-only torch first: the box has no GPU and the default wheel pulls ~4 GB of
# CUDA libs. sentence-transformers accepts any torch, so requirements.txt keeps it.
sudo -u "${SERVICE_USER}" .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
sudo -u "${SERVICE_USER}" .venv/bin/pip install -r requirements.txt

# Pre-fetch the reranker (BAAI/bge-reranker-v2-m3, ~2.2 GB) into HF_HOME so the
# first chat does not block on a download. The unit sets HF_HOME to this path;
# it lives outside the rsync tree so --delete never wipes it.
HF_CACHE=/var/cache/sevenlab/huggingface
sudo mkdir -p "${HF_CACHE}"
sudo chown -R "${SERVICE_USER}:${SERVICE_GROUP}" /var/cache/sevenlab
sudo -u "${SERVICE_USER}" env HF_HOME="${HF_CACHE}" .venv/bin/python -c \
    "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3', allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'])"

# -----------------------------------------------------------------------------
# 3. Frontend build
# -----------------------------------------------------------------------------
say "Building Next.js frontend"
cd "${INSTALL_ROOT}/frontend"
sudo -u "${SERVICE_USER}" npm ci --prefer-offline --no-audit --fund=false
sudo -u "${SERVICE_USER}" npm run build

# -----------------------------------------------------------------------------
# 3b. PDF sandbox image (PDF_SANDBOX_MODE=docker in backend/.env)
# -----------------------------------------------------------------------------
say "Building sevenlab-pdf-worker image"
need docker
cd "${INSTALL_ROOT}"
sudo -u "${SERVICE_USER}" bash backend/scripts/build_pdf_worker.sh

# -----------------------------------------------------------------------------
# 4. systemd units
# -----------------------------------------------------------------------------
say "Installing systemd units"
sudo install -m 0644 "${DEPLOY_DIR}/sevenlab-backend.service"  /etc/systemd/system/
sudo install -m 0644 "${DEPLOY_DIR}/sevenlab-frontend.service" /etc/systemd/system/
sudo systemctl daemon-reload

# -----------------------------------------------------------------------------
# 5. Caddy snippet
# -----------------------------------------------------------------------------
say "Installing Caddy snippet"
sudo mkdir -p /etc/caddy/conf.d
sudo install -m 0644 "${DEPLOY_DIR}/Caddyfile.sevenlab" /etc/caddy/conf.d/sevenlab.caddy

# `caddy validate` as root opens the log file with O_CREATE, leaving it root:root 0600,
# and the reload (User=caddy) then fails to open it. Pre-create it caddy-owned.
if [[ ! -e /var/log/caddy/sevenlab.log ]]; then
    sudo install -o caddy -g caddy -m 0640 /dev/null /var/log/caddy/sevenlab.log
fi

# The snippet only loads if the main Caddyfile imports conf.d — fail loudly, not silently.
if ! grep -qE '^\s*import\s+conf\.d/\*\.caddy' /etc/caddy/Caddyfile; then
    echo "WARNING: /etc/caddy/Caddyfile has no 'import conf.d/*.caddy' line — sevenlab vhost will NOT load." >&2
    echo "Append it:  echo 'import conf.d/*.caddy' | sudo tee -a /etc/caddy/Caddyfile" >&2
fi

# Only reload Caddy if the config validates — a bad snippet would 502 the whole box.
if sudo caddy validate --config /etc/caddy/Caddyfile >/dev/null 2>&1; then
    sudo systemctl reload caddy
    echo "caddy reloaded"
else
    echo "caddy validate failed — snippet copied but Caddy NOT reloaded" >&2
    echo "Run: sudo caddy validate --config /etc/caddy/Caddyfile" >&2
fi

# -----------------------------------------------------------------------------
# 6. Enable + start services
# -----------------------------------------------------------------------------
say "Enabling + starting services"
sudo systemctl enable sevenlab-backend.service sevenlab-frontend.service
sudo systemctl restart sevenlab-backend.service
sudo systemctl restart sevenlab-frontend.service

say "Status"
sudo systemctl --no-pager --lines=10 status sevenlab-backend.service || true
sudo systemctl --no-pager --lines=10 status sevenlab-frontend.service || true

cat <<EOF

Done. Verify from the box:
  curl -s http://127.0.0.1:8001/api/health/status
  curl -s http://127.0.0.1:3006/

Public: https://sevenlab.bosma.dev
Logs:   journalctl -u sevenlab-backend.service -f
        journalctl -u sevenlab-frontend.service -f
EOF
