#!/usr/bin/env bash
# Build the pdf-worker sandbox image (ISO 27001 A.8.7).
#
# Run from the repo root (where backend/ lives):
#   bash backend/scripts/build_pdf_worker.sh
#
# The Dockerfile COPYs backend/app/services/pdf_* and backend/pdf_worker.py,
# so the build context must be the repo root, not backend/.
set -euo pipefail

IMAGE="${PDF_SANDBOX_IMAGE:-sevenlab-pdf-worker:latest}"
DOCKERFILE="backend/Dockerfile.pdf-worker"

if [[ ! -f "${DOCKERFILE}" ]]; then
    echo "error: ${DOCKERFILE} not found — run from the repo root" >&2
    exit 1
fi

echo "building ${IMAGE} from ${DOCKERFILE}..."
docker build -f "${DOCKERFILE}" -t "${IMAGE}" .

echo "verifying worker contract (empty input must error cleanly)..."
set +e
echo -n "" | docker run --rm -i \
    --network=none --read-only \
    --tmpfs /tmp:rw,noexec,nosuid,size=64m \
    --user 65534:65534 --pids-limit 100 \
    "${IMAGE}" > /tmp/pdf-worker-smoke.json
rc=$?
set -e

if [[ "${rc}" -ne 2 ]]; then
    echo "error: smoke test expected rc=2 for empty input, got rc=${rc}" >&2
    cat /tmp/pdf-worker-smoke.json >&2 || true
    exit 1
fi

if ! grep -q '"error"' /tmp/pdf-worker-smoke.json; then
    echo "error: smoke test stdout missing error field" >&2
    cat /tmp/pdf-worker-smoke.json >&2
    exit 1
fi

echo "ok: ${IMAGE} built and smoke test passed"
