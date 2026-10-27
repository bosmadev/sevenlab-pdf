"""Operator entry point for provisioning MongoDB Atlas indexes.

Thin wrapper around `app.db.indices.ensure_all_indexes` so there is a stable
`python -m scripts.create_search_indexes` command alongside the module form
(`python -m app.db.indices`). All index logic lives in `app.db.indices`; this
file deliberately holds no definitions so the single source of truth cannot
drift. See T-009 (BM25) and T-008 (vector) for background.

Usage:
    cd backend && python -m scripts.create_search_indexes          # fire-and-forget
    cd backend && python -m scripts.create_search_indexes --wait   # block until READY
"""

from __future__ import annotations

import json
import logging
import os
import sys


def _ensure_backend_on_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.dirname(here)
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)


def main() -> int:
    _ensure_backend_on_path()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    from app.db.indices import ensure_all_indexes  # imported lazily so --help works offline

    try:
        summary = ensure_all_indexes(wait="--wait" in sys.argv)
    except Exception as exc:
        logging.exception("index creation failed")
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1

    print(json.dumps(summary, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
