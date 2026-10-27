"""Pytest shared fixtures.

UUID round-tripping on SQLite is handled by `app.db.types.UUIDType`, so no
sqlite3 adapter registration is needed here.

The PDF sandbox (`app.services.pdf_sandbox`) defaults to Docker whenever the
binary is on PATH. Dev and CI boxes rarely have the `sevenlab-pdf-worker`
image built, so the suite pins inline mode unless the operator or a test sets
`PDF_SANDBOX_MODE` explicitly. Docker-mode tests patch `subprocess.run`.
"""
from __future__ import annotations

import os

os.environ.setdefault("PDF_SANDBOX_MODE", "inline")
