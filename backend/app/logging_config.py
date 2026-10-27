"""Process-wide logging: console plus rotating files under ``backend/logs/``.

Files (directory from ``LOG_DIR``, default ``<backend>/logs``):

    app.log     every record at ``LOG_LEVEL`` (default INFO), one JSON object per line
    error.log   WARNING and above only — the operator's first stop on a 5xx
    audit.log   the security audit trail from ``app.services.audit`` (its own
                handler; we only default ``SEVENLAB_AUDIT_LOG_PATH`` into this dir)

Every record carries ``request_id``, read from the contextvar that
``RequestIdMiddleware`` sets per request, so the ``request_id`` in a frontend
error envelope joins directly to the backend line that produced it.

Rotation: ``LOG_MAX_MB`` (default 10) × ``LOG_BACKUPS`` (default 5) per file.
Under ``uvicorn --workers N`` every worker appends to the same files; single
lines stay intact (O_APPEND), rotation may race by one file. Point ``LOG_DIR``
at a per-worker path if that ever matters.
"""
from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from pathlib import Path

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
APP_LOG_NAME = "app.log"
ERROR_LOG_NAME = "error.log"
AUDIT_LOG_NAME = "audit.log"

_MARKER = "_sevenlab_handler"
_CONSOLE_FMT = "%(asctime)s %(levelname)s %(name)s rid=%(request_id)s - %(message)s"
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.access")


class RequestIdFilter(logging.Filter):
    """Stamp ``record.request_id`` from the current request context."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True


def _json_formatter() -> logging.Formatter:
    """python-json-logger when installed (it is pinned), plain text otherwise."""
    fmt = "%(asctime)s %(levelname)s %(name)s %(request_id)s %(message)s"
    rename = {"asctime": "ts", "levelname": "level", "name": "logger"}
    try:
        from pythonjsonlogger.json import JsonFormatter  # python-json-logger >= 3
    except Exception:  # noqa: BLE001
        try:
            from pythonjsonlogger.jsonlogger import JsonFormatter  # 2.x
        except Exception:  # noqa: BLE001
            return logging.Formatter(_CONSOLE_FMT)
    try:
        return JsonFormatter(fmt, rename_fields=rename)
    except TypeError:  # very old versions without rename_fields
        return JsonFormatter(fmt)


def _tagged(handler: logging.Handler, tag: str) -> logging.Handler:
    setattr(handler, _MARKER, tag)
    handler.addFilter(RequestIdFilter())
    return handler


def _drop_tagged(logger: logging.Logger, tags: tuple[str, ...]) -> None:
    for h in list(logger.handlers):
        if getattr(h, _MARKER, None) in tags:
            logger.removeHandler(h)
            h.close()


def configure_logging(
    log_dir: str | os.PathLike[str] | None = None,
    level: str | int | None = None,
) -> Path | None:
    """Attach console + file handlers. Idempotent; returns the log dir or None.

    Returns ``None`` when the directory cannot be created, in which case the
    console handler is still active so nothing is lost on stderr.
    """
    root = logging.getLogger()
    root.setLevel(level or os.environ.get("LOG_LEVEL", "INFO").upper())

    if not any(getattr(h, _MARKER, None) == "console" for h in root.handlers):
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(logging.Formatter(_CONSOLE_FMT))
        root.addHandler(_tagged(console, "console"))

    target = Path(log_dir or os.environ.get("LOG_DIR") or DEFAULT_LOG_DIR)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        root.warning("log dir %s not writable (%s); file logging disabled", target, exc)
        return None

    max_bytes = int(os.environ.get("LOG_MAX_MB", "10")) * 1024 * 1024
    backups = int(os.environ.get("LOG_BACKUPS", "5"))
    formatter = _json_formatter()

    file_handlers: list[logging.Handler] = []
    for name, min_level, tag in (
        (APP_LOG_NAME, logging.NOTSET, "app"),
        (ERROR_LOG_NAME, logging.WARNING, "error"),
    ):
        try:
            fh = RotatingFileHandler(
                target / name,
                maxBytes=max_bytes,
                backupCount=backups,
                encoding="utf-8",
            )
        except OSError as exc:
            root.warning("cannot open %s (%s); skipping", target / name, exc)
            continue
        fh.setLevel(min_level)
        fh.setFormatter(formatter)
        file_handlers.append(_tagged(fh, tag))

    _drop_tagged(root, ("app", "error"))
    for fh in file_handlers:
        root.addHandler(fh)

    # uvicorn's default dictConfig gives these loggers their own console
    # handlers with propagate=False, so they never reach root. Attach the file
    # handlers directly instead of flipping propagate (which would double the
    # console output).
    for name in _UVICORN_LOGGERS:
        lg = logging.getLogger(name)
        _drop_tagged(lg, ("app", "error"))
        for fh in file_handlers:
            lg.addHandler(fh)

    # Route the audit trail into the same directory unless the operator
    # already chose a path. audit.py reads this on first emit.
    os.environ.setdefault("SEVENLAB_AUDIT_LOG_PATH", str(target / AUDIT_LOG_NAME))

    return target


def teardown_logging() -> None:
    """Detach every handler this module installed (tests, re-configuration)."""
    tags = ("console", "app", "error")
    _drop_tagged(logging.getLogger(), tags)
    for name in _UVICORN_LOGGERS:
        _drop_tagged(logging.getLogger(name), tags)
