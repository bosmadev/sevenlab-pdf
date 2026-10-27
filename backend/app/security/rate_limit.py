"""In-process sliding-window login throttle.

Guards ``POST /api/auth`` from two concrete attacks on this box:

1. Unlimited bcrypt attempts make every configured password brute-forceable
   given time — there is no lockout anywhere else in the stack.
2. Each attempt costs ~250 ms of CPU at bcrypt cost 12. An attacker looping
   the endpoint starves everything else on the shared uvicorn workers, which
   is a denial-of-service, not just a login nuisance.

Deliberately in-process (no Redis): the app is a single service behind Caddy.
With uvicorn ``--workers 4`` the budget multiplies by worker count (3 tries
per minute x 4 workers = 12 tries before every worker has locked the client
out for 15 min) — acceptable for MVP; swap for an external Redis limiter in
Phase 2 (T-109).

Limits come from ``app.config.Settings`` (``LOGIN_RATE_MAX_ATTEMPTS``,
``LOGIN_RATE_WINDOW_SECONDS``, ``LOGIN_RATE_LOCKOUT_SECONDS``); the module
constants below are the documented defaults and the fallback when settings
cannot be loaded.

Port of ``~/source/scraper/main/lib/login-rate-limit.ts``.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict

from fastapi import Request

from app.config import get_settings

# Failed attempts permitted inside the window before throttling kicks in.
MAX_ATTEMPTS = 3
# Sliding window length (seconds).
WINDOW_SEC = 60
# How long a throttled client stays locked out (seconds).
LOCKOUT_SEC = 15 * 60
# Cap on tracked clients so a spoofed-IP flood cannot grow the Map forever.
MAX_TRACKED = 5_000


def _limits() -> tuple[int, int, int]:
    """``(max_attempts, window_sec, lockout_sec)`` — settings first, module defaults as floor."""
    try:
        s = get_settings()
    except Exception:  # noqa: BLE001 — a config error must never disable the gate
        return MAX_ATTEMPTS, WINDOW_SEC, LOCKOUT_SEC
    return (
        max(1, int(getattr(s, "login_rate_max_attempts", 0) or MAX_ATTEMPTS)),
        max(1, int(getattr(s, "login_rate_window_seconds", 0) or WINDOW_SEC)),
        max(1, int(getattr(s, "login_rate_lockout_seconds", 0) or LOCKOUT_SEC)),
    )


@dataclass
class _Bucket:
    failures: list[float] = field(default_factory=list)
    locked_until: float = 0.0


@dataclass(frozen=True)
class RateVerdict:
    allowed: bool
    client_key: str
    retry_after_sec: int


_lock = threading.Lock()
_buckets: Dict[str, _Bucket] = {}


def client_key_for(request: Request) -> str:
    """Identify the client behind a request.

    Caddy is the only ingress and sets ``X-Forwarded-For``, so the left-most
    entry is the real client. The header is only trusted because nothing but
    Caddy can reach the app port: ``127.0.0.1:8700``, bound in ``main.py``.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        first = xff.split(",")[0].strip()
        if first:
            return first
    real_ip = request.headers.get("x-real-ip")
    if real_ip and real_ip.strip():
        return real_ip.strip()
    client = request.client
    return client.host if client and client.host else "unknown"


def _prune(now: float) -> None:
    """Drop expired entries so the map cannot grow without bound."""
    _, window, _ = _limits()
    dead: list[str] = []
    for key, bucket in _buckets.items():
        if bucket.locked_until > now:
            continue
        bucket.failures = [t for t in bucket.failures if now - t < window]
        if not bucket.failures:
            dead.append(key)
    for key in dead:
        _buckets.pop(key, None)


def check_login_rate(request: Request) -> RateVerdict:
    """Check whether this client may attempt a login right now.

    Call BEFORE any bcrypt work — the CPU cost is the point.
    """
    now = time.time()
    client_key = client_key_for(request)
    max_attempts, window, lockout = _limits()

    with _lock:
        if len(_buckets) > MAX_TRACKED:
            _prune(now)

        bucket = _buckets.get(client_key)
        if bucket is None:
            return RateVerdict(allowed=True, client_key=client_key, retry_after_sec=0)

        if bucket.locked_until > now:
            return RateVerdict(
                allowed=False,
                client_key=client_key,
                retry_after_sec=max(1, int((bucket.locked_until - now) + 0.5)),
            )

        bucket.failures = [t for t in bucket.failures if now - t < window]
        if len(bucket.failures) >= max_attempts:
            bucket.locked_until = now + lockout
            return RateVerdict(
                allowed=False,
                client_key=client_key,
                retry_after_sec=lockout,
            )

        return RateVerdict(allowed=True, client_key=client_key, retry_after_sec=0)


def record_login_failure(client_key: str) -> None:
    """Record a failed password attempt against this client."""
    now = time.time()
    max_attempts, window, lockout = _limits()
    with _lock:
        bucket = _buckets.get(client_key) or _Bucket()
        bucket.failures = [t for t in bucket.failures if now - t < window]
        bucket.failures.append(now)
        if len(bucket.failures) >= max_attempts:
            bucket.locked_until = now + lockout
        _buckets[client_key] = bucket


def record_login_success(client_key: str) -> None:
    """Clear a client's failure history after a successful login."""
    with _lock:
        _buckets.pop(client_key, None)


def reset_login_rate_limit() -> None:
    """Test hook: wipe all tracked state."""
    with _lock:
        _buckets.clear()
