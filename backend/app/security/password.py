"""Password verification and tenant resolution.

Maps a plaintext login password to a stable ``owner_id`` used for per-tenant
scoping of documents and chats. The main password (``APP_PASSWORD_HASH``)
resolves to the admin owner, which sees every workspace. Every entry in
``APP_SECONDARY_HASHES`` resolves to its own tenant.

Supported ``APP_SECONDARY_HASHES`` entry formats (comma-separated):

* ``$2b$12$...``       → owner id derived from the hash: ``u_<sha256[0:12]>``
* ``danny:$2b$12$...`` → owner id is the label: ``danny``

Port of ``~/source/scraper/main/lib/owner.ts``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import List

import bcrypt

from app.config import get_settings

ADMIN_OWNER_ID = "admin"

# Deliberately excludes `.` so a token cannot be split into extra segments.
_OWNER_ID_RE = re.compile(r"^[a-z0-9_-]{1,32}$")


def is_valid_owner_id(owner_id: str) -> bool:
    """Validate an owner id parsed out of a session token."""
    return bool(_OWNER_ID_RE.match(owner_id or ""))


def is_admin_owner(owner_id: str | None) -> bool:
    """True when the supplied owner id is the all-seeing admin tenant."""
    return owner_id == ADMIN_OWNER_ID


def _label_to_owner_id(label: str) -> str | None:
    """Normalise a human label into a safe owner id.

    Returns ``None`` when nothing usable survives sanitisation. ``admin`` is
    reserved for the main password only, so a secondary entry labelled
    ``admin`` is rejected (would otherwise collapse into the main tenant).
    """
    cleaned = re.sub(r"[^a-z0-9_-]", "", label.strip().lower())[:32]
    if not cleaned:
        return None
    if cleaned == ADMIN_OWNER_ID:
        return None
    return cleaned


def _hash_to_owner_id(hash_: str) -> str:
    """Derive a stable, opaque owner id from a bcrypt hash."""
    digest = sha256(hash_.encode("utf-8")).hexdigest()
    return f"u_{digest[:12]}"


@dataclass(frozen=True)
class SecondaryEntry:
    owner_id: str
    hash: str


def parse_secondary_entries(raw: str | None) -> List[SecondaryEntry]:
    """Parse ``APP_SECONDARY_HASHES`` into (owner_id, hash) pairs.

    Malformed entries are skipped rather than raising, so one bad entry
    cannot lock every secondary tenant out.
    """
    if not raw:
        return []

    entries: list[SecondaryEntry] = []
    seen: set[str] = set()

    for part in raw.split(","):
        entry = part.strip()
        if not entry:
            continue

        label: str | None = None
        hash_ = entry

        colon = entry.find(":")
        if colon > 0:
            label = entry[:colon]
            hash_ = entry[colon + 1 :].strip()

        # bcrypt alphabet never contains `:`, so a stray colon is a label split.
        if not hash_.startswith("$2"):
            continue

        resolved = (label and _label_to_owner_id(label)) or _hash_to_owner_id(hash_)
        if resolved in seen:
            resolved = _hash_to_owner_id(hash_)
        if resolved in seen:
            continue

        seen.add(resolved)
        entries.append(SecondaryEntry(owner_id=resolved, hash=hash_))

    return entries


def _bcrypt_check(password: str, hash_: str) -> bool:
    """Timing-stable bcrypt compare that never throws on malformed hashes."""
    if not password or not hash_:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hash_.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def resolve_owner_id(password: str) -> str | None:
    """Resolve a plaintext password to its owner id.

    Every candidate hash is compared even after a match is found, so the
    response time does not leak which password was supplied.

    Returns:
        The owner id, or ``None`` when the password matches nothing.
    """
    primary_hash = get_settings().app_password_hash
    if not primary_hash:
        return None

    resolved: str | None = None

    if _bcrypt_check(password, primary_hash):
        resolved = ADMIN_OWNER_ID

    # `APP_SECONDARY_HASHES` is read from the live environment, not the cached
    # settings snapshot, so operators can add tenants without a restart.
    for entry in parse_secondary_entries(os.environ.get("APP_SECONDARY_HASHES")):
        if _bcrypt_check(password, entry.hash):
            if resolved is None:
                resolved = entry.owner_id

    return resolved
