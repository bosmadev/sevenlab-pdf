#!/usr/bin/env python3
"""Generate a bcrypt hash for APP_PASSWORD_HASH (cost 12).

Usage:
    python backend/scripts/hash_password.py          # prompts for password
    APP_PASSWORD=secret python backend/scripts/hash_password.py

Paste the printed hash into ``.env`` as ``APP_PASSWORD_HASH=<hash>``.
"""
from __future__ import annotations

import getpass
import os
import sys

import bcrypt

COST = 12


def main() -> int:
    pw = os.environ.get("APP_PASSWORD")
    if not pw:
        pw = getpass.getpass("Password: ")
        confirm = getpass.getpass("Confirm:  ")
        if pw != confirm:
            print("Passwords do not match", file=sys.stderr)
            return 2
    if not pw:
        print("Empty password rejected", file=sys.stderr)
        return 2
    hashed = bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt(rounds=COST)).decode("utf-8")
    print(hashed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
