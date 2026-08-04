"""Reading secrets from ``~/.majordomo/.env``.

Keys are read from the process environment everywhere in this codebase, which
is right — but it leaves the question of how they get there. Two facts decide
it:

- The wake trigger runs under Task Scheduler and the tray runs detached. Neither
  inherits anything you exported in a shell, so ``export`` / ``$env:`` alone
  gets you a briefing that works when you test it and silently fails on wake.
- Windows user-level variables (``setx``) *do* reach both, but put your
  ElevenLabs key into the environment of every process you ever start.

So: an optional ``.env`` beside ``config.yml``, loaded by us, which works
identically under the CLI, the tray and Task Scheduler because we do the
loading rather than the shell.

**A real environment variable always wins.** The file fills gaps, it never
overrides — so ``setx`` still works, CI still works, and a temporary
``$env:KEY=…`` still shadows the file for one run.

Deliberately not python-dotenv: this is thirty lines, and Majordomo's dependency
list is two packages on purpose.
"""
from __future__ import annotations

import os
from pathlib import Path

from majordomo.paths import majordomo_home


def env_path() -> Path:
    return majordomo_home() / ".env"


def parse(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines. Ignores blanks, comments and anything malformed."""
    values: dict[str, str] = {}

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # `export KEY=value` is what people paste out of shell instructions.
        if stripped.startswith("export "):
            stripped = stripped[len("export "):].lstrip()
        if "=" not in stripped:
            continue

        key, _, value = stripped.partition("=")
        key = key.strip()
        if not key:
            continue

        value = value.strip()
        # Strip one layer of matching quotes — a key wrapped in quotes that
        # reached the API verbatim would fail auth for a completely invisible
        # reason.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        values[key] = value

    return values


def load(path: Path | str | None = None) -> list[str]:
    """Fill missing env vars from the file. Returns the names actually set.

    Never raises: a missing or unreadable file just means nothing to load, and
    a broken .env should not be able to stop a briefing.
    """
    target = Path(path) if path is not None else env_path()
    if not target.is_file():
        return []

    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return []

    if text.startswith("﻿"):
        text = text[1:]

    applied: list[str] = []
    for key, value in parse(text).items():
        # A real environment variable always wins.
        if os.environ.get(key):
            continue
        os.environ[key] = value
        applied.append(key)

    return applied
