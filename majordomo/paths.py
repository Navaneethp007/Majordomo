"""Where Majordomo keeps its things.

Everything Majordomo owns lives under ``~/.majordomo``. The one file it writes
outside that directory is Claude Code's ``settings.json``, and only via
``majordomo.install`` — which backs it up first.
"""
from __future__ import annotations

import os
from pathlib import Path


def majordomo_home() -> Path:
    """Root of Majordomo's own state. Overridable for tests via MAJORDOMO_HOME."""
    override = os.environ.get("MAJORDOMO_HOME")
    if override:
        return Path(override)
    return Path.home() / ".majordomo"


def default_config_path() -> Path:
    return majordomo_home() / "config.yml"


def state_path() -> Path:
    """The append-only session event log written by the hook."""
    return majordomo_home() / "state.jsonl"


def error_log_path() -> Path:
    """Where the hook writes failures it swallowed. Never raises to the session."""
    return majordomo_home() / "error.log"


def activity_path() -> Path:
    """The append-only log of your own GitHub activity."""
    return majordomo_home() / "activity.jsonl"


def activity_fetched_path() -> Path:
    """When we last *asked* GitHub, which is not when you last did something.

    Kept apart from the log because they answer different questions: the log's
    newest entry is when you last pushed, and a quiet week would otherwise look
    exactly like a cold cache.
    """
    return majordomo_home() / "activity.fetched"


def memory_dir() -> Path:
    """One file per remembered fact. See ``majordomo.memory``."""
    return majordomo_home() / "memory"


def memory_index_path() -> Path:
    """The one-line-per-memory index. Small enough to load on every turn."""
    return memory_dir() / "INDEX.md"


def chats_dir() -> Path:
    """Saved ``mj chat`` transcripts, one JSONL file per session."""
    return majordomo_home() / "chats"


def backup_path(stamp: str) -> Path:
    return majordomo_home() / "backups" / f"settings.{stamp}.json"


def claude_settings_path() -> Path:
    """Claude Code's user settings — the only file we write outside our home."""
    override = os.environ.get("CLAUDE_CONFIG_DIR")
    base = Path(override) if override else Path.home() / ".claude"
    return base / "settings.json"


def ensure_home() -> Path:
    home = majordomo_home()
    home.mkdir(parents=True, exist_ok=True)
    return home
