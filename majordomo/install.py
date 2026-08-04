"""Wiring Majordomo into ``~/.claude/settings.json``.

Ported from Familiar's ``src/install.ts``, and for a sharper reason than reuse:
**Familiar is already in that file on this machine.** It owns handlers on
SessionStart, PostToolUse, Stop and SessionEnd, plus the statusLine. Two of
those events are ones Majordomo also needs.

So the rules are strict. Back up first. Deep-merge into the existing arrays
rather than replacing them. Never touch a handler that isn't ours. Make
uninstall remove exactly what we added and nothing else — including dropping an
event key entirely if we were the only thing in it, so the file ends up the way
we found it rather than littered with empties.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from majordomo.paths import backup_path, claude_settings_path, ensure_home


class SettingsUnreadable(Exception):
    """The settings file exists but isn't parseable. We decline rather than guess."""


#: Every event we hook, and the matcher each needs.
#:
#: SessionStart is scoped to ``startup|resume`` deliberately: unscoped, the
#: ``clear``/``compact``/``fork`` sources would each fire a spurious "session
#: started". Notification is listed twice because its payload carries only
#: ``message`` — it never says which matcher matched, so the matcher has to
#: travel in our own argv.
HOOK_SPECS: list[tuple[str, str | None]] = [
    ("SessionStart", "startup|resume"),
    ("UserPromptSubmit", None),
    ("Notification", "idle_prompt"),
    ("Notification", "permission_prompt"),
    ("SessionEnd", None),
]

#: Seconds. Generous enough for a cold Python start, short enough that a wedged
#: hook can't stall the session for long.
HOOK_TIMEOUT = 10


def _handler_for(cli_path: str, event: str, matcher: str | None) -> dict[str, Any]:
    """Build one hook handler entry.

    Exec form (command + args) rather than a shell string: no quoting rules to
    get wrong, which matters most on Windows. ``sys.executable`` pins the
    interpreter Majordomo was installed into, so a different `python` later on
    PATH cannot silently break the hook.
    """
    args = [
        "-m",
        "majordomo.cli",
        "hook",
        f"--event={event}",
    ]
    if matcher:
        args.append(f"--matcher={matcher}")
    return {
        "type": "command",
        "command": sys.executable.replace("\\", "/"),
        "args": args,
        "timeout": HOOK_TIMEOUT,
        # Our own marker. Path-independent, so a moved or reinstalled checkout
        # is still recognisably ours.
        "_majordomo": cli_path,
    }


def _is_ours(handler: dict[str, Any]) -> bool:
    """Identify a handler as Majordomo's, so uninstall removes only those."""
    if "_majordomo" in handler:
        return True
    args = handler.get("args")
    if not isinstance(args, list):
        return False
    return any(isinstance(a, str) and a == "majordomo.cli" for a in args)


def _strip_bom(text: str) -> str:
    return text[1:] if text.startswith("﻿") else text


def read_settings(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(_strip_bom(path.read_text(encoding="utf-8")))
    except ValueError as exc:
        raise SettingsUnreadable(f"Could not parse {path}. Fix or move it, then re-run.") from exc
    return data if isinstance(data, dict) else {}


def _write_settings(settings: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def backup_settings(path: Path) -> Path | None:
    if not path.is_file():
        return None
    ensure_home()
    stamp = datetime.now(timezone.utc).isoformat().replace(":", "-").replace(".", "-")
    target = backup_path(stamp)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(path.read_bytes())
    return target


@dataclass
class InstallResult:
    backup: Path | None
    added: list[str]
    settings_path: Path


@dataclass
class UninstallResult:
    backup: Path | None
    removed: int
    settings_path: Path


def _without_ours(groups: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Copy ``groups`` with every Majordomo handler removed. Returns the count."""
    removed = 0
    cleaned: list[dict[str, Any]] = []
    for group in groups:
        kept = []
        for handler in group.get("hooks", []) or []:
            if isinstance(handler, dict) and _is_ours(handler):
                removed += 1
            else:
                kept.append(handler)
        if kept:
            cleaned.append({**group, "hooks": kept})
    return cleaned, removed


def install(
    settings_path: Path | str | None = None,
    cli_path: str | None = None,
) -> InstallResult:
    """Add Majordomo's hooks, preserving everything already in the file."""
    path = Path(settings_path) if settings_path is not None else claude_settings_path()
    marker = cli_path if cli_path is not None else str(Path(__file__).resolve().parent.parent)

    settings = json.loads(json.dumps(read_settings(path)))  # deep copy, never alias
    backup = backup_settings(path)

    hooks: dict[str, Any] = dict(settings.get("hooks") or {})
    added: list[str] = []

    # Drop every previous Majordomo handler in one pass *before* adding any, so
    # re-running install upgrades a moved path in place instead of stacking
    # duplicates. This has to happen up front rather than per-spec: Notification
    # appears twice in HOOK_SPECS, and a per-spec strip would delete the
    # idle_prompt handler while installing the permission_prompt one.
    for event in {e for e, _ in HOOK_SPECS}:
        if event in hooks:
            hooks[event], _ = _without_ours(list(hooks[event] or []))

    for event, matcher in HOOK_SPECS:
        cleaned = list(hooks.get(event) or [])

        handler = _handler_for(marker, event, matcher)
        target = next((g for g in cleaned if (g.get("matcher") or None) == matcher), None)
        if target is not None:
            # Join the existing group rather than creating a duplicate beside it
            # — Familiar's SessionStart matcher is identical to ours.
            target["hooks"] = [*target.get("hooks", []), handler]
        else:
            group: dict[str, Any] = {"hooks": [handler]}
            if matcher:
                group = {"matcher": matcher, "hooks": [handler]}
            cleaned.append(group)

        hooks[event] = cleaned
        added.append(f"{event}{':' + matcher if matcher else ''}")

    settings["hooks"] = hooks
    _write_settings(settings, path)

    return InstallResult(backup=backup, added=added, settings_path=path)


def uninstall(
    settings_path: Path | str | None = None,
    cli_path: str | None = None,
) -> UninstallResult:
    """Remove exactly what install added, and nothing else."""
    path = Path(settings_path) if settings_path is not None else claude_settings_path()

    if not path.is_file():
        return UninstallResult(backup=None, removed=0, settings_path=path)

    settings = json.loads(json.dumps(read_settings(path)))
    backup = backup_settings(path)

    removed = 0
    existing_hooks = settings.get("hooks")
    if isinstance(existing_hooks, dict):
        hooks: dict[str, Any] = {}
        for event, groups in existing_hooks.items():
            cleaned, count = _without_ours(list(groups or []))
            removed += count
            # Drop the event key entirely if we were the only thing in it.
            if cleaned:
                hooks[event] = cleaned
        if hooks:
            settings["hooks"] = hooks
        else:
            settings.pop("hooks", None)

    _write_settings(settings, path)
    return UninstallResult(backup=backup, removed=removed, settings_path=path)
