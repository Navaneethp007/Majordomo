"""The session event log: append-only JSONL at ``~/.majordomo/state.jsonl``.

Ported from Familiar's ``src/state/log.ts``, whose two invariants are exactly
what this file needs — and for the same reason: it is written from inside a
Claude Code hook, where a raised exception surfaces to the user as a broken
editing session.

1. **Writes are single-line appends.** Several sessions can hook at once. An
   ``O_APPEND`` write of a sub-sector-sized line is effectively atomic, so
   concurrent writers do not interleave in practice. Anything that would embed a
   newline mid-record is escaped by ``json.dumps``.
2. **Reads never throw.** A torn final line from a process killed mid-write, a
   hand-edited file, a BOM some editor added, a record from a future version
   with a status we don't understand — all are skipped and counted, never
   raised.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

from majordomo.models import SessionEvent, SessionStatus, Surface
from majordomo.paths import state_path

_VALID_STATUSES = frozenset(get_args(SessionStatus))
_VALID_SURFACES = frozenset(get_args(Surface))


@dataclass
class ReadResult:
    events: list[SessionEvent] = field(default_factory=list)
    #: Lines that could not be parsed or did not describe a session event.
    #: Surfaced by ``mj sessions --debug`` rather than silently discarded.
    skipped: int = 0


def _parse_event(raw: Any) -> SessionEvent | None:
    """Turn one decoded JSON value into an event, or None if it isn't one."""
    if not isinstance(raw, dict):
        return None

    session_id = raw.get("session_id")
    surface = raw.get("surface")
    cwd = raw.get("cwd")
    status = raw.get("status")
    at = raw.get("at")

    if not isinstance(session_id, str) or not session_id:
        return None
    if surface not in _VALID_SURFACES:
        return None
    if status not in _VALID_STATUSES:
        return None
    if not isinstance(cwd, str) or not isinstance(at, str):
        return None

    topic = raw.get("topic")
    entrypoint = raw.get("entrypoint")

    return SessionEvent(
        session_id=session_id,
        surface=surface,  # type: ignore[arg-type]
        cwd=cwd,
        status=status,  # type: ignore[arg-type]
        at=at,
        topic=topic if isinstance(topic, str) else None,
        entrypoint=entrypoint if isinstance(entrypoint, str) else None,
    )


def read_events_detailed(path: Path | str | None = None) -> ReadResult:
    """Read the whole log. Never raises."""
    target = Path(path) if path is not None else state_path()
    if not target.is_file():
        return ReadResult()

    try:
        raw = target.read_text(encoding="utf-8")
    except OSError:
        return ReadResult()

    if raw.startswith("﻿"):
        raw = raw[1:]

    result = ReadResult()
    for line in raw.split("\n"):
        trimmed = line.strip()
        if not trimmed:
            continue
        try:
            decoded = json.loads(trimmed)
        except ValueError:
            # Garbage, or the last line of a file cut short mid-write.
            result.skipped += 1
            continue

        event = _parse_event(decoded)
        if event is None:
            result.skipped += 1
        else:
            result.events.append(event)

    return result


def read_events(path: Path | str | None = None) -> list[SessionEvent]:
    return read_events_detailed(path).events


def append_event(event: SessionEvent, path: Path | str | None = None) -> None:
    """Append one event as exactly one line. Never raises on a missing dir."""
    target = Path(path) if path is not None else state_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    # json.dumps escapes any newline inside a topic, so one event is always one
    # line — the property concurrent appends depend on.
    line = json.dumps(event.to_json(), ensure_ascii=False) + "\n"
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(line)
