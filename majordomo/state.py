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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, get_args

from majordomo.models import SessionEvent, SessionStatus, Surface
from majordomo import jsonlog
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
    kind = raw.get("kind")

    return SessionEvent(
        session_id=session_id,
        surface=surface,  # type: ignore[arg-type]
        cwd=cwd,
        status=status,  # type: ignore[arg-type]
        at=at,
        topic=topic if isinstance(topic, str) else None,
        entrypoint=entrypoint if isinstance(entrypoint, str) else None,
        kind=kind if isinstance(kind, str) else None,
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


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

#: Compact once the log passes this. Roughly 10k events — months of use.
#: Re-exported from ``jsonlog`` so callers of this module keep working.
COMPACT_OVER_BYTES = jsonlog.COMPACT_OVER_BYTES


def compact(
    path: Path | str | None = None,
    keep_hours: int = 72,
    now: datetime | None = None,
) -> int:
    """Rewrite the log keeping only what can still matter. Returns lines dropped.

    The hook appends on every prompt and nothing ever removed anything, so the
    file grew without bound — and every ``mj brief``, ``mj sessions`` and panel
    request JSON-parses the whole thing before ``fold`` throws most of it away.
    After months that is a multi-megabyte parse on a path that runs at every wake.

    What survives: every event inside ``keep_hours``, plus — for sessions that
    *do* have recent activity — the newest event older than the cutoff. That
    last part is why a long-running session keeps its topic: the topic is set
    once by ``UserPromptSubmit`` and later idle/blocked events don't repeat it,
    so dropping the old event would leave ``fold`` with nothing to carry.

    Note the direction. Old events belonging to sessions with *no* recent
    activity are discarded outright: ``fold`` drops those as stale regardless,
    so keeping them buys nothing and is precisely what makes a log grow forever
    — one dead session id per terminal you ever opened.

    Deliberately **not** called from the hook. Compaction costs a full read and
    rewrite, and the hook is the one path that runs on every keystroke-to-prompt.
    """
    target = Path(path) if path is not None else state_path()
    if not target.is_file():
        return 0

    moment = now or datetime.now(timezone.utc)
    cutoff = moment - timedelta(hours=keep_hours)

    events = read_events(target)
    if not events:
        return 0

    recent: list[SessionEvent] = []
    newest_older: dict[str, SessionEvent] = {}
    for event in events:
        stamp = _parse_at(event.at)
        if stamp is None or stamp >= cutoff:
            recent.append(event)
        else:
            newest_older[event.session_id] = event

    # Carry an old event forward only for a session that is still active, to
    # preserve its topic/cwd. Everything else older than the cutoff goes.
    live_recent = {e.session_id for e in recent}
    carried = [e for sid, e in newest_older.items() if sid in live_recent]

    kept = carried + recent
    dropped = len(events) - len(kept)
    if dropped <= 0:
        return 0

    # Atomic, via jsonlog: a reader sees the whole old log or the whole new
    # sees either the old file or the new one, never a half-written log. A hook
    # appending inside the swap window could lose one event — acceptable against
    # unbounded growth, and it is why this never runs from the hook itself.
    jsonlog.rewrite(target, kept)

    return dropped


def maybe_compact(
    path: Path | str | None = None,
    keep_hours: int = 72,
    max_bytes: int = COMPACT_OVER_BYTES,
) -> int:
    """Compact only if the log has actually got big. Never raises."""
    target = Path(path) if path is not None else state_path()
    try:
        if not jsonlog.is_large(target, max_bytes):
            return 0
        return compact(target, keep_hours=keep_hours)
    except OSError:
        # Housekeeping must never be the reason a briefing fails.
        return 0


def _parse_at(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
