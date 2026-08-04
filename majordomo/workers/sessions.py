"""The Local-Sessions worker.

Reads ``state.jsonl`` and nothing else — it never talks to Claude Code, never
parses a transcript, never touches the network. The hook writes; this reads.

It also makes no model call, deliberately. Its input is already structured, and
folding an append-only log into "which sessions are blocked" is a question with
one exactly-correct answer. Handing that to a free-tier model would add latency
and a chance to be wrong.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from majordomo import state
from majordomo.config import Config
from majordomo.models import NeedsYouItem, Session, SourceReport

NAME = "sessions"

#: Ordered worst-first — what a briefing should lead with.
_ATTENTION_ORDER = {"blocked": 0, "idle_awaiting_you": 1, "active": 2, "ended": 3}


def _parse_at(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def fold(events: list, now: datetime | None = None, stale_after_hours: int = 72) -> list[Session]:
    """Collapse the event log into each session's current state.

    The log is append-only, so a session's state is the *last* event about it —
    except for ``topic``, which is carried forward: a `UserPromptSubmit` sets it
    and the later idle/blocked events don't repeat it.

    Ended sessions are dropped (there is nothing to tell you about them), as are
    sessions whose last event is older than ``stale_after_hours`` — a machine
    that slept for a week should not brief you about last Tuesday's terminal.
    """
    moment = now or datetime.now(timezone.utc)
    cutoff = moment - timedelta(hours=stale_after_hours)

    latest: dict[str, Session] = {}
    for event in events:
        previous = latest.get(event.session_id)
        latest[event.session_id] = Session(
            session_id=event.session_id,
            surface=event.surface,
            cwd=event.cwd or (previous.cwd if previous else ""),
            status=event.status,
            at=event.at,
            # Carry the last known topic forward when this event doesn't have one.
            topic=event.topic or (previous.topic if previous else None),
        )

    live: list[Session] = []
    for session in latest.values():
        if session.status == "ended":
            continue
        stamp = _parse_at(session.at)
        if stamp is not None and stamp < cutoff:
            continue
        live.append(session)

    live.sort(key=lambda s: (_ATTENTION_ORDER.get(s.status, 9), s.at))
    return live


def describe(session: Session) -> str:
    where = Path(session.cwd).name or session.cwd or "unknown"
    topic = f" — {session.topic}" if session.topic else ""
    return f"{where} ({session.surface}){topic}"


def summarise(sessions: list[Session]) -> str:
    """A plain-language line the coordinator can fuse. No model needed."""
    if not sessions:
        return "No live coding sessions."

    blocked = [s for s in sessions if s.status == "blocked"]
    idle = [s for s in sessions if s.status == "idle_awaiting_you"]
    active = [s for s in sessions if s.status == "active"]

    parts = []
    if blocked:
        parts.append(
            f"{len(blocked)} session(s) blocked waiting for your approval: "
            + "; ".join(describe(s) for s in blocked)
        )
    if idle:
        parts.append(
            f"{len(idle)} session(s) finished and waiting on you: "
            + "; ".join(describe(s) for s in idle)
        )
    if active:
        parts.append(f"{len(active)} session(s) still working: " + "; ".join(describe(s) for s in active))
    return ". ".join(parts) + "."


def run(config: Config, state_file: Path | str | None = None) -> SourceReport:
    """Read the log, fold it, report. Never raises."""
    try:
        events = state.read_events(state_file)
        sessions = fold(
            events,
            stale_after_hours=config.sources.sessions.stale_after_hours,
        )

        items = [
            NeedsYouItem(
                kind="session_blocked" if s.status == "blocked" else "session_idle",
                title=describe(s),
                detail=(
                    "Waiting for your approval on a permission prompt."
                    if s.status == "blocked"
                    else "Finished and waiting for your next instruction."
                ),
                source=NAME,
                action=s.session_id,
            )
            for s in sessions
            if s.status in ("blocked", "idle_awaiting_you")
        ]

        return SourceReport(
            source=NAME,
            ok=True,
            summary=summarise(sessions),
            items=items,
            path="cheap",
            route_reason="local read, no model needed",
        )
    except Exception as exc:  # pragma: no cover - defensive; state.py doesn't raise
        return SourceReport.failed(NAME, str(exc))
