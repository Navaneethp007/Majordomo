"""The Local-Sessions worker.

Reads ``state.jsonl`` and nothing else — it never talks to Claude Code, never
parses a transcript, never touches the network. The hook writes; this reads.

It also makes no model call, deliberately. Its input is already structured, and
folding an append-only log into "which sessions are blocked" is a question with
one exactly-correct answer. Handing that to a free-tier model would add latency
and a chance to be wrong.
"""
from __future__ import annotations

from dataclasses import replace
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


def fold(
    events: list,
    now: datetime | None = None,
    stale_after_hours: int = 72,
    active_timeout_minutes: int = 90,
) -> list[Session]:
    """Collapse the event log into each session's current state.

    The log is append-only, so a session's state is the *last* event about it —
    except for ``topic``, which is carried forward: a `UserPromptSubmit` sets it
    and the later idle/blocked events don't repeat it.

    **Liveness is not the same as not-having-ended.** ``SessionEnd`` cannot run
    if the process is killed — a crashed VS Code window, a terminal closed with
    the X — so treating ``ended`` as the only exit left dead sessions listed as
    "active" for days. Two things fix that:

    - ``Stop`` fires at the end of every assistant turn, which both proves the
      session is alive **and** proves it is not blocked. A permission prompt
      happens *mid*-turn, so the turn cannot end while one is outstanding —
      a Stop after a block means you answered it.
    - An ``active`` session with no event for ``active_timeout_minutes`` is
      presumed gone. Nothing is claiming it is alive, so we stop claiming it.

    An earlier version treated Stop as a status-preserving heartbeat, on the
    theory that it should not clear a block. That was exactly backwards, and it
    was the worse failure: approving a prompt emits no ``UserPromptSubmit``, so
    the session stayed ``blocked`` forever — while every subsequent turn
    refreshed ``at`` and kept the staleness sweep from ever retiring it. The
    briefing then asserted a blocking obligation that had been resolved hours
    earlier, every single time it ran.

    ``blocked`` and ``idle_awaiting_you`` deliberately keep the much longer
    ``stale_after_hours`` window: those are genuinely waiting on you, and going
    quiet is exactly what they are supposed to do.
    """
    moment = now or datetime.now(timezone.utc)
    stale_cutoff = moment - timedelta(hours=stale_after_hours)
    active_cutoff = moment - timedelta(minutes=active_timeout_minutes)

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
        if stamp is not None:
            if stamp < stale_cutoff:
                continue
            if session.status == "active" and stamp < active_cutoff:
                continue  # nothing has claimed it is alive for a long time
        live.append(session)

    live.sort(key=lambda s: (_ATTENTION_ORDER.get(s.status, 9), s.at))
    return live


def describe(session: Session) -> str:
    where = Path(session.cwd).name or session.cwd or "unknown"
    topic = f" — {session.topic}" if session.topic else ""
    return f"{where} ({session.surface}){topic}"


def summarise(sessions: list[Session]) -> str:
    """A plain-language line the coordinator can fuse. No model needed.

    Each group states its own actionability inline. Without that, "2 sessions
    still working: fe-raad-erp — what does one payments[] element look like on
    the wire?" reads to a model exactly like an open question addressed to the
    user, and it duly reported it as a decision awaiting them. It is the
    opposite: a question they asked, already being worked on.
    """
    if not sessions:
        return "No live coding sessions."

    blocked = [s for s in sessions if s.status == "blocked"]
    idle = [s for s in sessions if s.status == "idle_awaiting_you"]
    active = [s for s in sessions if s.status == "active"]

    parts = []
    if blocked:
        parts.append(
            f"NEEDS ACTION — {len(blocked)} session(s) stopped at a permission prompt "
            "and cannot continue until approved: "
            + "; ".join(describe(s) for s in blocked)
        )
    if idle:
        parts.append(
            f"NEEDS ACTION — {len(idle)} session(s) finished and are waiting for the "
            "next instruction: " + "; ".join(describe(s) for s in idle)
        )
    if active:
        parts.append(
            f"NO ACTION NEEDED — {len(active)} session(s) currently running; the topic "
            "shown is what the user asked, not a question for them: "
            + "; ".join(describe(s) for s in active)
        )
    return ". ".join(parts) + "."


def run(
    config: Config,
    state_file: Path | str | None = None,
    now: datetime | None = None,
) -> SourceReport:
    """Read the log, fold it, report. Never raises.

    ``now`` is injectable because staleness is measured against it. Reading the
    wall clock directly would make every test with a fixed fixture timestamp
    pass today and fail three days from now.
    """
    try:
        events = state.read_events(state_file)
        sessions = fold(
            events,
            now=now,
            stale_after_hours=config.sources.sessions.stale_after_hours,
            active_timeout_minutes=config.sources.sessions.active_timeout_minutes,
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
            # summarise() already writes for a human. Sending it through the
            # cheap summarize call spent a request and lost the detail that
            # makes it useful — which repo, which topic.
            pre_summarised=True,
        )
    except Exception as exc:  # pragma: no cover - defensive; state.py doesn't raise
        return SourceReport.failed(NAME, str(exc))
