"""The record shapes that cross module boundaries.

All frozen: every one of these is produced by one component and read by another,
and nothing downstream has any business mutating what it was handed.

The two shapes pinned by the design spec (§8) are ``SessionEvent`` — one line of
``state.jsonl`` — and ``Briefing``, what the coordinator returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

#: Where a coding session is running. Decides how Resume reaches it (§7).
Surface = Literal["vscode", "terminal"]

#: A session's lifecycle state. ``idle_awaiting_you`` means it finished and is
#: waiting on you; ``blocked`` means it is sitting on a permission prompt.
SessionStatus = Literal["active", "idle_awaiting_you", "blocked", "ended"]


@dataclass(frozen=True)
class SessionEvent:
    """One appended line of ``state.jsonl`` — a session's state at a moment.

    Written only by the hook, read only by the sessions worker. The log is
    append-only, so a session's *current* state is the fold of its events, not
    any single record.
    """

    session_id: str
    surface: Surface
    cwd: str
    status: SessionStatus
    at: str  # ISO-8601 UTC
    topic: str | None = None
    #: The raw ``CLAUDE_CODE_ENTRYPOINT`` we detected ``surface`` from. Kept so
    #: that a wrong guess is diagnosable rather than invisible — the env var is
    #: undocumented and could change without warning.
    entrypoint: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Dict for JSONL serialisation, dropping unset optionals."""
        data: dict[str, Any] = {
            "session_id": self.session_id,
            "surface": self.surface,
            "cwd": self.cwd,
            "status": self.status,
            "at": self.at,
        }
        if self.topic is not None:
            data["topic"] = self.topic
        if self.entrypoint is not None:
            data["entrypoint"] = self.entrypoint
        return data


@dataclass(frozen=True)
class Session:
    """A session's *current* state, folded from all its events."""

    session_id: str
    surface: Surface
    cwd: str
    status: SessionStatus
    at: str
    topic: str | None = None


# ---------------------------------------------------------------------------
# Workers and the router
# ---------------------------------------------------------------------------

#: Which path through the gate a source actually took. Reported by
#: ``mj brief --explain`` so the escalation fork is observable, not theoretical.
RoutePath = Literal["cheap", "escalated", "error"]


@dataclass(frozen=True)
class SourceReport:
    """What one worker hands the coordinator.

    A worker never raises past its own boundary — a failed source returns a
    report with ``ok=False`` and an ``error`` note so the coordinator can brief
    everything else and mention the gap (spec §9).
    """

    source: str
    ok: bool
    summary: str
    items: list["NeedsYouItem"] = field(default_factory=list)
    path: RoutePath = "cheap"
    #: Why the gate routed this source the way it did — for ``--explain``.
    route_reason: str = ""
    error: str | None = None

    @classmethod
    def failed(cls, source: str, error: str) -> "SourceReport":
        """An error stub. The briefing goes ahead without this source."""
        return cls(
            source=source,
            ok=False,
            summary=f"{source} unavailable: {error}",
            path="error",
            route_reason="worker failed",
            error=error,
        )


@dataclass(frozen=True)
class NeedsYouItem:
    """One thing that actually requires a decision from you."""

    kind: str  # e.g. "review_request", "session_blocked"
    title: str
    detail: str
    source: str
    #: Opaque handle the surface uses to act — a session id, a PR url.
    action: str | None = None


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Briefing:
    """The one thing the whole pipeline exists to produce."""

    briefing_text: str
    needs_you: list[NeedsYouItem] = field(default_factory=list)
