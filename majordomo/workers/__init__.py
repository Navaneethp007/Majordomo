"""Workers — one per source.

A worker fetches its own raw data and reasons about its own domain: not "here is
everything", but "of this, what actually needs Nav?". Each returns a
``SourceReport``.

The hard rule is at the boundary: **a worker never raises past itself**. A
failed source returns ``SourceReport.failed(...)`` so the coordinator can brief
everything else and note the gap, rather than one dead token taking the whole
morning briefing down with it (spec §9).

Honest note on which workers are agents: **none of them are.**

No worker calls a model. Each one fetches, flattens, and applies rules written
here in code — ``if s.status in ("blocked", "idle_awaiting_you")`` for sessions,
review-requests-only for GitHub, ``items=[]`` unconditionally for Gmail. The
reasoning happens later and uniformly, in ``coordinator.route_report``, which is
why a worker's report leaves here unreasoned with ``route_reason="pending
routing"``.

That is the safety property, not an accident of layering. The one time a model
was allowed to judge actionability from prose it invented three obligations that
did not exist (see ``workers.gmail``), so ``needs_you`` is derived structurally
and a model cannot add to it however it behaves.
"""
from __future__ import annotations

from typing import Protocol

from majordomo.config import Config
from majordomo.models import SourceReport


class Worker(Protocol):
    """What every worker must provide."""

    name: str

    def run(self, config: Config) -> SourceReport:
        """Fetch, reason, and report. Must not raise."""
        ...
