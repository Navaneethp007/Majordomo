"""Workers — one per source.

A worker fetches its own raw data and reasons about its own domain: not "here is
everything", but "of this, what actually needs Nav?". Each returns a
``SourceReport``.

The hard rule is at the boundary: **a worker never raises past itself**. A
failed source returns ``SourceReport.failed(...)`` so the coordinator can brief
everything else and note the gap, rather than one dead token taking the whole
morning briefing down with it (spec §9).

Honest note on which workers are agents. The GitHub worker genuinely reasons —
it takes a pile of PRs and notifications and judges what matters. The sessions
worker does not call a model at all: its input is already structured, and a fold
over the event log answers "which sessions are blocked" exactly, where a model
would only add latency and a chance to be wrong.
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
