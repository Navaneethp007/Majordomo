"""One wake cycle, start to finish.

    workers (parallel) → router → coordinator → briefing

Kept out of ``cli.py`` so the tray and the wake trigger can call exactly the
same pipeline the CLI does, rather than reimplementing it or shelling out.

Workers run in threads because they are I/O-bound — an HTTP round trip to GitHub
and a file read — so the GIL is irrelevant and processes would cost more than
they save.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from majordomo import coordinator, state
from majordomo.config import Config
from majordomo.models import Briefing, SourceReport
from majordomo.workers import github, gmail, sessions


@dataclass
class BriefResult:
    briefing: Briefing
    reports: list[SourceReport] = field(default_factory=list)

    def explain(self) -> str:
        """Which path each source took, and why — for ``mj brief --explain``.

        This exists because the escalation fork is otherwise invisible: on an
        ordinary morning every source takes the cheap path, and "we route per
        source" would be a claim you have to take on faith.
        """
        lines = ["Routing:"]
        for report in self.reports:
            lines.append(f"  {report.source:<10} {report.path:<10} {report.route_reason}")
        if self.briefing.note:
            # The fusing step belongs to no source, so it has nowhere else to
            # report from — and a fallback briefing otherwise looks exactly like
            # a successful terse one.
            lines.append(f"  {'fuse':<10} {'degraded':<10} {self.briefing.note}")
        return "\n".join(lines)


def gather(config: Config, state_file: Path | str | None = None) -> list[SourceReport]:
    """Run every enabled worker in parallel. Workers never raise."""
    tasks = []
    if config.sources.sessions.enabled:
        tasks.append(("sessions", lambda: sessions.run(config, state_file)))
    if config.sources.github.enabled:
        tasks.append(("github", lambda: github.run(config)))
    if config.sources.gmail.enabled:
        tasks.append(("gmail", lambda: gmail.run(config)))

    if not tasks:
        return []

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = [(name, pool.submit(fn)) for name, fn in tasks]
        reports = []
        for name, future in futures:
            try:
                reports.append(future.result())
            except Exception as exc:
                # Workers are contracted not to raise, but a thread that dies
                # anyway must not take the briefing with it.
                reports.append(SourceReport.failed(name, str(exc)))

    return reports


def run(config: Config, state_file: Path | str | None = None) -> BriefResult:
    """Fetch, route, fuse. Raises only MissingApiKey, which the CLI turns into an exit."""
    # Housekeeping on the read path rather than in the hook: this runs a few
    # times a day, the hook runs on every prompt. No-ops unless the log is big.
    state.maybe_compact(state_file, keep_hours=config.sources.sessions.stale_after_hours)

    raw = gather(config, state_file)
    routed = [coordinator.route_report(report, config) for report in raw]
    briefing = coordinator.fuse(routed, config)
    return BriefResult(briefing=briefing, reports=routed)
