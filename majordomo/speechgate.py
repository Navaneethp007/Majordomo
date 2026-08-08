"""Deciding whether a briefing has already been said.

``--speak-if-needed`` alone is not enough to stop the briefing becoming noise,
because of how the triggers actually fire:

- **Cold boot says everything twice.** OnLogon fires immediately and OnBoot
  fires a minute later. ``MultipleInstancesPolicy=IgnoreNew`` is per-task, so it
  cannot dedupe across them — two ``winsound.PlaySound`` calls overlap and you
  are billed for two syntheses of identical text.
- **Modern standby resumes constantly.** Kernel-Power 107 fires every time the
  lid opens. A session blocked since 9am would be read aloud, word for word,
  every one of those times.

So speech is gated on *novelty*, not just need. We keep a fingerprint of what
was last spoken:

- the situation changed  → speak, however recently we last spoke
- the situation is the same → stay quiet until ``repeat_after_minutes`` passes

The fingerprint deliberately covers unavailable sources as well as needs-you
items, so "GitHub token expired" is a change worth announcing rather than
something that hashes to the same silence as "all quiet".
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from majordomo.models import Briefing, SourceReport
from majordomo.paths import majordomo_home


def spoken_path() -> Path:
    return majordomo_home() / "last_spoken.json"


def fingerprint(briefing: Briefing, reports: list[SourceReport] | None = None) -> str:
    """A stable digest of *what the situation is*, not how it was worded.

    Hashing ``briefing_text`` would be useless — the model rephrases itself
    every run, so identical circumstances would look novel each time. Hashing
    the structured facts means a rewording stays silent and a genuine change
    speaks.
    """
    parts = [f"{item.kind}|{item.title}" for item in briefing.needs_you]
    parts += [f"down|{r.source}" for r in (reports or []) if not r.ok]
    parts.sort()
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:32]


def read_last(path: Path | str | None = None) -> tuple[str, datetime] | None:
    """The last spoken fingerprint and when. Never raises."""
    target = Path(path) if path is not None else spoken_path()
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        stamp = datetime.fromisoformat(data["at"])
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return str(data["fingerprint"]), stamp
    except (OSError, ValueError, KeyError, TypeError):
        # A corrupt marker must not silence the briefing forever — treat it as
        # "nothing spoken yet" and let this run speak.
        return None


def record(fp: str, now: datetime | None = None, path: Path | str | None = None) -> None:
    """Remember what we just said. Never raises — failing to record is not
    worth losing a briefing over; the cost is one duplicate."""
    target = Path(path) if path is not None else spoken_path()
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"fingerprint": fp, "at": stamp}), encoding="utf-8"
        )
    except OSError:
        pass


def is_repeat(
    fp: str,
    repeat_after_minutes: int,
    now: datetime | None = None,
    path: Path | str | None = None,
) -> bool:
    """True when this exact situation was already spoken recently enough."""
    last = read_last(path)
    if last is None:
        return False

    last_fp, last_at = last
    if last_fp != fp:
        return False  # something changed — worth saying again

    moment = now or datetime.now(timezone.utc)
    return moment - last_at < timedelta(minutes=repeat_after_minutes)
