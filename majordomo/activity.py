"""Your own GitHub activity, cached locally.

The GitHub *worker* answers "what is waiting on me right now?" — a live question
whose answer is worthless if it's five minutes old. This answers a different
one: "what have I been doing?" That has no reason to touch the network, so it
doesn't. Fetch occasionally, store, and read from disk.

Two consequences worth stating, because they are the point:

- ``mj ask "what did I work on in June?"`` is instant and works on a plane.
- A GitHub outage costs you nothing. The cache answers, and says how old it is.

── STORAGE ──────────────────────────────────────────────────────────────────
``~/.majordomo/activity.jsonl`` — append-only, one event per line, exactly the
discipline ``state.py`` uses and for the same reason: a torn final line from a
process killed mid-write must cost one event, not the whole history.

Pruning happens **on read**, never by rewriting the file. Rewriting an
append-only log is where append-only logs go to get corrupted, and the file is
small enough that carrying a few thousand dead lines costs nothing.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from majordomo.config import Config, GitHubConfig
from majordomo.paths import activity_fetched_path, activity_path
from majordomo.workers.github import GitHubError, get_json, resolve_token

#: How the three searches map onto something readable. Ordered: the first
#: search to claim a URL wins, and "you opened this" beats "you commented on
#: this" as a description of the same pull request.
KIND_LABELS = {
    "pr": "opened a PR",
    "commit": "committed",
    "comment": "commented",
}

#: A YYYY-MM heading. Anything else groups under "undated".
_MONTH = re.compile(r"^\d{4}-\d{2}$")


@dataclass(frozen=True)
class ActivityEvent:
    """One thing you did, flattened out of three different API shapes."""

    #: Dedup key. The html_url — unique, stable, and meaningful to a human.
    id: str
    kind: str  # "pr" | "commit" | "comment"
    at: str  # ISO-8601
    repo: str  # owner/name
    title: str
    url: str | None = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "at": self.at,
            "repo": self.repo,
            "title": self.title,
            "url": self.url,
        }


@dataclass
class ReadResult:
    events: list[ActivityEvent] = field(default_factory=list)
    #: Lines that were not parseable as an event. Surfaced by
    #: ``mj activity --debug`` rather than silently dropped.
    skipped: int = 0


@dataclass
class RefreshResult:
    fetched: int = 0
    added: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Reading — never raises
# ---------------------------------------------------------------------------


def _parse_event(raw: object) -> ActivityEvent | None:
    if not isinstance(raw, dict):
        return None
    ident = raw.get("id")
    kind = raw.get("kind")
    at = raw.get("at")
    if not isinstance(ident, str) or not ident:
        return None
    if not isinstance(kind, str) or not isinstance(at, str):
        return None
    url = raw.get("url")
    return ActivityEvent(
        id=ident,
        kind=kind,
        at=at,
        repo=raw.get("repo") if isinstance(raw.get("repo"), str) else "",
        title=raw.get("title") if isinstance(raw.get("title"), str) else "",
        url=url if isinstance(url, str) else None,
    )


def read_events_detailed(path: Path | str | None = None) -> ReadResult:
    """Read the whole log. Never raises."""
    target = Path(path) if path is not None else activity_path()
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
            result.skipped += 1
            continue
        event = _parse_event(decoded)
        if event is None:
            result.skipped += 1
        else:
            result.events.append(event)
    return result


def read_events(path: Path | str | None = None) -> list[ActivityEvent]:
    return read_events_detailed(path).events


def _parse_at(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def recent(
    days: int = 90,
    now: datetime | None = None,
    path: Path | str | None = None,
) -> list[ActivityEvent]:
    """Events inside the window, newest first. Never raises.

    ``now`` is injectable for the same reason it is in the sessions worker:
    reading the wall clock directly makes every fixture-dated test pass today
    and fail next quarter.
    """
    moment = now or datetime.now(timezone.utc)
    cutoff = moment - timedelta(days=days)

    kept = []
    for event in read_events(path):
        stamp = _parse_at(event.at)
        if stamp is None or stamp >= cutoff:
            kept.append(event)

    kept.sort(key=lambda e: e.at, reverse=True)
    return kept


def newest_at(path: Path | str | None = None) -> datetime | None:
    """When the most recent cached event happened, or None if empty."""
    stamps = [s for s in (_parse_at(e.at) for e in read_events(path)) if s]
    return max(stamps) if stamps else None


def _marker_for(path: Path | str | None) -> Path:
    """The fetch marker beside a given log, so tests can use a temp path."""
    if path is None:
        return activity_fetched_path()
    return Path(str(path) + ".fetched")


def record_fetch(
    now: datetime | None = None, path: Path | str | None = None
) -> None:
    """Note that we asked GitHub just now. Never raises."""
    marker = _marker_for(path)
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            (now or datetime.now(timezone.utc)).isoformat(), encoding="utf-8"
        )
    except OSError:
        # A missing marker only means we refresh more often than needed.
        pass


def last_fetch(path: Path | str | None = None) -> datetime | None:
    """When we last asked GitHub, or None if never. Never raises."""
    marker = _marker_for(path)
    if not marker.is_file():
        return None
    try:
        return _parse_at(marker.read_text(encoding="utf-8").strip())
    except OSError:
        return None


def is_stale(
    max_age_hours: float = 6.0,
    now: datetime | None = None,
    path: Path | str | None = None,
) -> bool:
    """Should we refresh before answering? Never fetched counts as stale.

    Measured against the last **fetch**, not the newest event. Using the newest
    event means a week without pushing makes the cache permanently stale, so
    every single question fires three search calls before answering — the cache
    failing hardest for exactly the quiet weeks it exists to cover.
    """
    fetched = last_fetch(path)
    if fetched is None:
        return True
    moment = now or datetime.now(timezone.utc)
    return (moment - fetched) > timedelta(hours=max_age_hours)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def append_events(
    events: list[ActivityEvent],
    path: Path | str | None = None,
) -> int:
    """Append events not already present. Returns how many were new.

    Dedup is against what is on disk, by ``id``. Refreshing twice in a row must
    be a no-op, because the search windows overlap by design.
    """
    target = Path(path) if path is not None else activity_path()
    known = {e.id for e in read_events(target)}

    fresh = []
    for event in events:
        if event.id in known:
            continue
        known.add(event.id)  # also dedups within this batch
        fresh.append(event)

    if not fresh:
        return 0

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as fh:
        for event in fresh:
            fh.write(json.dumps(event.to_json(), ensure_ascii=False) + "\n")
    return len(fresh)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _repo_from_url(url: object) -> str:
    """'https://api.github.com/repos/nav/majordomo' -> 'nav/majordomo'."""
    if not isinstance(url, str):
        return ""
    parts = [p for p in url.rstrip("/").split("/") if p]
    return "/".join(parts[-2:]) if len(parts) >= 2 else ""


def _items(payload: object) -> list:
    if isinstance(payload, dict):
        found = payload.get("items")
        return found if isinstance(found, list) else []
    return payload if isinstance(payload, list) else []


def _as_pr(item: dict) -> ActivityEvent | None:
    url = item.get("html_url")
    if not isinstance(url, str):
        return None
    return ActivityEvent(
        id=url,
        kind="pr",
        at=str(item.get("updated_at") or item.get("created_at") or ""),
        repo=_repo_from_url(item.get("repository_url")),
        title=str(item.get("title") or "(untitled)"),
        url=url,
    )


def _as_commit(item: dict) -> ActivityEvent | None:
    url = item.get("html_url")
    if not isinstance(url, str):
        return None
    commit = item.get("commit") or {}
    author = commit.get("author") or {}
    repository = item.get("repository") or {}
    message = str(commit.get("message") or "")
    return ActivityEvent(
        id=url,
        kind="commit",
        at=str(author.get("date") or ""),
        repo=str(repository.get("full_name") or ""),
        # A commit message is a paragraph; the subject is the fact.
        title=message.split("\n", 1)[0][:200] or "(no message)",
        url=url,
    )


def _as_comment(item: dict) -> ActivityEvent | None:
    url = item.get("html_url")
    if not isinstance(url, str):
        return None
    return ActivityEvent(
        id=url,
        kind="comment",
        at=str(item.get("updated_at") or ""),
        repo=_repo_from_url(item.get("repository_url")),
        title=str(item.get("title") or "(untitled)"),
        url=url,
    )


def fetch(
    cfg: GitHubConfig, token: str, since: str
) -> tuple[list[ActivityEvent], list[str]]:
    """Pull your activity since ``since`` (a YYYY-MM-DD date).

    Returns ``(events, errors)``. Each search is independent and a failure in
    one does **not** discard what the others returned: GitHub's search endpoints
    rate-limit separately and commit search is the flakiest of the three, so
    all-or-nothing meant one 403 on the last call threw away a complete set of
    pull requests already in hand.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    per_page = cfg.activity_per_page

    # Ordered: the first search to claim a URL wins in append_events, and
    # "you opened this PR" beats "you commented" as a description of the same PR.
    searches = (
        ("/search/issues", f"author:@me type:pr updated:>={since}", _as_pr),
        ("/search/commits", f"author:@me author-date:>={since}", _as_commit),
        ("/search/issues", f"commenter:@me updated:>={since}", _as_comment),
    )

    events: list[ActivityEvent] = []
    errors: list[str] = []

    with httpx.Client(base_url=cfg.api_base, headers=headers, timeout=cfg.timeout) as client:
        for url, query, convert in searches:
            try:
                payload = get_json(client, url, {"q": query, "per_page": per_page})
            except GitHubError as exc:
                errors.append(f"{query.split()[0]}: {exc}")
                continue
            for item in _items(payload):
                event = convert(item) if isinstance(item, dict) else None
                if event is not None:
                    events.append(event)

    return events, errors


def refresh(
    config: Config,
    token: str | None = None,
    now: datetime | None = None,
    path: Path | str | None = None,
) -> RefreshResult:
    """Fetch and store. Never raises — a failure is reported, not thrown.

    Same contract as a worker: this runs on the way to answering a question, and
    a GitHub outage must cost you the *freshness* of the answer, not the answer.
    """
    cfg = config.sources.github
    moment = now or datetime.now(timezone.utc)
    since = (moment - timedelta(days=cfg.activity_days)).date().isoformat()

    try:
        resolved = token or resolve_token(cfg)
        events, errors = fetch(cfg, resolved, since)
    except GitHubError as exc:
        return RefreshResult(error=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        return RefreshResult(error=f"unexpected: {exc}")

    added = append_events(events, path)

    # Recorded even on a partial failure: we *did* ask, and the point of the
    # marker is to stop every question re-firing the same searches.
    record_fetch(moment, path)

    return RefreshResult(
        fetched=len(events),
        added=added,
        error="; ".join(errors) if errors else None,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def describe(event: ActivityEvent) -> str:
    label = KIND_LABELS.get(event.kind, event.kind)
    where = f" in {event.repo}" if event.repo else ""
    day = event.at[:10]
    return f"{day}  {label}{where}: {event.title}"


def digest(events: list[ActivityEvent], limit: int = 60) -> str:
    """The activity block that goes into a prompt.

    Grouped by month rather than listed flat: "what was I doing in June" is the
    question this exists to answer, and a model reads a dated heading far more
    reliably than it infers month boundaries from sixty ISO timestamps.
    """
    if not events:
        return "No recorded GitHub activity."

    shown = events[:limit]
    lines: list[str] = []
    current = ""
    for event in shown:
        month = event.at[:7]
        # An unparseable timestamp is kept (losing an event to a bad date is
        # worse than showing it) but must not become a heading — "not-a-d:" is
        # how a rendering bug looks in production.
        if not _MONTH.match(month):
            month = "undated"
        if month != current:
            current = month
            lines.append(f"\n{month}:")
        lines.append(f"  {describe(event)}")

    if len(events) > limit:
        lines.append(f"\n({len(events) - limit} older entries not listed.)")

    return "\n".join(lines).strip()
