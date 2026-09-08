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

Two different prunings, and the distinction matters:

- **The read window** (``recent``) filters in memory and never touches the file.
  This is what ``activity_days`` controls, and it is cheap.
- **Compaction** (``compact``) does rewrite, but only past a size threshold and
  only from ``refresh`` — never from a read path. It keeps a window twice as
  wide as the read window, because dropping an event the moment it leaves view
  means a later widening of ``activity_days`` finds nothing behind it.

Rewriting an append-only log is where such logs get corrupted, so the rewrite
goes through ``jsonlog.rewrite`` — write beside the target, then replace, which
is atomic. Shared with ``state.py`` so a fix to it reaches both.
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
from majordomo import jsonlog
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

    The window is **calendar days**, counted back from midnight today. A rolling
    ``now - days`` made ``--days 0`` a cutoff of *this instant*, so it could only
    ever match events in the future and always returned nothing — while the CLI
    described it as "just today". Counting from midnight makes 0 mean today, 1
    mean since yesterday morning, and 7 mean the last week, which is what the
    flag reads as.

    ``now`` is injectable for the same reason it is in the sessions worker:
    reading the wall clock directly makes every fixture-dated test pass today
    and fail next quarter.
    """
    moment = now or datetime.now(timezone.utc)
    # Midnight *where you are*, not midnight UTC. "Today" is a local idea: at
    # UTC-5, snapping to UTC midnight puts the cutoff five hours into your
    # morning and `--days 0` silently drops everything you did before lunch.
    local = moment.astimezone()
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = midnight - timedelta(days=max(0, days))

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
# Compaction — the same shape as state.compact, deliberately
# ---------------------------------------------------------------------------

#: Compact once the log passes this. Roughly 10k events, or years of activity.
#: Re-exported from ``jsonlog`` so callers of this module keep working.
COMPACT_OVER_BYTES = jsonlog.COMPACT_OVER_BYTES

#: Keep this multiple of ``activity_days`` when compacting. Wider than the read
#: window on purpose: dropping an event the moment it leaves the window means a
#: later widening of ``activity_days`` finds nothing behind it, and re-fetching
#: only reaches as far back as GitHub's search will go.
KEEP_WINDOW_MULTIPLE = 2

#: Never compact to a window narrower than this, whatever ``activity_days`` says.
#: The read window is a display preference and can legitimately be 0 ("today");
#: the *cache* is history, and history you throw away does not come back — the
#: search API stops at 1000 results however far you ask it to reach.
MIN_KEEP_DAYS = 30


def compact(
    path: Path | str | None = None,
    keep_days: int = 180,
    now: datetime | None = None,
) -> int:
    """Rewrite the log keeping only what is still in range. Returns lines dropped.

    Shares the rewrite with :func:`majordomo.state.compact` via ``jsonlog`` —
    the atomic part, which is what matters. What each keeps differs and stays
    here: sessions carry an old event forward to preserve a topic, and activity
    simply drops anything outside the window.

    Simpler than the state version in one respect: there is no topic to carry
    forward, so an event outside the window is simply gone.
    """
    target = Path(path) if path is not None else activity_path()
    if not target.is_file():
        return 0

    moment = now or datetime.now(timezone.utc)
    cutoff = moment - timedelta(days=keep_days)

    events = read_events(target)
    if not events:
        return 0

    # An unparseable timestamp is kept, matching `recent`: losing an event to a
    # malformed date is worse than carrying it.
    kept = [
        e for e in events
        if (stamp := _parse_at(e.at)) is None or stamp >= cutoff
    ]

    dropped = len(events) - len(kept)
    if dropped <= 0:
        return 0

    jsonlog.rewrite(target, kept)

    return dropped


def maybe_compact(
    path: Path | str | None = None,
    keep_days: int = 180,
    max_bytes: int = COMPACT_OVER_BYTES,
) -> int:
    """Compact only if the log has actually got big. Never raises."""
    target = Path(path) if path is not None else activity_path()
    try:
        if not jsonlog.is_large(target, max_bytes):
            return 0
        return compact(target, keep_days=keep_days)
    except OSError:
        # Housekeeping must never be the reason a question goes unanswered.
        return 0


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


def _total_count(payload: object) -> int | None:
    """How many results GitHub says exist, or None if it didn't say.

    This is what makes truncation visible. Search responses carry it and we
    used to read only ``items`` — so a search returning exactly ``per_page``
    results was indistinguishable from one that happened to have that many.
    """
    if isinstance(payload, dict):
        total = payload.get("total_count")
        if isinstance(total, int):
            return total
    return None


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


#: GitHub's search API will not return results beyond this offset. Ask for page
#: 11 at 100 per page and you get a 422, not an empty page — so this is a wall,
#: not a preference, and no configuration can move it.
SEARCH_RESULT_CEILING = 1_000


#: GitHub rejects a per_page above this and substitutes its own default.
MAX_PER_PAGE = 100


def _per_page(cfg: GitHubConfig) -> int:
    """Results per request, clamped to what the API will actually honour.

    Clamping only inside ``_last_page`` was not enough: the unclamped value also
    went into the request *and* into the ``len(items) < per_page`` stop
    condition, where a 0 can never be reached — so every page was fetched
    whether or not there was anything left. Same shape as the ``max_messages: 0``
    bug in the Gmail worker: a limit that is only enforced in one of the places
    it is read.
    """
    return max(1, min(cfg.activity_per_page, MAX_PER_PAGE))


def _last_page(cfg: GitHubConfig) -> int:
    """The highest page worth requesting: your setting, or the API's wall."""
    return max(1, min(cfg.activity_max_pages, SEARCH_RESULT_CEILING // _per_page(cfg)))


def _how_to_get_more(cfg: GitHubConfig) -> str:
    """What to actually do about a truncated search.

    This used to say "raise sources.github.activity_max_pages" unconditionally.
    Past the API's ceiling that advice is worse than none: following it turns a
    silent truncation into a 422 on every refresh. Once you are at the wall the
    only thing that works is asking for a narrower window.
    """
    if cfg.activity_max_pages < SEARCH_RESULT_CEILING // _per_page(cfg):
        return "raise sources.github.activity_max_pages to reach the rest"
    return (
        f"that is GitHub's {SEARCH_RESULT_CEILING}-result search limit, not a "
        f"setting — narrow the window with --days to see further back"
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
    per_page = _per_page(cfg)

    # Ordered: the first search to claim a URL wins in append_events, and
    # "you opened this PR" beats "you commented" as a description of the same PR.
    # Each carries its own label. Deriving one from the query gave `author:@me`
    # for both the pull-request and the commit search, so a truncation or error
    # warning could not say which endpoint it came from — and those two fail for
    # different reasons and at different rates.
    searches = (
        ("pull requests", "/search/issues", f"author:@me type:pr updated:>={since}", _as_pr),
        ("commits", "/search/commits", f"author:@me author-date:>={since}", _as_commit),
        ("comments", "/search/issues", f"commenter:@me updated:>={since}", _as_comment),
    )

    events: list[ActivityEvent] = []
    errors: list[str] = []
    notices: list[str] = []

    with httpx.Client(base_url=cfg.api_base, headers=headers, timeout=cfg.timeout) as client:
        for label, url, query, convert in searches:
            collected = 0
            total = None

            for page in range(1, _last_page(cfg) + 1):
                try:
                    payload = get_json(
                        client,
                        url,
                        {"q": query, "per_page": per_page, "page": page},
                    )
                except GitHubError as exc:
                    # Page 1 failing means this search returned nothing; a later
                    # page failing still leaves the earlier ones in `events`.
                    errors.append(f"{label}: {exc}")
                    break

                if total is None:
                    total = _total_count(payload)

                items = _items(payload)
                for item in items:
                    event = convert(item) if isinstance(item, dict) else None
                    if event is not None:
                        events.append(event)
                collected += len(items)

                # A short page is the last page — asking for another wastes a
                # request against a rate-limited endpoint.
                if len(items) < per_page:
                    break

            # Say so when GitHub had more than we took. Before this, a search
            # returning exactly `per_page` results looked identical to one that
            # had exactly that many — 100 commits in a 90-day window was a
            # ceiling being reported as a count.
            if total is not None and total > collected:
                # A *notice*, not an error. Sharing the errors list made a
                # completely successful refresh report "could not refresh",
                # which is the opposite of what happened.
                notices.append(
                    f"{label}: took {collected} of {total} — {_how_to_get_more(cfg)}"
                )

    return events, errors + notices


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

    # Housekeeping at the natural mutation point. Not on read: reads happen on
    # the way to answering a question and must stay cheap, and this is already
    # the slow path.
    # Floored, because `activity_days: 0` is a legitimate setting — it means
    # "show me today" on the read path — and multiplying it gives a keep-window
    # of zero, which compacts the entire history away. The cache cannot be
    # rebuilt past GitHub's 1000-result search ceiling, so that is permanent
    # data loss triggered by a config value we deliberately made valid.
    keep_days = max(MIN_KEEP_DAYS, cfg.activity_days * KEEP_WINDOW_MULTIPLE)
    maybe_compact(path, keep_days=keep_days)

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
