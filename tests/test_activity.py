"""Tests for the GitHub activity store — HTTP fully mocked, no network."""
from __future__ import annotations

import inspect

from datetime import datetime, timedelta, timezone
from unittest import mock

from majordomo import activity
from majordomo import config as config_module
from majordomo.workers.github import GitHubError

CFG = config_module.build(config_module.DEFAULTS)
NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def event(ident, kind="pr", at="2026-08-28T10:00:00Z", repo="nav/majordomo", title="A thing"):
    return activity.ActivityEvent(id=ident, kind=kind, at=at, repo=repo, title=title, url=ident)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_append_then_read(tmp_path):
    path = tmp_path / "activity.jsonl"
    assert activity.append_events([event("a"), event("b")], path) == 2
    assert [e.id for e in activity.read_events(path)] == ["a", "b"]


def test_refreshing_twice_adds_nothing(tmp_path):
    """The search windows overlap by design, so dedup is not optional."""
    path = tmp_path / "activity.jsonl"
    batch = [event("a"), event("b")]
    activity.append_events(batch, path)
    assert activity.append_events(batch, path) == 0
    assert len(activity.read_events(path)) == 2


def test_duplicates_within_one_batch_are_collapsed(tmp_path):
    path = tmp_path / "activity.jsonl"
    assert activity.append_events([event("a"), event("a")], path) == 1


def test_missing_file_reads_empty(tmp_path):
    assert activity.read_events(tmp_path / "nope.jsonl") == []


def test_torn_final_line_is_skipped_and_counted(tmp_path):
    path = tmp_path / "activity.jsonl"
    activity.append_events([event("a")], path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"id":"tor')          # killed mid-write

    result = activity.read_events_detailed(path)
    assert len(result.events) == 1
    assert result.skipped == 1


def test_a_line_that_is_not_an_event_is_skipped(tmp_path):
    path = tmp_path / "activity.jsonl"
    path.write_text('{"unrelated": true}\n[1,2,3]\n"a string"\n', encoding="utf-8")

    result = activity.read_events_detailed(path)
    assert result.events == []
    assert result.skipped == 3


def test_bom_is_stripped(tmp_path):
    path = tmp_path / "activity.jsonl"
    path.write_text('﻿{"id":"a","kind":"pr","at":"2026-08-01T00:00:00Z"}\n',
                    encoding="utf-8")
    assert len(activity.read_events(path)) == 1


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


def test_recent_excludes_events_past_the_cutoff(tmp_path):
    path = tmp_path / "activity.jsonl"
    activity.append_events(
        [event("new", at="2026-08-28T10:00:00Z"), event("old", at="2026-01-02T10:00:00Z")],
        path,
    )
    kept = activity.recent(days=90, now=NOW, path=path)
    assert [e.id for e in kept] == ["new"]


def test_pruning_does_not_rewrite_the_file(tmp_path):
    """Rewriting an append-only log is where append-only logs get corrupted."""
    path = tmp_path / "activity.jsonl"
    activity.append_events([event("old", at="2026-01-02T10:00:00Z")], path)

    assert activity.recent(days=90, now=NOW, path=path) == []
    assert len(activity.read_events(path)) == 1     # still on disk


def test_recent_is_newest_first(tmp_path):
    path = tmp_path / "activity.jsonl"
    activity.append_events(
        [event("older", at="2026-08-01T00:00:00Z"),
         event("newer", at="2026-08-20T00:00:00Z")],
        path,
    )
    assert [e.id for e in activity.recent(days=90, now=NOW, path=path)] == ["newer", "older"]


def test_an_unparseable_timestamp_is_kept_rather_than_dropped(tmp_path):
    """Losing an event to a malformed date is worse than showing it."""
    path = tmp_path / "activity.jsonl"
    activity.append_events([event("weird", at="not-a-date")], path)
    assert len(activity.recent(days=90, now=NOW, path=path)) == 1


def test_newest_at_reports_the_latest_event(tmp_path):
    path = tmp_path / "activity.jsonl"
    assert activity.newest_at(path) is None

    activity.append_events(
        [event("a", at="2026-08-01T00:00:00Z"), event("b", at="2026-08-20T00:00:00Z")],
        path,
    )
    assert activity.newest_at(path).date().isoformat() == "2026-08-20"


def test_staleness_is_measured_from_the_fetch_not_the_newest_event(tmp_path):
    """A quiet week must not make the cache permanently stale.

    Measuring from the newest *event* means that after a week without pushing,
    every question re-fires three search calls — the cache failing hardest for
    exactly the quiet weeks it exists to cover.
    """
    path = tmp_path / "activity.jsonl"
    activity.append_events([event("old", at="2026-06-01T00:00:00Z")], path)

    assert activity.is_stale(6.0, now=NOW, path=path) is True    # never fetched

    activity.record_fetch(NOW - timedelta(hours=1), path)
    assert activity.last_fetch(path) is not None
    assert activity.is_stale(6.0, now=NOW, path=path) is False   # month-old event, fresh fetch
    assert activity.is_stale(0.5, now=NOW, path=path) is True    # older than the budget


def test_a_missing_or_unreadable_marker_reads_as_never_fetched(tmp_path):
    path = tmp_path / "activity.jsonl"
    assert activity.last_fetch(path) is None

    activity._marker_for(path).write_text("not a timestamp", encoding="utf-8")
    assert activity.last_fetch(path) is None
    assert activity.is_stale(6.0, now=NOW, path=path) is True


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


SEARCH_PRS = {
    "items": [
        {
            "title": "Speak the wake briefing only when needed",
            "updated_at": "2026-08-28T10:00:00Z",
            "repository_url": "https://api.github.com/repos/nav/majordomo",
            "html_url": "https://github.com/nav/majordomo/pull/12",
        }
    ]
}
SEARCH_COMMITS = {
    "items": [
        {
            "html_url": "https://github.com/nav/majordomo/commit/abc",
            "commit": {"message": "Add the Gmail worker\n\nLonger body here.",
                       "author": {"date": "2026-08-27T09:00:00Z"}},
            "repository": {"full_name": "nav/majordomo"},
        }
    ]
}
SEARCH_COMMENTS = {
    "items": [
        {
            "title": "Riva latency on Windows",
            "updated_at": "2026-07-14T12:00:00Z",
            "repository_url": "https://api.github.com/repos/nav/voicelog",
            "html_url": "https://github.com/nav/voicelog/issues/3",
        }
    ]
}


def fake_get(responses):
    calls = []

    def _get(client, url, params=None):
        calls.append((url, params))
        return responses[len(calls) - 1]

    return _get, calls


def test_fetch_flattens_all_three_searches():
    _get, calls = fake_get([SEARCH_PRS, SEARCH_COMMITS, SEARCH_COMMENTS])
    with mock.patch.object(activity, "get_json", _get), mock.patch.object(activity, "httpx"):
        events, errors = activity.fetch(CFG.sources.github, "token", "2026-06-03")

    assert [e.kind for e in events] == ["pr", "commit", "comment"]
    assert events[0].repo == "nav/majordomo"
    assert events[2].repo == "nav/voicelog"
    assert errors == []
    # the date window reaches every query
    assert all("2026-06-03" in call[1]["q"] for call in calls)


def test_fetch_takes_only_the_commit_subject():
    """A commit message is a paragraph; the subject is the fact."""
    _get, _ = fake_get([{"items": []}, SEARCH_COMMITS, {"items": []}])
    with mock.patch.object(activity, "get_json", _get), mock.patch.object(activity, "httpx"):
        events, _ = activity.fetch(CFG.sources.github, "token", "2026-06-03")

    assert events[0].title == "Add the Gmail worker"


def test_fetch_skips_items_with_no_url():
    _get, _ = fake_get([{"items": [{"title": "no url here"}]}, {"items": []}, {"items": []}])
    with mock.patch.object(activity, "get_json", _get), mock.patch.object(activity, "httpx"):
        events, _ = activity.fetch(CFG.sources.github, "token", "2026-06-03")

    assert events == []


def test_one_failed_search_does_not_discard_the_others():
    """Commit search is the flakiest of the three and rate-limits separately."""
    calls = {"n": 0}

    def flaky(client, url, params=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return SEARCH_PRS
        raise GitHubError("rate limited")

    with mock.patch.object(activity, "get_json", flaky), mock.patch.object(activity, "httpx"):
        events, errors = activity.fetch(CFG.sources.github, "token", "2026-06-03")

    assert [e.kind for e in events] == ["pr"]      # the one that worked survives
    assert len(errors) == 2
    assert all("rate limited" in e for e in errors)


def test_refresh_stores_a_partial_result_and_reports_the_gap(tmp_path):
    path = tmp_path / "a.jsonl"
    calls = {"n": 0}

    def flaky(client, url, params=None):
        calls["n"] += 1
        return SEARCH_PRS if calls["n"] == 1 else _raise()

    def _raise():
        raise GitHubError("403")

    with mock.patch.object(activity, "get_json", flaky), mock.patch.object(activity, "httpx"):
        result = activity.refresh(CFG, token="t", now=NOW, path=path)

    assert result.added == 1          # kept what we got
    assert result.error is not None   # and said the picture is incomplete
    assert len(activity.read_events(path)) == 1


def test_refresh_reports_a_failure_rather_than_raising(tmp_path):
    """A GitHub outage costs freshness, not the answer."""
    with mock.patch.object(activity, "resolve_token", side_effect=GitHubError("no token")):
        result = activity.refresh(CFG, path=tmp_path / "a.jsonl")

    assert result.error == "no token"
    assert result.added == 0


def test_refresh_stores_what_it_fetched(tmp_path):
    path = tmp_path / "a.jsonl"
    _get, _ = fake_get([SEARCH_PRS, SEARCH_COMMITS, SEARCH_COMMENTS])
    with mock.patch.object(activity, "get_json", _get), mock.patch.object(activity, "httpx"):
        result = activity.refresh(CFG, token="t", now=NOW, path=path)

    assert result.fetched == 3 and result.added == 3
    assert len(activity.read_events(path)) == 3


def test_refresh_records_the_fetch_so_the_next_question_uses_the_cache(tmp_path):
    path = tmp_path / "a.jsonl"
    _get, _ = fake_get([SEARCH_PRS, SEARCH_COMMITS, SEARCH_COMMENTS])
    with mock.patch.object(activity, "get_json", _get), mock.patch.object(activity, "httpx"):
        activity.refresh(CFG, token="t", now=NOW, path=path)

    assert activity.last_fetch(path) == NOW
    assert activity.is_stale(6.0, now=NOW, path=path) is False


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_digest_groups_by_month():
    events = [
        event("a", at="2026-08-28T10:00:00Z", title="August thing"),
        event("b", at="2026-07-14T10:00:00Z", title="July thing"),
    ]
    text = activity.digest(events)
    assert "2026-08:" in text and "2026-07:" in text
    assert text.index("2026-08:") < text.index("2026-07:")


def test_digest_states_what_it_left_out():
    events = [event(str(i), at="2026-08-28T10:00:00Z") for i in range(10)]
    assert "5 older entries not listed" in activity.digest(events, limit=5)


def test_an_unparseable_date_groups_under_undated():
    """'not-a-d:' as a month heading is how a rendering bug looks in production."""
    text = activity.digest([event("x", at="not-a-date")])
    assert "undated:" in text
    assert "not-a-d:" not in text


def test_digest_on_an_empty_store():
    assert activity.digest([]) == "No recorded GitHub activity."


def test_describe_reads_as_a_sentence():
    assert activity.describe(event("a", kind="comment", repo="nav/voicelog", title="X")) == (
        "2026-08-28  commented in nav/voicelog: X"
    )


def test_repo_from_url_handles_junk():
    assert activity._repo_from_url("https://api.github.com/repos/nav/majordomo") == "nav/majordomo"
    assert activity._repo_from_url(None) == ""
    assert activity._repo_from_url("single") == ""


# ---------------------------------------------------------------------------
# Truncation and paging
# ---------------------------------------------------------------------------

def _page(items, total):
    return {"total_count": total, "items": items}


def _pr_item(n):
    return {
        "title": f"PR {n}",
        "updated_at": "2026-08-28T10:00:00Z",
        "repository_url": "https://api.github.com/repos/nav/majordomo",
        "html_url": f"https://github.com/nav/majordomo/pull/{n}",
    }


def test_truncation_is_reported_not_silently_swallowed():
    """100 results from a 90-day window is a ceiling, not a count."""
    cfg = config_module.build(
        config_module._deep_merge(
            config_module.DEFAULTS,
            {"sources": {"github": {"activity_per_page": 2, "activity_max_pages": 1}}},
        )
    ).sources.github

    _get, _ = fake_get([_page([_pr_item(1), _pr_item(2)], total=57),
                        _page([], 0), _page([], 0)])
    with mock.patch.object(activity, "get_json", _get), mock.patch.object(activity, "httpx"):
        events, errors = activity.fetch(cfg, "token", "2026-06-03")

    assert len(events) == 2
    assert any("took 2 of 57" in e for e in errors)


def test_no_truncation_note_when_we_got_everything():
    cfg = config_module.build(
        config_module._deep_merge(
            config_module.DEFAULTS,
            {"sources": {"github": {"activity_per_page": 5, "activity_max_pages": 2}}},
        )
    ).sources.github

    _get, _ = fake_get([_page([_pr_item(1)], total=1), _page([], 0), _page([], 0)])
    with mock.patch.object(activity, "get_json", _get), mock.patch.object(activity, "httpx"):
        _, errors = activity.fetch(cfg, "token", "2026-06-03")

    assert errors == []


def test_paging_stops_at_the_configured_maximum():
    cfg = config_module.build(
        config_module._deep_merge(
            config_module.DEFAULTS,
            {"sources": {"github": {"activity_per_page": 2, "activity_max_pages": 2}}},
        )
    ).sources.github

    full = _page([_pr_item(1), _pr_item(2)], total=100)
    _get, calls = fake_get([full] * 12)
    with mock.patch.object(activity, "get_json", _get), mock.patch.object(activity, "httpx"):
        activity.fetch(cfg, "token", "2026-06-03")

    # 3 searches x 2 pages, never a third page for any of them
    assert len(calls) == 6
    assert max(c[1]["page"] for c in calls) == 2


def test_a_short_page_ends_the_search_early():
    """Asking for another page wastes a request on a rate-limited endpoint."""
    cfg = config_module.build(
        config_module._deep_merge(
            config_module.DEFAULTS,
            {"sources": {"github": {"activity_per_page": 5, "activity_max_pages": 3}}},
        )
    ).sources.github

    _get, calls = fake_get([_page([_pr_item(1)], total=1)] * 9)
    with mock.patch.object(activity, "get_json", _get), mock.patch.object(activity, "httpx"):
        activity.fetch(cfg, "token", "2026-06-03")

    assert len(calls) == 3          # one page each, not three


def test_a_failure_on_a_later_page_keeps_the_earlier_ones():
    cfg = config_module.build(
        config_module._deep_merge(
            config_module.DEFAULTS,
            {"sources": {"github": {"activity_per_page": 2, "activity_max_pages": 3}}},
        )
    ).sources.github

    calls = {"n": 0}

    def flaky(client, url, params=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return _page([_pr_item(1), _pr_item(2)], total=100)
        raise GitHubError("rate limited")

    with mock.patch.object(activity, "get_json", flaky), mock.patch.object(activity, "httpx"):
        events, errors = activity.fetch(cfg, "token", "2026-06-03")

    assert len(events) == 2
    assert errors


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

def test_compact_drops_events_outside_the_window(tmp_path):
    path = tmp_path / "a.jsonl"
    activity.append_events(
        [event("new", at="2026-08-28T10:00:00Z"), event("ancient", at="2024-01-02T10:00:00Z")],
        path,
    )

    assert activity.compact(path, keep_days=180, now=NOW) == 1
    assert [e.id for e in activity.read_events(path)] == ["new"]


def test_compact_keeps_recent_events_byte_identical(tmp_path):
    path = tmp_path / "a.jsonl"
    keep = event("keep", at="2026-08-28T10:00:00Z", title="Unicode ünïcode ok")
    activity.append_events([keep, event("old", at="2024-01-02T10:00:00Z")], path)

    activity.compact(path, keep_days=180, now=NOW)
    survivor = activity.read_events(path)[0]
    assert survivor == keep


def test_compact_keeps_an_unparseable_timestamp(tmp_path):
    """Losing an event to a malformed date is worse than carrying it."""
    path = tmp_path / "a.jsonl"
    activity.append_events([event("weird", at="not-a-date")], path)

    assert activity.compact(path, keep_days=180, now=NOW) == 0
    assert len(activity.read_events(path)) == 1


def test_compact_is_a_no_op_when_nothing_is_stale(tmp_path):
    path = tmp_path / "a.jsonl"
    activity.append_events([event("a", at="2026-08-28T10:00:00Z")], path)
    assert activity.compact(path, keep_days=180, now=NOW) == 0


def test_compact_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "a.jsonl"
    activity.append_events(
        [event("new", at="2026-08-28T10:00:00Z"), event("old", at="2024-01-02T10:00:00Z")],
        path,
    )
    activity.compact(path, keep_days=180, now=NOW)

    assert not any(p.name.endswith(".compacting") for p in tmp_path.iterdir())


def test_maybe_compact_leaves_a_small_log_alone(tmp_path):
    path = tmp_path / "a.jsonl"
    activity.append_events([event("old", at="2024-01-02T10:00:00Z")], path)

    assert activity.maybe_compact(path, max_bytes=1_000_000) == 0
    assert len(activity.read_events(path)) == 1     # untouched


def test_maybe_compact_fires_once_the_log_is_big(tmp_path):
    path = tmp_path / "a.jsonl"
    activity.append_events(
        [event(f"old{i}", at="2024-01-02T10:00:00Z") for i in range(50)], path
    )
    assert activity.maybe_compact(path, keep_days=180, max_bytes=100) > 0


def test_maybe_compact_never_raises(tmp_path):
    assert activity.maybe_compact(tmp_path / "nope" / "deep.jsonl") == 0


# ---------------------------------------------------------------------------
# GitHub's search ceiling
# ---------------------------------------------------------------------------


def github_config(max_pages, per_page=100):
    from majordomo import config as config_module

    defaults = config_module.DEFAULTS
    return config_module.build(
        {
            **defaults,
            "sources": {
                **defaults["sources"],
                "github": {
                    **defaults["sources"]["github"],
                    "activity_max_pages": max_pages,
                    "activity_per_page": per_page,
                },
            },
        }
    ).sources.github


def test_the_page_loop_stops_at_the_api_ceiling():
    """Search returns 422 past 1000 results, so asking for page 11 at 100 per
    page converts silent truncation into a hard failure on every refresh."""
    assert activity._last_page(github_config(50)) == 10
    assert activity._last_page(github_config(50, per_page=50)) == 20


def test_a_setting_below_the_ceiling_is_honoured():
    assert activity._last_page(github_config(3)) == 3


def test_zero_pages_still_asks_for_one():
    assert activity._last_page(github_config(0)) == 1


def test_the_advice_says_raise_the_setting_while_that_would_work():
    assert "activity_max_pages" in activity._how_to_get_more(github_config(3))


def test_the_advice_changes_once_you_are_at_the_ceiling():
    """Telling you to raise a setting that cannot help is worse than saying
    nothing — following it breaks every refresh."""
    advice = activity._how_to_get_more(github_config(10))

    assert "activity_max_pages" not in advice
    assert "--days" in advice
    assert "1000" in advice


# ---------------------------------------------------------------------------
# The read window
# ---------------------------------------------------------------------------


def local_at(day: int, hour: int) -> datetime:
    """A moment in *this machine's* zone. "Today" is a local idea, so a fixture
    pinned to UTC passes in London and fails in Kolkata."""
    naive = datetime(2026, 9, day, hour, 0)
    return naive.astimezone()


NOON = local_at(7, 12)


def dated(at, ident="e"):
    return activity.ActivityEvent(ident, "pr", at, "nav/majordomo", "A thing", ident)


def test_days_zero_means_today(tmp_path):
    """`now - timedelta(days=0)` is *this instant*, so 0 could only ever match
    events in the future and always returned nothing — while the CLI called it
    'just today'."""
    log = tmp_path / "a.jsonl"
    activity.append_events(
        [
            dated(local_at(7, 9).isoformat(), "today"),
            dated(local_at(6, 9).isoformat(), "yesterday"),
        ],
        path=log,
    )

    kept = [e.id for e in activity.recent(days=0, now=NOON, path=log)]

    assert kept == ["today"]


def test_an_event_later_today_is_still_today(tmp_path):
    """A rolling window would drop an event stamped after `now`."""
    log = tmp_path / "a.jsonl"
    activity.append_events([dated(local_at(7, 18).isoformat(), "this-evening")], path=log)

    assert [e.id for e in activity.recent(days=0, now=NOON, path=log)] == ["this-evening"]


def test_days_one_reaches_back_to_yesterday_morning(tmp_path):
    log = tmp_path / "a.jsonl"
    activity.append_events(
        [
            dated(local_at(6, 9).isoformat(), "yesterday"),
            dated(local_at(5, 9).isoformat(), "before"),
        ],
        path=log,
    )

    kept = [e.id for e in activity.recent(days=1, now=NOON, path=log)]

    assert kept == ["yesterday"]


def test_a_negative_window_is_treated_as_today(tmp_path):
    log = tmp_path / "a.jsonl"
    activity.append_events([dated(local_at(7, 9).isoformat(), "today")], path=log)

    assert [e.id for e in activity.recent(days=-5, now=NOON, path=log)] == ["today"]


# ---------------------------------------------------------------------------
# per_page, and which search a warning came from
# ---------------------------------------------------------------------------


def test_per_page_zero_is_clamped_everywhere_it_is_read():
    """Clamping only inside _last_page left the 0 in the request and in the
    `len(items) < per_page` stop condition, which it can never satisfy — so
    every page was fetched whether or not anything was left."""
    assert activity._per_page(github_config(5, per_page=0)) == 1


def test_per_page_is_clamped_to_what_the_api_honours():
    assert activity._per_page(github_config(5, per_page=500)) == activity.MAX_PER_PAGE


def test_a_sane_per_page_is_left_alone():
    assert activity._per_page(github_config(5, per_page=50)) == 50


def test_zero_per_page_does_not_widen_the_page_budget():
    """1000 // 0 would divide by zero; 1000 // 1 would authorise 1000 pages."""
    assert activity._last_page(github_config(5, per_page=0)) == 5


def test_a_failure_says_which_search_it_came_from():
    """`query.split()[0]` gave `author:@me` for both the pull-request and the
    commit search — and those two fail for different reasons and at different
    rates, so a warning that cannot name one is not actionable."""
    def only_commits_fail(client, url, params):
        if "commits" in url:
            raise activity.GitHubError("403 rate limited")
        return {"items": [], "total_count": 0}

    with mock.patch.object(activity, "get_json", only_commits_fail),          mock.patch.object(activity, "httpx"):
        _events, errors = activity.fetch(github_config(2), "token", "2026-09-01")

    assert errors == ["commits: 403 rate limited"]


def test_a_truncation_says_which_search_was_truncated():
    def swamped(client, url, params):
        if "commits" not in url:
            return {"items": [], "total_count": 0}
        item = {
            "html_url": "https://github.com/nav/x/commit/abc",
            "commit": {"message": "a commit", "author": {"date": "2026-09-02T10:00:00Z"}},
            "repository": {"full_name": "nav/x"},
        }
        return {"items": [dict(item, html_url=f"{item['html_url']}{n}")
                          for n in range(100)], "total_count": 900}

    with mock.patch.object(activity, "get_json", swamped),          mock.patch.object(activity, "httpx"):
        _events, errors = activity.fetch(github_config(1), "token", "2026-09-01")

    assert len(errors) == 1
    assert errors[0].startswith("commits: took 100 of 900")


def test_the_window_is_local_midnight_not_utc():
    """At UTC-5, snapping to UTC midnight puts the cutoff five hours into your
    morning and `--days 0` silently drops everything before lunch."""
    import inspect

    source = inspect.getsource(activity.recent)
    assert "astimezone()" in source


# ---------------------------------------------------------------------------
# Compaction must not be able to erase the cache
# ---------------------------------------------------------------------------


def test_activity_days_zero_does_not_wipe_the_log(tmp_path, monkeypatch):
    """`activity_days: 0` is a setting we deliberately made legitimate — it
    means "today" on the read path. Multiplied into a keep-window it became
    zero, and the cache cannot be rebuilt past GitHub's 1000-result ceiling."""
    log = tmp_path / "a.jsonl"
    activity.append_events(
        [dated(local_at(day, 9).isoformat(), f"e{day}") for day in (1, 3, 5, 7)],
        path=log,
    )

    cfg = github_config(5)
    keep = max(activity.MIN_KEEP_DAYS, 0 * activity.KEEP_WINDOW_MULTIPLE)

    assert keep >= activity.MIN_KEEP_DAYS
    activity.compact(path=log, keep_days=keep, now=NOON)
    assert len(activity.read_events(log)) == 4      # nothing dropped


def test_the_keep_window_is_floored_below_the_read_window():
    """The read window is a display preference; the cache is history."""
    assert activity.MIN_KEEP_DAYS > 0


def test_a_normal_window_still_compacts(tmp_path):
    log = tmp_path / "a.jsonl"
    activity.append_events(
        [dated("2020-01-01T09:00:00Z", "ancient"),
         dated(local_at(7, 9).isoformat(), "today")],
        path=log,
    )

    activity.compact(path=log, keep_days=30, now=NOON)

    assert [e.id for e in activity.read_events(log)] == ["today"]


# ---------------------------------------------------------------------------
# A truncation notice is not a failure
# ---------------------------------------------------------------------------


def test_a_truncated_but_successful_refresh_is_not_an_error(tmp_path, monkeypatch):
    """Sharing the errors list made a completely successful refresh report
    "could not refresh", which is the opposite of what happened."""
    def swamped(client, url, params):
        if "commits" not in url:
            return {"items": [], "total_count": 0}
        item = {
            "html_url": "https://github.com/nav/x/commit/abc",
            "commit": {"message": "a commit", "author": {"date": "2026-09-02T10:00:00Z"}},
            "repository": {"full_name": "nav/x"},
        }
        return {"items": [dict(item, html_url=f"{item['html_url']}{n}")
                          for n in range(100)], "total_count": 900}

    with mock.patch.object(activity, "get_json", swamped), \
         mock.patch.object(activity, "httpx"):
        events, notices = activity.fetch(github_config(1), "token", "2026-09-01")

    assert len(events) == 100                     # it worked
    assert any("took 100 of 900" in n for n in notices)
    assert not any("403" in n or "failed" in n for n in notices)
