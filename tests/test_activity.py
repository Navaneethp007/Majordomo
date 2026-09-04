"""Tests for the GitHub activity store — HTTP fully mocked, no network."""
from __future__ import annotations

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
