"""Tests for the Local-Sessions worker — pure over a fixture log, no I/O."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from majordomo import config as config_module
from majordomo.models import SessionEvent
from majordomo.workers import sessions

NOW = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)


def event(session_id, status, at=NOW, topic=None, surface="terminal", cwd="c:/repos/thing"):
    return SessionEvent(
        session_id=session_id,
        surface=surface,
        cwd=cwd,
        status=status,
        at=at.isoformat(),
        topic=topic,
    )


# ---------------------------------------------------------------------------
# fold
# ---------------------------------------------------------------------------

def test_last_event_wins():
    folded = sessions.fold(
        [event("s1", "active"), event("s1", "blocked")], now=NOW
    )
    assert [s.status for s in folded] == ["blocked"]


def test_topic_is_carried_forward():
    """UserPromptSubmit sets the topic; the later blocked event doesn't repeat
    it, but the session is still about that thing."""
    folded = sessions.fold(
        [
            event("s1", "active", topic="wire up the router"),
            event("s1", "blocked"),
        ],
        now=NOW,
    )
    assert folded[0].topic == "wire up the router"


def test_newer_topic_replaces_older():
    folded = sessions.fold(
        [
            event("s1", "active", topic="first thing"),
            event("s1", "active", topic="second thing"),
        ],
        now=NOW,
    )
    assert folded[0].topic == "second thing"


def test_ended_sessions_are_dropped():
    folded = sessions.fold(
        [event("s1", "active"), event("s1", "ended"), event("s2", "active")], now=NOW
    )
    assert [s.session_id for s in folded] == ["s2"]


def test_stale_sessions_are_dropped():
    """A machine that slept for a week shouldn't brief last Tuesday's terminal."""
    old = NOW - timedelta(hours=100)
    folded = sessions.fold(
        [event("old", "idle_awaiting_you", at=old), event("new", "idle_awaiting_you")],
        now=NOW,
        stale_after_hours=72,
    )
    assert [s.session_id for s in folded] == ["new"]


def test_blocked_sorts_before_idle_before_active():
    folded = sessions.fold(
        [
            event("a", "active"),
            event("i", "idle_awaiting_you"),
            event("b", "blocked"),
        ],
        now=NOW,
    )
    assert [s.session_id for s in folded] == ["b", "i", "a"]


def test_cwd_is_carried_forward_when_missing():
    folded = sessions.fold(
        [event("s1", "active", cwd="c:/repos/thing"), event("s1", "blocked", cwd="")],
        now=NOW,
    )
    assert folded[0].cwd == "c:/repos/thing"


def test_empty_log_folds_to_nothing():
    assert sessions.fold([], now=NOW) == []


def test_unparseable_timestamp_does_not_drop_the_session():
    """Better to brief a session with a bad clock than to silently lose it."""
    broken = SessionEvent(
        session_id="s1", surface="terminal", cwd="c:/x", status="blocked", at="not-a-date"
    )
    assert len(sessions.fold([broken], now=NOW)) == 1


# ---------------------------------------------------------------------------
# summarise / run
# ---------------------------------------------------------------------------

def test_summarise_empty_says_so():
    assert sessions.summarise([]) == "No live coding sessions."


def test_summarise_counts_each_state():
    folded = sessions.fold(
        [event("b", "blocked"), event("i", "idle_awaiting_you"), event("a", "active")],
        now=NOW,
    )
    text = sessions.summarise(folded)
    assert "1 session(s) blocked" in text
    assert "finished and waiting" in text
    assert "still working" in text


def test_run_produces_needs_you_for_blocked_and_idle(tmp_path):
    from majordomo import state

    path = tmp_path / "state.jsonl"
    for e in (
        event("b", "blocked", topic="rm -rf something"),
        event("i", "idle_awaiting_you"),
        event("a", "active"),
    ):
        state.append_event(e, path)

    cfg = config_module.build(config_module.DEFAULTS)
    report = sessions.run(cfg, path)

    assert report.ok
    kinds = {item.kind for item in report.items}
    assert kinds == {"session_blocked", "session_idle"}
    # Active sessions need nothing from you — they're still working.
    assert len(report.items) == 2


def test_run_carries_session_id_as_the_action(tmp_path):
    from majordomo import state

    path = tmp_path / "state.jsonl"
    state.append_event(event("abc123", "blocked"), path)

    report = sessions.run(config_module.build(config_module.DEFAULTS), path)

    assert report.items[0].action == "abc123"


def test_run_on_missing_log_is_ok_and_empty(tmp_path):
    report = sessions.run(config_module.build(config_module.DEFAULTS), tmp_path / "none.jsonl")
    assert report.ok
    assert report.items == []


def test_run_makes_no_model_call(tmp_path, monkeypatch):
    """The sessions worker is deliberately deterministic — a fold has one exactly
    correct answer, and a free-tier model could only get it wrong more slowly."""
    import majordomo.llm as llm

    def boom(*a, **k):
        raise AssertionError("the sessions worker must not call the model")

    monkeypatch.setattr(llm, "complete", boom)
    sessions.run(config_module.build(config_module.DEFAULTS), tmp_path / "none.jsonl")
