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
    assert "1 session(s) stopped at a permission prompt" in text
    assert "finished and are waiting" in text
    assert "currently running" in text


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
    report = sessions.run(cfg, path, now=NOW)

    assert report.ok
    kinds = {item.kind for item in report.items}
    assert kinds == {"session_blocked", "session_idle"}
    # Active sessions need nothing from you — they're still working.
    assert len(report.items) == 2


def test_run_carries_session_id_as_the_action(tmp_path):
    from majordomo import state

    path = tmp_path / "state.jsonl"
    state.append_event(event("abc123", "blocked"), path)

    report = sessions.run(config_module.build(config_module.DEFAULTS), path, now=NOW)

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


def test_staleness_is_measured_against_an_injectable_clock(tmp_path):
    """Reading the wall clock directly would make a test with a fixed fixture
    timestamp pass today and fail three days from now — which is exactly what
    happened before `now` was injectable."""
    from majordomo import state

    path = tmp_path / "state.jsonl"
    state.append_event(event("s1", "blocked"), path)
    cfg = config_module.build(config_module.DEFAULTS)

    assert sessions.run(cfg, path, now=NOW).items          # fresh at NOW
    assert not sessions.run(cfg, path, now=NOW + timedelta(days=30)).items  # stale later


def test_active_sessions_are_marked_as_needing_nothing():
    """An active session's topic is a question the user asked Claude. Presented
    bare, the model reported it as a decision awaiting the user."""
    folded = sessions.fold([event("a", "active", topic="what shape is payments[]?")], now=NOW)
    text = sessions.summarise(folded)

    assert "NO ACTION NEEDED" in text
    assert "not a question for them" in text


def test_blocked_and_idle_are_marked_as_needing_action():
    folded = sessions.fold(
        [event("b", "blocked"), event("i", "idle_awaiting_you")], now=NOW
    )
    text = sessions.summarise(folded)

    assert text.count("NEEDS ACTION") == 2
    assert "cannot continue until approved" in text


# ---------------------------------------------------------------------------
# Liveness — SessionEnd is not guaranteed to fire
# ---------------------------------------------------------------------------

def stop(session_id, at):
    return SessionEvent(session_id=session_id, surface="terminal", cwd="c:/r",
                        status="active", at=at.isoformat(), kind="Stop")


def test_a_session_killed_without_sessionend_stops_being_active():
    """The reported bug: a VS Code window that dies cannot run SessionEnd, so
    the session sat at 'active' for 10 hours with nothing contradicting it."""
    dead = sessions.fold([event("gone", "active", at=NOW - timedelta(hours=10))], now=NOW)
    assert dead == []


def test_a_working_session_stays_active_via_its_heartbeat():
    """Stop fires at every turn end, so a live session keeps proving it exists
    even during a long agentic run with no prompts submitted."""
    folded = sessions.fold(
        [
            event("live", "active", at=NOW - timedelta(hours=10), topic="the refactor"),
            stop("live", NOW - timedelta(minutes=2)),
        ],
        now=NOW,
    )
    assert [s.session_id for s in folded] == ["live"]
    assert folded[0].topic == "the refactor", "a heartbeat must not lose the topic"


def test_a_stop_after_a_block_means_it_was_approved():
    """A permission prompt happens mid-turn, so the turn cannot end while one is
    outstanding — a Stop after a block is proof you answered it.

    The inverse (treating Stop as status-preserving) meant approving a prompt
    left the session `blocked` forever, because approving emits no
    UserPromptSubmit and every later turn refreshed `at` so the staleness sweep
    never retired it either."""
    folded = sessions.fold(
        [event("s1", "blocked", at=NOW - timedelta(minutes=30)),
         stop("s1", NOW - timedelta(minutes=1))],
        now=NOW,
    )
    assert folded[0].status == "active"


def test_a_blocked_session_does_not_stay_blocked_across_a_working_day():
    """The concrete regression: get blocked, approve, keep working for hours."""
    events = [event("s1", "active", at=NOW - timedelta(hours=5), topic="the migration"),
              event("s1", "blocked", at=NOW - timedelta(hours=4, minutes=50))]
    events += [stop("s1", NOW - timedelta(minutes=m)) for m in (280, 200, 120, 30, 2)]

    folded = sessions.fold(events, now=NOW)

    assert folded[0].status == "active"
    assert folded[0].topic == "the migration", "the heartbeat must not lose the topic"


def test_a_block_with_no_stop_after_it_stays_blocked():
    """The genuine case must still survive: nothing has happened since."""
    folded = sessions.fold(
        [stop("s1", NOW - timedelta(minutes=40)),
         event("s1", "blocked", at=NOW - timedelta(minutes=20))],
        now=NOW,
    )
    assert folded[0].status == "blocked"


def test_blocked_sessions_survive_the_active_timeout():
    """Going quiet is exactly what a blocked session is supposed to do."""
    folded = sessions.fold(
        [event("b", "blocked", at=NOW - timedelta(hours=10))],
        now=NOW, active_timeout_minutes=90,
    )
    assert [s.session_id for s in folded] == ["b"]


def test_idle_sessions_also_survive_the_active_timeout():
    folded = sessions.fold(
        [event("i", "idle_awaiting_you", at=NOW - timedelta(hours=10))],
        now=NOW, active_timeout_minutes=90,
    )
    assert [s.session_id for s in folded] == ["i"]


def test_active_timeout_is_configurable():
    old = [event("a", "active", at=NOW - timedelta(minutes=45))]
    assert sessions.fold(old, now=NOW, active_timeout_minutes=90) != []
    assert sessions.fold(old, now=NOW, active_timeout_minutes=30) == []


def test_a_stop_for_an_unseen_session_still_registers_it():
    """If we somehow miss SessionStart, a Stop should not be discarded."""
    folded = sessions.fold([stop("orphan", NOW - timedelta(minutes=1))], now=NOW)
    assert [s.session_id for s in folded] == ["orphan"]
