"""Tests for majordomo.state — the append-only session log.

Two properties matter more than anything else, because this file is written
from inside a Claude Code hook:

1. Writes are single-line appends, so concurrent sessions don't interleave.
2. Reads never throw. A torn final line from a process killed mid-write, a
   hand-edited file, a stray blank line — all are skipped, not raised.
"""
from __future__ import annotations

from majordomo import state
from majordomo.models import SessionEvent


def make_event(session_id: str = "s1", status: str = "active", **kw) -> SessionEvent:
    defaults = dict(
        session_id=session_id,
        surface="terminal",
        cwd="c:/repo",
        status=status,
        at="2026-08-04T09:00:00+00:00",
    )
    defaults.update(kw)
    return SessionEvent(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

def test_append_then_read_round_trips(tmp_path):
    path = tmp_path / "state.jsonl"
    event = make_event(topic="fix the parser")

    state.append_event(event, path)
    events = state.read_events(path)

    assert len(events) == 1
    assert events[0] == event


def test_append_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "deeper" / "state.jsonl"
    state.append_event(make_event(), path)
    assert path.is_file()


def test_read_missing_file_is_empty_not_error(tmp_path):
    assert state.read_events(tmp_path / "absent.jsonl") == []


def test_appends_accumulate_in_order(tmp_path):
    path = tmp_path / "state.jsonl"
    state.append_event(make_event("s1", "active"), path)
    state.append_event(make_event("s1", "blocked"), path)
    state.append_event(make_event("s2", "active"), path)

    events = state.read_events(path)

    assert [(e.session_id, e.status) for e in events] == [
        ("s1", "active"),
        ("s1", "blocked"),
        ("s2", "active"),
    ]


def test_each_append_writes_exactly_one_line(tmp_path):
    """Concurrent hooks rely on one append == one line. Multi-line writes would
    let two sessions interleave into an unparseable mess."""
    path = tmp_path / "state.jsonl"
    state.append_event(make_event(topic="line one\nline two"), path)

    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert raw.count("\n") == 1


# ---------------------------------------------------------------------------
# Tolerant reading — the property that keeps a hook from breaking a session
# ---------------------------------------------------------------------------

def test_torn_final_line_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "state.jsonl"
    state.append_event(make_event("s1"), path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"session_id": "s2", "surf')  # killed mid-write

    events = state.read_events(path)

    assert [e.session_id for e in events] == ["s1"]


def test_blank_and_garbage_lines_are_skipped(tmp_path):
    path = tmp_path / "state.jsonl"
    state.append_event(make_event("s1"), path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n\n   \nnot json at all\n")
    state.append_event(make_event("s2"), path)

    events = state.read_events(path)

    assert [e.session_id for e in events] == ["s1", "s2"]


def test_valid_json_wrong_shape_is_skipped(tmp_path):
    path = tmp_path / "state.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"hello": "world"}\n')
        fh.write("[1, 2, 3]\n")
    state.append_event(make_event("s1"), path)

    events = state.read_events(path)

    assert [e.session_id for e in events] == ["s1"]


def test_bom_does_not_swallow_the_file(tmp_path):
    """PowerShell's Out-File writes a UTF-8 BOM; losing the whole log to an
    invisible byte would be a miserable way to fail."""
    path = tmp_path / "state.jsonl"
    state.append_event(make_event("s1"), path)
    raw = path.read_text(encoding="utf-8")
    path.write_text("\ufeff" + raw, encoding="utf-8")

    assert [e.session_id for e in state.read_events(path)] == ["s1"]


def test_read_detailed_counts_skipped(tmp_path):
    path = tmp_path / "state.jsonl"
    state.append_event(make_event("s1"), path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("garbage\n{}\n")

    result = state.read_events_detailed(path)

    assert len(result.events) == 1
    assert result.skipped == 2


def test_unknown_status_is_skipped(tmp_path):
    """A future version writing a status we don't understand must not become a
    session in an impossible state."""
    path = tmp_path / "state.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            '{"session_id":"s1","surface":"terminal","cwd":"c:/r",'
            '"status":"transcending","at":"2026-08-04T09:00:00+00:00"}\n'
        )

    assert state.read_events(path) == []


# ---------------------------------------------------------------------------
# Compaction — the hook appends forever; every read parses the whole file
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone  # noqa: E402

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)


def at(delta_hours: int) -> str:
    return (NOW - timedelta(hours=delta_hours)).isoformat()


def test_compact_drops_events_past_the_window(tmp_path):
    path = tmp_path / "state.jsonl"
    for i in range(50):
        state.append_event(make_event("old", "active", at=at(200)), path)
    state.append_event(make_event("new", "active", at=at(1)), path)

    dropped = state.compact(path, keep_hours=72, now=NOW)

    assert dropped > 0
    assert len(state.read_events(path)) < 51


def test_compact_keeps_recent_events_intact(tmp_path):
    path = tmp_path / "state.jsonl"
    for i in range(5):
        state.append_event(make_event(f"s{i}", "blocked", at=at(1)), path)

    state.compact(path, keep_hours=72, now=NOW)

    assert len(state.read_events(path)) == 5


def test_compact_preserves_the_topic_of_a_long_running_session(tmp_path):
    """A topic is set once by UserPromptSubmit and never repeated. A session
    started four days ago and still active must keep it, or folding afterwards
    loses what the session was about."""
    path = tmp_path / "state.jsonl"
    state.append_event(make_event("long", "active", at=at(100), topic="the big refactor"), path)
    state.append_event(make_event("long", "blocked", at=at(1)), path)  # recent, no topic

    state.compact(path, keep_hours=72, now=NOW)

    from majordomo.workers.sessions import fold
    folded = fold(state.read_events(path), now=NOW)
    assert folded[0].topic == "the big refactor"


def test_compact_discards_old_sessions_with_no_recent_activity(tmp_path):
    """These are what make the log grow forever — one dead id per terminal ever
    opened. fold drops them as stale anyway, so keeping them buys nothing."""
    path = tmp_path / "state.jsonl"
    for i in range(200):
        state.append_event(make_event(f"dead{i}", "active", at=at(500)), path)
    state.append_event(make_event("live", "active", at=at(1)), path)

    dropped = state.compact(path, keep_hours=72, now=NOW)

    assert dropped == 200
    assert [e.session_id for e in state.read_events(path)] == ["live"]


def test_compact_discards_old_ended_sessions(tmp_path):
    path = tmp_path / "state.jsonl"
    state.append_event(make_event("done", "ended", at=at(100)), path)
    state.append_event(make_event("live", "active", at=at(1)), path)

    state.compact(path, keep_hours=72, now=NOW)

    assert [e.session_id for e in state.read_events(path)] == ["live"]


def test_compaction_is_idempotent(tmp_path):
    """Running it twice must not keep churning the file."""
    path = tmp_path / "state.jsonl"
    for i in range(30):
        state.append_event(make_event(f"o{i}", "active", at=at(500)), path)
    state.append_event(make_event("live", "active", at=at(1)), path)

    state.compact(path, keep_hours=72, now=NOW)
    assert state.compact(path, keep_hours=72, now=NOW) == 0


def test_compact_is_a_noop_when_nothing_to_drop(tmp_path):
    path = tmp_path / "state.jsonl"
    state.append_event(make_event("s1", "active", at=at(1)), path)
    assert state.compact(path, keep_hours=72, now=NOW) == 0


def test_compact_on_missing_file_is_zero(tmp_path):
    assert state.compact(tmp_path / "absent.jsonl") == 0


def test_maybe_compact_leaves_a_small_log_alone(tmp_path):
    path = tmp_path / "state.jsonl"
    state.append_event(make_event("s1", "active", at=at(500)), path)

    assert state.maybe_compact(path, max_bytes=1_000_000) == 0
    assert len(state.read_events(path)) == 1, "a small log must not be touched"


def test_maybe_compact_fires_once_the_log_is_big(tmp_path):
    path = tmp_path / "state.jsonl"
    for i in range(200):
        state.append_event(make_event(f"old{i}", "active", at=at(500)), path)
    state.append_event(make_event("live", "active", at=at(1)), path)

    assert state.maybe_compact(path, keep_hours=72, max_bytes=100) > 0
    assert len(state.read_events(path)) < 50, "the log must actually shrink"


def test_maybe_compact_never_raises(tmp_path):
    """Housekeeping must not be the reason a briefing fails."""
    assert state.maybe_compact(tmp_path / "nope" / "deep.jsonl") == 0


def test_compaction_leaves_no_temp_file_behind(tmp_path):
    path = tmp_path / "state.jsonl"
    for i in range(20):
        state.append_event(make_event(f"o{i}", "active", at=at(500)), path)
    state.append_event(make_event("new", "active", at=at(1)), path)

    state.compact(path, keep_hours=72, now=NOW)

    assert list(tmp_path.glob("*.compacting")) == []
