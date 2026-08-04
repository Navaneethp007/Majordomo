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
