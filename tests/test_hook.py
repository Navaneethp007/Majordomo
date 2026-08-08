"""Tests for majordomo.hook.

The hook runs inside the user's live Claude Code session. Its contract is
absolute: **it always exits 0 and never writes to stdout/stderr in a way that
could be read as a control decision.** Claude Code reads exit code 2 from a
`Stop` hook as "do not stop" and from `PreToolUse` as "block this tool", so a
crash here would not degrade Majordomo — it would break the user's editing.
"""
from __future__ import annotations

import io
import json

import pytest

from majordomo import hook, state

COMMON = {
    "session_id": "abc-123",
    "cwd": "c:/Users/nvps7/majordomo",
    "permission_mode": "default",
    "transcript_path": "c:/whatever.jsonl",
}


def run(monkeypatch, tmp_path, event, matcher=None, payload=None, env=None):
    """Drive the hook end-to-end with a payload on stdin, return written events."""
    path = tmp_path / "state.jsonl"
    body = json.dumps({**COMMON, "hook_event_name": event, **(payload or {})})
    monkeypatch.setattr(hook, "read_stdin", lambda: body)
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", (env or {}).get("entrypoint", "cli"))

    code = hook.run_hook(event, matcher, state_file=path)

    assert code == 0, "the hook must always exit 0"
    return state.read_events(path)


# ---------------------------------------------------------------------------
# Event → status mapping (spec §6)
# ---------------------------------------------------------------------------

def test_session_start_records_active(monkeypatch, tmp_path):
    events = run(monkeypatch, tmp_path, "SessionStart", "startup")

    assert len(events) == 1
    assert events[0].session_id == "abc-123"
    assert events[0].status == "active"
    assert events[0].cwd == "c:/Users/nvps7/majordomo"


def test_user_prompt_submit_records_topic(monkeypatch, tmp_path):
    events = run(
        monkeypatch, tmp_path, "UserPromptSubmit",
        payload={"prompt": "make the router escalate on size"},
    )

    assert events[0].topic == "make the router escalate on size"
    assert events[0].status == "active"


def test_long_prompt_is_truncated(monkeypatch, tmp_path):
    """A pasted stack trace as 'topic' would blow up every downstream prompt."""
    events = run(
        monkeypatch, tmp_path, "UserPromptSubmit",
        payload={"prompt": "x" * 5000},
    )

    assert len(events[0].topic) <= hook.MAX_TOPIC_CHARS


def test_idle_notification_records_awaiting_you(monkeypatch, tmp_path):
    events = run(
        monkeypatch, tmp_path, "Notification", "idle_prompt",
        payload={"message": "Claude is waiting for your input"},
    )

    assert events[0].status == "idle_awaiting_you"


def test_permission_notification_records_blocked(monkeypatch, tmp_path):
    events = run(
        monkeypatch, tmp_path, "Notification", "permission_prompt",
        payload={"message": "Claude needs your permission to use Bash"},
    )

    assert events[0].status == "blocked"


def test_session_end_records_ended(monkeypatch, tmp_path):
    events = run(monkeypatch, tmp_path, "SessionEnd")
    assert events[0].status == "ended"


def test_unknown_notification_matcher_writes_nothing(monkeypatch, tmp_path):
    """`auth_success` and friends are noise — only the two we asked for count."""
    events = run(monkeypatch, tmp_path, "Notification", "auth_success")
    assert events == []


def test_unhandled_event_writes_nothing(monkeypatch, tmp_path):
    events = run(monkeypatch, tmp_path, "PreToolUse")
    assert events == []


# ---------------------------------------------------------------------------
# Surface detection — the spec's gap, filled by CLAUDE_CODE_ENTRYPOINT
# ---------------------------------------------------------------------------

def test_vscode_entrypoint_detected(monkeypatch, tmp_path):
    events = run(
        monkeypatch, tmp_path, "SessionStart", "startup",
        env={"entrypoint": "claude-vscode"},
    )

    assert events[0].surface == "vscode"
    assert events[0].entrypoint == "claude-vscode"


def test_cli_entrypoint_detected(monkeypatch, tmp_path):
    events = run(monkeypatch, tmp_path, "SessionStart", "startup", env={"entrypoint": "cli"})
    assert events[0].surface == "terminal"


def test_unknown_entrypoint_falls_back_to_terminal(monkeypatch, tmp_path):
    """CLAUDE_CODE_ENTRYPOINT is undocumented and may gain values. An unknown
    one must degrade to the safe default — and be recorded raw, so a wrong guess
    is diagnosable rather than invisible."""
    events = run(
        monkeypatch, tmp_path, "SessionStart", "startup",
        env={"entrypoint": "some-future-surface"},
    )

    assert events[0].surface == "terminal"
    assert events[0].entrypoint == "some-future-surface"


def test_missing_entrypoint_falls_back_to_terminal(monkeypatch, tmp_path):
    path = tmp_path / "state.jsonl"
    monkeypatch.delenv("CLAUDE_CODE_ENTRYPOINT", raising=False)
    monkeypatch.setattr(
        hook, "read_stdin",
        lambda: json.dumps({**COMMON, "hook_event_name": "SessionStart"}),
    )

    assert hook.run_hook("SessionStart", "startup", state_file=path) == 0
    assert state.read_events(path)[0].surface == "terminal"


# ---------------------------------------------------------------------------
# The contract: never break the session
# ---------------------------------------------------------------------------

def test_malformed_stdin_exits_zero_and_writes_nothing(monkeypatch, tmp_path):
    path = tmp_path / "state.jsonl"
    monkeypatch.setattr(hook, "read_stdin", lambda: "{not json at all")

    assert hook.run_hook("SessionStart", "startup", state_file=path) == 0
    assert state.read_events(path) == []


def test_empty_stdin_exits_zero(monkeypatch, tmp_path):
    path = tmp_path / "state.jsonl"
    monkeypatch.setattr(hook, "read_stdin", lambda: "")

    assert hook.run_hook("SessionStart", "startup", state_file=path) == 0
    assert state.read_events(path) == []


def test_missing_session_id_writes_nothing(monkeypatch, tmp_path):
    """Without a session id the record could never be folded or resumed."""
    path = tmp_path / "state.jsonl"
    monkeypatch.setattr(
        hook, "read_stdin",
        lambda: json.dumps({"hook_event_name": "SessionStart", "cwd": "c:/x"}),
    )

    assert hook.run_hook("SessionStart", "startup", state_file=path) == 0
    assert state.read_events(path) == []


def test_bom_prefixed_payload_still_parses(monkeypatch, tmp_path):
    path = tmp_path / "state.jsonl"
    body = json.dumps({**COMMON, "hook_event_name": "SessionStart"})
    monkeypatch.setattr(hook, "read_stdin", lambda: "\ufeff" + body)

    assert hook.run_hook("SessionStart", "startup", state_file=path) == 0
    assert len(state.read_events(path)) == 1


def test_write_failure_is_swallowed(monkeypatch, tmp_path):
    """Even a disk error must not raise into the session."""
    path = tmp_path / "state.jsonl"
    monkeypatch.setattr(
        hook, "read_stdin",
        lambda: json.dumps({**COMMON, "hook_event_name": "SessionStart"}),
    )

    def boom(event, target=None):
        raise OSError("disk full")

    monkeypatch.setattr(hook.state, "append_event", boom)

    assert hook.run_hook("SessionStart", "startup", state_file=path) == 0


def test_hook_writes_nothing_to_stdout(monkeypatch, tmp_path, capsys):
    """Stdout from a hook is interpreted by Claude Code. Ours must stay silent."""
    path = tmp_path / "state.jsonl"
    monkeypatch.setattr(
        hook, "read_stdin",
        lambda: json.dumps({**COMMON, "hook_event_name": "SessionStart"}),
    )

    hook.run_hook("SessionStart", "startup", state_file=path)

    assert capsys.readouterr().out == ""


def test_errors_are_logged_to_file(monkeypatch, tmp_path):
    """Swallowed does not mean lost — a failure has to leave a trace somewhere."""
    log = tmp_path / "error.log"
    monkeypatch.setattr(hook, "read_stdin", lambda: "{broken")
    monkeypatch.setattr(hook, "error_log_path", lambda: log)

    hook.run_hook("SessionStart", "startup", state_file=tmp_path / "state.jsonl")

    assert log.is_file()
    assert "hook:parse-stdin" in log.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# read_stdin
# ---------------------------------------------------------------------------

def test_read_stdin_returns_empty_on_tty(monkeypatch):
    """Run by hand in a terminal there is no payload — don't hang forever."""
    fake = io.StringIO("")
    fake.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.setattr("sys.stdin", fake)

    assert hook.read_stdin() == ""


def test_errors_never_touch_the_real_home(monkeypatch, tmp_path):
    """The hook logs swallowed failures to ~/.majordomo/error.log. Provoking one
    in a test must land in the isolated home, not the developer's own log —
    where a pytest traceback would later read as a real session failure."""
    import os
    from majordomo import paths

    monkeypatch.setattr(hook, "read_stdin", lambda: "{broken json")
    hook.run_hook("SessionStart", "startup", state_file=tmp_path / "state.jsonl")

    written = paths.error_log_path()
    assert written.is_file()
    assert str(written).startswith(os.environ["MAJORDOMO_HOME"])


def test_stop_records_a_heartbeat(monkeypatch, tmp_path):
    events = run(monkeypatch, tmp_path, "Stop", payload={"last_assistant_message": "done"})
    assert len(events) == 1
    assert events[0].kind == "Stop"
    assert events[0].status == "active"


def test_every_event_records_which_hook_wrote_it(monkeypatch, tmp_path):
    """SessionStart and UserPromptSubmit both write 'active', so without `kind`
    the log cannot be read back to tell what actually happened."""
    for hook_event, matcher in [("SessionStart", "startup"), ("UserPromptSubmit", None),
                                ("Stop", None), ("SessionEnd", None)]:
        events = run(monkeypatch, tmp_path / hook_event, hook_event, matcher)
        assert events[0].kind == hook_event
