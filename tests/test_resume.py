"""Tests for resume — assert the command is CONSTRUCTED, never launch anything."""
from __future__ import annotations

import pytest

from majordomo import resume
from majordomo.models import Session


def session(surface="terminal", cwd="c:/Users/nvps7/majordomo", sid="abc-123"):
    return Session(session_id=sid, surface=surface, cwd=cwd, status="blocked", at="2026-08-04T09:00:00+00:00")


# ---------------------------------------------------------------------------
# VS Code
# ---------------------------------------------------------------------------

def test_vscode_builds_the_documented_uri():
    """Verified against the installed extension: its registerUriHandler matches
    path /open and reads `session` from the query."""
    command = resume.build(session(surface="vscode"))

    assert command.uri == "vscode://Anthropic.claude-code/open?session=abc-123"
    assert command.argv is None


def test_vscode_session_id_is_url_encoded():
    command = resume.build(session(surface="vscode", sid="a b/c"))
    assert "a%20b%2Fc" in command.uri


def test_vscode_does_not_need_a_cwd():
    """The extension resolves the session itself; only the terminal needs cwd."""
    command = resume.build(session(surface="vscode", cwd=""))
    assert command.uri is not None


# ---------------------------------------------------------------------------
# Terminal
# ---------------------------------------------------------------------------

def test_terminal_builds_the_resume_argv():
    command = resume.build(session(surface="terminal"))

    assert command.argv == ["claude", "--resume", "abc-123"]
    assert command.uri is None


def test_terminal_runs_from_the_recorded_cwd():
    """Session-id lookup is scoped to the project dir — run it anywhere else and
    Claude Code simply won't find the session."""
    command = resume.build(session(surface="terminal", cwd="c:/repos/voicelog"))
    assert command.cwd == "c:/repos/voicelog"


def test_terminal_without_cwd_fails_loudly():
    """Better an explanatory error than `claude --resume` reporting the session
    doesn't exist because it searched the wrong project."""
    with pytest.raises(resume.ResumeError) as exc:
        resume.build(session(surface="terminal", cwd=""))
    assert "cwd" in str(exc.value)


def test_missing_session_id_fails():
    with pytest.raises(resume.ResumeError):
        resume.build(session(sid=""))


# ---------------------------------------------------------------------------
# launch — mocked; nothing is ever really started
# ---------------------------------------------------------------------------

def test_launch_uri_uses_start_with_empty_title(monkeypatch):
    """Windows `start` eats the first quoted argument as a window title."""
    calls = []
    monkeypatch.setattr(resume.subprocess, "run", lambda *a, **k: calls.append(a[0]))
    monkeypatch.setattr(resume.sys, "platform", "win32")

    resume.launch(resume.build(session(surface="vscode")))

    assert calls[0][:3] == ["cmd", "/c", "start"]
    assert calls[0][3] == ""


def test_launch_terminal_passes_cwd(monkeypatch):
    captured = {}

    def fake_popen(argv, cwd=None):
        captured["argv"] = argv
        captured["cwd"] = cwd

    monkeypatch.setattr(resume.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(resume.sys, "platform", "win32")

    resume.launch(resume.build(session(surface="terminal", cwd="c:/repos/x")))

    assert captured["cwd"] == "c:/repos/x"
    assert "--resume" in captured["argv"]
