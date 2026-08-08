"""Tests for the pipeline wiring, the CLI's degradation policy, and triggers."""
from __future__ import annotations

import pytest

from majordomo import brief, cli, hook, resume, trigger, tts
from majordomo import config as config_module
from majordomo.llm import LLMError, MissingApiKey
from majordomo.models import Briefing, NeedsYouItem, SourceReport

CFG = config_module.build(config_module.DEFAULTS)


# ---------------------------------------------------------------------------
# brief.gather — the parallel fan-out
# ---------------------------------------------------------------------------

def test_gather_runs_every_enabled_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(brief.github, "run", lambda c: SourceReport("github", True, "gh"))

    reports = brief.gather(CFG, tmp_path / "none.jsonl")

    assert {r.source for r in reports} == {"github", "sessions"}


def test_gather_skips_disabled_sources(monkeypatch, tmp_path):
    data = config_module._deep_merge(
        config_module.DEFAULTS, {"sources": {"github": {"enabled": False}}}
    )
    reports = brief.gather(config_module.build(data), tmp_path / "none.jsonl")

    assert [r.source for r in reports] == ["sessions"]


def test_a_worker_thread_that_dies_becomes_a_stub(monkeypatch, tmp_path):
    """Workers are contracted not to raise, but a thread that dies anyway must
    not take the whole briefing with it."""
    def boom(config):
        raise RuntimeError("thread died")

    monkeypatch.setattr(brief.github, "run", boom)

    reports = brief.gather(CFG, tmp_path / "none.jsonl")
    github_report = next(r for r in reports if r.source == "github")

    assert github_report.ok is False
    assert "thread died" in github_report.error


def test_explain_names_every_source_and_path():
    result = brief.BriefResult(
        briefing=Briefing("text"),
        reports=[
            SourceReport("github", True, "s", path="escalated", route_reason="too big"),
            SourceReport("sessions", True, "s", path="cheap", route_reason="local"),
        ],
    )
    explained = result.explain()

    assert "github" in explained and "escalated" in explained and "too big" in explained
    assert "sessions" in explained and "cheap" in explained


# ---------------------------------------------------------------------------
# CLI degradation policy
# ---------------------------------------------------------------------------

def test_brief_prints_text_before_speaking(monkeypatch, capsys):
    """Text output is never gated on audio."""
    spoken = []
    monkeypatch.setattr(
        brief, "run",
        lambda c, s=None: brief.BriefResult(briefing=Briefing("All quiet.")),
    )
    monkeypatch.setattr(tts, "speak", lambda t, c: spoken.append(t))

    cli.main(["brief"])

    assert "All quiet." in capsys.readouterr().out
    assert spoken == ["All quiet."]


def test_tts_failure_only_warns(monkeypatch, capsys):
    monkeypatch.setattr(
        brief, "run",
        lambda c, s=None: brief.BriefResult(briefing=Briefing("All quiet.")),
    )

    def boom(text, cfg):
        raise tts.TTSError("no audio device")

    monkeypatch.setattr(tts, "speak", boom)

    cli.main(["brief"])  # must not raise, must not exit non-zero

    captured = capsys.readouterr()
    assert "All quiet." in captured.out
    assert "could not speak" in captured.err


def test_missing_api_key_exits_one(monkeypatch, capsys):
    """Unrecoverable and yours to fix — the one case that stops."""
    def boom(config, state_file=None):
        raise MissingApiKey("Set the OPENROUTER_API_KEY environment variable")

    monkeypatch.setattr(brief, "run", boom)

    with pytest.raises(SystemExit) as exc:
        cli.main(["brief"])

    assert exc.value.code == 1
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_no_speak_skips_audio(monkeypatch, capsys):
    def boom(text, cfg):
        raise AssertionError("--no-speak must not synthesize")

    monkeypatch.setattr(
        brief, "run", lambda c, s=None: brief.BriefResult(briefing=Briefing("Quiet."))
    )
    monkeypatch.setattr(tts, "speak", boom)

    cli.main(["brief", "--no-speak"])
    assert "Quiet." in capsys.readouterr().out


def test_needs_you_is_printed(monkeypatch, capsys):
    item = NeedsYouItem(
        kind="session_blocked", title="majordomo (vscode)",
        detail="Waiting for your approval.", source="sessions",
    )
    monkeypatch.setattr(
        brief, "run",
        lambda c, s=None: brief.BriefResult(briefing=Briefing("One thing.", [item])),
    )

    cli.main(["brief", "--no-speak"])

    out = capsys.readouterr().out
    assert "majordomo (vscode)" in out
    assert "Waiting for your approval." in out


def test_explain_flag_prints_routing(monkeypatch, capsys):
    monkeypatch.setattr(
        brief, "run",
        lambda c, s=None: brief.BriefResult(
            briefing=Briefing("Quiet."),
            reports=[SourceReport("github", True, "s", path="cheap", route_reason="small")],
        ),
    )

    cli.main(["brief", "--no-speak", "--explain"])

    assert "Routing:" in capsys.readouterr().out


def test_resume_dry_run_launches_nothing(monkeypatch, capsys):
    from majordomo.models import Session

    def boom(cmd):
        raise AssertionError("--dry-run must not launch")

    monkeypatch.setattr(resume, "launch", boom)
    monkeypatch.setattr(
        cli, "_live_sessions",
        lambda c: ([Session("abc-123", "vscode", "c:/x", "blocked", "2026-08-04T09:00:00+00:00")], 0),
    )

    cli.main(["resume", "abc", "--dry-run"])

    assert "vscode://Anthropic.claude-code/open?session=abc-123" in capsys.readouterr().out


def test_resume_unknown_id_exits_one(monkeypatch):
    monkeypatch.setattr(cli, "_live_sessions", lambda c: ([], 0))
    with pytest.raises(SystemExit) as exc:
        cli.main(["resume", "nope"])
    assert exc.value.code == 1


def test_resume_ambiguous_prefix_exits_one(monkeypatch):
    from majordomo.models import Session

    two = [
        Session("abc-1", "vscode", "c:/x", "blocked", "2026-08-04T09:00:00+00:00"),
        Session("abc-2", "vscode", "c:/x", "blocked", "2026-08-04T09:00:00+00:00"),
    ]
    monkeypatch.setattr(cli, "_live_sessions", lambda c: (two, 0))

    with pytest.raises(SystemExit) as exc:
        cli.main(["resume", "abc"])
    assert exc.value.code == 1


def test_hook_subcommand_always_exits_zero(monkeypatch):
    monkeypatch.setattr(hook, "run_hook", lambda e, m: 0)
    with pytest.raises(SystemExit) as exc:
        cli.main(["hook", "--event=SessionStart", "--matcher=startup"])
    assert exc.value.code == 0


def test_bare_invocation_prints_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 0
    assert "briefing" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Wake / boot / login triggers
# ---------------------------------------------------------------------------

def test_wake_task_uses_the_kernel_power_event():
    """'Woke from sleep' has no first-class trigger — it only exists as an
    event-log record, so it needs an XPath subscription."""
    xml = trigger.build_task_xml(trigger.TASKS[2], "python.exe", "-m majordomo.cli brief")

    assert "EventTrigger" in xml
    assert "Kernel-Power" in xml
    assert "EventID=107" in xml


def test_logon_and_boot_use_first_class_triggers():
    logon = trigger.build_task_xml(trigger.TASKS[0], "py.exe", "args")
    boot = trigger.build_task_xml(trigger.TASKS[1], "py.exe", "args")

    assert "LogonTrigger" in logon
    assert "BootTrigger" in boot


def test_task_xml_is_wellformed_with_awkward_characters():
    """Built with etree rather than string formatting so an & can't corrupt it."""
    import xml.etree.ElementTree as ET

    spec = trigger.TaskSpec("Majordomo\\Odd", "logon", "Brief me <now> & later")
    xml = trigger.build_task_xml(spec, "c:/py.exe", "-m majordomo.cli brief")

    parsed = ET.fromstring(xml.split("\n", 1)[1])
    assert parsed is not None


def test_task_does_not_wake_the_machine():
    """A briefing must never be the reason a laptop wakes up or stays awake."""
    xml = trigger.build_task_xml(trigger.TASKS[0], "py.exe", "args")
    assert "<StartWhenAvailable>false</StartWhenAvailable>" in xml


# ---------------------------------------------------------------------------
# Import weight — the hook runs on every prompt you submit
# ---------------------------------------------------------------------------

def test_cli_import_does_not_drag_in_the_network_stack():
    """`mj hook` fires on every prompt. Importing httpx and yaml on that path
    cost ~250ms per keystroke-to-response; every heavy import in cli.py is
    deliberately inside the command that needs it. This test is what stops a
    convenient top-level import from quietly putting it back."""
    import subprocess
    import sys

    probe = (
        "import sys, majordomo.cli; "
        "print(','.join(m for m in ('httpx','yaml','ssl') if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=60
    )

    assert out.stdout.strip() == "", f"cli.py now imports: {out.stdout.strip()}"


# ---------------------------------------------------------------------------
# should_speak — the wake trigger's voice policy
# ---------------------------------------------------------------------------

class _Args:
    def __init__(self, no_speak=False, speak_if_needed=False):
        self.no_speak = no_speak
        self.speak_if_needed = speak_if_needed


ITEM = NeedsYouItem(kind="session_blocked", title="t", detail="d", source="sessions")


def test_default_always_speaks():
    """You asked for a briefing at the keyboard; you get one aloud."""
    assert cli.should_speak(_Args(), CFG, Briefing("quiet", [])) is True


def test_no_speak_wins_over_everything():
    assert cli.should_speak(_Args(no_speak=True), CFG, Briefing("x", [ITEM])) is False


def test_speak_if_needed_is_silent_when_nothing_needs_you():
    """The whole point: no being told 'nothing is happening' out loud."""
    assert cli.should_speak(_Args(speak_if_needed=True), CFG, Briefing("quiet", [])) is False


def test_speak_if_needed_speaks_when_something_does():
    assert cli.should_speak(_Args(speak_if_needed=True), CFG, Briefing("x", [ITEM])) is True


def test_voice_disabled_in_config_silences_everything():
    data = config_module._deep_merge(config_module.DEFAULTS, {"voice": {"enabled": False}})
    muted = config_module.build(data)
    assert cli.should_speak(_Args(), muted, Briefing("x", [ITEM])) is False


def test_wake_trigger_uses_the_conditional_flag():
    """spec §8 says the wake cycle speaks; speaking unconditionally makes it
    noise. Asserted against the module constant the installer actually uses —
    passing the string into build_task_xml and checking it came back only
    proved the XML builder echoes its input, and stayed green if this reverted
    to --no-speak."""
    assert "--speak-if-needed" in trigger.BRIEF_ARGUMENTS
    assert "--no-speak" not in trigger.BRIEF_ARGUMENTS
    assert trigger.BRIEF_ARGUMENTS.startswith("-m majordomo.cli brief")


def test_task_xml_carries_the_real_arguments():
    xml = trigger.build_task_xml(trigger.TASKS[2], "pythonw.exe", trigger.BRIEF_ARGUMENTS)
    assert trigger.BRIEF_ARGUMENTS in xml


# ---------------------------------------------------------------------------
# Silence must not be ambiguous, and must not repeat forever
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone  # noqa: E402

T0 = datetime(2026, 8, 8, 9, 0, tzinfo=timezone.utc)
OK_REPORT = SourceReport("github", True, "all fine")
DEAD_REPORT = SourceReport.failed("github", "401 token expired")


def test_a_failed_source_is_worth_speaking(monkeypatch):
    """Under pythonw.exe there is no console — sys.stdout is None and prints go
    nowhere, so audio is the only channel. An expired token producing the same
    silence as a quiet morning is indistinguishable from 'all clear'."""
    assert cli.should_speak(
        _Args(speak_if_needed=True), CFG, Briefing("github is down", []),
        [DEAD_REPORT], now=T0,
    ) is True


def test_all_healthy_and_nothing_pending_stays_silent(monkeypatch):
    assert cli.should_speak(
        _Args(speak_if_needed=True), CFG, Briefing("quiet", []), [OK_REPORT], now=T0,
    ) is False


def test_cold_boot_does_not_speak_twice(monkeypatch):
    """OnLogon fires at once and OnBoot a minute later; MultipleInstancesPolicy
    is per-task and cannot dedupe across them."""
    briefing = Briefing("one blocked session", [ITEM])
    args = _Args(speak_if_needed=True)

    first = cli.should_speak(args, CFG, briefing, [OK_REPORT], now=T0)
    second = cli.should_speak(args, CFG, briefing, [OK_REPORT], now=T0 + timedelta(minutes=1))

    assert first is True
    assert second is False


def test_unchanged_situation_stops_repeating(monkeypatch):
    """Kernel-Power 107 fires on every modern-standby resume. A session blocked
    since morning must not be read out on every lid open."""
    briefing = Briefing("still blocked", [ITEM])
    args = _Args(speak_if_needed=True)

    cli.should_speak(args, CFG, briefing, [OK_REPORT], now=T0)
    for minutes in (5, 30, 90):
        assert cli.should_speak(
            args, CFG, briefing, [OK_REPORT], now=T0 + timedelta(minutes=minutes)
        ) is False


def test_a_new_item_speaks_immediately(monkeypatch):
    """Cooldown must suppress repetition, never news."""
    args = _Args(speak_if_needed=True)
    cli.should_speak(args, CFG, Briefing("x", [ITEM]), [OK_REPORT], now=T0)

    fresh = NeedsYouItem(kind="review_request", title="repo#99", detail="d", source="github")
    assert cli.should_speak(
        args, CFG, Briefing("x", [ITEM, fresh]), [OK_REPORT], now=T0 + timedelta(minutes=2)
    ) is True


def test_the_same_situation_speaks_again_after_the_cooldown(monkeypatch):
    args = _Args(speak_if_needed=True)
    cli.should_speak(args, CFG, Briefing("x", [ITEM]), [OK_REPORT], now=T0)

    assert cli.should_speak(
        args, CFG, Briefing("x", [ITEM]), [OK_REPORT], now=T0 + timedelta(hours=3)
    ) is True


def test_manual_brief_ignores_the_gate_entirely():
    """You typed `mj brief`. You get it aloud, however recently it last spoke."""
    args = _Args()
    assert cli.should_speak(args, CFG, Briefing("x", [ITEM]), [OK_REPORT], now=T0) is True
    assert cli.should_speak(args, CFG, Briefing("x", [ITEM]), [OK_REPORT], now=T0) is True
