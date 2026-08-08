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
    """Gmail ships disabled, so enable it explicitly here."""
    monkeypatch.setattr(brief.github, "run", lambda c: SourceReport("github", True, "gh"))
    monkeypatch.setattr(brief.gmail, "run", lambda c: SourceReport("gmail", True, "mail"))
    data = config_module._deep_merge(
        config_module.DEFAULTS, {"sources": {"gmail": {"enabled": True}}}
    )

    reports = brief.gather(config_module.build(data), tmp_path / "none.jsonl")

    assert {r.source for r in reports} == {"github", "gmail", "sessions"}


def test_gmail_ships_disabled():
    """Nothing provisions Gmail credentials, so left on it would fail every run
    — and a failing source makes the wake trigger speak, which is the noise
    --speak-if-needed exists to prevent."""
    assert CFG.sources.gmail.enabled is False


def test_gather_skips_disabled_sources(monkeypatch, tmp_path):
    data = config_module._deep_merge(
        config_module.DEFAULTS,
        {"sources": {"github": {"enabled": False}, "gmail": {"enabled": False}}},
    )
    reports = brief.gather(config_module.build(data), tmp_path / "none.jsonl")

    assert [r.source for r in reports] == ["sessions"]


def test_a_worker_thread_that_dies_becomes_a_stub(monkeypatch, tmp_path):
    """Workers are contracted not to raise, but a thread that dies anyway must
    not take the whole briefing with it."""
    def boom(config):
        raise RuntimeError("thread died")

    monkeypatch.setattr(brief.github, "run", boom)
    monkeypatch.setattr(brief.gmail, "run", lambda c: SourceReport("gmail", True, "mail"))

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
# Wake / login triggers
# ---------------------------------------------------------------------------

LOGON = next(t for t in trigger.TASKS if t.kind == "logon")
WAKE = next(t for t in trigger.TASKS if t.kind == "wake")

def test_wake_task_uses_the_kernel_power_event():
    """'Woke from sleep' has no first-class trigger — it only exists as an
    event-log record, so it needs an XPath subscription."""
    xml = trigger.build_task_xml(WAKE, "python.exe", "-m majordomo.cli brief")

    assert "EventTrigger" in xml
    assert "Kernel-Power" in xml
    assert "EventID=107" in xml


def test_logon_uses_a_first_class_trigger():
    assert "LogonTrigger" in trigger.build_task_xml(LOGON, "py.exe", "args")


def test_logon_task_is_scoped_to_the_current_user():
    """Without a UserId the trigger covers every account on the machine, which
    needs administrator rights — that was the real cause of the
    `mj install-trigger` "Access is denied." failure."""
    xml = trigger.build_task_xml(LOGON, "py.exe", "args")
    assert f"<UserId>{trigger.current_user()}</UserId>" in xml
    assert xml.count("<UserId>") == 2, "both the trigger and the principal need it"


def test_there_is_no_boot_trigger():
    """A BootTrigger runs before login, so Windows treats it as machine-level
    and refuses without elevation. It also buys nothing: a boot is always
    followed by a logon, so it only ever added a duplicate briefing."""
    kinds = {t.kind for t in trigger.TASKS}
    assert kinds == {"logon", "wake"}
    for spec in trigger.TASKS:
        assert "BootTrigger" not in trigger.build_task_xml(spec, "py.exe", "args")


def test_task_xml_is_wellformed_with_awkward_characters():
    """Built with etree rather than string formatting so an & can't corrupt it."""
    import xml.etree.ElementTree as ET

    spec = trigger.TaskSpec("Majordomo\\Odd", "logon", "Brief me <now> & later")
    xml = trigger.build_task_xml(spec, "c:/py.exe", "-m majordomo.cli brief")

    parsed = ET.fromstring(xml.split("\n", 1)[1])
    assert parsed is not None


def test_task_does_not_wake_the_machine():
    """A briefing must never be the reason a laptop wakes up or stays awake."""
    xml = trigger.build_task_xml(LOGON, "py.exe", "args")
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
    xml = trigger.build_task_xml(WAKE, "pythonw.exe", trigger.BRIEF_ARGUMENTS)
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


# ---------------------------------------------------------------------------
# safe_print — a cp1252 console must not be able to kill a briefing
# ---------------------------------------------------------------------------

def test_safe_print_survives_an_unencodable_character():
    """sys.stdout.encoding is cp1252 on a default Windows console, where
    print('\U0001f916') raises UnicodeEncodeError. Marketing subject lines are
    full of emoji, so this would crash mj brief on most runs."""
    import io

    buffer = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")
    cli.safe_print("Automated run finished \U0001f916", file=buffer)

    buffer.seek(0)
    assert "Automated run finished" in buffer.read()


def test_safe_print_is_a_noop_without_a_console():
    """pythonw.exe under Task Scheduler may have no stdout at all."""
    cli.safe_print("anything", file=None) if False else None
    import majordomo.cli as m
    real, m.sys.stdout = m.sys.stdout, None
    try:
        m.safe_print("must not raise")
    finally:
        m.sys.stdout = real


def test_context_is_printed_under_its_own_heading(monkeypatch, capsys):
    """'Also waiting' must be visually separate from 'Needs you' — merging them
    is how a digest turns into a to-do list."""
    from majordomo.models import ContextItem

    mail = ContextItem(kind="unread_mail", title="Priya: Invoice rounding",
                       detail="Unread.", source="gmail")
    monkeypatch.setattr(
        brief, "run",
        lambda c, s=None: brief.BriefResult(briefing=Briefing("Quiet.", [], [mail])),
    )

    cli.main(["brief", "--no-speak"])
    out = capsys.readouterr().out

    assert "Also waiting" in out
    assert "Needs you" not in out
    assert "Priya: Invoice rounding" in out


# ---------------------------------------------------------------------------
# An unconfigured source is not an outage
# ---------------------------------------------------------------------------

def test_an_unconfigured_source_does_not_make_the_wake_trigger_speak():
    """"You never gave me a Gmail password" is a setup state, not news. Treating
    it as an outage made the trigger talk aloud on every single wake."""
    stub = SourceReport.failed("gmail", "no Gmail credentials", unconfigured=True)

    assert cli.should_speak(
        _Args(speak_if_needed=True), CFG, Briefing("quiet", []), [stub], now=T0
    ) is False


def test_a_genuinely_broken_source_still_speaks():
    """A source that was working and stopped is worth interrupting for."""
    stub = SourceReport.failed("github", "401 token expired")

    assert cli.should_speak(
        _Args(speak_if_needed=True), CFG, Briefing("github is down", []), [stub], now=T0
    ) is True


def test_sessions_and_brief_agree_on_liveness(monkeypatch):
    """_live_sessions used to drop active_timeout_minutes, so `mj sessions` used
    the hardcoded default while `mj brief` used the configured value."""
    captured = {}
    import majordomo.workers.sessions as sessions_mod

    def spy(events, now=None, stale_after_hours=72, active_timeout_minutes=90):
        captured["timeout"] = active_timeout_minutes
        return []

    monkeypatch.setattr("majordomo.workers.sessions.fold", spy)
    data = config_module._deep_merge(
        config_module.DEFAULTS, {"sources": {"sessions": {"active_timeout_minutes": 15}}}
    )
    cli._live_sessions(config_module.build(data))

    assert captured["timeout"] == 15


# ---------------------------------------------------------------------------
# A retired scheduled task must still be removable
# ---------------------------------------------------------------------------

def test_uninstall_sweeps_retired_task_names():
    """Dropping OnBoot from TASKS without this would make it unremovable — a
    live duplicate briefing beyond the reach of the fix that removed it."""
    deleted = []
    import majordomo.trigger as trig

    real = trig._run_schtasks
    trig._run_schtasks = lambda args: (
        deleted.append(args[2]) or type("R", (), {"returncode": 0})()
    )
    try:
        trig.uninstall()
    finally:
        trig._run_schtasks = real

    assert r"Majordomo\OnBoot" in deleted
    assert r"Majordomo\OnLogon" in deleted
    assert r"Majordomo\OnWake" in deleted
