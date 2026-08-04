"""The `mj` command line.

This is the only module that catches the typed exceptions raised below it, which
is what makes the degradation policy legible in one place — Voicelog's pattern,
and the reason its failures never feel like crashes:

    MissingApiKey  → error, exit 1   (unrecoverable, and your fault to fix)
    LLMError       → warn, fall back to the raw per-source list, exit 0
    TTSError       → warn, text is already printed, exit 0

Text output is never gated on audio.

── ON THE IMPORTS ────────────────────────────────────────────────────────────
Every import here is deliberately **inside** the command that needs it.

This module is the hook entrypoint, and the hook runs on every prompt you
submit, every notification, every session start and end. Importing ``brief`` at
module scope pulled in the coordinator, the workers, ``httpx`` and ``yaml`` —
about 250ms of import work — on a code path that touches none of them. It made
every prompt you typed measurably slower to start.

So: nothing heavier than ``argparse`` at module scope. Measured, not assumed.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import sys

from majordomo import __version__


def _load_config(args):
    from majordomo import config as config_module
    from majordomo.config import ConfigFileNotFound

    try:
        return config_module.load(args.config)
    except ConfigFileNotFound as exc:
        print(f"error: no config file at {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# mj brief
# ---------------------------------------------------------------------------

def cmd_brief(args) -> None:
    from majordomo import brief, tts
    from majordomo.llm import MissingApiKey

    config = _load_config(args)

    try:
        result = brief.run(config)
    except MissingApiKey as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Text first, always — never gated on TTS.
    print(result.briefing.briefing_text)

    if result.briefing.needs_you:
        print("\nNeeds you:")
        for item in result.briefing.needs_you:
            print(f"  • [{item.source}] {item.title}")
            print(f"      {item.detail}")

    if args.explain:
        print()
        print(result.explain())

    if config.voice.enabled and not args.no_speak:
        try:
            print("Speaking… (Ctrl+C to skip)", file=sys.stderr)
            tts.speak(result.briefing.briefing_text, config.voice)
        except tts.TTSError as exc:
            print(f"warning: could not speak ({exc})", file=sys.stderr)
        except KeyboardInterrupt:
            print("\nskipped audio.", file=sys.stderr)


# ---------------------------------------------------------------------------
# mj sessions / mj resume
# ---------------------------------------------------------------------------

def _live_sessions(config):
    from majordomo import state
    from majordomo.workers.sessions import fold

    result = state.read_events_detailed()
    return (
        fold(result.events, stale_after_hours=config.sources.sessions.stale_after_hours),
        result.skipped,
    )


def cmd_sessions(args) -> None:
    from majordomo.workers.sessions import describe

    config = _load_config(args)
    live, skipped = _live_sessions(config)

    if not live:
        print("No live coding sessions.")
        print("(If you expected some, run `mj install-hooks` first.)", file=sys.stderr)
    for session in live:
        print(f"{session.status:<20} {session.session_id[:8]}  {describe(session)}")

    if args.debug and skipped:
        print(f"\n{skipped} unparseable line(s) in the state log.", file=sys.stderr)


def cmd_resume(args) -> None:
    from majordomo import resume as resume_mod

    config = _load_config(args)
    live, _ = _live_sessions(config)

    matches = [s for s in live if s.session_id.startswith(args.session_id)]
    if not matches:
        print(f"error: no live session starting with {args.session_id!r}", file=sys.stderr)
        sys.exit(1)
    if len(matches) > 1:
        print(f"error: {args.session_id!r} matches {len(matches)} sessions", file=sys.stderr)
        sys.exit(1)

    try:
        command = resume_mod.build(matches[0])
    except resume_mod.ResumeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(command.uri if command.uri else f"{' '.join(command.argv)}  (in {command.cwd})")
        return

    resume_mod.launch(command)


# ---------------------------------------------------------------------------
# mj install-hooks / uninstall-hooks
# ---------------------------------------------------------------------------

def cmd_install_hooks(args) -> None:
    from majordomo import install as install_mod

    try:
        result = install_mod.install()
    except install_mod.SettingsUnreadable as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Wired into {result.settings_path}")
    for entry in result.added:
        print(f"  + {entry}")
    if result.backup:
        print(f"Backed up first to {result.backup}")
    print("\nRestart any running Claude Code sessions to pick the hooks up.")


def cmd_uninstall_hooks(args) -> None:
    from majordomo import install as install_mod

    try:
        result = install_mod.uninstall()
    except install_mod.SettingsUnreadable as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Removed {result.removed} hook handler(s) from {result.settings_path}")
    if result.backup:
        print(f"Backed up first to {result.backup}")


# ---------------------------------------------------------------------------
# mj tray / mj install-trigger
# ---------------------------------------------------------------------------

def cmd_tray(args) -> None:
    from majordomo import tray

    config = _load_config(args)
    try:
        tray.run(config)
    except tray.TrayUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        pass


def cmd_install_trigger(args) -> None:
    from majordomo import trigger

    try:
        registered = trigger.install()
    except trigger.TriggerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Registered scheduled tasks:")
    for name in registered:
        print(f"  + {name}")


def cmd_uninstall_trigger(args) -> None:
    from majordomo import trigger

    try:
        removed = trigger.uninstall()
    except trigger.TriggerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Removed {removed} scheduled task(s).")


# ---------------------------------------------------------------------------
# mj hook (invoked by Claude Code, not by you)
# ---------------------------------------------------------------------------

def cmd_hook(args) -> None:
    # The hot path: this runs on every prompt you submit. It must import
    # nothing but the hook itself and the log it writes to.
    from majordomo.hook import run_hook

    # run_hook swallows everything and always returns 0. Anything that escaped
    # to here would still not be allowed to change the exit code.
    sys.exit(run_hook(args.event, args.matcher))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mj",
        description="One spoken briefing of everything that needs you.",
    )
    parser.add_argument("--version", action="version", version=f"majordomo {__version__}")
    parser.add_argument("--config", help="path to a config file (default: ~/.majordomo/config.yml)")

    sub = parser.add_subparsers(dest="command")

    p_brief = sub.add_parser("brief", help="fetch, fuse, print and speak the briefing")
    p_brief.add_argument("--no-speak", action="store_true", help="text only")
    p_brief.add_argument(
        "--explain",
        action="store_true",
        help="show which path each source took through the router, and why",
    )
    p_brief.set_defaults(func=cmd_brief)

    p_sessions = sub.add_parser("sessions", help="list live / idle / blocked coding sessions")
    p_sessions.add_argument("--debug", action="store_true", help="report unparseable log lines")
    p_sessions.set_defaults(func=cmd_sessions)

    p_resume = sub.add_parser("resume", help="jump back into a session")
    p_resume.add_argument("session_id", help="full id or a unique prefix")
    p_resume.add_argument("--dry-run", action="store_true", help="print the command, launch nothing")
    p_resume.set_defaults(func=cmd_resume)

    sub.add_parser("install-hooks", help="wire session-awareness into Claude Code").set_defaults(
        func=cmd_install_hooks
    )
    sub.add_parser("uninstall-hooks", help="remove Majordomo's Claude Code hooks").set_defaults(
        func=cmd_uninstall_hooks
    )

    sub.add_parser("tray", help="run the resident tray icon").set_defaults(func=cmd_tray)
    sub.add_parser(
        "install-trigger", help="brief me automatically on wake / boot / login"
    ).set_defaults(func=cmd_install_trigger)
    sub.add_parser("uninstall-trigger", help="remove the scheduled tasks").set_defaults(
        func=cmd_uninstall_trigger
    )

    p_hook = sub.add_parser("hook", help=argparse.SUPPRESS)
    p_hook.add_argument("--event", required=True)
    p_hook.add_argument("--matcher", default=None)
    p_hook.set_defaults(func=cmd_hook)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(0)

    # Fill missing env vars from ~/.majordomo/.env before anything reads them.
    # Real environment variables still win, so this only fills gaps — and it
    # means the tray and the Task Scheduler trigger find the same keys the CLI
    # does, which a shell export would never have reached.
    #
    # Skipped for the hook: it needs no credentials, and it runs on every prompt.
    if args.command != "hook":
        from majordomo import dotenv

        dotenv.load()

    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
