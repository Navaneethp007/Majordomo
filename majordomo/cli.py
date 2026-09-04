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


def safe_print(text: str = "", file=None) -> None:
    """Print text that came from outside, without letting the console kill us.

    ``sys.stdout.encoding`` is **cp1252** on a default Windows console, and
    ``print('\\U0001f916')`` raises UnicodeEncodeError there. Email subject
    lines are full of emoji, so anything printing a subject would crash on most
    runs — and a session topic containing one would do the same.

    Losing a glyph is a cosmetic failure; losing the whole briefing to a
    traceback is not. So unprintable characters degrade to a replacement mark
    and the text still lands.
    """
    stream = file or sys.stdout
    if stream is None:  # pythonw.exe: no console at all
        return
    try:
        print(text, file=stream)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        stream.write(text.encode(encoding, errors="replace").decode(encoding) + "\n")
    except (OSError, ValueError):
        # A closed or detached stream. The briefing is not worth a crash.
        pass


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

def should_speak(args, config, briefing, reports=None, now=None) -> bool:
    """Decide whether this briefing gets read aloud.

    Three modes, because the right answer differs by who asked:

    - ``--no-speak``            never. Text only.
    - ``--speak-if-needed``     only when something needs you, and only if we
                                have not already said this exact thing recently.
    - default                   always (you asked for a briefing; you get one).

    The middle mode is what the wake/boot/login trigger uses, and it has to
    handle two things beyond "is anything pending".

    **Silence must not be ambiguous.** Under ``pythonw.exe`` there is no console
    — ``sys.stdout`` is None and every print is discarded — so audio is the only
    channel the scheduled task has. A failed source carries no ``needs_you``
    items, so gating purely on that made an expired GitHub token sound exactly
    like a quiet morning. A source being down is itself worth saying.

    **Repetition must not be endless.** See ``speechgate``: the same situation
    stays quiet until it changes or ``repeat_after_minutes`` elapses.
    """
    from majordomo import speechgate

    if not config.voice.enabled or args.no_speak:
        return False
    if not args.speak_if_needed:
        return True

    reports = reports or []
    something_needs_you = bool(briefing.needs_you)
    # A source that was never set up is not an outage. Counting it as one made
    # the wake trigger speak on every single wake — the precise noise this flag
    # exists to prevent.
    something_is_down = any(not r.ok and not r.unconfigured for r in reports)
    if not (something_needs_you or something_is_down):
        return False

    fingerprint = speechgate.fingerprint(briefing, reports)
    if speechgate.is_repeat(fingerprint, config.voice.repeat_after_minutes, now=now):
        return False

    speechgate.record(fingerprint, now=now)
    return True


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
    safe_print(result.briefing.briefing_text)

    if result.briefing.needs_you:
        safe_print("\nNeeds you:")
        for item in result.briefing.needs_you:
            safe_print(f"  • [{item.source}] {item.title}")
            safe_print(f"      {item.detail}")

    # Separate heading, deliberately. These are things to look at, not things to
    # do — collapsing them into "Needs you" is how a briefing starts lying.
    if result.briefing.context:
        safe_print("\nAlso waiting (nothing required):")
        for item in result.briefing.context:
            safe_print(f"  · [{item.source}] {item.title}")

    if args.explain:
        safe_print()
        safe_print(result.explain())

    if should_speak(args, config, result.briefing, result.reports):
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
        fold(
            result.events,
            stale_after_hours=config.sources.sessions.stale_after_hours,
            # Passed explicitly, or `mj sessions` silently falls back to the
            # hardcoded default and disagrees with `mj brief` about which
            # sessions are live.
            active_timeout_minutes=config.sources.sessions.active_timeout_minutes,
        ),
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
        safe_print(f"{session.status:<20} {session.session_id[:8]}  {describe(session)}")

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
        safe_print(command.uri if command.uri else f"{' '.join(command.argv)}  (in {command.cwd})")
        return

    resume_mod.launch(command)


# ---------------------------------------------------------------------------
# mj chat / mj start
# ---------------------------------------------------------------------------

def cmd_chat(args) -> None:
    from majordomo import chat as chat_mod

    config = _load_config(args)
    try:
        chat_mod.run(config, resume=args.resume)
    except KeyboardInterrupt:  # pragma: no cover - the loop catches its own
        safe_print("")


def cmd_start(args) -> None:
    from majordomo import scaffold

    config = _load_config(args)
    idea = " ".join(args.idea).strip()
    if not idea:
        print("error: start what?", file=sys.stderr)
        sys.exit(1)

    try:
        scaffold.start(idea, config, dry_run=args.dry_run, write=safe_print)
    except scaffold.ScaffoldError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# mj ask
# ---------------------------------------------------------------------------

def cmd_ask(args) -> None:
    """One question, one answer, no conversation state."""
    from majordomo import activity as activity_mod
    from majordomo import context as context_mod
    from majordomo import prompts
    from majordomo.llm import LLMError, MissingApiKey, complete

    config = _load_config(args)
    question = " ".join(args.question).strip()
    if not question:
        print("error: ask what?", file=sys.stderr)
        sys.exit(1)

    # Opportunistic, never blocking: if the cache is old we try to top it up,
    # but a failure here costs freshness, not the answer.
    if not args.no_refresh and activity_mod.is_stale():
        result = activity_mod.refresh(config)
        if result.error:
            print(f"warning: activity is stale ({result.error})", file=sys.stderr)

    ctx = context_mod.build(config, query=question)
    messages = prompts.build_ask_prompt(ctx.render(), question)

    if args.explain:
        print(f"context: {ctx.summary()}", file=sys.stderr)

    try:
        reply = complete(messages, config.brain, config.brain.chat_model)
    except MissingApiKey as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except LLMError as exc:
        print(f"error: the model call failed ({exc})", file=sys.stderr)
        sys.exit(1)

    safe_print((reply or "").strip() or "(empty response)")


# ---------------------------------------------------------------------------
# mj activity
# ---------------------------------------------------------------------------

def cmd_activity(args) -> None:
    """Show what you have been doing on GitHub, from the local cache."""
    from majordomo import activity as activity_mod

    config = _load_config(args)
    # `or` would swallow --days 0, which is a legitimate "just today".
    days = args.days if args.days is not None else config.sources.github.activity_days

    if args.refresh:
        print("Fetching…", file=sys.stderr)
        result = activity_mod.refresh(config)
        # Not exclusive: one search can fail while the others return. Reporting
        # only the error would hide events we did get, and reporting only the
        # count would hide that the picture is incomplete.
        if result.fetched:
            print(f"Fetched {result.fetched}, {result.added} new.", file=sys.stderr)
        if result.error:
            print(f"warning: could not refresh ({result.error})", file=sys.stderr)

    events = activity_mod.recent(days=days)
    if not events:
        safe_print(f"No recorded activity in the last {days} days.")
        if not args.refresh:
            safe_print("(Run `mj activity --refresh` to fetch.)")
        return

    safe_print(activity_mod.digest(events, limit=args.limit))

    newest = activity_mod.newest_at()
    if newest is not None:
        safe_print(f"\nCache newest entry: {newest.date().isoformat()}")

    if args.debug:
        skipped = activity_mod.read_events_detailed().skipped
        if skipped:
            print(f"{skipped} unparseable line(s) in the log.", file=sys.stderr)


# ---------------------------------------------------------------------------
# mj remember
# ---------------------------------------------------------------------------

def cmd_remember(args) -> None:
    """Write, list, or forget one memory.

    Deliberately non-interactive. A prompt-and-confirm flow reads well in a demo
    and is miserable to script, to test, and to run from the tray — so a
    near-duplicate is *refused* with the name of what it collides with, and you
    re-run with --update or --force. The decision is still yours; it just
    happens in argv instead of at a blocking prompt.
    """
    from majordomo import memory as memory_mod

    config = _load_config(args)
    if not config.memory.enabled:
        print("error: memory is disabled in config", file=sys.stderr)
        sys.exit(1)

    if args.forget:
        if memory_mod.delete_memory(args.forget):
            safe_print(f"Forgot {args.forget}.")
        else:
            print(f"error: no memory named {args.forget!r}", file=sys.stderr)
            sys.exit(1)
        return

    if args.update and not args.text:
        # Silently listing here would look like the update succeeded.
        print(
            f"error: --update {args.update} needs the replacement text",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.update and memory_mod.read_memory(args.update) is None:
        # A typo would otherwise create a *new* memory alongside the one it was
        # meant to replace — the exact duplicate this flag exists to avoid.
        print(f"error: no memory named {args.update!r} to update", file=sys.stderr)
        sys.exit(1)

    if not args.text:
        result = memory_mod.read_all_detailed()
        if not result.memories:
            safe_print("Nothing remembered yet.")
            safe_print('Add one with: mj remember "some fact about you"')
        for item in sorted(result.memories, key=lambda m: m.name):
            safe_print(f"  [{item.type:<10}] {item.name}")
            safe_print(f"               {item.description}")
        if result.skipped:
            print(
                f"\n{result.skipped} unreadable file(s) in the memory directory.",
                file=sys.stderr,
            )
        return

    text = " ".join(args.text).strip()

    if not args.update and not args.force:
        similar = memory_mod.find_similar(text)
        if similar is not None:
            print(
                f"error: this looks like it already covers {similar.name!r}:\n"
                f"         {similar.description}\n"
                f"       re-run with --update {similar.name} to replace it, "
                f"or --force to keep both",
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        written = memory_mod.write_memory(
            memory_mod.MemoryCandidate(description=text, type=args.type),
            overwrite=args.update,
        )
    except memory_mod.MemoryError_ as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    safe_print(f"Remembered as {written.name}.")


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
        "--speak-if-needed",
        action="store_true",
        help="speak only when something actually needs you (what the wake trigger uses)",
    )
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

    p_chat = sub.add_parser("chat", help="an interactive session with your context loaded")
    p_chat.add_argument(
        "--resume", action="store_true", help="continue the most recent conversation"
    )
    p_chat.set_defaults(func=cmd_chat)

    p_start = sub.add_parser("start", help="scaffold a repo and open Claude Code in it")
    p_start.add_argument("idea", nargs="+", help="what you want to build")
    p_start.add_argument(
        "--dry-run", action="store_true", help="print what would happen, create nothing"
    )
    p_start.set_defaults(func=cmd_start)

    p_ask = sub.add_parser("ask", help="ask one question, with your context loaded")
    p_ask.add_argument("question", nargs="+", help="what to ask")
    p_ask.add_argument(
        "--no-refresh", action="store_true", help="use the activity cache as-is"
    )
    p_ask.add_argument(
        "--explain", action="store_true", help="report what context was loaded"
    )
    p_ask.set_defaults(func=cmd_ask)

    p_activity = sub.add_parser("activity", help="what you have been doing on GitHub")
    p_activity.add_argument(
        "--refresh", action="store_true", help="fetch from GitHub before showing"
    )
    p_activity.add_argument(
        "--days", type=int, default=None, help="window to show (default: config)"
    )
    p_activity.add_argument(
        "--limit", type=int, default=60, help="most entries to print (default: 60)"
    )
    p_activity.add_argument(
        "--debug", action="store_true", help="report unparseable log lines"
    )
    p_activity.set_defaults(func=cmd_activity)

    p_remember = sub.add_parser("remember", help="write, list, or forget a memory")
    p_remember.add_argument(
        "text", nargs="*", help="the fact to remember; omit to list what is remembered"
    )
    p_remember.add_argument(
        "--type",
        default="user",
        choices=["user", "preference", "project", "reference"],
        help="what kind of fact this is (default: user)",
    )
    p_remember.add_argument(
        "--update", metavar="NAME", help="replace an existing memory instead of adding one"
    )
    p_remember.add_argument(
        "--force", action="store_true", help="write even if it looks like a duplicate"
    )
    p_remember.add_argument("--forget", metavar="NAME", help="delete a memory by name")
    p_remember.set_defaults(func=cmd_remember)

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
