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
from pathlib import Path

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
        # Flushed, because the other half of this program's output goes to
        # stderr, which is not buffered. Without it, a redirected run showed the
        # "stopped:" line *before* the work it stopped after — the failure
        # arriving ahead of the result it was reporting on.
        print(text, file=stream, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "ascii"
        stream.write(text.encode(encoding, errors="replace").decode(encoding) + "\n")
        # Flushed here too. This branch exists *for* emoji-bearing lines, so
        # without it the stderr-arrives-first ordering bug survived in exactly
        # the case the flush above was added to fix.
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
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

    if args.list:
        saved = chat_mod.saved_sessions()
        if not saved:
            safe_print("No saved conversations yet.")
            return
        for path in reversed(saved):  # newest first, the way you think of them
            safe_print("  " + chat_mod.describe_session(path))
        return

    transcript = None
    if args.resume and args.resume is not True:
        # A specific one, by id or unique prefix — the same matching `mj resume`
        # uses for sessions, so the two commands behave alike.
        transcript = chat_mod.find_session(args.resume)
        if transcript is None:
            count = chat_mod.match_count(args.resume)
            problem = (
                f"{args.resume!r} matches {count} conversations"
                if count > 1
                else f"no saved conversation starting with {args.resume!r}"
            )
            print(f"error: {problem}", file=sys.stderr)
            print("(`mj chat --list` shows them)", file=sys.stderr)
            sys.exit(1)

    try:
        chat_mod.run(
            config,
            resume=bool(args.resume),
            transcript=transcript,
            terminal=terminal(),
        )
    except KeyboardInterrupt:  # pragma: no cover - the loop catches its own
        safe_print("")


def ask(question: str) -> str:
    """Put a question to whoever is at the keyboard. "" when nobody is.

    The counterpart to ``safe_print``: the one place a library module's request
    for an answer becomes an actual read of stdin.

    ``EOFError`` is "" — nobody is there, which every caller treats as a
    refusal. **Ctrl+C propagates**, because it is not an answer. Swallowing it
    here collapsed "declined" and "interrupted" into one value, and a loop over
    memory candidates then declined the first and cheerfully asked about the
    second. What an interrupt should abort differs by caller, so the callers
    decide.
    """
    try:
        return input(question)
    except EOFError:
        safe_print("")
        return ""


def terminal():
    """The real console, wired into ``chat``'s three callables.

    `chat` used to import ``safe_print`` and ``confirm_action`` from here, which
    pointed the dependency the wrong way round — the conversation depending on
    the command line rather than the other way about. Handing it these three
    instead leaves `chat` importable without a console at all.
    """
    from majordomo.chat import Terminal

    return Terminal(write=safe_print, ask=ask, confirm=confirm_action)


def confirm_action(name: str, arguments: dict) -> bool:
    """Show what is about to happen and ask. The safety gate, at the terminal.

    Shows the *content* of a write and the whole command for a run — approving
    something you cannot see is not approval. Defaults to no: a stray Enter
    should decline, not authorise a change to your files.
    """
    from majordomo import tools

    safe_print(f"\n  {tools.describe_call(name, arguments)}")
    # (see _edit_preview for why an edit is not just two truncated blocks)

    if name == "write_file":
        content = tools.as_text(arguments.get("content"))
        lines = content.splitlines()
        for line in lines[:12]:
            safe_print(f"    | {line}")
        if len(lines) > 12:
            safe_print(f"    | … {len(lines) - 12} more lines")
    elif name == "edit_file":
        for line in _edit_preview(
            tools.as_text(arguments.get("old")), tools.as_text(arguments.get("new"))
        ):
            safe_print(f"    {line}")

    if not sys.stdin.isatty():
        # Piped or scripted: nobody can answer, so every gated call would be
        # declined and the agent would spend its whole turn budget being told
        # no. Say why once, and let the caller decide to pass --yes.
        print(
            "error: this needs approval and stdin is not a terminal — "
            "re-run interactively, or pass --yes to approve everything",
            file=sys.stderr,
        )
        raise SystemExit(1)

    try:
        answer = input("  allow this? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        safe_print("")
        return False
    return answer in ("y", "yes")


#: Lines shown per side of an edit before eliding. Small enough to read at a
#: glance, which is the only way a confirmation prompt actually gets read.
EDIT_PREVIEW_LINES = 8


def _edit_preview(old: str, new: str) -> list[str]:
    """Render an edit as the part that actually changes.

    Showing the first N lines of each side is wrong when they share a prefix —
    an append produces two blocks that look identical, the change falls off the
    bottom, and you approve a no-op that isn't one. That happened: an edit
    adding a whole function previewed as three unchanged lines.

    So trim the common prefix and suffix first, spend the budget on the
    difference, and always say when something was elided. A gate that hides
    what it is gating is worse than no gate — it manufactures confidence.
    """
    before, after = old.splitlines(), new.splitlines()

    head = 0
    while head < len(before) and head < len(after) and before[head] == after[head]:
        head += 1

    tail = 0
    while (
        tail < len(before) - head
        and tail < len(after) - head
        and before[-1 - tail] == after[-1 - tail]
    ):
        tail += 1

    removed = before[head : len(before) - tail]
    added = after[head : len(after) - tail]

    lines: list[str] = []
    if head:
        lines.append(f"  {head} unchanged line(s)")
    for marker, block in (("-", removed), ("+", added)):
        for line in block[:EDIT_PREVIEW_LINES]:
            lines.append(f"{marker} {line}")
        if len(block) > EDIT_PREVIEW_LINES:
            lines.append(f"{marker} … {len(block) - EDIT_PREVIEW_LINES} more line(s)")
    if tail:
        lines.append(f"  {tail} unchanged line(s)")

    if not removed and not added:
        # Whitespace-only, or a genuine no-op. Either way, say so rather than
        # printing nothing and leaving the prompt looking like a bug.
        lines.append("  (no visible change — whitespace only)")
    return lines


def cmd_do(args) -> None:
    """Give the agent a task in the current directory."""
    from majordomo import agent

    config = _load_config(args)
    task = " ".join(args.task).strip()
    if not task:
        print("error: do what?", file=sys.stderr)
        sys.exit(1)

    root = Path(args.directory or ".")
    if not root.is_dir():
        # write_file creates parent directories, which is what lets the agent
        # build src/calc/ from nothing — but it also means a typo here would
        # quietly populate a whole bogus tree instead of failing. The project
        # root is the one path we will not create on your behalf.
        print(f"error: no directory at {root}", file=sys.stderr)
        sys.exit(1)

    confirm = agent.always_allow if args.yes else confirm_action

    if args.yes:
        print(
            "warning: --yes approves every write and command without asking",
            file=sys.stderr,
        )

    from majordomo.llm import MissingApiKey

    try:
        outcome = agent.run(task, config, root=root, confirm=confirm, write=safe_print)
    except MissingApiKey as exc:
        # agent.run swallows LLMError so a half-finished task keeps its trail,
        # but a missing key is unrecoverable and identical on every retry —
        # same policy as every other command here.
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    # One rendering, shared with `/agent` — two of them drift, and the last
    # time they did the transcript recorded a stop message while the screen
    # showed the work that had actually succeeded.
    from majordomo import chat as chat_mod
    from majordomo import render as render_mod

    report = chat_mod.agent_report(outcome)
    safe_print("")
    safe_print(render_mod.render(report))

    # On stderr as well, because the exit status is 0 either way and that is
    # how a script notices — but only when the report has not already said
    # it. With nothing to salvage the report *is* the stop message, and
    # printing it twice reads as two separate problems.
    if outcome.stopped_because and outcome.stopped_because not in report:
        print(f"\nstopped: {outcome.stopped_because}", file=sys.stderr)

    declined = [s for s in outcome.steps if not s.approved]
    if declined:
        print(f"({len(declined)} action(s) declined)", file=sys.stderr)


def cmd_config(args) -> None:
    """Show what configuration is in force, or install the annotated example.

    Every model here is a config value rather than code, but nothing said so —
    the defaults were only visible by reading ``config.py``. This makes the
    roles and their current values something you can look at.
    """
    from majordomo.paths import default_config_path

    target = Path(args.config) if args.config else default_config_path()

    if args.init:
        # Beside this module, not at the repo root — a wheel install has no repo
        # root, and this has to work for someone who ran `pip install`.
        example = Path(__file__).resolve().parent / "config.example.yml"
        if not example.is_file():
            print(f"error: no example at {example}", file=sys.stderr)
            sys.exit(1)
        if target.exists() and not args.force:
            print(
                f"error: {target} already exists — pass --force to overwrite, "
                f"or edit it directly",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError as exc:
            print(f"error: could not write {target}: {exc}", file=sys.stderr)
            sys.exit(1)
        safe_print(f"Wrote {target}.")
        safe_print("Every value is commented. Nothing in it is required.")
        return

    config = _load_config(args)
    brain = config.brain

    safe_print(f"  file       {target}{'' if target.is_file() else '  (absent — using defaults)'}")
    safe_print(f"  provider   {brain.provider}  ->  {brain.base_url}")
    import os

    key_state = "set" if os.environ.get(brain.api_key_env) else "NOT SET"
    safe_print(f"  key from   {brain.api_key_env}  ({key_state})")
    safe_print("")
    safe_print("  Models, by role:")
    for role, model, purpose in (
        ("worker", brain.worker_model, "compress each source"),
        ("fuser", brain.fuser_model, "write the spoken briefing"),
        ("reducer", brain.reducer_model, "handle an oversized payload"),
        ("chat", brain.chat_model, "mj ask, mj chat"),
        ("agent", brain.agent_model or f"{brain.chat_model}  (via chat)", "mj do, /agent"),
        ("fallback", brain.fallback_model or "(none)", "when a role's model fails"),
    ):
        safe_print(f"    {role:<9} {model}")
        safe_print(f"    {'':<9}   {purpose}")
    safe_print("")
    if not target.is_file():
        safe_print("  `mj config --init` writes an annotated file you can edit.")


def cmd_mic(args) -> None:
    """Report what your microphone sounds like, in the numbers voice input uses.

    Voice failing is almost always one threshold being wrong for one microphone,
    and nothing about the symptom says so — the recording just stops early. This
    turns that into a number you can put in the config.
    """
    from majordomo import asr

    config = _load_config(args)

    try:
        result = asr.measure(config.voice, seconds=args.seconds, write=safe_print)
    except asr.ASRError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    safe_print("")
    safe_print(f"  quietest chunk    {result['min']:.0f}")
    safe_print(f"  median            {result['p50']:.0f}")
    safe_print(f"  90th percentile   {result['p90']:.0f}")
    safe_print(f"  loudest chunk     {result['max']:.0f}")
    safe_print(f"  threshold in use  {result['threshold']:.0f}")
    safe_print(f"  counted as quiet  {result['quiet_fraction']:.0%} of the time")
    safe_print(
        f"  longest quiet run {result['longest_quiet_ms']}ms "
        f"(recording stops at {result['stops_at_ms']}ms)"
    )
    safe_print("")

    # p10 stands in for your pauses, p90 for your speech. The threshold has to
    # sit between them — and if there is no gap, the recording contained no
    # speech at all, which is a different problem with a different fix. Advising
    # a number in that case is how you end up told to set a threshold below your
    # own room tone.
    quiet_level = result["p10"]
    loud_level = result["p90"]

    if loud_level < max(quiet_level * 3, 1):
        safe_print(
            "Nothing here looks like speech — the loud and quiet parts sit at "
            "the same level, so this was room tone throughout. Run it again and "
            "talk for the whole measurement."
        )
        return

    if quiet_level < result["threshold"] < loud_level:
        safe_print("These look healthy — speech clears the threshold, pauses do not.")
        return

    if result["threshold"] >= loud_level:
        safe_print(
            "This would have cut you off: the threshold sits above your "
            "speaking level, so every chunk read as silence."
        )
    else:
        safe_print(
            "This would never stop on its own: the threshold sits below your "
            "room tone, so nothing ever reads as silence."
        )
    safe_print("Put this in ~/.majordomo/config.yml:")
    safe_print(f"\n  voice:\n    silence_rms: {int((quiet_level * loud_level) ** 0.5)}\n")


def cmd_review(args) -> None:
    """Hand a repository to Claude Code for a review.

    Majordomo cannot read code — its context is memory and GitHub *activity*.
    Rather than build a second code reviewer, it opens the one you already have.
    """
    from majordomo.resume import spawn_detached

    target = Path(args.directory or ".").resolve()
    if not target.is_dir():
        print(f"error: no directory at {target}", file=sys.stderr)
        sys.exit(1)

    safe_print(f"Opening Claude Code in {target}…")
    spawn_detached(["claude", "/code-review"], cwd=str(target))


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

def _shell_quote(text: str) -> str:
    """Quote a task so the printed command survives being pasted.

    Interpolating into double quotes broke on any task containing one:

        mj do "find the "TODO" markers"     <- three arguments, not one

    Double quotes with the inner ones escaped, always. The previous version
    reached for single quotes when the text held a double, which is correct in
    bash and PowerShell and **wrong in cmd.exe** — cmd has no single-quote
    syntax and passes them through as literal characters, so the command it
    printed could not be pasted into the shell this project is mostly used from.

    `\\"` is not cmd.exe's own escape either; cmd cannot represent an embedded
    double quote in a way the other two also accept. Rather than pick a form
    that is wrong somewhere and claim otherwise, this picks the one that works
    in two of three and `describe_quoting` says so out loud.
    """
    return '"' + text.replace('"', '\\"') + '"'


def describe_quoting(text: str) -> str:
    """A warning when the printed command will not paste everywhere, else "".

    Said only when it applies. A caveat attached to every command would be
    ignored by the time it mattered.
    """
    if '"' not in text:
        return ""
    return (
        "(the quotes are escaped for bash and PowerShell; cmd.exe cannot "
        "represent an embedded quote — retype it there)"
    )


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

    from majordomo import render as render_mod

    answer = (reply or "").strip() or "(empty response)"

    # One-shot, so there is nobody to ask — print the command instead of an
    # offer. Anything `/agent` would say here is wrong: there is no chat to stay
    # in, and two different instructions in one output is worse than none.
    # The same shape as `chat._offer_agent`, deliberately. These two have
    # disagreed twice: first about whether a leaked tool call keeps its prose,
    # then about whether the marker branch does. Both times the branch that was
    # already right stayed right and the sibling stayed wrong, because there was
    # nothing making them one algorithm.
    task = prompts.needs_agent(answer)
    leaked = task is None and prompts.looks_like_a_tool_call(answer)

    if task is not None or leaked:
        # Whichever marker it is, keep what was written before it. A model that
        # wrote a useful paragraph and then asked for the agent should lose the
        # marker, not the paragraph.
        kept = (
            prompts.strip_tool_call(answer) if leaked
            else prompts.strip_needs_agent(answer)
        )
        if kept:
            safe_print(render_mod.render(kept))
            safe_print("")
        if leaked:
            safe_print(prompts.NO_TOOLS_HERE)
            safe_print("")

        # `not task` means the marker carried nothing; the question is the best
        # available description of the job.
        task = task or question
        safe_print("That needs the agent, which can read and change files:")
        safe_print(f"  mj do {_shell_quote(task)}")
        caveat = describe_quoting(task)
        if caveat:
            safe_print(f"  {caveat}")
        return

    safe_print(render_mod.render(answer))


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

def _resolve_similar(similar) -> str:
    """Ask what to do about a memory that looks like an existing one.

    Returns "update", "keep" or "cancel".

    This used to refuse outright and tell you to re-run with a flag. But the
    similarity check is a noisy heuristic — see ``memory.find_similar`` — so a
    refusal is wrong often enough that the habit it teaches is reflexive
    ``--force``, which is the same as not having the check at all. A question
    costs one keystroke when it guesses wrong.

    Non-interactive callers still get the old refusal: a script must not block
    on a prompt nobody is there to answer.
    """
    safe_print(f"\n  This looks close to {similar.name!r}:")
    safe_print(f"    {similar.description}")

    if not sys.stdin.isatty():
        print(
            f"error: re-run with --update {similar.name} to replace it, "
            f"or --force to keep both",
            file=sys.stderr,
        )
        return "cancel"

    try:
        answer = input("  [u]pdate it, [k]eep both, [c]ancel? ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        safe_print("")
        return "cancel"

    if answer.startswith("u"):
        return "update"
    if answer.startswith("k"):
        return "keep"
    return "cancel"


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
        try:
            existed = memory_mod.delete_memory(args.forget)
        except memory_mod.MemoryError_ as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        if existed:
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
            choice = _resolve_similar(similar)
            if choice == "cancel":
                sys.exit(1)
            if choice == "update":
                args.update = similar.name

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
        registered, warnings = trigger.install()
    except trigger.TriggerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Registered scheduled tasks:")
    for name in registered:
        print(f"  + {name}")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)


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
        "--resume",
        nargs="?",
        const=True,
        metavar="ID",
        help="continue a conversation: the most recent, or one by id/prefix",
    )
    p_chat.add_argument(
        "--list", action="store_true", help="list saved conversations and exit"
    )
    p_chat.set_defaults(func=cmd_chat)

    p_do = sub.add_parser("do", help="give the agent a task in this directory")
    p_do.add_argument("task", nargs="+", help="what you want done")
    p_do.add_argument(
        "--directory", "-C", default=None, help="work here instead of the current directory"
    )
    p_do.add_argument(
        "--yes",
        action="store_true",
        help="approve every write and command without asking (careful)",
    )
    p_do.set_defaults(func=cmd_do)

    p_review = sub.add_parser("review", help="open Claude Code on a repo to review it")
    p_review.add_argument(
        "directory", nargs="?", default=None, help="the repo (default: here)"
    )
    p_review.set_defaults(func=cmd_review)

    p_config = sub.add_parser(
        "config", help="show the active configuration, or install the example"
    )
    p_config.add_argument(
        "--init", action="store_true", help="write the annotated example config"
    )
    p_config.add_argument(
        "--force", action="store_true", help="overwrite an existing config file"
    )
    p_config.set_defaults(func=cmd_config)

    p_mic = sub.add_parser("mic", help="measure your microphone, to tune voice input")
    p_mic.add_argument(
        "--seconds", type=float, default=10.0, help="how long to record (default: 10)"
    )
    p_mic.set_defaults(func=cmd_mic)

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
