"""The interactive session — the loop, the prompt, and what the commands do.

The conversation itself lives in ``session.py``: turns, compaction, the file on
disk. This is the half that touches a person — keys, rendering, the offers, the
waiting indicator.

They were one module, and the bugs clustered precisely at the seam between them.
See ``session.py``'s header for the three worst; the short version is that a
loop free to reach into ``session.turns[-1]`` and rewrite it will keep doing so,
and each time the stored conversation and the one on screen drift a little
further apart.

The rule that fell out of fixing them, and the one to keep: **decide what to
show first, then store exactly that.** ``agent_report`` and
``_restate_last_answer`` are both that rule made mechanical.

── WHERE THE I/O IS ─────────────────────────────────────────────────────────
Nothing here reads stdin or writes stdout directly. A ``Terminal`` carries the
three callables a conversation needs — write, ask, confirm — and the default is
headless: silence and a refusal. That is what lets ``chat`` be imported by
something with no console, and what stopped this module importing the CLI for
``safe_print`` and ``confirm_action``, which had the dependency pointing the
wrong way round.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from majordomo.config import Config

# Re-exported, not owned. The REPL is one consumer of the conversation model,
# and callers that already say `chat.Session` keep working — the split is about
# where the code lives, not about breaking every import in one commit.
from majordomo.session import (  # noqa: F401
    KEEP_RECENT_TURNS,
    MIN_RECENT_TURNS,
    SUMMARY_MARKER,
    ChatFailed,
    Session,
    Turn,
    clear,
    compact,
    describe_session,
    estimate_tokens,
    find_session,
    latest_transcript,
    load_turns,
    match_count,
    needs_compaction,
    new_session,
    save,
    saved_sessions,
    send,
)

# One clock, not two. Both files defined `_now`, identically — harmless until
# either is made injectable for testing compaction, at which point the
# terminal layer and the session model start stamping turns from different
# clocks and the transcript's ordering quietly stops being reliable.
from majordomo.session import _now  # noqa: F401

HELP = """\
  /context    what memory and activity are loaded
  /clear      start a fresh conversation, keeping the same loaded context
  /remember [FACT] keep a fact now, or propose some from this conversation
  /build NAME scaffold a repo from this conversation and open Claude Code
  /agent TASK put the agent to work here — it asks before writing or running
  /voice      speak your next message instead of typing it
  /help       this
  /exit       leave (also Ctrl+D)"""


@dataclass(frozen=True)
class Terminal:
    """Where a conversation puts text and where it gets answers.

    Everything else in this module is a pure function of a ``Session`` and a
    ``Config``. These three callables are the only places it touches a human,
    and collecting them here means the module never imports the CLI — which it
    did, for ``safe_print`` and ``confirm_action``, making the dependency point
    the wrong way.

    The default is **headless**: it writes nowhere and declines everything. A
    caller that forgets to pass a real one gets silence and a refusal, not an
    unattended write or a blocked read on a stdin nobody is watching.

    ``agent.run`` already takes ``confirm`` and ``write`` the same way and for
    the same reason. This is that pattern, with the third callable a
    conversation also needs.
    """

    #: One line of output.
    write: Callable[[str], None] = lambda _text: None
    #: A question, returning whatever was typed. "" means no answer.
    ask: Callable[[str], str] = lambda _question: ""
    #: Approval for a tool call the agent wants to make: ``(name, arguments)``.
    confirm: Callable[[str, dict], bool] = lambda _name, _arguments: False

    def yes(self, question: str) -> bool:
        """Was the answer to this yes? Anything else, including nothing, is no."""
        return self.ask(question).strip().lower() in ("y", "yes")


#: Used when a caller passes nothing. Writes nowhere, agrees to nothing.
HEADLESS = Terminal()


class Waiting:
    """Ticking seconds on one line while a reply is in flight.

    **Seconds, not a spinner.** Measured, chat latency here ranges from 2s to
    31s on the same prompt; a spinner looks identical at both, while the number
    tells you whether it is slow or stuck. That is the whole reason this exists
    — the gap was already survivable, it just gave you nothing to judge.

    Two constraints shape it:

    - **Silent when stdout is not a terminal.** Redirected, a session would fill
      with timer frames. ``render.supports_ansi`` already answers this question.
    - **It never competes with the line editor for the cursor.** It runs only
      while ``send`` is blocking, which is strictly between reads — the editor
      has returned a line and has not been called again. That ordering is why a
      carriage return here is safe, and it is the reason to keep this wrapped
      around ``send`` rather than started anywhere more convenient.
    """

    #: How often to repaint. Fast enough to look live, slow enough that a
    #: 30-second wait is not 300 writes.
    INTERVAL = 0.25

    def __init__(self, label: str = "thinking", stream=None):
        import sys

        self.label = label
        self.stream = stream if stream is not None else sys.stdout
        self._stop = None
        self._thread = None
        self._painted = 0

    def _enabled(self) -> bool:
        from majordomo import render

        return self.stream is not None and render.supports_ansi(self.stream)

    def _paint(self, text: str) -> None:
        # Pad to erase the previous frame: "10s" over "9s" would otherwise
        # leave the stray digit behind.
        padding = " " * max(0, self._painted - len(text))
        try:
            self.stream.write("\r" + text + padding)
            self.stream.flush()
        except (OSError, ValueError):
            # A closed or detached stream. A progress indicator is never worth
            # taking the conversation down with it.
            return
        self._painted = len(text)

    def _clear(self) -> None:
        if self._painted:
            self._paint("")
            try:
                self.stream.write("\r")
                self.stream.flush()
            except (OSError, ValueError):
                pass
            self._painted = 0

    def __enter__(self):
        if not self._enabled():
            return self

        import threading
        import time

        self._stop = threading.Event()
        started = time.monotonic()

        def tick():
            while not self._stop.wait(self.INTERVAL):
                self._paint(f"  {self.label}… {int(time.monotonic() - started)}s")

        self._thread = threading.Thread(target=tick, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        if self._stop is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
        self._clear()
        return False


def run(
    config: Config,
    resume: bool = False,
    transcript: Path | None = None,
    terminal: Terminal = HEADLESS,
) -> None:
    """The interactive loop. Returns when the user leaves."""
    from majordomo import keys, render as render_mod

    session = new_session(config, resume=resume, transcript=transcript)

    terminal.write("Majordomo. /help for commands, /exit to leave.")
    terminal.write(f"({session.context_summary})")
    if session.turns:
        terminal.write(f"Resumed {len(session.turns)} turns from {session.path.name}.")
    terminal.write("")

    trigger = config.voice.listen_key if config.voice.enabled else ""

    while True:
        try:
            typed = keys.read_line("you › ", trigger)
        except (EOFError, KeyboardInterrupt):
            terminal.write("")
            break

        # Two ways in, one path out: the key at the prompt, or /voice typed.
        # Both land here rather than in _handle_command, which has no business
        # knowing about the send loop.
        if typed is keys.TRIGGERED or (
            isinstance(typed, str) and typed.strip().lower() == "/voice"
        ):
            spoken = _listen(config, terminal)
            if spoken is None:
                continue
            line = spoken
        else:
            line = typed.strip()

        if not line:
            continue

        if line.startswith("/"):
            if _handle_command(line, session, config, terminal):
                break
            continue

        try:
            with Waiting():
                reply = send(session, line, config)
        except ChatFailed as exc:
            terminal.write(f"\n[the model call failed: {exc}]")
            terminal.write("[your message was not sent — try again]\n")
            continue

        # Strip the marker first — it was never meant to be read, and
        # storing it would put the protocol token in the transcript.
        reply, remember = _proposed_memory(reply)
        _restate_last_answer(session, reply)

        handled = _offer_agent(session, config, reply, line, terminal)
        if not handled:
            # Rendered, not raw: models write markdown regardless of
            # instruction, and literal `**bold**` reads worse than the
            # prose would have.
            terminal.write(f"\nmj  › {render_mod.render(reply)}\n")
        save(session)

        # Asked *after* the answer is on screen. It used to block before it,
        # so you approved a memory without having seen what produced it.
        if remember:
            try:
                _save_memory(remember, config, terminal)
            except KeyboardInterrupt:
                # Declines the memory and stays in the conversation.
                terminal.write("")

    # Guarded, not unconditional. An empty transcript still sorts newest by
    # mtime, so opening the REPL and quitting made *that* the latest session —
    # and `--resume` then restored nothing over yesterday's conversation. The
    # `/clear` path always guarded this; the two disagreed.
    #
    # On `log` rather than `turns`: `turns` shrinks when compaction folds it, so
    # a long conversation that had just been compacted could look empty enough
    # to skip saving. `log` is what `save` writes.
    if session.log:
        save(session)
        terminal.write(f"Saved to {session.path}.")
        _offer_memories(session, config, terminal)


def agent_report(outcome) -> str:
    """What the agent has to show for itself, as one piece of text.

    Built **before** anything is printed or stored, because those two must be
    the same string. The previous shape decided the stored turn from
    ``outcome.answer`` and only afterwards printed a salvaged result — so you
    watched tool output appear while the transcript recorded "(the agent
    stopped: …)", and the next turn's context and ``--resume`` both lost it.

    Shared with ``cmd_do`` for the same reason ``run_agent`` is shared with
    ``/agent``: two renderings of one outcome drift.
    """
    from majordomo import agent

    if outcome.answer:
        return outcome.answer

    salvaged = agent.last_result(outcome)
    if salvaged:
        # Labelled. Printed bare, a pytest trace reads exactly like the agent's
        # own answer — and this is reached precisely when there is no answer to
        # confuse it with.
        return f"[no summary — the last thing it got back]\n\n{salvaged}"

    if outcome.stopped_because:
        return f"(the agent stopped: {outcome.stopped_because})"
    return "(the agent finished without saying anything)"


def run_agent(session: Session, config: Config, task: str, terminal: Terminal = HEADLESS) -> None:
    """Put the agent to work and fold what happened back into the conversation.

    Shared by ``/agent`` and by the hand-off offer, so the two cannot drift. The
    offer exists precisely to *be* the same thing as typing the command, and a
    second copy of this would eventually stop being that.
    """
    from majordomo import agent
    from majordomo.llm import MissingApiKey

    try:
        outcome = agent.run(
            task, config, confirm=terminal.confirm, write=terminal.write
        )
    except MissingApiKey as exc:
        # `agent.run` swallows LLMError so a half-finished task keeps its trail,
        # but MissingApiKey is a bare Exception and went straight through —
        # taking the REPL and the conversation with it. `cmd_do` guards the
        # identical call; the two paths disagreed.
        terminal.write(f"[{exc}]")
        return

    # One string, decided first, then both shown and stored. Anything else
    # reintroduces the gap this exists to close.
    report = agent_report(outcome)

    # Fold the result back in. Without this the agent's work is invisible to the
    # next turn, and you would have to re-explain what just happened to the
    # thing that was watching it happen.
    asked = Turn(role="user", content=f"[I asked the agent to: {task}]", at=_now())
    answered = Turn(role="assistant", content=report, at=_now())
    # Both lists: `turns` is the working context, `log` is what gets saved.
    # Appending to `turns` alone silently dropped these from the transcript.
    session.turns.extend((asked, answered))
    session.log.extend((asked, answered))
    save(session)

    terminal.write(f"\n{report}\n")
    # Only when the report has not already carried it — with nothing to
    # salvage the report *is* the stop message, and saying it twice reads
    # as two separate problems.
    if outcome.stopped_because and outcome.stopped_because not in report:
        terminal.write(f"[stopped: {outcome.stopped_because}]")


def _restate_last_answer(session: Session, content: str) -> None:
    """Rewrite the assistant turn just stored, in both the context and the record.

    ``send`` appends the reply the moment it arrives, so by the time anything
    inspects it the raw text is already in ``turns`` *and* ``log``. Suppressing
    it on screen alone left a bare ``NEEDS_AGENT:`` line — or a leaked tool-call
    block — replayed to the model next turn, restored by ``--resume``, and
    handed to the memory proposer at exit. Hidden in the one place it did no
    harm, kept in every place it did.

    The rule: what is stored is what you were shown.
    """
    replacement = Turn(role="assistant", content=content, at=_now())
    for record in (session.turns, session.log):
        if record and record[-1].role == "assistant":
            record[-1] = replacement


def _offer_agent(session, config, reply, asked_for, terminal: Terminal = HEADLESS) -> bool:
    """Offer to hand a request to the agent. Returns whether it took over.

    Chat has no tools, deliberately — the agent reads files freely, with no
    prompt, and slipping that into the most casual surface you have is not a
    thing to do silently. But "review that folder" is a perfectly reasonable
    sentence to say in a conversation, and making you retype it as a command is
    friction for its own sake. So: it asks, and a `y` runs exactly what
    ``/agent`` would.

    Two ways in. The model is told to emit ``NEEDS_AGENT: <task>`` when a
    question genuinely needs the disk — a structural signal, parsed in Python,
    the same discipline as ``needs_you``. And when it ignores that and writes
    out a tool call instead, the markup is dropped rather than printed: raw
    ``<function_call>`` text reads as though something ran.
    """
    from majordomo import prompts
    from majordomo import render as render_mod

    task = prompts.needs_agent(reply)
    leaked = task is None and prompts.looks_like_a_tool_call(reply)
    if task is None and not leaked:
        return False

    # An empty task means the marker arrived carrying nothing. It is still a
    # hand-off — the alternative is printing the protocol token — so fall back
    # to what was actually asked for.
    if not task:
        task = asked_for

    shown: list[str] = []

    # Whichever branch we are on, keep whatever prose came with it. A model that
    # wrote three good paragraphs and one stray marker should lose the marker,
    # not the paragraphs — and declining the offer must not cost you the answer.
    # The leaked branch always did this; the marker branch did not, and the two
    # doing the same job differently is how one of them stays wrong.
    kept = (
        prompts.strip_tool_call(reply) if leaked else prompts.strip_needs_agent(reply)
    )
    if kept:
        shown.append(render_mod.render(kept))
    if leaked:
        shown.append(prompts.NO_TOOLS_HERE)

    shown.append("That needs the agent, which can read and change files here.")
    shown.append(f"  {task}")

    body = "\n\n".join(shown)
    terminal.write(f"\nmj  > {body}\n")
    _restate_last_answer(session, body)

    try:
        go = terminal.yes("  Run it? [y/N] ")
    except KeyboardInterrupt:
        # Declines this offer and returns to the prompt. Letting it through
        # would end the conversation over a change of mind about one task.
        terminal.write("")
        go = False

    if not go:
        terminal.write("[not run - /agent <task> whenever you want it]\n")
        return True

    run_agent(session, config, task, terminal)
    return True


def _handle_command(line: str, session: Session, config: Config, terminal: Terminal = HEADLESS) -> bool:
    """Run a /command. Returns True when the loop should end."""
    parts = line.split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command in ("/exit", "/quit"):
        return True

    if command == "/help":
        terminal.write(HELP)
        return False

    if command == "/context":
        terminal.write(session.context_summary)
        terminal.write(
            f"{len(session.turns)} turns, ~{estimate_tokens(session)} tokens"
            + (f", {session.compactions} compaction(s)" if session.compactions else "")
        )
        if session.compaction_failed:
            terminal.write(
                "  warning: this is over the compaction threshold and could not "
                "be summarised — it will keep growing. /clear starts fresh."
            )
        return False

    if command == "/clear":
        if session.turns:
            save(session)
            # `save` already tolerates a session with no file; naming the file
            # afterwards did not, so a session built without a path crashed on
            # the report rather than on the write.
            where = f" to {session.path.name}" if session.path else ""
            terminal.write(f"Saved {len(session.turns)} turns{where}.")
        clear(session)
        terminal.write("Cleared. Same context loaded, fresh conversation.")
        return False

    if command == "/remember":
        if argument:
            # The explicit path, for when you already know what you want kept.
            # Same confirmation as the offer — nothing here writes unasked.
            try:
                _save_memory(argument, config, terminal)
            except KeyboardInterrupt:
                terminal.write("")
        else:
            _offer_memories(session, config, terminal)
        return False

    if command == "/agent":
        if not argument:
            terminal.write("usage: /agent <what you want done>")
            return False
        run_agent(session, config, argument, terminal)
        return False

    if command == "/build":
        from majordomo import scaffold

        if not argument:
            terminal.write("usage: /build <name>")
            return False
        try:
            scaffold.from_chat(
                argument, session.transcript(), config, write=terminal.write
            )
        except scaffold.ScaffoldError as exc:
            terminal.write(f"[{exc}]")
        return False

    terminal.write(f"unknown command {command}. /help for the list.")
    return False


def _listen(config: Config, terminal: Terminal = HEADLESS) -> str | None:
    """Record one spoken message. Returns None if nothing usable was captured.

    Never raises past here. Voice is an input convenience — losing a spoken
    message should cost you the message, not drop you out of the conversation,
    so every failure prints a line and returns you to the prompt to type.
    """
    from majordomo import asr

    try:
        text = asr.listen(config.voice, on_start=lambda: terminal.write("listening… (speak now)"))
    except asr.MicrophoneUnavailable as exc:
        terminal.write(f"[no microphone: {exc}]")
        return None
    except asr.ASRError as exc:
        terminal.write(f"[could not hear you: {exc}]")
        return None

    terminal.write(f"you › {text}")
    return text


def _save_memory(description: str, config: Config, terminal: Terminal = HEADLESS, kind: str = "user") -> bool:
    """Ask, then write one memory. The single place chat writes to memory.

    Shared by the mid-conversation offer and by ``/remember <text>`` so the two
    cannot diverge — and so the confirmation is not something either of them can
    forget. ``propose`` never writes unattended for the same reason: a wrong
    memory replays into every future conversation that matches it.
    """
    from majordomo import memory as memory_mod

    if not config.memory.enabled:
        terminal.write("[memory is disabled in config]")
        return False

    terminal.write(f"\n  [{kind}] {description}")
    if not terminal.yes("  remember this? [y/N] "):
        return False

    try:
        written = memory_mod.write_memory(
            memory_mod.MemoryCandidate(description=description, type=kind)
        )
    except memory_mod.MemoryError_ as exc:
        terminal.write(f"  not saved: {exc}")
        return False

    terminal.write(f"  saved as {written.name}")
    return True


def _proposed_memory(reply: str) -> tuple[str, str | None]:
    """Split a reply into what to show and the fact it asked to keep.

    Pure, and separate from the asking on purpose. The marker has to come out of
    the text *before* anything is printed or stored — it was never meant to be
    read — but the question about it has to come *after*, or you are approving a
    memory before you have seen what produced it. One function doing both forced
    those two moments to be the same one.
    """
    from majordomo import prompts

    fact = prompts.wants_remembered(reply)
    if fact is None:
        return reply, None

    # A reply that was *only* the marker strips to nothing, and an empty
    # assistant turn is stored, replayed and resumed as a blank. `send` guards
    # the same case with "(empty response)".
    return prompts.strip_remember(reply) or "(noted)", fact


def _offer_memories(session: Session, config: Config, terminal: Terminal = HEADLESS) -> None:
    """Propose what to remember. Writes nothing without an explicit yes."""
    from majordomo import memory as memory_mod

    if not config.memory.enabled or not session.turns:
        return

    terminal.write("\nLooking for anything worth remembering…")
    candidates = propose(session.transcript(), config)
    if not candidates:
        terminal.write("Nothing worth keeping.")
        return

    for candidate in candidates:
        terminal.write(f"\n  [{candidate.type}] {candidate.description}")
        try:
            keep = terminal.yes("  remember this? [y/N] ")
        except KeyboardInterrupt:
            # Stops the review. Declining one candidate and moving on to
            # the next is what an interrupt used to do, and it is not what
            # an interrupt means.
            terminal.write("")
            return
        if not keep:
            continue
        try:
            written = memory_mod.write_memory(candidate)
            terminal.write(f"  saved as {written.name}")
        except memory_mod.MemoryError_ as exc:
            terminal.write(f"  not saved: {exc}")


def propose(transcript: str, config: Config):
    """Ask the model what is worth remembering. Never writes — returns candidates.

    The response format is deliberately dumb (``type | description`` per line)
    rather than JSON: a small model gets a one-field-per-line format right far
    more often than it produces valid JSON, and a parse failure here should cost
    a suggestion, not the whole exit path.
    """
    from majordomo import prompts
    from majordomo.llm import LLMError, MissingApiKey, complete
    from majordomo.memory import KNOWN_TYPES, MemoryCandidate

    try:
        reply = complete(
            prompts.build_memory_proposal_prompt(transcript),
            config.brain,
            config.brain.chat_model,
        )
    except (LLMError, MissingApiKey):
        return []

    text = (reply or "").strip()
    if not text or text.upper().startswith("NOTHING"):
        return []

    candidates = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*").strip()
        if "|" not in line:
            continue
        kind, _, description = line.partition("|")
        kind = kind.strip().lower()
        description = description.strip()
        if not description:
            continue
        candidates.append(
            MemoryCandidate(
                description=description,
                type=kind if kind in KNOWN_TYPES else "user",
            )
        )
    return candidates
