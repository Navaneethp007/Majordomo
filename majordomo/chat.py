"""The interactive session — where brainstorming happens.

Structurally this is one idea: a frozen prefix plus a growing list of turns.

    [ system + memory + activity ]   built once at startup, never rebuilt
    [ user | assistant | user | ... ] appended to on every exchange

The freeze is the load-bearing part. The API is stateless, so every turn
re-sends everything before it — a twenty-turn conversation sends the prefix
twenty times. Byte-identical, that prefix bills at roughly a tenth; rebuilt each
turn, or seasoned with a timestamp, it bills in full and nothing tells you.
``context.py`` explains the ordering; this module's job is to not undo it.

── ON DEGRADATION ───────────────────────────────────────────────────────────
A failed request must not cost you the conversation. Losing forty minutes of
brainstorming to one 503 would be the worst failure this feature has, and it is
entirely avoidable: the transcript lives here, not on the server. So an
``LLMError`` prints a warning, drops the unanswered user turn, and returns you
to the prompt with everything intact.

``MissingApiKey`` is handled the same way here, unlike everywhere else. It is
unrecoverable and the other commands exit on it — but exiting *this* command
discards a conversation to fix an environment variable, and the exit-path save
never runs. The message still names the variable, so it stays as fixable as it
was; you just do not lose the session finding that out.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from majordomo.config import Config
from majordomo.paths import chats_dir

#: Turns kept verbatim when compacting. Enough that the immediate thread of the
#: conversation survives intact — losing the last few exchanges to a summary is
#: exactly when a model starts contradicting itself.
KEEP_RECENT_TURNS = 8

#: Never folded away, however large they are: the exchange you are in the middle
#: of. Compaction that eats the current question produces a model answering
#: something nobody asked.
MIN_RECENT_TURNS = 2

#: Prefix identifying a turn as folded notes rather than something anyone said.
#: `compact` looks for it so a summary is carried forward instead of being
#: summarised again.
SUMMARY_MARKER = "[Earlier in this conversation, summarised]"

HELP = """\
  /context    what memory and activity are loaded
  /clear      start a fresh conversation, keeping the same loaded context
  /remember   propose what is worth remembering from this conversation
  /build NAME scaffold a repo from this conversation and open Claude Code
  /agent TASK put the agent to work here — it asks before writing or running
  /voice      speak your next message instead of typing it
  /help       this
  /exit       leave (also Ctrl+D)"""


@dataclass(frozen=True)
class Turn:
    role: str  # "user" | "assistant"
    content: str
    at: str = ""

    def to_json(self) -> dict:
        return {"role": self.role, "content": self.content, "at": self.at}


@dataclass
class Session:
    """One conversation. The system prompt is frozen at construction."""

    system: str
    turns: list[Turn] = field(default_factory=list)
    path: Path | None = None
    #: Set when compaction has folded earlier turns away, so the UI can say so.
    compactions: int = 0
    #: True when compaction was needed on the last turn and did not happen. The
    #: conversation still works; it is growing without a ceiling.
    compaction_failed: bool = False
    #: Everything ever said, in order — the durable record, never compacted.
    #: `turns` is the *working context* and shrinks when compaction folds it;
    #: this does not. Writing `turns` to disk meant a compaction overwrote the
    #: transcript with its own summary, destroying the original text. Losing it
    #: from context is the feature; losing it from disk was data loss.
    log: list[Turn] = field(default_factory=list)
    #: One line describing what context was loaded, for ``/context``. Kept here
    #: so reporting it never rebuilds the prefix — rebuilding is what breaks the
    #: byte-identical guarantee this whole module is arranged around.
    context_summary: str = ""

    def messages(self) -> list[dict]:
        """The full request body: frozen prefix, then every turn in order."""
        return [{"role": "system", "content": self.system}] + [
            {"role": t.role, "content": t.content} for t in self.turns
        ]

    def transcript(self) -> str:
        """The conversation as plain text, for memory proposals and scaffolding.

        Reads ``log``, not ``turns``. Compaction shrinks the working context on
        purpose, but this feeds `/remember` and `/build` — the two moments a
        session decides what to keep permanently. Handing those a summary of
        what was said, rather than what was said, is the wrong input at exactly
        the wrong time.
        """
        source = self.log or self.turns
        return "\n\n".join(
            f"{'Me' if t.role == 'user' else 'Majordomo'}: {t.content}"
            for t in source
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Size and compaction
# ---------------------------------------------------------------------------


def estimate_tokens(session: Session) -> int:
    """Rough size of the next request. Four chars per token, as in the router."""
    total = len(session.system)
    for turn in session.turns:
        total += len(turn.content)
    return total // 4


def needs_compaction(session: Session, config: Config) -> bool:
    return estimate_tokens(session) > config.brain.chat_compact_threshold_tokens


def _recent_to_keep(session: Session, config: Config) -> int:
    """How many recent turns can stay verbatim and still leave room to shrink.

    Walks back from the newest turn, taking turns while they fit in half the
    threshold. Half rather than all of it because compacting down to exactly the
    limit means compacting again on the very next message.
    """
    budget = max(config.brain.chat_compact_threshold_tokens // 2, 1)
    kept = 0
    used = 0
    for turn in reversed(session.turns):
        used += len(turn.content) // 4
        if used > budget and kept >= MIN_RECENT_TURNS:
            break
        kept += 1
        if kept >= KEEP_RECENT_TURNS:
            break
    return max(MIN_RECENT_TURNS, kept)


def compact(session: Session, config: Config) -> bool:
    """Fold the older turns into one summary, keeping recent ones verbatim.

    Deliberately the same shape as ``coordinator.reduce_source``: hand the
    oversized part to the reducer model, keep a stated note that it happened.
    Returns whether anything was folded.

    A failed summarisation is not fatal — the conversation continues uncompacted
    and simply costs more. Dropping turns because a summary call failed would
    silently lose the thing the user came for.

    How many turns stay verbatim adapts to how big they are. It used to be a
    flat eight, with a guard refusing to compact below ten turns — but the
    *trigger* is token count, so a conversation of four pasted files crossed the
    threshold, hit the guard, and could never compact again. It just grew until
    the provider rejected it. Anything that can grow without bound needs its
    limit expressed in the same units as its trigger.
    """
    from majordomo.llm import LLMError, MissingApiKey, complete

    keep = _recent_to_keep(session, config)
    if len(session.turns) <= keep:
        return False

    old = session.turns[:-keep]
    recent = session.turns[-keep:]

    # A previous summary is carried separately, never re-summarised as if it
    # were conversation. Folding it back into the body means every compaction
    # re-compresses the last one: a telephone game, where a 450-character
    # distillation competes with 6,000-character answers and loses a little each
    # round. Measured — a fact stated in turn one survived the first compaction
    # and was gone by the third.
    carried = ""
    if old and old[0].content.startswith(SUMMARY_MARKER):
        carried = old[0].content[len(SUMMARY_MARKER) :].strip()
        old = old[1:]
        if not old:
            return False          # nothing new to fold; leave the notes alone

    body = "\n\n".join(
        f"{'Me' if t.role == 'user' else 'You'}: {t.content}" for t in old
    )

    instructions = (
        "You are maintaining running notes on a conversation. The notes "
        "replace the turns they cover, so anything you leave out is gone.\n\n"
        "Keep:\n"
        "- Facts the user stated about themselves, their work, their "
        "preferences or their situation — names, numbers, dates, tools, "
        "anything they said to remember.\n"
        "- Decisions made and constraints agreed.\n"
        "- Questions still open.\n\n"
        "Drop pleasantries, and explanations the user asked for and received "
        "— those were answers, not context.\n\n"
        "Write notes, not prose.\n\n"
    )
    if carried:
        instructions += (
            "These are the existing notes. Reproduce every fact in them, "
            "changing one only if the new exchanges below correct it:\n\n"
            f"{carried}\n\n"
            "New exchanges to fold in:\n\n"
        )
    else:
        instructions += "Conversation to summarise:\n\n"

    try:
        summary = complete(
            [{"role": "user", "content": instructions + body}],
            config.brain,
            config.brain.reducer_model,
        )
    except (LLMError, MissingApiKey):
        # A summary we cannot get is not fatal — the conversation continues
        # uncompacted and simply costs more. Raising here would lose it.
        return False

    cleaned = (summary or "").strip()
    if not cleaned:
        return False

    session.turns = [
        Turn(
            role="user",
            content=(
                f"[Earlier in this conversation, summarised]\n\n{cleaned}"
            ),
            at=_now(),
        ),
        *recent,
    ]
    session.compactions += 1
    return True


# ---------------------------------------------------------------------------
# Talking
# ---------------------------------------------------------------------------


class ChatFailed(Exception):
    """The model call failed. The transcript is intact; try again."""


def send(session: Session, text: str, config: Config) -> str:
    """Add a user turn, get a reply, add it. Raises ChatFailed on a bad call.

    On failure the user turn is removed again, so a retry does not stack two
    copies of the same message into the history.
    """
    from majordomo.llm import LLMError, MissingApiKey, complete

    asked_at = _now()
    session.turns.append(Turn(role="user", content=text, at=asked_at))

    if needs_compaction(session, config):
        # Recorded rather than discarded. Compaction failing is the one problem
        # here that gets worse the longer it goes unnoticed: every turn after it
        # is larger than the last, and the visible symptom arrives only when the
        # provider rejects the request outright. `/context` reports this.
        session.compaction_failed = not compact(session, config)

    try:
        reply = complete(session.messages(), config.brain, config.brain.chat_model)
    except (LLMError, MissingApiKey) as exc:
        # MissingApiKey is a bare Exception, not an LLMError. `cmd_do` and
        # `/agent` were both given guards; this — every ordinary message you
        # type — was not, so the first one with no key killed the REPL and the
        # exit-path save() never ran. Treated as ChatFailed because the outcome
        # is the same from here: the turn is rolled back and you keep the
        # conversation. The message names the variable, so it is still fixable.
        session.turns.pop()
        raise ChatFailed(str(exc)) from exc

    cleaned = (reply or "").strip() or "(empty response)"
    answer = Turn(role="assistant", content=cleaned, at=_now())
    session.turns.append(answer)
    # Appended only now, so a failed call leaves nothing half-written.
    session.log.append(Turn(role="user", content=text, at=asked_at))
    session.log.append(answer)
    return cleaned


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save(session: Session) -> None:
    """Write the transcript. Never raises — a chat is worth more than a log."""
    if session.path is None:
        return
    try:
        session.path.parent.mkdir(parents=True, exist_ok=True)
        with open(session.path, "w", encoding="utf-8") as fh:
            for turn in session.log:
                fh.write(json.dumps(turn.to_json(), ensure_ascii=False) + "\n")
    except OSError:
        pass


def load_turns(path: Path | str) -> list[Turn]:
    """Read a saved transcript. Never raises; bad lines are skipped."""
    target = Path(path)
    if not target.is_file():
        return []
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError:
        return []
    if raw.startswith("﻿"):
        raw = raw[1:]

    turns: list[Turn] = []
    for line in raw.split("\n"):
        trimmed = line.strip()
        if not trimmed:
            continue
        try:
            decoded = json.loads(trimmed)
        except ValueError:
            continue
        role = decoded.get("role")
        content = decoded.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            turns.append(
                Turn(role=role, content=content, at=str(decoded.get("at") or ""))
            )
    return turns


def saved_sessions() -> list[Path]:
    """Every saved transcript, oldest first. Never raises."""
    directory = chats_dir()
    if not directory.is_dir():
        return []
    try:
        return sorted(directory.glob("*.jsonl"))
    except OSError:
        return []


def latest_transcript() -> Path | None:
    """The most recent chat that has anything in it.

    Empty ones are skipped rather than returned. ``run`` no longer writes them,
    but any already on disk would still sort newest and shadow real work — and
    "resume" restoring nothing is a worse answer than reaching one file further
    back. Nothing is deleted here; an empty file is ignored, not cleaned up.
    """
    for path in reversed(saved_sessions()):
        if load_turns(path):
            return path
    return None


def find_session(session_id: str) -> Path | None:
    """A saved chat by id or unique prefix, the way ``mj resume`` matches.

    Returns None for both "no match" and "several matches" — the caller reports
    which, since the two need different advice.
    """
    matches = [p for p in saved_sessions() if p.stem.startswith(session_id)]
    return matches[0] if len(matches) == 1 else None


def match_count(session_id: str) -> int:
    return sum(1 for p in saved_sessions() if p.stem.startswith(session_id))


def describe_session(path: Path) -> str:
    """One line for ``mj chat --list``: when, how long, what it opened with."""
    turns = load_turns(path)
    opener = next((t.content for t in turns if t.role == "user"), "")
    opener = " ".join(opener.split())[:60] or "(empty)"
    return f"{path.stem}  {len(turns):>3} turns  {opener}"


def _new_transcript_path() -> Path:
    return chats_dir() / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.jsonl"


def new_session(
    config: Config,
    resume: bool = False,
    transcript: Path | None = None,
) -> Session:
    """Build a session: frozen prefix, optionally with prior turns restored.

    ``transcript`` names a specific saved chat to continue; ``resume`` without
    one continues the most recent.
    """
    from majordomo import context as context_mod
    from majordomo import prompts

    ctx = context_mod.build(config)
    system = prompts.build_chat_system_prompt(ctx.render())

    turns: list[Turn] = []
    path: Path | None = transcript

    if path is None and resume:
        path = latest_transcript()

    if path is not None:
        turns = load_turns(path)
    else:
        path = _new_transcript_path()

    session = Session(
        system=system,
        turns=turns,
        log=list(turns),
        path=path,
        context_summary=ctx.summary(),
    )

    # A resumed conversation arrives at full length, because the file is the
    # record and the record is never compacted. Fold it back down now rather
    # than on the first message: otherwise every resume re-sends the whole
    # history once, at full price, before deciding it was too long.
    if turns and needs_compaction(session, config):
        session.compaction_failed = not compact(session, config)

    return session


def clear(session: Session) -> None:
    """Start a fresh conversation without leaving, or reloading context.

    The frozen system prefix is deliberately *kept*. Rebuilding it would reread
    memory and activity and produce different bytes, which breaks the
    byte-identical guarantee ``context.py`` is arranged around — the whole
    reason the prefix is assembled once. Clearing is about the turns.
    """
    session.turns = []
    session.log = []
    session.compactions = 0
    session.compaction_failed = False
    session.path = _new_transcript_path()


# ---------------------------------------------------------------------------
# The REPL
# ---------------------------------------------------------------------------


def run(
    config: Config,
    resume: bool = False,
    transcript: Path | None = None,
) -> None:
    """The interactive loop. Returns when the user leaves."""
    from majordomo import keys, render as render_mod
    from majordomo.cli import safe_print

    session = new_session(config, resume=resume, transcript=transcript)

    safe_print("Majordomo. /help for commands, /exit to leave.")
    safe_print(f"({session.context_summary})")
    if session.turns:
        safe_print(f"Resumed {len(session.turns)} turns from {session.path.name}.")
    safe_print("")

    trigger = config.voice.listen_key if config.voice.enabled else ""

    while True:
        try:
            typed = keys.read_line("you › ", trigger)
        except (EOFError, KeyboardInterrupt):
            safe_print("")
            break

        # Two ways in, one path out: the key at the prompt, or /voice typed.
        # Both land here rather than in _handle_command, which has no business
        # knowing about the send loop.
        if typed is keys.TRIGGERED or (
            isinstance(typed, str) and typed.strip().lower() == "/voice"
        ):
            spoken = _listen(config, safe_print)
            if spoken is None:
                continue
            line = spoken
        else:
            line = typed.strip()

        if not line:
            continue

        if line.startswith("/"):
            if _handle_command(line, session, config, safe_print):
                break
            continue

        try:
            reply = send(session, line, config)
        except ChatFailed as exc:
            safe_print(f"\n[the model call failed: {exc}]")
            safe_print("[your message was not sent — try again]\n")
            continue

        # Rendered, not raw: models write markdown regardless of instruction,
        # and literal `**bold**` reads worse than the prose would have.
        safe_print(f"\nmj  › {render_mod.render(reply)}\n")
        save(session)

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
        safe_print(f"Saved to {session.path}.")
        _offer_memories(session, config, safe_print)


def _handle_command(line: str, session: Session, config: Config, write) -> bool:
    """Run a /command. Returns True when the loop should end."""
    parts = line.split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""

    if command in ("/exit", "/quit"):
        return True

    if command == "/help":
        write(HELP)
        return False

    if command == "/context":
        write(session.context_summary)
        write(
            f"{len(session.turns)} turns, ~{estimate_tokens(session)} tokens"
            + (f", {session.compactions} compaction(s)" if session.compactions else "")
        )
        if session.compaction_failed:
            write(
                "  warning: this is over the compaction threshold and could not "
                "be summarised — it will keep growing. /clear starts fresh."
            )
        return False

    if command == "/clear":
        if session.turns:
            save(session)
            write(f"Saved {len(session.turns)} turns to {session.path.name}.")
        clear(session)
        write("Cleared. Same context loaded, fresh conversation.")
        return False

    if command == "/remember":
        _offer_memories(session, config, write)
        return False


    if command == "/agent":
        from majordomo import agent
        from majordomo.cli import confirm_action

        if not argument:
            write("usage: /agent <what you want done>")
            return False

        from majordomo.llm import MissingApiKey

        try:
            outcome = agent.run(argument, config, confirm=confirm_action, write=write)
        except MissingApiKey as exc:
            # `agent.run` swallows LLMError so a half-finished task keeps its
            # trail, but MissingApiKey is a bare Exception and went straight
            # through — taking the REPL and the conversation with it. `cmd_do`
            # guards the identical call; the two paths disagreed.
            write(f"[{exc}]")
            return False

        # Fold the result back into the conversation. Without this the agent's
        # work is invisible to the next turn, and you would have to re-explain
        # what just happened to the thing that was watching it happen.
        asked = Turn(
            role="user", content=f"[I asked the agent to: {argument}]", at=_now()
        )
        answered = Turn(
            role="assistant",
            content=outcome.answer
            or f"(the agent stopped: {outcome.stopped_because})",
            at=_now(),
        )
        # Both lists: `turns` is the working context, `log` is what gets saved.
        # Appending to `turns` alone silently dropped these from the transcript.
        session.turns.extend((asked, answered))
        session.log.extend((asked, answered))
        save(session)

        if outcome.answer:
            write(f"\n{outcome.answer}\n")
        if outcome.stopped_because:
            write(f"[stopped: {outcome.stopped_because}]")
        return False

    if command == "/build":
        from majordomo import scaffold

        if not argument:
            write("usage: /build <name>")
            return False
        try:
            scaffold.from_chat(argument, session.transcript(), config, write=write)
        except scaffold.ScaffoldError as exc:
            write(f"[{exc}]")
        return False

    write(f"unknown command {command}. /help for the list.")
    return False


def _listen(config: Config, write) -> str | None:
    """Record one spoken message. Returns None if nothing usable was captured.

    Never raises past here. Voice is an input convenience — losing a spoken
    message should cost you the message, not drop you out of the conversation,
    so every failure prints a line and returns you to the prompt to type.
    """
    from majordomo import asr

    try:
        text = asr.listen(config.voice, on_start=lambda: write("listening… (speak now)"))
    except asr.MicrophoneUnavailable as exc:
        write(f"[no microphone: {exc}]")
        return None
    except asr.ASRError as exc:
        write(f"[could not hear you: {exc}]")
        return None

    write(f"you › {text}")
    return text


def _offer_memories(session: Session, config: Config, write) -> None:
    """Propose what to remember. Writes nothing without an explicit yes."""
    from majordomo import memory as memory_mod

    if not config.memory.enabled or not session.turns:
        return

    write("\nLooking for anything worth remembering…")
    candidates = propose(session.transcript(), config)
    if not candidates:
        write("Nothing worth keeping.")
        return

    for candidate in candidates:
        write(f"\n  [{candidate.type}] {candidate.description}")
        try:
            answer = input("  remember this? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            write("")
            return
        if answer not in ("y", "yes"):
            continue
        try:
            written = memory_mod.write_memory(candidate)
            write(f"  saved as {written.name}")
        except memory_mod.MemoryError_ as exc:
            write(f"  not saved: {exc}")


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
