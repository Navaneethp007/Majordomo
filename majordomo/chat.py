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
to the prompt with everything intact. ``MissingApiKey`` still exits, because
retrying cannot fix it.
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

HELP = """\
  /context    what memory and activity are loaded
  /remember   propose what is worth remembering from this conversation
  /build NAME scaffold a repo from this conversation and open Claude Code
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
        """The conversation as plain text, for memory proposals and scaffolding."""
        return "\n\n".join(
            f"{'Me' if t.role == 'user' else 'Majordomo'}: {t.content}"
            for t in self.turns
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


def compact(session: Session, config: Config) -> bool:
    """Fold the older turns into one summary, keeping recent ones verbatim.

    Deliberately the same shape as ``coordinator.reduce_source``: hand the
    oversized part to the reducer model, keep a stated note that it happened.
    Returns whether anything was folded.

    A failed summarisation is not fatal — the conversation continues uncompacted
    and simply costs more. Dropping turns because a summary call failed would
    silently lose the thing the user came for.
    """
    from majordomo.llm import LLMError, complete

    if len(session.turns) <= KEEP_RECENT_TURNS + 2:
        return False

    old = session.turns[:-KEEP_RECENT_TURNS]
    recent = session.turns[-KEEP_RECENT_TURNS:]

    body = "\n\n".join(
        f"{'Me' if t.role == 'user' else 'You'}: {t.content}" for t in old
    )

    try:
        summary = complete(
            [
                {
                    "role": "user",
                    "content": (
                        "Summarise the earlier part of this conversation so it "
                        "can be carried forward. Keep decisions made, "
                        "constraints agreed, and anything still open. Drop "
                        "pleasantries and abandoned tangents.\n\n" + body
                    ),
                }
            ],
            config.brain,
            config.brain.reducer_model,
        )
    except LLMError:
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
    from majordomo.llm import LLMError, complete

    session.turns.append(Turn(role="user", content=text, at=_now()))

    if needs_compaction(session, config):
        compact(session, config)

    try:
        reply = complete(session.messages(), config.brain, config.brain.chat_model)
    except LLMError as exc:
        session.turns.pop()
        raise ChatFailed(str(exc)) from exc

    cleaned = (reply or "").strip() or "(empty response)"
    session.turns.append(Turn(role="assistant", content=cleaned, at=_now()))
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
            for turn in session.turns:
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


def latest_transcript() -> Path | None:
    """The most recently saved chat, if there is one."""
    directory = chats_dir()
    if not directory.is_dir():
        return None
    try:
        saved = sorted(directory.glob("*.jsonl"))
    except OSError:
        return None
    return saved[-1] if saved else None


def new_session(config: Config, resume: bool = False) -> Session:
    """Build a session: frozen prefix, optionally with prior turns restored."""
    from majordomo import context as context_mod
    from majordomo import prompts

    ctx = context_mod.build(config)
    system = prompts.build_chat_system_prompt(ctx.render())

    turns: list[Turn] = []
    path: Path | None = None

    if resume:
        previous = latest_transcript()
        if previous is not None:
            turns = load_turns(previous)
            path = previous

    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = chats_dir() / f"{stamp}.jsonl"

    return Session(
        system=system, turns=turns, path=path, context_summary=ctx.summary()
    )


# ---------------------------------------------------------------------------
# The REPL
# ---------------------------------------------------------------------------


def run(config: Config, resume: bool = False) -> None:
    """The interactive loop. Returns when the user leaves."""
    from majordomo.cli import safe_print

    session = new_session(config, resume=resume)

    safe_print("Majordomo. /help for commands, /exit to leave.")
    safe_print(f"({session.context_summary})")
    if session.turns:
        safe_print(f"Resumed {len(session.turns)} turns from {session.path.name}.")
    safe_print("")

    while True:
        try:
            line = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            safe_print("")
            break

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

        safe_print(f"\nmj  › {reply}\n")
        save(session)

    save(session)
    if session.turns:
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
        return False

    if command == "/remember":
        _offer_memories(session, config, write)
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
