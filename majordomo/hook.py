"""The hook entrypoint, invoked by Claude Code as `mj hook --event=X`.

── THE RULE THAT MATTERS MOST ────────────────────────────────────────────────
This process ALWAYS exits 0, and never prints to stdout.

Claude Code reads exit code 2 from a `Stop` hook as "do not stop" and from
`PreToolUse` as "block this tool", and it parses hook stdout. A crash or a stray
print here would not degrade Majordomo — it would break the user's editing
session. A tool that does that is uninstalled within the hour.

So: every failure is caught, appended to ``~/.majordomo/error.log``, and
swallowed.
─────────────────────────────────────────────────────────────────────────────

Two things the design spec left open are resolved here.

**Which Notification fired.** The `Notification` payload carries only `message`
— it does not say which matcher matched. So the matcher is encoded in our own
argv (`--event=Notification --matcher=idle_prompt`) at install time, rather than
guessed from the message text, which is prose and will change.

**Which surface the session is on.** No hook field reports it, but Claude Code
exports ``CLAUDE_CODE_ENTRYPOINT`` and hook subprocesses inherit it
(`claude-vscode` in the extension, `cli` in the terminal). That variable is
undocumented, so an unrecognised value degrades to `terminal` and the raw value
is recorded on the event — a wrong guess should be diagnosable, not invisible.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from majordomo import state
from majordomo.models import SessionEvent, SessionStatus, Surface
from majordomo.paths import error_log_path

#: A topic is a label, not a document. A pasted stack trace as the "topic" of a
#: session would blow up every downstream prompt that lists sessions.
MAX_TOPIC_CHARS = 200

#: Entrypoint values we recognise. Anything else → terminal.
_VSCODE_ENTRYPOINTS = frozenset({"claude-vscode", "vscode"})

#: Only these two Notification matchers mean anything to us. `auth_success`,
#: `elicitation_*` and friends are noise.
_NOTIFICATION_STATUS: dict[str, SessionStatus] = {
    "idle_prompt": "idle_awaiting_you",
    "permission_prompt": "blocked",
}


def log_error(context: str, error: object) -> None:
    """Record a swallowed failure. Never raises — this is the last line of defence."""
    try:
        path = error_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat()
        detail = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ) if isinstance(error, BaseException) else str(error)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{stamp} {context}: {detail.strip()}\n")
    except Exception:  # pragma: no cover - nothing left to do about it
        pass


def read_stdin() -> str:
    """Read the hook payload. Returns '' when there is no payload to read."""
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return ""
        return sys.stdin.read()
    except Exception:
        return ""


def detect_surface(env: dict[str, str] | None = None) -> tuple[Surface, str | None]:
    """Return ``(surface, raw_entrypoint)`` from the inherited environment."""
    source = env if env is not None else os.environ
    raw = source.get("CLAUDE_CODE_ENTRYPOINT")
    if raw and raw.lower() in _VSCODE_ENTRYPOINTS:
        return "vscode", raw
    return "terminal", raw


def status_for(event: str, matcher: str | None) -> SessionStatus | None:
    """Map a hook event (plus its matcher) to a session status, or None to ignore."""
    if event == "SessionStart":
        return "active"
    if event == "UserPromptSubmit":
        # A prompt was submitted, so the session is working again — this both
        # clears a previous idle/blocked state and carries the new topic.
        return "active"
    if event == "Stop":
        # A turn ended, which proves two things at once: the session is alive,
        # and it is not blocked. A permission prompt happens *mid*-turn, so the
        # turn cannot end while one is outstanding — a Stop after a block is how
        # we learn you answered it, since approving emits no event of its own.
        #
        # This is deliberately a plain status like any other. An earlier version
        # treated Stop as status-preserving, which left an approved session
        # `blocked` forever while refreshing `at` on every turn kept the
        # staleness sweep from ever retiring it. See `workers.sessions.fold`.
        return "active"
    if event == "SessionEnd":
        return "ended"
    if event == "Notification":
        return _NOTIFICATION_STATUS.get(matcher or "")
    return None


def event_from_payload(
    payload: dict,
    event: str,
    matcher: str | None,
    now: datetime | None = None,
    env: dict[str, str] | None = None,
) -> SessionEvent | None:
    """Translate one hook payload into a session event, or None to write nothing."""
    status = status_for(event, matcher)
    if status is None:
        return None

    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        # Without an id the record could never be folded or resumed.
        return None

    topic = None
    prompt = payload.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        topic = prompt.strip()[:MAX_TOPIC_CHARS]

    surface, entrypoint = detect_surface(env)
    cwd = payload.get("cwd")
    stamp = (now or datetime.now(timezone.utc)).isoformat()

    return SessionEvent(
        session_id=session_id,
        surface=surface,
        cwd=cwd if isinstance(cwd, str) else "",
        status=status,
        at=stamp,
        topic=topic,
        entrypoint=entrypoint,
        kind=event,
    )


def run_hook(
    event: str,
    matcher: str | None = None,
    state_file: Path | str | None = None,
) -> int:
    """Handle one hook invocation. Returns 0. Always."""
    try:
        raw = read_stdin()
        if not raw.strip():
            return 0

        # Strip a UTF-8 BOM. Claude Code does not send one, but anything that
        # pipes a payload in on Windows might, and losing the whole payload to
        # an invisible byte is a miserable way to fail.
        text = raw[1:] if raw.startswith("﻿") else raw

        try:
            payload = json.loads(text)
        except ValueError as exc:
            log_error("hook:parse-stdin", exc)
            return 0
        if not isinstance(payload, dict):
            return 0

        session_event = event_from_payload(payload, event, matcher)
        if session_event is None:
            return 0

        state.append_event(session_event, state_file)
    except Exception as exc:
        log_error(f"hook:{event}", exc)

    return 0
