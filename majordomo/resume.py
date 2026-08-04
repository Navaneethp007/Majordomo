"""Jumping back into the exact session you left.

The method depends on the surface, which is why the hook records it:

- **VS Code** — the ``vscode://`` URI the extension registers. Verified against
  the installed bundle: its ``registerUriHandler`` matches path ``/open``, reads
  ``session`` and ``prompt`` from the query, and dispatches
  ``claude-vscode.primaryEditor.open``. It focuses the tab if already open.
- **Terminal** — ``claude --resume <id>``, run **from the session's own cwd**.
  Session-id lookup is scoped to the project directory, so running it from
  anywhere else simply won't find the session.

Constructing the command and running it are deliberately separate functions, so
tests can assert the exact URI and argv without launching anything.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from urllib.parse import quote

from majordomo.models import Session

#: The extension id from its package.json (publisher `Anthropic`, name
#: `claude-code`). VS Code matches the URI authority case-insensitively.
VSCODE_EXTENSION_ID = "Anthropic.claude-code"


class ResumeError(Exception):
    """Resume could not be constructed or launched."""


@dataclass(frozen=True)
class ResumeCommand:
    """What to run, and where. ``uri`` and ``argv`` are mutually exclusive."""

    surface: str
    uri: str | None = None
    argv: list[str] | None = None
    cwd: str | None = None


def vscode_uri(session_id: str) -> str:
    return f"vscode://{VSCODE_EXTENSION_ID}/open?session={quote(session_id, safe='')}"


def build(session: Session) -> ResumeCommand:
    """Build the resume command for a session. Launches nothing."""
    if not session.session_id:
        raise ResumeError("session has no id")

    if session.surface == "vscode":
        return ResumeCommand(surface="vscode", uri=vscode_uri(session.session_id))

    if not session.cwd:
        # Without the recorded cwd, `claude --resume` would search the wrong
        # project and report the session doesn't exist — a confusing failure.
        raise ResumeError(
            f"session {session.session_id} has no recorded cwd; "
            "terminal resume needs the project directory"
        )

    return ResumeCommand(
        surface="terminal",
        argv=["claude", "--resume", session.session_id],
        cwd=session.cwd,
    )


def launch(command: ResumeCommand) -> None:
    """Actually run it. Separate from ``build`` so tests never launch anything."""
    if command.uri:
        if sys.platform == "win32":
            # `start` needs an empty title argument first, else it eats the URI.
            subprocess.run(["cmd", "/c", "start", "", command.uri], check=True)
        elif sys.platform == "darwin":  # pragma: no cover
            subprocess.run(["open", command.uri], check=True)
        else:  # pragma: no cover
            subprocess.run(["xdg-open", command.uri], check=True)
        return

    if not command.argv:  # pragma: no cover - build never produces this
        raise ResumeError("nothing to launch")

    if sys.platform == "win32":
        # A new console, so resuming doesn't hijack the terminal running `mj`.
        subprocess.Popen(
            ["cmd", "/c", "start", "", *command.argv],
            cwd=command.cwd,
        )
    else:  # pragma: no cover
        subprocess.Popen(command.argv, cwd=command.cwd)
