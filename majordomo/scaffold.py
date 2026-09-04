"""Turning an idea into a repository with Claude Code already working in it.

This is the payoff for everything else in phase 1. The context store and the
memory exist so that when you say "let's build this", the session that opens
already knows who you are and what you have been doing — rather than starting
from nothing the way a bare ``claude`` in an empty folder does.

── WHY THE BRIEF IS A FILE, NOT AN ARGUMENT ─────────────────────────────────
The obvious implementation is ``claude "<the whole brief>"``. It breaks: Windows
caps a command line around 8191 characters and a brainstorm transcript passes
that easily, failing in a way that looks like Claude Code ignoring you rather
than an argument being truncated.

So the brief is written into the repo as ``BRIEF.md`` and Claude Code is asked
to read it. No length limit, and the brief stays in the project as a record of
what it was meant to be — which is worth having on its own.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from majordomo.config import Config

#: What Claude Code is told to do first. Short by necessity (see the docstring)
#: — everything substantive lives in the file it points at.
OPENING_PROMPT = (
    "Read BRIEF.md in this directory — it is the brief for this project, "
    "written just now. Start by telling me whether the plan makes sense and "
    "what you would change, then we will build it."
)


class ScaffoldError(Exception):
    """The project could not be created.

    Raised before anything is written when the target is unusable, and from
    ``create`` if the filesystem refuses partway — in which case the message
    names the path, because a half-written directory is worth knowing about.
    A failed ``git init`` is *not* this: see ``create``.
    """


@dataclass(frozen=True)
class Plan:
    """What ``create`` would do. Printed verbatim by ``--dry-run``."""

    name: str
    path: Path
    brief: str

    def describe(self) -> str:
        return "\n".join(
            [
                f"  mkdir      {self.path}",
                f"  git init   {self.path}",
                f"  write      {self.path / 'README.md'}",
                f"  write      {self.path / 'BRIEF.md'}  ({len(self.brief)} chars)",
                f"  launch     claude {OPENING_PROMPT[:40]}…  (in {self.path})",
            ]
        )


def slug(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return cleaned[:50].strip("-") or "project"


def build_brief(idea: str, transcript: str, context_text: str) -> str:
    """The BRIEF.md handed to Claude Code.

    Ordered deliberately: what we are building, then how we got here, then who
    is asking. A reader who stops after the first section still has the task.
    """
    parts = [f"# {idea.strip()}\n"]

    if transcript.strip():
        parts.append(
            "## How we got here\n\n"
            "This came out of a conversation with Majordomo. The transcript is "
            "below — it holds the reasoning, the constraints agreed, and "
            "anything still open.\n\n"
            f"{transcript.strip()}\n"
        )

    if context_text.strip():
        parts.append(
            "## Context on who you are working with\n\n"
            f"{context_text.strip()}\n"
        )

    return "\n".join(parts)


def plan(name: str, brief: str, config: Config) -> Plan:
    """Resolve where this would go. Raises rather than overwrite anything."""
    root = Path(config.scaffold.root or Path.home() / "projects")
    path = root / slug(name)

    # A file at the target is checked separately: `path.exists()` is true for
    # one, and `iterdir()` then raises NotADirectoryError — a traceback where
    # the user should get a sentence.
    if path.is_file():
        raise ScaffoldError(f"{path} already exists as a file")
    if path.is_dir() and any(path.iterdir()):
        raise ScaffoldError(f"{path} already exists and is not empty")

    return Plan(name=name, path=path, brief=brief)


def create(target: Plan) -> tuple[Path, list[str]]:
    """Make the directory, write the files, init git.

    Returns ``(path, warnings)``. A failed ``git init`` is a **warning**, not an
    error, and the distinction is load-bearing: by the time git runs, the
    directory and the brief already exist, so raising would abandon a usable
    project *and* leave the path non-empty — which ``plan`` then rejects,
    making every retry fail too. Warning and carrying on leaves the user with a
    working project and one command to run themselves.
    """
    try:
        target.path.mkdir(parents=True, exist_ok=True)
        (target.path / "README.md").write_text(
            f"# {target.name}\n\nSee BRIEF.md.\n", encoding="utf-8"
        )
        (target.path / "BRIEF.md").write_text(target.brief, encoding="utf-8")
    except OSError as exc:
        raise ScaffoldError(f"could not write to {target.path}: {exc}") from exc

    warnings: list[str] = []
    try:
        result = subprocess.run(
            ["git", "init"],
            cwd=target.path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            warnings.append(
                "`git init` failed "
                f"({result.stderr.strip() or result.stdout.strip()}) — "
                f"run it yourself in {target.path}"
            )
    except (OSError, subprocess.SubprocessError) as exc:
        warnings.append(
            f"could not run git ({exc}) — run `git init` yourself in {target.path}"
        )

    return target.path, warnings


def launch(path: Path) -> None:
    """Open Claude Code in the new project. Reuses the resume spawn helper."""
    from majordomo.resume import spawn_detached

    spawn_detached(["claude", OPENING_PROMPT], cwd=str(path))


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def start(
    idea: str,
    config: Config,
    transcript: str = "",
    dry_run: bool = False,
    write=print,
) -> Path | None:
    """Scaffold from an idea (and optionally a chat transcript), then hand off.

    Raises:
        ScaffoldError: the target exists, or the filesystem refused.
    """
    from majordomo import context as context_mod

    ctx = context_mod.build(config, query=idea)
    brief = build_brief(idea, transcript, ctx.render())
    target = plan(idea, brief, config)

    if dry_run:
        write(f"Would create {target.path}:")
        write(target.describe())
        return None

    path, warnings = create(target)
    write(f"Created {path}")
    for warning in warnings:
        write(f"warning: {warning}")
    write("Opening Claude Code…")
    launch(path)
    return path


def from_chat(name: str, transcript: str, config: Config, write=print) -> Path | None:
    """The ``/build`` path: same as ``start``, with the conversation attached."""
    return start(name, config, transcript=transcript, write=write)
