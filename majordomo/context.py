"""What Majordomo knows, assembled into the block that leads every prompt.

This module does the I/O — reads memory, reads the activity cache — and hands
back plain text. ``prompts.py`` stays pure and takes that text as an argument.

── WHY THE ORDER IS THE WHOLE POINT ─────────────────────────────────────────
Every request to a stateless model re-sends everything before it. A twenty-turn
chat with a 6k context block sends that block twenty times. Prompt caching makes
the repeated prefix roughly a tenth of the price — but only if the prefix is
**byte-identical** every time.

So this block is built once, frozen, and contains nothing that varies between
turns. No timestamp, no run id, no "as of now". The staleness of the activity
cache is stated as a *date* the cache already carries, not as a clock reading,
precisely so that two calls a minute apart produce the same bytes.

Get this wrong and nothing breaks — you simply pay full price on every turn
forever, silently. That is why it is a module with a docstring rather than three
lines inlined into a command.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from majordomo.config import Config


@dataclass(frozen=True)
class Context:
    """The stable prefix: what is known, before anything is asked."""

    memory_index: str
    memory_bodies: list[tuple[str, str]]  # (name, body)
    activity: str
    #: Date of the newest cached activity entry, or "" when nothing is cached.
    activity_through: str

    def render(self) -> str:
        parts: list[str] = []

        if self.memory_index.strip():
            # INDEX.md carries its own "# Majordomo memory" title for whoever
            # opens the file; nested under a section heading here it is noise.
            body = "\n".join(
                line
                for line in self.memory_index.strip().splitlines()
                if not line.startswith("# ")
            ).strip()
            if body:
                parts.append("## What I know about you\n\n" + body)

        if self.memory_bodies:
            detail = "\n\n".join(
                f"### {name}\n{body}" for name, body in self.memory_bodies
            )
            parts.append("## Detail on the relevant ones\n\n" + detail)

        if self.activity.strip():
            header = "## Your recent GitHub activity"
            if self.activity_through:
                # A date the data already carries — not a clock reading, which
                # would change the bytes on every call and kill the cache.
                header += f"\n\n(cache covers up to {self.activity_through})"
            parts.append(header + "\n\n" + self.activity.strip())

        return "\n\n".join(parts)

    def summary(self) -> str:
        """One line for ``/context`` and ``--explain``. Not sent to the model."""
        return (
            f"{len(self.memory_index.splitlines()) - 2 if self.memory_index else 0} "
            f"memories indexed, {len(self.memory_bodies)} loaded in full, "
            f"activity through {self.activity_through or 'never fetched'}"
        )


def build(
    config: Config,
    query: str | None = None,
    now: datetime | None = None,
    activity_limit: int = 60,
) -> Context:
    """Gather everything known. Never raises — a missing piece is an empty one.

    Args:
        query: what the user asked, used to pick which memory *bodies* come
            along. ``None`` (the chat case, where nothing has been asked yet)
            loads the most recently written ones instead.
        now: injected for testability; only used to window the activity cache.
    """
    from majordomo import activity as activity_mod
    from majordomo import memory as memory_mod

    index = ""
    bodies: list[tuple[str, str]] = []

    if config.memory.enabled:
        # Read the directory once and derive both the index and the bodies from
        # it — a second pass would be a second chance to disagree with itself.
        all_memories = memory_mod.read_all()

        # An empty store contributes nothing. Sending "(nothing remembered yet)"
        # spends tokens on every turn to say the section is empty.
        if all_memories:
            index = memory_mod.build_index(all_memories)

            limit = config.memory.max_bodies_loaded
            if query:
                scored = [(memory_mod.score(m, query), m) for m in all_memories]
                hits = sorted(
                    ((s, m) for s, m in scored if s > 0),
                    key=lambda pair: (-pair[0], pair[1].name),
                )
                chosen = [m for _, m in hits[:limit]]
            else:
                # No question yet. Newest-first is the best available proxy for
                # "most likely to matter" without spending a model call on it.
                chosen = sorted(
                    all_memories, key=lambda m: m.created, reverse=True
                )[:limit]

            bodies = [
                (m.name, m.body) for m in chosen if m.body and m.body != m.description
            ]

    events = activity_mod.recent(
        days=config.sources.github.activity_days, now=now
    )
    digest = activity_mod.digest(events, limit=activity_limit)
    newest = activity_mod.newest_at()

    return Context(
        memory_index=index,
        memory_bodies=bodies,
        activity=digest,
        activity_through=newest.date().isoformat() if newest else "",
    )
