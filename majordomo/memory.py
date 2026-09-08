"""What Majordomo has learned about you.

One file per fact under ``~/.majordomo/memory/``, plus an ``INDEX.md`` holding a
single line per memory.

── WHY A DIRECTORY AND NOT ONE PROFILE FILE ─────────────────────────────────
Because the cost of remembering has to stay flat as the profile grows.

A single profile.json or PROFILE.md is sent in full on every turn, so a year of
accumulated facts is a year of tokens paid for on every question — including the
ones where none of it is relevant. The index here is one line per memory, small
enough to send always; the *bodies* are the expensive part and only the ones
whose description matches the question come along.

That is the same progressive-disclosure shape Claude Code's own memory uses, and
it is what stops a profile from rotting into an unreadable append-only log.
─────────────────────────────────────────────────────────────────────────────

Two invariants, both learned from ``state.py``:

1. **Reads never raise.** A hand-edited file with broken frontmatter, a file
   from a future version with a type we don't understand, a stray ``.md`` that
   isn't a memory at all — each is skipped and counted, never raised. This is
   read on the path to a briefing; one bad file must not cost you the answer.
2. **The files are the source of truth; the index is derived.** ``INDEX.md`` is
   regenerated from the directory after every write and delete, so it cannot
   drift out of sync with what is actually there. Hand-edit a memory and the
   index corrects itself on the next write.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from majordomo.paths import memory_dir

#: A description is a one-line summary used for relevance matching. Longer than
#: this and it stops being scannable in an index that ships on every turn.
MAX_DESCRIPTION_CHARS = 200

#: A memory is a fact, not a document. Anything longer belongs in a file the
#: memory *points at*.
MAX_BODY_CHARS = 4_000

#: Recognised kinds. An unknown value is kept rather than rejected — a memory
#: from a future version should still be readable, just uncategorised.
KNOWN_TYPES = ("user", "preference", "project", "reference")

#: Words carrying no signal for relevance matching.
_STOPWORDS = frozenset(
    """
    a an the and or but if of in on at to for with from by is are was were be
    been being do does did doing have has had i me my we our you your it its
    this that these those what which who whom when where why how all any both
    each more most other some such no nor not only own same so than too very
    can will just should now about into over under again then once
    """.split()
)

#: Obvious credential shapes. Not a security boundary — a cheap guard against
#: the one mistake that would be genuinely bad to make silently, since a memory
#: is replayed into every future conversation that matches it.
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"),
)


class MemoryError_(Exception):
    """A *write* was refused. Reads never raise — see the module docstring."""


@dataclass(frozen=True)
class Memory:
    """One remembered fact."""

    name: str
    description: str
    body: str
    type: str = "user"
    created: str = ""

    def index_line(self) -> str:
        return f"- [{self.name}] {self.description}"


@dataclass(frozen=True)
class MemoryCandidate:
    """A proposed memory, not yet written. Produced by ``propose``.

    ``body`` is optional: most memories are a single sentence, and repeating it
    in both fields is noise. ``write_memory`` falls back to the description.
    """

    description: str
    body: str = ""
    type: str = "user"
    name: str = ""


@dataclass
class ReadResult:
    memories: list[Memory] = field(default_factory=list)
    #: Files that were not parseable as a memory. Surfaced by ``mj remember
    #: --debug`` rather than silently discarded.
    skipped: int = 0


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """'Prefers Windows-native!' -> 'prefers-windows-native'.

    Truncates on a word boundary, never mid-word — the slug is the filename and
    the index label, and 'cross-platform-as-a-later-co' reads like a bug.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(cleaned) <= 60:
        return cleaned or "memory"
    cut = cleaned[:60].rsplit("-", 1)[0]
    return cut.strip("-") or cleaned[:60].strip("-") or "memory"


def auto_name(description: str) -> str:
    """A short slug from a description's first few meaningful words.

    Naming from the whole description gives a filename nobody can read at a
    glance; the first handful of content words is enough to identify it and
    short enough to scan in a directory listing.
    """
    words = [w for w in re.split(r"[^a-z0-9]+", description.lower()) if w]
    kept: list[str] = []
    for word in words:
        if word in _STOPWORDS and kept:
            continue
        kept.append(word)
        if len(kept) >= 6:
            break
    return slugify("-".join(kept)) if kept else "memory"


def _tokens(text: str) -> set[str]:
    words = re.split(r"[^a-z0-9]+", text.lower())
    return {w for w in words if w and w not in _STOPWORDS and len(w) > 1}


#: Suffixes stripped before comparing two facts. Crude on purpose — a real
#: stemmer is a dependency, and the only job here is that "roast" and "roasted"
#: stop looking like different subjects.
_SUFFIXES = ("edly", "ing", "ed", "es", "ly", "s")


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _same_word(a: str, b: str) -> bool:
    """Close enough to be the same word: equal, or one abbreviates the other.

    The prefix rule is what connects "med" to "medium". It needs a floor of
    three characters or short tokens start matching everything.
    """
    if a == b:
        return True
    return min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))


def looks_like_secret(text: str) -> bool:
    """Does this contain something that is obviously a credential?"""
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def parse(text: str) -> Memory | None:
    """Parse one memory file. Returns None if it isn't one — never raises."""
    if text.startswith("﻿"):
        text = text[1:]

    # Split on fence *lines*, never on the bare substring "---".
    #
    # Splitting on the substring silently mangles any memory whose description
    # or body contains one — a markdown horizontal rule, a CLI flag, an ASCII
    # divider. The field is cut in half and the remaining metadata spills into
    # the body, so `type` and `created` vanish too. Data loss with no error,
    # which is the worst kind there is.
    lines = text.lstrip().splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    closing = next(
        (i for i in range(1, len(lines)) if lines[i].strip() == "---"), None
    )
    if closing is None:
        return None

    try:
        meta = yaml.safe_load("\n".join(lines[1:closing]))
    except Exception:
        # A hand-edit that broke the YAML. Skip the file, keep the briefing.
        return None
    if not isinstance(meta, dict):
        return None

    name = meta.get("name")
    description = meta.get("description")
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(description, str) or not description.strip():
        return None

    kind = meta.get("type")
    created = meta.get("created")

    return Memory(
        name=name.strip(),
        description=description.strip(),
        body="\n".join(lines[closing + 1:]).strip(),
        type=kind.strip() if isinstance(kind, str) and kind.strip() else "user",
        # A date lands as a datetime.date from YAML, not a string.
        created=str(created) if created else "",
    )


def render(memory: Memory) -> str:
    """A memory as the text of its file. ``safe_dump`` handles the quoting.

    Hand-writing ``description: {value}`` would break the moment a description
    contained a colon, which they routinely do.
    """
    meta = {
        "name": memory.name,
        "description": memory.description,
        "type": memory.type,
        "created": memory.created or date.today().isoformat(),
    }
    front = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{front}\n---\n\n{memory.body.strip()}\n"


def score(memory: Memory, query: str) -> int:
    """How many meaningful words the query shares with this memory's label.

    Deliberately matched against the name and description only, never the body:
    the description is what the index promises, and scoring on hidden text would
    make ``relevant`` unpredictable from what you can see.
    """
    return len(_tokens(query) & _tokens(f"{memory.name} {memory.description}"))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _root(root: Path | str | None) -> Path:
    return Path(root) if root is not None else memory_dir()


def read_all_detailed(root: Path | str | None = None) -> ReadResult:
    """Every memory on disk. Never raises."""
    target = _root(root)
    result = ReadResult()
    if not target.is_dir():
        return result

    try:
        paths = sorted(target.glob("*.md"))
    except OSError:
        return result

    for path in paths:
        if path.name == "INDEX.md":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            result.skipped += 1
            continue
        memory = parse(text)
        if memory is None:
            result.skipped += 1
        else:
            result.memories.append(memory)

    return result


def read_all(root: Path | str | None = None) -> list[Memory]:
    return read_all_detailed(root).memories


def read_memory(name: str, root: Path | str | None = None) -> Memory | None:
    """One memory by name. Never raises."""
    path = _root(root) / f"{slugify(name)}.md"
    if not path.is_file():
        return None
    try:
        return parse(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def relevant(
    query: str,
    limit: int = 8,
    root: Path | str | None = None,
) -> list[Memory]:
    """The memories whose label overlaps the query, best first.

    Keyword overlap rather than embeddings: it needs no model call, no extra
    dependency, and no index to keep warm — and on a set of a few hundred short
    descriptions the difference in quality does not pay for any of that.
    """
    scored = [(score(m, query), m) for m in read_all(root)]
    hits = [(s, m) for s, m in scored if s > 0]
    hits.sort(key=lambda pair: (-pair[0], pair[1].name))
    return [m for _, m in hits[:limit]]


def build_index(memories: list[Memory]) -> str:
    """The index text for a set of memories. Pure — takes no filesystem."""
    if not memories:
        return "# Majordomo memory\n\n(nothing remembered yet)\n"
    lines = ["# Majordomo memory", ""]
    lines += [m.index_line() for m in sorted(memories, key=lambda m: m.name)]
    return "\n".join(lines) + "\n"


def read_index(root: Path | str | None = None) -> str:
    """The index as text.

    Built from the files rather than read from INDEX.md, so a stale or
    hand-mangled index can never be what gets sent to the model. INDEX.md exists
    for *you* to read; this is what the prompt uses.
    """
    return build_index(read_all(root))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def find_similar(
    description: str,
    root: Path | str | None = None,
    threshold: float = 0.5,
) -> Memory | None:
    """An existing memory covering roughly the same ground, if there is one.

    This is what makes ``write_memory`` able to offer update-instead-of-duplicate.
    Without it the set fills with six near-identical statements of the same
    preference and stops being worth reading.

    **This is a hint, not a verdict.** Measured against a set of real pairs, no
    threshold cleanly separates a restatement from a different fact: "Works
    mainly in Python" and "Works mainly in Rust" share every word that word
    overlap can see, and differ only in the one carrying the meaning. So it is
    tuned to catch rather than to be certain, and the caller asks the user —
    a wrong guess costs a keystroke, and a hard refusal here would just teach
    everyone to reach for ``--force``.
    """
    incoming = {_stem(t) for t in _tokens(description)}
    if not incoming:
        return None

    best: tuple[float, Memory] | None = None
    for memory in read_all(root):
        existing = {_stem(t) for t in _tokens(memory.description)}
        if not existing:
            continue
        hits = sum(any(_same_word(a, b) for b in existing) for a in incoming)
        overlap = hits / min(len(incoming), len(existing))
        if overlap >= threshold and (best is None or overlap > best[0]):
            best = (overlap, memory)

    return best[1] if best else None


def _unique_name(base: str, root: Path) -> str:
    """A slug not already taken. 'topic', then 'topic-2', 'topic-3'..."""
    name = base
    suffix = 2
    while (root / f"{name}.md").exists():
        name = f"{base}-{suffix}"
        suffix += 1
    return name


def write_memory(
    candidate: MemoryCandidate,
    root: Path | str | None = None,
    overwrite: str | None = None,
) -> Memory:
    """Write one memory and refresh the index.

    Args:
        candidate: what to remember.
        root:      memory directory; defaults to ``~/.majordomo/memory``.
        overwrite: the name of an existing memory to replace, rather than
                   creating a new one. This is the update half of
                   update-instead-of-duplicate.

    Raises:
        MemoryError_: the directory or the file could not be written, or the
            candidate is empty, too long, or contains something
            that looks like a credential. Writes are the one operation here that
            is allowed to refuse — a bad memory is replayed into every future
            conversation, so it is worth stopping at the door.
    """
    description = candidate.description.strip()
    body = candidate.body.strip()

    if not description:
        raise MemoryError_("a memory needs a description")
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise MemoryError_(
            f"description is {len(description)} chars; keep it under "
            f"{MAX_DESCRIPTION_CHARS} so the index stays scannable"
        )
    if len(body) > MAX_BODY_CHARS:
        raise MemoryError_(
            f"body is {len(body)} chars; a memory is a fact, not a document — "
            f"keep it under {MAX_BODY_CHARS} or point at a file instead"
        )
    if looks_like_secret(f"{description}\n{body}"):
        raise MemoryError_(
            "that looks like it contains a credential. Memories are replayed "
            "into future conversations — put secrets in ~/.majordomo/.env"
        )

    target = _root(root)
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MemoryError_(f"could not create {target}: {exc}") from exc

    if overwrite:
        name = slugify(overwrite)
    else:
        base = slugify(candidate.name) if candidate.name else auto_name(description)
        name = _unique_name(base, target)

    memory = Memory(
        name=name,
        description=description,
        body=body or description,
        type=candidate.type if candidate.type in KNOWN_TYPES else "user",
        created=date.today().isoformat(),
    )

    # A raw OSError here escapes every caller: they catch MemoryError_, and
    # this surfaces on the chat REPL's exit path and out of `mj remember` — the
    # two places least able to afford a traceback. A read-only directory, a full
    # disk or a permission change is a refusal to write, which is exactly what
    # MemoryError_ means.
    try:
        (target / f"{name}.md").write_text(render(memory), encoding="utf-8")
    except OSError as exc:
        raise MemoryError_(f"could not write {name}.md: {exc}") from exc

    _refresh_index(target)
    return memory


def delete_memory(name: str, root: Path | str | None = None) -> bool:
    """Remove one memory. Returns whether it existed.

    Raises:
        MemoryError_: the file exists but could not be removed.
    """
    target = _root(root)
    path = target / f"{slugify(name)}.md"
    if not path.is_file():
        return False
    try:
        path.unlink()
    except OSError as exc:
        raise MemoryError_(f"could not delete {path.name}: {exc}") from exc

    _refresh_index(target)
    return True


def _refresh_index(target: Path) -> None:
    """Regenerate INDEX.md from the directory. The files are the truth."""
    try:
        target.mkdir(parents=True, exist_ok=True)
        (target / "INDEX.md").write_text(
            build_index(read_all(target)), encoding="utf-8"
        )
    except OSError:
        # An unwritable index is cosmetic — read_index() derives from the files
        # anyway, so nothing downstream depends on this having succeeded.
        pass


__all__ = [
    "Memory",
    "MemoryCandidate",
    "MemoryError_",
    "ReadResult",
    "auto_name",
    "build_index",
    "delete_memory",
    "find_similar",
    "looks_like_secret",
    "parse",
    "read_all",
    "read_all_detailed",
    "read_index",
    "read_memory",
    "relevant",
    "render",
    "score",
    "slugify",
    "write_memory",
]
