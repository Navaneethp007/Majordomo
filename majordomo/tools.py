"""What the agent can actually do, and what it must ask before doing.

Two halves, and the split is the safety model:

- **Reads** — ``read_file``, ``list_files``, ``grep`` — run freely. Looking at
  code cannot break anything.
- **Writes and commands** — ``write_file``, ``edit_file``, ``run_command`` — are
  shown and confirmed first. A free model driving a loop *will* be wrong
  sometimes; the gate is what makes that survivable rather than expensive.

── ON PATH CONFINEMENT ──────────────────────────────────────────────────────
Every path is resolved to canonical form and checked against the project root
before anything opens it. The check is on the *resolved* path deliberately:
``../../.ssh/id_rsa`` and a symlink pointing outside both look innocent until
resolved, and a model that has read something it should not cannot unread it.

This is the same check ``scaffold.plan`` makes before creating a directory —
kept here rather than shared because the failure modes differ: scaffold refuses
and stops, an agent tool refuses and reports back so the model can try
something else.

Confinement is not sufficient on its own, because the root is whatever
directory you launched from — often a home directory, which contains ``.env``,
``.ssh`` and the rest. So there is a second rule: ``is_sensitive`` refuses
credential files *wherever* they sit, inside the root included. See its
docstring for why that one is a flat refusal rather than a prompt.

``run_command`` is outside both rules — a shell is a shell, and ``type .env``
would work. That is precisely why commands are confirmed and reads are not.
─────────────────────────────────────────────────────────────────────────────

Every tool returns a string, including its errors. An exception here would end
the agent's turn; a string saying "no such file" is something the model can read
and act on, which is almost always what you want.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: Longer than this and a tool result starts crowding out the conversation.
#: Truncation is always stated, so the model can ask for a narrower slice.
MAX_RESULT_CHARS = 8_000

#: How long a command may run before it is killed.
COMMAND_TIMEOUT = 120


@dataclass(frozen=True)
class Tool:
    """One capability, plus whether using it needs a human to say yes."""

    name: str
    description: str
    parameters: dict
    #: Reads run freely; anything that changes the world asks first.
    needs_confirmation: bool = False

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _obj(**properties) -> dict:
    required = [name for name, spec in properties.items() if spec.pop("_required", True)]
    return {"type": "object", "properties": properties, "required": required}


TOOLS: dict[str, Tool] = {
    "read_file": Tool(
        name="read_file",
        description=(
            "Read a file and return its contents. Text files come back with "
            "line numbers; PDFs and Word (.docx) files are converted to text "
            "for you, so use this for those too rather than reaching for a "
            "shell command. Read a file before editing it, so the edit matches "
            "what is actually there."
        ),
        parameters=_obj(path={"type": "string", "description": "Path within the project"}),
    ),
    "list_files": Tool(
        name="list_files",
        description=(
            "List files under a directory, recursively. Returns paths relative "
            "to the project root."
        ),
        parameters=_obj(
            directory={"type": "string", "description": "Directory within the project"},
            pattern={
                "type": "string",
                "description": "Optional glob, e.g. '*.py'",
                "_required": False,
            },
        ),
    ),
    "grep": Tool(
        name="grep",
        description=(
            "Search file contents for a regular expression. Returns matching "
            "lines with their file and line number."
        ),
        parameters=_obj(
            pattern={"type": "string", "description": "Regular expression"},
            directory={
                "type": "string",
                "description": "Where to search; defaults to the project root",
                "_required": False,
            },
            glob={
                "type": "string",
                "description": "Limit to files matching this, e.g. '*.py'",
                "_required": False,
            },
        ),
    ),
    "write_file": Tool(
        name="write_file",
        description=(
            "Create a file, or replace one entirely. Prefer edit_file for a "
            "change to an existing file — a full rewrite loses anything you did "
            "not know was there."
        ),
        parameters=_obj(
            path={"type": "string", "description": "Path within the project"},
            content={"type": "string", "description": "The complete file contents"},
        ),
        needs_confirmation=True,
    ),
    "edit_file": Tool(
        name="edit_file",
        description=(
            "Replace an exact string in a file. The old text must appear "
            "exactly once — if it appears twice or not at all the edit is "
            "refused, because a near-miss silently changing the wrong line is "
            "worse than a failure you can see. Include surrounding lines to "
            "make it unique."
        ),
        parameters=_obj(
            path={"type": "string", "description": "Path within the project"},
            old={"type": "string", "description": "Exact text to replace"},
            new={"type": "string", "description": "What to replace it with"},
        ),
        needs_confirmation=True,
    ),
    "run_command": Tool(
        name="run_command",
        description=(
            "Run a shell command in the project directory and return its "
            "output. Use it for tests, linters and git — not for editing files, "
            "which edit_file does more safely."
        ),
        parameters=_obj(
            command={"type": "string", "description": "The command to run"},
        ),
        needs_confirmation=True,
    ),
}


def schemas() -> list[dict]:
    """Every tool, in the shape the API wants."""
    return [tool.schema() for tool in TOOLS.values()]


# ---------------------------------------------------------------------------
# Confinement
# ---------------------------------------------------------------------------


class Refused(Exception):
    """A path a tool will not touch. Caught as one thing by every tool."""


class OutsideProject(Refused):
    """A path resolved to somewhere outside the project root."""


class Sensitive(Refused):
    """A path names a credential file. Refused wherever it lives."""


#: Directories whose entire contents are off limits, matched on any path segment.
_SECRET_DIRS = frozenset({".ssh", ".aws", ".gnupg", ".azure", ".kube", ".docker"})

#: Exact filenames that hold credentials.
_SECRET_NAMES = frozenset(
    {
        "credentials",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        ".netrc",
        ".pgpass",
        ".htpasswd",
        ".npmrc",
        ".pypirc",
        "secrets.json",
        "credentials.json",
        "service-account.json",
    }
)

#: Filename patterns: .env and its variants, private keys, keystores.
_SECRET_PATTERNS = (
    re.compile(r"^\.env(\..+)?$"),
    re.compile(r"^.+\.(pem|key|p12|pfx|jks|keystore|ppk)$"),
    re.compile(r"^\.?(secrets?|credentials)\.(ya?ml|json|toml|ini)$"),
)


def within(base: Path, target: Path) -> bool:
    """Is ``target`` at or below ``base``, after both are resolved?

    ``resolve`` checks the *directory* a tool was pointed at. It does not check
    what comes back from walking it — and two things escape that way:

    - **A model-supplied glob.** ``list_files(pattern="../*")`` and
      ``grep(glob="../*")`` are handed straight to ``rglob``, which happily
      walks upward. No symlink required.
    - **A junction or symlink inside the tree**, which ``rglob`` follows out.

    Both land on tools with ``needs_confirmation=False``, so there is no prompt,
    and ``grep`` prints matching *lines* — the contents leave the machine for a
    model provider. Every path a walk yields is checked, not just the root.
    """
    try:
        real = target.resolve()
    except OSError:
        return False
    return real == base or base in real.parents


def is_sensitive(target: Path) -> bool:
    """Does this path name something that holds a credential?

    Confinement alone is not enough once the root is a home directory, which is
    where this will usually be run from. Reads are ungated by design — asking
    before every file would make the agent useless — but a read means the
    contents go into a request to a third-party model provider. So a question
    as ordinary as "why is my API key not working?" would ship ``.env`` off the
    machine without a single prompt.

    A bad edit is recoverable with git. A leaked key is not: it is already gone
    by the time you see it, and the remedy is revocation. That asymmetry is why
    this is a flat refusal rather than a confirmation — there is no answer to
    "shall I send your private key to OpenRouter?" that should be yes.
    """
    name = target.name.lower()
    if name in _SECRET_NAMES:
        return True
    if any(pattern.match(name) for pattern in _SECRET_PATTERNS):
        return True
    return any(part.lower() in _SECRET_DIRS for part in target.parts)


def resolve(root: Path, candidate: str) -> Path:
    """Resolve a model-supplied path, refusing anything outside ``root``.

    Resolution happens *before* the check because that is the only order that
    works: ``../../.ssh/id_rsa`` and a symlink out of the tree both look
    ordinary until resolved.

    Raises:
        OutsideProject: the path escapes ``root``.
        Sensitive: the path names a credential file, wherever it sits. This one
            applies even *inside* the root — a project's own ``.env`` is exactly
            as leakable as one in your home directory.
    """
    base = Path(root).resolve()
    try:
        target = (base / candidate).resolve()
    except OSError as exc:
        raise OutsideProject(f"could not resolve {candidate!r}: {exc}") from exc

    if target != base and base not in target.parents:
        raise OutsideProject(
            f"{candidate!r} is outside the project directory ({base})"
        )
    if is_sensitive(target):
        raise Sensitive(
            f"{candidate!r} holds credentials and cannot be read or written. "
            f"If you need what is configured there, ask the user rather than "
            f"opening the file."
        )
    return target


#: The notice ``_truncate`` appends. Named so a caller can *ask* whether a result
#: was cut rather than reasoning about its length: comparing against
#: ``MAX_RESULT_CHARS`` is arithmetic about a side effect, and gets the boundary
#: wrong for a body that is exactly the limit and was never truncated.
TRUNCATION_NOTICE = f"… truncated at {MAX_RESULT_CHARS} characters."


def was_truncated(text: str) -> bool:
    """Did ``_truncate`` cut this? The fact, rather than an inference from it."""
    return TRUNCATION_NOTICE in text


def _truncate(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return (
        text[:MAX_RESULT_CHARS]
        + f"\n{TRUNCATION_NOTICE} "
        "Narrow the request — a line range, a tighter pattern — to see more."
    )


# ---------------------------------------------------------------------------
# The tools themselves. Every one returns a string, errors included.
# ---------------------------------------------------------------------------


#: Directories never worth a model's context, and never worth walking.
_NOISE = frozenset({".git", "__pycache__", ".pytest_cache", "node_modules",
                    ".venv", "venv", ".mypy_cache", "dist", "build"})


def _worth_reading(path: Path) -> bool:
    return not any(part in _NOISE for part in path.parts)


#: Formats recognised by their first bytes, mapped to what to call them.
#:
#: Named rather than lumped together as "binary", because the remedy differs and
#: the model acts on what it is told. "report.pdf is a PDF" leads somewhere;
#: "report.pdf is not text" invites a second attempt at the same file.
_MAGIC = (
    (b"%PDF-", "a PDF"),
    (b"PK\x03\x04", "a zip-based file (.docx, .xlsx, .pptx and .zip all look like this)"),
    (b"\xd0\xcf\x11\xe0", "an old Office file (.doc/.xls/.ppt)"),
    (b"\x89PNG", "a PNG image"),
    (b"\xff\xd8\xff", "a JPEG image"),
    (b"GIF8", "a GIF image"),
    (b"\x7fELF", "a compiled binary"),
    (b"\x1f\x8b", "a gzip archive"),
    (b"SQLite format 3", "a SQLite database"),
)

#: Magic that is also ordinary text, so it needs corroboration before it means
#: anything. ``MZ`` is a DOS executable header *and* two printable letters — a
#: note opening "MZ is the prefix used by PE binaries" was refused by name, and
#: the refusal tells the model not to retry, so it could not recover. The other
#: printable signatures here are long enough to be unambiguous on their own.
_AMBIGUOUS_MAGIC = ((b"MZ", "a Windows executable"),)

#: How much of a file to inspect. A text file's first kilobytes settle it, and
#: reading the whole of a 400MB binary to decide it is binary is absurd.
_SNIFF_BYTES = 4096


def _decode(raw: bytes) -> str:
    """Bytes to text, honouring a byte-order mark.

    ``describe_binary`` already decides UTF-16 is text rather than binary, so
    reading it back as UTF-8 produced a line of characters separated by spaces —
    technically not a refusal, practically still garbage. Windows tools write
    UTF-16 with a BOM often enough to be worth the four lines.
    """
    boms = (
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
        (b"\xef\xbb\xbf", "utf-8-sig"),
    )
    for bom, encoding in boms:
        if raw.startswith(bom):
            try:
                return raw.decode(encoding)
            except (UnicodeDecodeError, ValueError):
                break
    return raw.decode("utf-8", errors="replace")


def describe_binary(path: Path, head: bytes | None = None) -> str:
    """What kind of non-text file this is, or ``""`` if it reads as text.

    The reason this exists: ``read_text(errors="replace")`` does not fail on a
    PDF. It returns the bytes as replacement characters, the model summarises
    the noise, and you get a confident answer about a document nobody read. A
    tool that cannot do something must say so — silently returning garbage is
    the worst of the three options.

    ``head`` lets a caller that has already read the file hand the bytes over
    rather than have them read a second time.
    """
    if head is None:
        try:
            with open(path, "rb") as handle:
                head = handle.read(_SNIFF_BYTES)
        except OSError:
            return ""

    for magic, description in _MAGIC:
        if head.startswith(magic):
            return description

    # Only believed when something else in the head is non-text.
    has_nul = b"\x00" in head
    for magic, description in _AMBIGUOUS_MAGIC:
        if head.startswith(magic) and has_nul:
            return description

    # A NUL byte is the classic tell, and the one that catches formats not
    # listed above. UTF-16 text trips it too — but only a byte-order mark
    # settles that, because `decode("utf-16")` succeeds on almost any
    # even-length byte string and happily called a binary blob text.
    utf16_bom = (b"\xff\xfe", b"\xfe\xff")
    if b"\x00" in head and not head.startswith(utf16_bom):
        return "a binary file"
    return ""


def read_file(root: Path, path: str = "", **_ignored) -> str:
    try:
        target = resolve(root, path)
    except Refused as exc:
        return f"ERROR: {exc}"
    if not target.is_file():
        return f"ERROR: no file at {path}"

    # Extraction is tried before the refusal, so a PDF reads as text where the
    # extra is installed and refuses clearly where it is not. Each failure names
    # itself — the extra, an encrypted file and a scan need different answers,
    # and "cannot read it" for all three sends the model in circles.
    from majordomo import documents

    if documents.can_extract(target):
        try:
            return _truncate(documents.extract(target)) or "(no text)"
        except documents.ExtractionError as exc:
            return f"ERROR: {exc}"

    kind = describe_binary(target)
    if kind:
        return (
            f"ERROR: {path} is {kind}, not text — I cannot read it. Do not try "
            f"again with a different tool; tell the user what the file is."
        )

    try:
        text = _decode(target.read_bytes())
    except OSError as exc:
        return f"ERROR: could not read {path}: {exc}"

    numbered = "\n".join(
        f"{n:>5}  {line}" for n, line in enumerate(text.splitlines(), 1)
    )
    return _truncate(numbered) or "(empty file)"


def list_files(root: Path, directory: str = ".", pattern: str = "", **_ignored) -> str:
    try:
        target = resolve(root, directory or ".")
    except Refused as exc:
        return f"ERROR: {exc}"
    if not target.is_dir():
        return f"ERROR: no directory at {directory}"

    base = Path(root).resolve()
    found: list[str] = []
    for path in sorted(target.rglob(pattern or "*")):
        if not path.is_file():
            continue
        # The glob came from the model, and rglob follows links out of the tree.
        if not within(base, path) or is_sensitive(path):
            continue
        if not _worth_reading(path):
            continue
        found.append(str(path.relative_to(base)).replace("\\", "/"))

    return _truncate("\n".join(found)) if found else "(nothing matched)"


def grep(
    root: Path, pattern: str = "", directory: str = ".", glob: str = "", **_ignored
) -> str:
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"ERROR: bad regular expression: {exc}"
    try:
        target = resolve(root, directory or ".")
    except Refused as exc:
        return f"ERROR: {exc}"

    base = Path(root).resolve()
    hits: list[str] = []
    # Filtered before sorting: sorting first means ordering every path in
    # node_modules before discarding it.
    for path in sorted(p for p in target.rglob(glob or "*") if _worth_reading(p)):
        if not path.is_file():
            continue
        # grep prints matching *lines*, so confining the directory is not
        # enough on its own: a "../*" glob, or a junction rglob followed, would
        # print a file's contents one line at a time from outside the project.
        if not within(base, path) or is_sensitive(path):
            continue
        # Same reason as read_file, one step worse: grep prints matching
        # *lines*, so a binary file contributes mangled bytes that look like
        # findings.
        # Read once. `describe_binary` opened every candidate and then the
        # search opened it again — and the search used a bare UTF-8 decode, so
        # the BOM-marked UTF-16 files the sniff deliberately let through came
        # back as garbage matches.
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if describe_binary(path, head=raw[:_SNIFF_BYTES]):
            continue
        lines = _decode(raw).splitlines()
        rel = str(path.relative_to(base)).replace("\\", "/")
        for number, line in enumerate(lines, 1):
            if regex.search(line):
                hits.append(f"{rel}:{number}: {line.strip()[:200]}")

    return _truncate("\n".join(hits)) if hits else "(no matches)"


def write_file(root: Path, path: str = "", content: str = "", **_ignored) -> str:
    if not isinstance(content, str):
        # Say what was wrong rather than coercing. A list of lines is ambiguous
        # about its line endings and trailing newline, and quietly guessing
        # produces a file that looks right and diffs wrong.
        return (
            f"ERROR: `content` must be a string, not "
            f"{type(content).__name__} — send the whole file as one string"
        )
    try:
        target = resolve(root, path)
    except Refused as exc:
        return f"ERROR: {exc}"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.is_file()
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"ERROR: could not write {path}: {exc}"
    return f"{'Replaced' if existed else 'Created'} {path} ({len(content)} chars)"


def edit_file(root: Path, path: str = "", old: str = "", new: str = "", **_ignored) -> str:
    for label, value in (("old", old), ("new", new)):
        if not isinstance(value, str):
            return (
                f"ERROR: `{label}` must be a string, not "
                f"{type(value).__name__} — send the exact text, newlines included"
            )
    try:
        target = resolve(root, path)
    except Refused as exc:
        return f"ERROR: {exc}"
    if not target.is_file():
        return f"ERROR: no file at {path}"
    if not old:
        return "ERROR: `old` is empty — use write_file to create a file"

    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        return f"ERROR: could not read {path}: {exc}"

    count = text.count(old)
    if count == 0:
        return f"ERROR: that text does not appear in {path}"
    if count > 1:
        # Replacing the first of several is how an edit silently lands on the
        # wrong line. Make the model disambiguate instead.
        return (
            f"ERROR: that text appears {count} times in {path} — include more "
            f"surrounding lines so it is unique"
        )

    try:
        target.write_text(text.replace(old, new, 1), encoding="utf-8")
    except OSError as exc:
        return f"ERROR: could not write {path}: {exc}"
    return f"Edited {path}"


def run_command(root: Path, command: str = "", **_ignored) -> str:
    # Same guard as write_file and edit_file. A list here would reach
    # subprocess with shell=True and fail somewhere less legible.
    if not isinstance(command, str):
        return (
            f"ERROR: `command` must be a string, not {type(command).__name__} "
            f"— send the whole command line as one string"
        )
    if not command.strip():
        return "ERROR: no command given"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(Path(root).resolve()),
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: command exceeded {COMMAND_TIMEOUT}s and was stopped"
    except OSError as exc:
        return f"ERROR: could not run it: {exc}"

    parts = [f"exit code {result.returncode}"]
    if result.stdout.strip():
        parts.append("stdout:\n" + result.stdout.strip())
    if result.stderr.strip():
        parts.append("stderr:\n" + result.stderr.strip())
    return _truncate("\n".join(parts))


HANDLERS = {
    "read_file": read_file,
    "list_files": list_files,
    "grep": grep,
    "write_file": write_file,
    "edit_file": edit_file,
    "run_command": run_command,
}


def as_text(value) -> str:
    """A model-supplied argument as text, whatever the model actually sent.

    The schema says these are strings; a model may send a list of lines anyway.
    This runs in the confirmation prompt, *before* the tool handler gets a
    chance to refuse the call — so an unchecked ``.splitlines()`` here took down
    the whole run and discarded its trail, over a preview. Nothing in here can
    raise. The handler still refuses the call properly and tells the model what
    was wrong, which is the part that helps it recover.

    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(as_text(item) for item in value)
    return str(value)


def describe_call(name: str, arguments: dict) -> str:
    """One line for the confirmation prompt. What you are about to approve.

    Deliberately shows the *content* for a write and the whole command for a
    run — approving something you cannot see is not approval.
    """
    if name == "write_file":
        content = as_text(arguments.get("content"))
        return f"write {arguments.get('path')} ({len(content)} chars)"
    if name == "edit_file":
        old = as_text(arguments.get("old")).strip().splitlines()
        first = old[0][:60] if old else ""
        return f"edit {arguments.get('path')} — replace {first!r}…"
    if name == "run_command":
        return f"run: {arguments.get('command')}"
    return f"{name}({', '.join(f'{k}={v!r}' for k, v in arguments.items())})"
