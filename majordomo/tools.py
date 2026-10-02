"""What the agent can actually do, and what it must ask before doing.

Two halves, and the split is the safety model:

- **Reads** — ``read_file``, ``list_files``, ``grep`` — run freely. Looking at
  code cannot break anything.
- **Writes and commands** — ``write_file``, ``edit_file``, ``run_command`` — are
  shown and confirmed first. A free model driving a loop *will* be wrong
  sometimes; the gate is what makes that survivable rather than expensive.

── A THIRD CASE: TOOLS THAT ARE BOTH ────────────────────────────────────────
``git`` and ``github`` do not fit either half, because what they are depends on
their arguments: ``git log`` changes nothing and ``git commit`` changes
everything. A per-tool flag cannot express that, so ``Tool.gate`` narrows the
decision to the individual call — see ``Tool.requires_approval`` for why that is
a method rather than a widened boolean, and ``_git_needs_approval`` for the
classification itself.

The important consequence is that **a read is not automatically safe**. Deciding
``git`` reads were free would have opened a new ungated path to credentials,
because ``git show HEAD:.env`` prints a file ``read_file`` flatly refuses. So
content-printing git commands are gated even though they only read, and the
denylist reaches into git's arguments too. Three places now enforce one rule:
``resolve`` for paths, ``within`` for what a walk yields, and
``_refuse_sensitive_argument`` for what git is asked to print.

Outward-facing writes are a fourth case again, handled in ``cli``: a comment
posted under your name cannot be withdrawn, so ``--yes`` refuses them when
nobody is at the terminal.

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
from typing import Callable

#: Longer than this and a tool result starts crowding out the conversation.
#: Truncation is always stated, so the model can ask for a narrower slice.
MAX_RESULT_CHARS = 8_000

#: How long a command may run before it is killed.
COMMAND_TIMEOUT = 120

#: How long a networked `gh` call may take. Separate from COMMAND_TIMEOUT
#: because the trade-off differs: a local command that hangs for two minutes is
#: a stuck test run, while `gh pr create` is a round trip to GitHub and 10s — the
#: value `workers/github.py` uses for minting a token — would fail on a slow
#: connection for no reason.
NETWORK_TIMEOUT = 60


class ExecutableMissing(Exception):
    """A required external program is not on PATH."""


def _resolve_exe(name: str) -> str:
    """Find an executable, by path rather than by name.

    Resolved through ``shutil.which`` rather than handed to the OS as a bare
    name, for the reason ``workers/github.py`` records: on Windows a scoop or npm
    install is a ``.cmd`` shim, which ``subprocess`` will not execute without a
    shell. ``which`` finds the shim by PATHEXT and returns a path that runs
    directly, so ``shell=False`` stays possible.

    (``scaffold.py`` passes a bare ``"git"`` and gets away with it because a
    failed ``git init`` there is only a warning. A tool cannot be so relaxed.)
    """
    import shutil

    found = shutil.which(name)
    if not found:
        raise ExecutableMissing(f"{name} is not on PATH")
    return found


def _decode_output(raw: bytes) -> str:
    """Bytes from a child process to text, trying UTF-8 before the locale codec.

    Separate from ``_decode``, which is for *file* contents and leads with a
    byte-order mark. Process output has no BOM and a different problem: what
    codec it is in depends on which program wrote it.

    ── WHY A FALLBACK AND NOT A CHOICE ──────────────────────────────────────
    Neither single answer is right, which I established by measuring rather
    than reasoning. On this machine, for one em dash:

        git            emits e2 80 94   — UTF-8, whatever the console code page
        a python child emits 97         — cp1252, from its own locale

    So forcing UTF-8 makes ``git log`` correct and breaks every Windows-native
    command (``97`` is not valid UTF-8 at all — "invalid start byte"), and
    keeping the locale codec does the exact opposite, silently, which is how
    ``git log`` came back mangled.

    The way out is that **UTF-8 is self-validating**: byte sequences that are not
    UTF-8 are overwhelmingly detectable as such, while cp1252 maps almost
    everything to *something* and so can never report a failure. Trying the
    strict, checkable codec first and falling back to the permissive one is
    therefore reliable in a way the reverse ordering could never be.
    """
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        import locale

        return raw.decode(locale.getpreferredencoding(False), errors="replace")


def _run(argv: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    """Run a program directly — no shell — and hand back the completed process.

    The one seam both the ``git`` and ``github`` tools go through, which is what
    lets tests run git for real and stub ``gh`` at a single point. Two seams
    (``which`` *and* ``subprocess.run``) is the fragility the network guard in
    ``conftest`` was written about.

    ── ON THE DECODING ──────────────────────────────────────────────────────
    ``encoding="utf-8"`` is not decoration. ``text=True`` alone decodes with the
    locale codec — **cp1252** on a default Windows install — while git and ``gh``
    emit UTF-8 regardless of the console code page. Measured across em dashes,
    ellipses, emoji, CJK, Devanagari and accented Latin: every one of them
    *silently mojibakes*, and only a C1 control byte raises. Silent corruption is
    the worse outcome, because a ``UnicodeDecodeError`` would at least be
    noticed — instead the model is handed mangled text and told it is the commit
    message. This project's own history is full of em dashes, so ``git log`` here
    was the trigger case.

    Raises:
        ExecutableMissing: via ``_resolve_exe``.
        subprocess.TimeoutExpired, OSError: left to the caller, because what a
            timeout *means* differs per tool — for ``git`` it is a retry, and for
            ``gh pr create`` the pull request may already exist.
    """
    return subprocess.run(
        [_resolve_exe(argv[0]), *argv[1:]],
        cwd=str(Path(cwd).resolve()),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _format_result(result: subprocess.CompletedProcess) -> str:
    """The shape every command result takes. Shared so they cannot drift.

    Accepts a process captured either as bytes or as text, because its two
    callers differ: ``_run`` knows it is talking to git or ``gh`` and can demand
    UTF-8, while ``run_command`` runs anything and has to sniff. Both end up
    here so the ``exit code N`` / ``stdout:`` / ``stderr:`` shape is written once.
    """

    def text(stream) -> str:
        if stream is None:
            return ""
        return _decode_output(stream) if isinstance(stream, bytes) else stream

    parts = [f"exit code {result.returncode}"]
    out, err = text(result.stdout), text(result.stderr)
    if out.strip():
        parts.append("stdout:\n" + out.strip())
    if err.strip():
        parts.append("stderr:\n" + err.strip())
    return _truncate("\n".join(parts))


@dataclass(frozen=True)
class Tool:
    """One capability, plus whether using it needs a human to say yes."""

    name: str
    description: str
    parameters: dict
    #: Reads run freely; anything that changes the world asks first.
    needs_confirmation: bool = False
    #: Narrows the gate to *particular calls*. Consulted only when
    #: ``needs_confirmation`` is already true, so a tool is either never gated,
    #: always gated, or gated by this predicate — never accidentally ungated.
    gate: Callable[[dict], bool] | None = None
    #: Returns an ``"ERROR: …"`` string when these arguments cannot be used at
    #: all, else "". Checked *before* the gate, so a call that could never run
    #: does not interrupt anyone to ask about it — see ``Tool.unusable``.
    reject: Callable[[dict], str] | None = None

    def unusable(self, arguments: dict) -> str:
        """Why this call cannot run, or "" if it can.

        Exists because of what a real session looked like::

            git (bad arguments)
            allow this? [y/N] y
            · git (bad arguments)

        The model sent the wrong shape, the gate dutifully asked permission for
        a call already known to be impossible, and the user approved nothing
        happening. A prompt that cannot lead anywhere is worse than no prompt:
        it spends the one thing the gate depends on, which is being worth
        reading.
        """
        if self.reject is None:
            return ""
        try:
            return self.reject(arguments)
        except Exception:  # pragma: no cover - a validator must never raise
            return ""

    def requires_approval(self, arguments: dict) -> bool:
        """Does **this** call need a human? The single answer to that question.

        Some tools are read *and* write depending on their arguments — ``git``
        is the reason this exists, where ``log`` changes nothing and ``commit``
        changes everything. A per-tool boolean cannot express that, and the
        alternatives are worse:

        - Splitting into ``git`` and ``git_write`` hands the model a lever on its
          own gate, and the argument classifier would still be needed to stop it
          calling the wrong one.
        - Widening ``needs_confirmation`` to ``bool | Callable`` is the trap. A
          function object is **truthy**, so every ``if tool.needs_confirmation``
          in the codebase would keep compiling, keep passing, and silently mean
          "always gate". A union that degrades to the wrong answer rather than an
          error is not a refactor.

        Note what is *not* here: nothing in ``schema()`` reports this. The model
        is told about the split in prose, and finds out it guessed wrong from an
        error message. Declaring it on the wire would be write-ness asserted by
        the least trustworthy party, and we would then have to either trust it
        (a hole) or ignore it (dead weight).
        """
        if not self.needs_confirmation:
            return False
        return self.gate is None or self.gate(arguments)

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


# ---------------------------------------------------------------------------
# Classifying a git invocation
#
# ── WHY THIS IS NOT JUST "READS ARE FREE" ──────────────────────────────────
# The obvious split — log/status/diff free, commit/push gated — is wrong, and
# it took measuring to see why. ``read_file`` flatly refuses ``.env`` because
# "there is no answer to 'shall I send your private key to OpenRouter?' that
# should be yes". But git prints file contents too:
#
#     read_file(".env")      ERROR: holds credentials and cannot be read
#     git show HEAD:.env     prints it, verbatim
#     git log -p             prints it, verbatim
#
# So a free ``git show`` would be a brand-new ungated path to exactly the thing
# the denylist exists to stop, and the model would not even have to try: it is
# the normal way to read an old version of a file. This is the same hole
# ``within`` closes for ``grep``, for the reason recorded there — grep prints
# matching *lines*, so confining the directory is not enough.
#
# Hence two rules rather than one. Content-printing reads are gated, and any
# argument naming a credential file is refused outright.
# ---------------------------------------------------------------------------

#: Subcommands that only ever report. An **allowlist**: a subcommand nobody
#: thought of must be gated, because next year's git will have one and being
#: wrong in that direction costs a prompt rather than a mistake.
#:
#: ── WHY ``cat-file`` IS NOT HERE ─────────────────────────────────────────────
#: It was, and that was a hole as bad as the one this whole classification
#: exists to close. Two free calls:
#:
#:     git ls-tree HEAD            100644 blob 15f649ce…  .env
#:     git cat-file blob 15f649ce  OPENROUTER_API_KEY=sk-SUPERSECRET
#:
#: and the key is in a provider's logs with no human in the loop.
#: ``_refuse_sensitive_argument`` cannot help, because there is no path in that
#: argv — only a hash.
#:
#: The rule that actually holds is not "does it print file contents": ``blame``
#: prints a whole file and is free, which is fine, because ``read_file`` is free
#: too and the path can be checked. The rule is **whether the thing it prints
#: can be reviewed before it runs**. A path can; an object id cannot. So any
#: subcommand addressing content by object id is gated, and ``cat-file`` and
#: ``diff-tree`` are plumbing whose whole job is exactly that.
#:
#: Worth noting that ``cat-file -p`` *was* already gated — by coincidence, since
#: ``-p`` is in ``_GIT_PATCH_FLAGS`` for ``log -p``. The ``blob`` spelling takes
#: no flag and sailed straight through. Protection that holds for one spelling
#: of a command and not the obvious adjacent one is not protection.
#:
#: This is the same lesson as ``within`` two docstrings up: naming which file is
#: allowed is not enough when the tool can reach the contents another way.
_GIT_READS = frozenset({
    "status", "log", "show-branch", "blame", "rev-parse", "rev-list",
    "describe", "shortlog", "ls-files", "ls-tree",
    "merge-base", "name-rev", "whatchanged", "count-objects", "version",
})

#: Two-word reads, where the subcommand alone does not settle it.
_GIT_READ_PAIRS = frozenset({("stash", "list")})

#: Reads *only* while they are listing. Bare they report; with an argument they
#: create or delete, and `branch -D` must never run unasked.
#:
#: ``stash`` was here and should never have been: bare ``git stash`` is
#: ``git stash push``, not ``git stash list``. It ran free and silently reverted
#: uncommitted work — and the inversion was complete, because ``stash pop``, the
#: *recovery*, asked permission while the destruction did not. Every other member
#: genuinely lists when bare.
_GIT_LISTERS = frozenset({"branch", "tag", "remote", "config"})

#: Flags that make a listing subcommand do something else.
_GIT_LIST_FLAGS = frozenset({
    "-a", "-r", "-v", "-vv", "-l", "--all", "--list", "--verbose", "--remotes",
    "--show-current", "--contains", "--merged", "--no-merged", "--sort",
})

#: Flags that turn a report into a dump of file contents. ``log`` is free until
#: one of these appears, at which point it is as revealing as ``read_file``.
_GIT_PATCH_FLAGS = frozenset({"-p", "-u", "--patch", "--full-diff", "--raw"})

#: Subcommands that print file contents by default, so they are always gated.
_GIT_PRINTS_CONTENT = frozenset({"show", "diff", "difftool"})

#: ...unless the output is restricted to names and counts.
_GIT_SUMMARY_FLAGS = frozenset({
    "--stat", "--numstat", "--shortstat", "--dirstat", "--summary",
    "--name-only", "--name-status", "--check", "--quiet",
})


def _git_reject(args) -> str:
    """Everything about a git call that can be refused without running anything.

    ── WHY REFUSALS BELONG BEFORE THE GATE ──────────────────────────────────
    A refusal is a call that cannot run, so asking about it is the same mistake
    as asking about a malformed one — and worse here, because of what it looks
    like::

        run: git push --force
        allow this? [y/N] y
        ERROR: force-pushing is refused by this tool…

    The user is asked to authorise a force push, agrees, and is then told no.
    That does not merely waste the prompt, it misrepresents what the tool will
    do. ``unusable``'s docstring had the principle right and this was wired to
    only one of its two sources.

    **One refusal deliberately stays after the gate:** ``_refuse_foreign_repo``
    shells out to ``git rev-parse``, and running git before the user has agreed
    to anything is a different trade from reading the argv. Everything here is a
    pure function of the arguments; that one is not. If a refusal is ever added,
    the question to ask is which of those two it is.

    The handler calls this too, so the gate and the handler cannot disagree
    about what is refusable — the only thing that makes skipping a prompt safe.
    """
    argv = _argv(args, "git")
    if isinstance(argv, str):
        return argv

    if argv[0].startswith("-"):
        return (
            f"ERROR: {argv[0]!r} comes before the subcommand, and this tool "
            f"accepts no options there — they can redirect git outside the "
            f"project or change how it behaves. Start with the subcommand."
        )

    return _refuse_sensitive_argument(argv) or _refuse_force_push(argv)


def _gh_reject(args) -> str:
    """The same, for gh. See ``_git_reject`` for why these run before the gate."""
    argv = _argv(args, "gh")
    if isinstance(argv, str):
        return argv

    noun, _verb = _gh_pair(argv)
    if noun in _GH_REFUSED:
        return (
            f"ERROR: `gh {noun}` is refused by this tool and will be refused "
            f"every time — it handles credentials, or bypasses the read/write "
            f"split entirely. If the user needs it, they should run it "
            f"themselves."
        )

    if noun == "api":
        return _refuse_sensitive_api_path(argv)
    return ""


#: Decoding passes before giving up. Three is well past anything a real path
#: needs; the cap exists so a pathological input cannot spin here.
_MAX_UNQUOTE_PASSES = 4


def _fully_unquoted(text: str) -> str:
    """Percent-decode until it stops changing.

    One pass was enough for ``%2Eenv``, which GitHub reads as ``.env``, and not
    enough for ``%252Eenv`` — that decodes to ``%2Eenv``, which a single pass
    then compares literally and lets through.

    Whether GitHub itself decodes twice is not something I can establish without
    a live API call, and guessing is the wrong move either way: decoding to a
    fixed point removes the question instead of betting on the answer. ``unquote``
    is idempotent once no ``%`` remains, so the loop terminates on its own and
    the cap is belt rather than braces.
    """
    from urllib.parse import unquote

    for _ in range(_MAX_UNQUOTE_PASSES):
        decoded = unquote(text)
        if decoded == text:
            return text
        text = decoded
    return text


def _refuse_sensitive_api_path(argv: list[str]) -> str:
    """"ERROR: …" if an API path names a credential file, else "".

    ``gh api repos/o/n/contents/.env`` returns the file base64-encoded, and was
    a free read — while ``read_file(".env")`` refuses and ``git show HEAD:.env``
    refuses. The denylist had two doors covered and a third standing open.

    This does **not** inherit the ``gh pr diff`` exception. That one is scoped to
    the diff of a pull request somebody asked to have reviewed, and is justified
    on those grounds. ``api …/contents/…`` is arbitrary file read across the
    whole scope of the user's token — including every private repository it
    reaches — which is a different thing wearing the same "it is only remote"
    label. The asset ``is_sensitive`` protects is the user's own secrets, and a
    private repo of theirs is exactly where those live.
    """
    for item in argv[1:]:
        if item.startswith("-"):
            continue
        candidate = _fully_unquoted(item.split("?")[0])
        # As segments: `.ssh/id_rsa` has to be caught on either.
        for segment in candidate.split("/"):
            if segment and is_sensitive(Path(segment)):
                return (
                    f"ERROR: {item!r} names a credential file. It cannot be read "
                    f"through the API any more than through read_file — a read "
                    f"means the contents reach a model provider, and the token "
                    f"here reaches the user's private repositories. Ask the user "
                    f"what is configured there instead."
                )
    return ""


def _reject_argv(args, program: str) -> str:
    """"ERROR: …" if these arguments are unusable, else "".

    The same `_argv` the handler calls, so the gate and the handler cannot
    disagree about what is runnable — which is the only way this is safe, since
    skipping the prompt for an invalid call is only correct while the call truly
    cannot execute.
    """
    result = _argv(args, program)
    return result if isinstance(result, str) else ""


def _git_subcommand(args) -> str:
    """The subcommand, or "" when there is not one."""
    for item in args or []:
        text = str(item)
        if not text.startswith("-"):
            return text
    return ""


def _git_needs_approval(args) -> bool:
    """Does this git invocation need a human? Defaults to yes.

    ── WHY THIS IS TWO FUNCTIONS ────────────────────────────────────────────
    It was one, as a chain of short-circuits — which meant the *order* of the
    branches was part of the policy, and nothing said so. Adding
    ``stash list`` as a read put a special case above the patch-flag gate, so
    ``git stash list -p`` dumped a stashed diff for free. Three defects in this
    one classifier had that shape: ``-p`` protecting ``cat-file`` only by
    coincidence, ``stash`` being the one lister that does not list, and then a
    special case placed ahead of a general guard while explaining why it could
    not go through the general path.

    So the unconditional refusals live here and the allowlist lives in
    ``_git_is_read``, which can only ever answer "is this one of the known
    reads". A branch added to the allowlist *cannot* be placed above a gate,
    because the gate is not in that function. Order stops being load-bearing.
    """
    items = [str(item) for item in (args or [])]
    subcommand = _git_subcommand(items)
    if not subcommand:
        return True

    flags = {item for item in items if item.startswith("-")}
    rest = [item for item in items if not item.startswith("-")][1:]

    # Unconditional: a report that has become a file dump. `log -p`,
    # `show --raw`, `stash list -p`. Nothing in the allowlist can undo this,
    # which is the whole reason it is not in the allowlist's function.
    if flags & _GIT_PATCH_FLAGS:
        return True

    return not _git_is_read(subcommand, flags, rest)


def _git_is_read(subcommand: str, flags: set[str], rest: list[str]) -> bool:
    """Is this one of the enumerated reads? Unknown is not a read.

    Only ever consulted *after* the unconditional gates in
    ``_git_needs_approval``, so nothing here can make a dangerous call free.
    """
    if (subcommand, rest[0] if rest else "") in _GIT_READ_PAIRS:
        # `stash list`: harmless, and unreachable through `_GIT_LISTERS` because
        # bare `stash` is a write.
        return True

    if subcommand in _GIT_PRINTS_CONTENT:
        # Free only when the output is reduced to names and counts, which is
        # what "what changed?" usually means and which reveals nothing.
        return bool(flags & _GIT_SUMMARY_FLAGS)

    if subcommand in _GIT_READS:
        return True

    if subcommand in _GIT_LISTERS:
        # Listing only: no positional arguments, and no flag we do not know to
        # be a listing flag. `branch` is free, `branch -D old` is not.
        unknown = {f for f in flags if f.split("=")[0] not in _GIT_LIST_FLAGS}
        return not (rest or unknown)

    return False


# ---------------------------------------------------------------------------
# Classifying a gh invocation
#
# The same allowlist discipline as git, with one difference that changes the
# stakes: a write here is **outward facing**. A bad commit is recoverable with
# git; a pull request opened or a comment posted under your name on somebody
# else's repository is not. So the gated set is shown with its body (see
# ``preview_call``), and ``cli`` refuses it outright when nobody is watching.
# ---------------------------------------------------------------------------

#: ``(noun, verb)`` pairs that only ever report.
_GH_READS = frozenset({
    ("pr", "view"), ("pr", "list"), ("pr", "diff"), ("pr", "checks"),
    ("pr", "status"),
    ("issue", "view"), ("issue", "list"), ("issue", "status"),
    ("repo", "view"), ("release", "view"), ("release", "list"),
    # `run list` only. `run view --log` dumps a CI log, and CI logs routinely
    # contain tokens that were masked in the web UI but not in the raw output —
    # a realistic secret path, and one no local denylist can reach.
    ("run", "list"), ("workflow", "list"),
    ("cache", "list"), ("label", "list"), ("search", "prs"),
    ("search", "issues"), ("search", "repos"),
})

#: Nouns refused outright, whatever the verb. Not classified — excluded.
#:
#: ``auth`` would print or change a credential through a model. The rest manage
#: secrets and keys, which is never something to do by proxy.
#:
#: ``api`` **was** here, refused on the grounds that it is "an arbitrary HTTP
#: client carrying the user's token". That argument does not survive contact:
#: every ``gh`` subcommand carries the token, ``pr view`` as much as ``api``, and
#: nothing here can read it either way — it is held by a CLI the user
#: authenticated themselves.
#:
#: The real distinction is narrower and does not justify a refusal. ``api`` is
#: the one subcommand whose read/write-ness is not in its name: everything else
#: is decided by ``(noun, verb)``, while ``gh api repos/x`` reads and
#: ``gh api -X POST repos/x/issues`` writes. That makes it harder to classify,
#: not impossible — see ``_gh_api_writes``. Refusing it also cost something
#: real: reading the files of a repository you do not have locally, which is
#: exactly what someone asking "summarise this repo" wants.
_GH_REFUSED = frozenset({
    "auth", "secret", "ssh-key", "gpg-key", "config", "alias",
    "codespace", "variable",
})

#: Flags that set the method explicitly. ``--method GET`` is a read even with
#: fields attached — gh then sends them as a query string.
_GH_METHOD_FLAGS = ("-X", "--method")

#: Flags that add request parameters. Their mere presence makes gh switch to
#: POST unless a method says otherwise, which its own ``--help`` states: "adding
#: request parameters will automatically switch the request method to POST". So
#: the absence of ``-X`` is **not** enough to call something a read.
_GH_FIELD_FLAGS = ("-f", "--raw-field", "-F", "--field", "--input")

#: Methods that only read.
_GH_READ_METHODS = frozenset({"get", "head"})

#: Endpoints that return bytes nobody can review before the call is made.
#:
#: ── THE QUESTION, ASKED OF ENDPOINTS THIS TIME ───────────────────────────────
#: ``_GIT_READS`` settled the rule locally: it is not "does this print file
#: contents" — ``blame`` does and is free, because the path can be checked —
#: it is **whether what gets printed can be reviewed before it runs**. A path
#: can be; an object id cannot.
#:
#: That rule was then applied to subcommand names and not to this endpoint
#: surface, so the same two-step came straight back over HTTP::
#:
#:     gh api repos/o/n/git/trees/HEAD?recursive=1   paths and blob shas
#:     gh api repos/o/n/git/blobs/<sha>              the file, by sha
#:
#: which is ``ls-tree`` → ``cat-file blob`` with a different transport. The
#: archives belong here for the same reason at larger scale — a tarball is the
#: whole repository including whatever was committed to it — and so do the raw
#: log endpoints, which are what ``gh run view --log`` wraps. Gating the wrapper
#: and leaving the endpoint free was the same mistake twice in one feature.
#:
#: ``git/trees`` stays free: names and shas, no content, exactly as ``ls-tree``
#: is free locally. The sha oracle is harmless once nothing free resolves a sha.
#:
#: Gated rather than refused, matching ``show`` and ``diff`` locally — a human
#: can reasonably say yes to any of these. Only credential *paths* are refused.
_GH_OPAQUE_CONTENT = (
    "/git/blobs/",      # a file addressed by sha, with no path to inspect
    "/tarball",         # the whole repository
    "/zipball",
    "/logs",            # CI output; what `gh run view --log` wraps
)


def _gh_api_returns_unreviewable_content(argv: list[str]) -> bool:
    """Does this endpoint hand back bytes nobody vouched for? See above."""
    for item in argv[1:]:
        if item.startswith("-"):
            continue
        path = "/" + item.split("?")[0].strip("/")
        if any(marker in path or path.endswith(marker.rstrip("/"))
               for marker in _GH_OPAQUE_CONTENT):
            return True
    return False

#: Flags a read may carry. An **allowlist**, for the reason the subcommands are
#: one: the first version detected dangerous spellings instead, and
#: ``gh api -XDELETE repos/owner/repo`` went through ungated — Go's ``pflag``
#: accepts an attached value, so ``-XDELETE`` is ``-X DELETE``, which matched
#: nothing and fell through to "read". That is a repository deleted with no
#: prompt, and ``-XPOST …/comments`` posting under the user's name past the one
#: gate that refuses outward posts even under ``--yes``.
#:
#: Detecting spellings means enumerating them correctly forever, against a CLI
#: whose parser we do not control: after ``-XDELETE`` comes ``-sXDELETE``
#: clustering, and then whatever pflag adds next. Allowlisting means an
#: unrecognised flag is simply gated, and the question stops being "did I think
#: of this spelling".
_GH_API_READ_FLAGS = frozenset({
    "-H", "--header", "--cache", "-i", "--include", "-q", "--jq",
    "-t", "--template", "--paginate", "--slurp", "--silent", "-s",
    "--verbose", "--hostname",
})


def _gh_api_writes(argv: list[str]) -> bool:
    """Would this ``gh api`` call change something? Defaults to yes.

    Three things decide it, and all three are in the argv — which is the whole
    reason this can be classified rather than refused:

    - an explicit ``--method``, which wins outright;
    - otherwise the presence of field flags, because gh silently switches to
      POST when parameters are added;
    - otherwise GET, which is a read.

    ``gh api graphql`` is the honest exception and stays a write. A GraphQL
    mutation lives in the *query body*, not in a flag, so no amount of argv
    inspection can tell a read from a write — and a classifier that cannot state
    its own rule is the thing this project keeps getting wrong.
    """
    words = [a for a in argv[1:] if not a.startswith("-")]
    if words and words[0].lower() == "graphql":
        return True

    if _gh_api_returns_unreviewable_content(argv):
        return True

    # The method first, in its own pass. It outranks everything else: with an
    # explicit `--method GET`, gh sends any fields as a query string, so they no
    # longer imply a write. A single pass got that wrong by returning on the
    # field before it had read the method.
    method = ""
    asked_for_a_method = False
    for index, item in enumerate(argv):
        flag, _, inline = item.partition("=")
        if flag in _GH_METHOD_FLAGS:
            asked_for_a_method = True
            method = inline or (argv[index + 1] if index + 1 < len(argv) else "")
        elif item.startswith("-X") and len(item) > 2:
            # `-XDELETE`, which pflag reads as `-X DELETE`. The field branch
            # below already knew about attached values and this one did not,
            # which is how a repository-deleting call classified as a read.
            asked_for_a_method = True
            method = item[2:].lstrip("=")

    if asked_for_a_method:
        # A method flag with nothing after it reads as "" — unknown intent, so
        # gated, rather than falling through to the flag allowlist and out as a
        # read.
        return method.strip().lower() not in _GH_READ_METHODS

    for item in argv:
        if not item.startswith("-"):
            continue
        flag = item.partition("=")[0]

        if flag in _GH_FIELD_FLAGS or (
            # The attached form, `-fkey=value`, which gh also accepts.
            item.startswith(("-f", "-F")) and len(item) > 2
        ):
            # Parameters with no method, so gh switches to POST by itself.
            return True

        if flag not in _GH_API_READ_FLAGS and not item.startswith("-X"):
            # Unrecognised. Gated, which is what makes the spelling of `-X`
            # stop mattering — this is an allowlist, so a flag nobody enumerated
            # is a flag nobody vouched for.
            return True

    return False


def _gh_pair(args) -> tuple[str, str]:
    """The ``(noun, verb)`` of a gh invocation, each "" when absent.

    Only the **leading** non-flag words, stopping at the first option. Taking
    them from wherever they sat let a flag's value fill the slot::

        ["--repo", "pr", "--template", "view", "issue", "create"]

    classified as ``pr view`` while the command was ``issue create``. Nothing
    executed, because ``gh`` parses the command before its flags and rejects
    that form — but that left the safety resting on an external CLI's argument
    parser rather than on this classifier, and the tool description actively
    teaches ``--repo``. Stopping at the first ``-`` makes the above ``("", "")``,
    which is gated, and leaves every real form working.
    """
    words: list[str] = []
    for item in args or []:
        text = str(item)
        if text.startswith("-"):
            break
        words.append(text)
        if len(words) == 2:
            break
    return (words[0] if words else ""), (words[1] if len(words) > 1 else "")


def _gh_needs_approval(args) -> bool:
    """Does this gh invocation need a human? Defaults to yes.

    One deliberate exception to "content-printing reads are gated", stated here
    rather than left as a silent inconsistency: ``gh pr diff`` prints the diff of
    the change you are being asked about. It is remote content the user already
    has in their browser, it is the *point* of reading a pull request, and gating
    it would make "review this PR" a prompt for every file. The local denylist
    was never going to reach a remote diff either way.

    ``gh run view`` is **not** in the read set, for the opposite reason — see
    ``_GH_READS``.
    """
    argv = [str(item) for item in (args or [])]
    noun, _verb = _gh_pair(argv)

    if noun == "api":
        # The one subcommand whose read/write-ness is in its flags rather than
        # its name, so it is classified separately rather than by the pair.
        return _gh_api_writes(argv)

    return _gh_pair(args) not in _GH_READS


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
            "output. Use it for tests, linters and build steps — not for "
            "editing files, which edit_file does more safely, and not for git, "
            "which the git tool does without a shell and without asking for "
            "read-only commands."
        ),
        parameters=_obj(
            command={"type": "string", "description": "The command to run"},
        ),
        needs_confirmation=True,
    ),
    "git": Tool(
        name="git",
        description=(
            "Run git in the project repository. Pass the arguments as a list, "
            "e.g. [\"log\", \"--oneline\", \"-5\"] or "
            "[\"commit\", \"-m\", \"Fix the parser\"]. Inspecting the "
            "repository — status, log, branch listings, diff --stat — runs "
            "straight away. Anything that changes it, and anything that prints "
            "file contents, is shown to the user for approval first."
        ),
        parameters=_obj(
            args={
                "type": "array",
                "items": {"type": "string"},
                "description": "Arguments to git, without the word 'git'",
            },
        ),
        needs_confirmation=True,
        gate=lambda arguments: _git_needs_approval(arguments.get("args")),
        reject=lambda arguments: _git_reject(arguments.get("args")),
    ),
    "github": Tool(
        name="github",
        description=(
            "Work with GitHub through the gh CLI. Pass the arguments as a "
            "list. Defaults to the repository you are in, and works on **any** "
            "repository with --repo, so this is how you look at a repo on "
            "GitHub — do not go searching the local disk for it:\n"
            "  [\"repo\", \"view\", \"owner/name\"]\n"
            "  [\"issue\", \"list\", \"--repo\", \"owner/name\"]\n"
            "  [\"pr\", \"view\", \"12\", \"--comments\"]\n"
            "If you do not know the owner, find it with "
            "[\"search\", \"repos\", \"<name>\"] rather than guessing one. "
            "Reading — repo view, pr view, pr list, pr diff, issue view, issue "
            "list, and search repos/prs/issues — runs straight away. Anything "
            "that creates, merges, closes or comments is shown to the user for "
            "approval first, because other people can see it.\n"
            "To read a repository's files without having it locally, use the "
            "REST API, which is also a free read:\n"
            "  [\"api\", \"repos/owner/name/readme\"]\n"
            "  [\"api\", \"repos/owner/name/contents/path/to/file\"]\n"
            "  [\"api\", \"repos/owner/name/git/trees/HEAD?recursive=1\"]\n"
            "Writing through the API — -X POST, or any -f/-F field, which makes "
            "gh POST by itself — is shown for approval. Clone only when you need "
            "many files or to run the code."
        ),
        parameters=_obj(
            args={
                "type": "array",
                "items": {"type": "string"},
                "description": "Arguments to gh, without the word 'gh'",
            },
        ),
        needs_confirmation=True,
        gate=lambda arguments: _gh_needs_approval(arguments.get("args")),
        reject=lambda arguments: _gh_reject(arguments.get("args")),
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
            # Bytes, decoded by `_decode_output` below rather than by
            # `text=True`. This runs *arbitrary* commands, so the codec depends
            # on which program wrote the output: `text=True` assumed the locale
            # and silently mangled `git log`, while forcing UTF-8 would break
            # every Windows-native command instead. See `_decode_output`.
            timeout=COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: command exceeded {COMMAND_TIMEOUT}s and was stopped"
    except OSError as exc:
        return f"ERROR: could not run it: {exc}"

    return _format_result(result)


def _argv(args, program: str) -> list[str] | str:
    """Model-supplied arguments as a clean argv, or an ERROR string.

    Says what it wants rather than guessing, the same policy ``write_file`` takes
    with a non-string body — except for two coercions that are unambiguous and
    which models get wrong constantly:

    - a leading literal ``"git"`` is **stripped**, because the model will send it
      about half the time and the alternative is an error the user has to read;
    - an ``int`` element becomes ``str(5)``, which has exactly one sensible
      reading, unlike joining a list of lines.

    A bare string is *not* split. ``shlex.split`` would re-guess the quoting the
    list exists to remove, and quoting is unresolvable across the three shells
    this project has to live with — see ``cli._shell_quote``.
    """
    if isinstance(args, str):
        return (
            "ERROR: `args` must be a list of separate arguments, not one "
            'string — send ["log", "--oneline"] rather than "log --oneline"'
        )
    if not isinstance(args, (list, tuple)):
        return (
            f"ERROR: `args` must be a list of strings, not "
            f"{type(args).__name__}"
        )

    argv = [str(item) for item in args if str(item) != ""]
    if argv and argv[0] == program:
        argv = argv[1:]
    if not argv:
        return f"ERROR: no arguments given — what should {program} do?"
    return argv


def _refuse_sensitive_argument(argv: list[str]) -> str:
    """"ERROR: ..." if any argument names a credential file, else "".

    ``git show HEAD:.env`` and ``git log -p -- .env`` both print a file the
    denylist refuses outright, so the denylist has to reach in here too. Checked
    on the *last path-looking segment* of each argument, because git spells the
    same file three ways: ``.env``, ``HEAD:.env``, ``-- .env``.

    This does not catch a bare ``git log -p`` over a repository whose history
    contains a ``.env`` — nothing short of scanning the output would, and
    scanning output cannot be done safely. That case is covered by ``-p`` being
    gated instead.
    """
    for item in argv:
        candidate = item.split(":")[-1]
        if not candidate or candidate.startswith("-"):
            continue
        if is_sensitive(Path(candidate)):
            return (
                f"ERROR: {item!r} names a credential file, which cannot be read "
                f"through git any more than through read_file — a read means the "
                f"contents reach a model provider. Ask the user what is "
                f"configured there instead."
            )
    return ""


def _refuse_force_push(argv: list[str]) -> str:
    """"ERROR: ..." for a force push, else "".

    Scoped to ``push`` deliberately: ``git clean -f``, ``branch -f``,
    ``checkout -f`` and ``tag -f`` all exist, are local, and are ordinary gated
    writes rather than refusals.

    ``--force-with-lease`` and ``--force-if-includes`` are **allowed** (still
    gated). They are the safe form — they fail rather than clobber when the
    remote has moved — so refusing them would push the model toward plain
    ``--force``, which is the opposite of the intent. Note that a substring test
    for ``"--force"`` catches the lease form; that is the over-eager detector
    this project keeps writing, pre-loaded, which is why the match is on whole
    tokens.

    A known gap, stated rather than half-caught: ``git push origin +main`` is a
    force push written as a refspec. Catching it means parsing every non-option
    argument as a refspec without reimplementing ``git check-ref-format``, and a
    detector that cannot be stated precisely is worse than a documented gap. The
    gate still shows the whole argv and a human still says yes.
    """
    if not argv or argv[0] != "push":
        return ""

    for item in argv[1:]:
        long_form = item.split("=")[0]
        if long_form in ("--force-with-lease", "--force-if-includes"):
            continue
        if item == "--force" or (
            item.startswith("-")
            and not item.startswith("--")
            and "f" in item[1:]          # the cluster form: -f, -qf, -fu
        ):
            return (
                "ERROR: force-pushing is refused by this tool and will be "
                "refused every time — it can destroy commits that exist only on "
                "the remote, which nothing here can undo. Use "
                "`--force-with-lease` if the history genuinely needs replacing, "
                "or stop and ask the user to run it themselves."
            )
    return ""


def _refuse_foreign_repo(root: Path) -> str:
    """"ERROR: ..." if the repository is not at or below the project root.

    ``cwd=root`` is not the same as "the repository is the root": git walks
    *upward* looking for a ``.git``, so running in a subdirectory of a checkout —
    or in a plain folder that happens to sit inside one — operates on the parent
    repository. ``git log -p`` there would print files from outside the project
    root, which is exactly the confinement ``within`` exists to enforce.

    This matters more than it looks because ``chat.run_agent`` calls
    ``agent.run`` with no ``root=``, so the root defaults to the working
    directory of the ``mj`` process — arbitrary under the tray or the scheduled
    trigger.

    Not being in a repository at all is **not** refused: ``git init`` and
    ``git status`` both have useful things to say there, and git's own message is
    clearer than anything written here.

    Cached per root, because this runs *before every git call* and otherwise
    makes ``git status`` cost two processes. Where the repository is cannot
    change under a single agent run without someone moving a ``.git`` directory
    mid-task, and the short timeout is because ``rev-parse`` is a local lookup —
    the full 120s belongs to the command the user actually asked for.
    """
    resolved = Path(root).resolve()
    if resolved in _TOPLEVEL_CACHE:
        return _TOPLEVEL_CACHE[resolved]

    verdict = _check_toplevel(resolved)
    _TOPLEVEL_CACHE[resolved] = verdict
    return verdict


#: Memoised ``_refuse_foreign_repo`` answers, keyed by resolved root.
_TOPLEVEL_CACHE: dict[Path, str] = {}

#: Long enough for a local object lookup, short enough not to double the cost of
#: every git call when something is wrong.
_REV_PARSE_TIMEOUT = 15


def _check_toplevel(root: Path) -> str:
    try:
        found = _run(
            ["git", "rev-parse", "--show-toplevel"], root, _REV_PARSE_TIMEOUT
        )
    except (ExecutableMissing, subprocess.TimeoutExpired, OSError):
        # Let the real call report it, so the message is about the real command.
        return ""

    if found.returncode != 0:
        return ""   # not a repository; git will say so more clearly

    toplevel = Path((found.stdout or "").strip())
    if not toplevel.name or within(Path(root).resolve(), toplevel):
        return ""

    return (
        f"ERROR: the git repository here is {toplevel}, which is outside the "
        f"project directory ({Path(root).resolve()}). Reading it would reach "
        f"files the project does not contain. Ask the user if this is what they "
        f"meant."
    )


def git(root: Path, args=None, **_ignored) -> str:
    """Run git in the project repository. See ``_git_needs_approval`` for gating."""
    # The same refusals the gate consulted, so the two cannot disagree about
    # what is refusable. See `_git_reject` for the leading-option rule and why
    # these run before anyone is asked for permission.
    refusal = _git_reject(args) or _refuse_foreign_repo(root)
    if refusal:
        return refusal

    argv = _argv(args, "git")

    try:
        result = _run(["git", *argv], root, COMMAND_TIMEOUT)
    except ExecutableMissing:
        return "ERROR: git is not on PATH. Ask the user to install it."
    except subprocess.TimeoutExpired:
        return f"ERROR: git exceeded {COMMAND_TIMEOUT}s and was stopped"
    except OSError as exc:
        return f"ERROR: could not run git: {exc}"

    formatted = _format_result(result)
    if was_truncated(formatted):
        # `_truncate`'s own advice — "a line range, a tighter pattern" — is
        # vocabulary from `read_file` and `grep`, and means nothing to a model
        # looking at a diff. 8,000 characters is roughly a hundred lines, so this
        # fires constantly on `diff` and `log -p`; naming the git-shaped way out
        # is the difference between adapting and guessing.
        formatted += (
            "\n(For git, narrow it with --stat, `-- <path>` to one file, "
            "or -n to fewer commits.)"
        )
    return formatted


def github(root: Path, args=None, **_ignored) -> str:
    """Work with GitHub through ``gh``. See ``_gh_needs_approval`` for gating."""
    refusal = _gh_reject(args)
    if refusal:
        return refusal

    argv = _argv(args, "gh")

    try:
        result = _run(["gh", *argv], root, NETWORK_TIMEOUT)
    except ExecutableMissing:
        # Deliberately distinct from "not logged in". One message for both sends
        # the model round in circles trying to fix the wrong thing — the lesson
        # `read_file`'s per-failure messages were written to record.
        return (
            "ERROR: the gh CLI is not on PATH. Ask the user to install it from "
            "cli.github.com — nothing here can reach GitHub without it."
        )
    except subprocess.TimeoutExpired:
        # The one error in this project that is **not** safe to retry. Every
        # other ERROR string means "that did not happen, try again"; a timed-out
        # `pr create` may well have succeeded, and a retry opens a second pull
        # request. Say so, because nothing else in the vocabulary does.
        return (
            f"ERROR: gh did not answer within {NETWORK_TIMEOUT}s, and the "
            f"outcome is UNKNOWN — if this was creating or commenting on "
            f"something, it may have succeeded. Do not retry it. Check with "
            f"`pr view` or `issue view`, or tell the user to look."
        )
    except OSError as exc:
        return f"ERROR: could not run gh: {exc}"

    formatted = _format_result(result)
    if result.returncode != 0 and "gh auth login" in formatted:
        formatted += (
            "\n(gh is installed but not authenticated — the user needs to run "
            "`gh auth login`. Nothing here can do that for them.)"
        )
    return formatted


HANDLERS = {
    "read_file": read_file,
    "list_files": list_files,
    "grep": grep,
    "write_file": write_file,
    "edit_file": edit_file,
    "run_command": run_command,
    "git": git,
    "github": github,
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


# ---------------------------------------------------------------------------
# Previewing a call, for the confirmation gate
#
# Lives here rather than in ``cli`` because it is a property of the *call*, not
# of the terminal. It used to be a second ``name ==`` chain inside
# ``confirm_action``, which meant every new tool needed branches in two files
# and in two different modules — and the rule a preview exists to serve belongs
# next to the tools it describes: **a gate that hides what it is gating is worse
# than no gate, because it manufactures confidence.**
#
# ``cli`` still adds the indentation. Nothing here decides what a terminal looks
# like.
# ---------------------------------------------------------------------------

#: Lines of a file or a posted body to show before eliding. One number, where
#: there used to be two that disagreed for no recorded reason — ``write_file``
#: had an inline 12 and an edit had 8.
PREVIEW_LINES = 12


def _elide(lines: list[str], limit: int = PREVIEW_LINES, marker: str = "| ") -> list[str]:
    """At most ``limit`` lines, and always say when something was left out."""
    shown = [f"{marker}{line}" for line in lines[:limit]]
    if len(lines) > limit:
        shown.append(f"{marker}… {len(lines) - limit} more line(s)")
    return shown


def _preview_write(arguments: dict) -> list[str]:
    return _elide(as_text(arguments.get("content")).splitlines())


def _preview_edit(arguments: dict) -> list[str]:
    return _edit_preview(
        as_text(arguments.get("old")), as_text(arguments.get("new"))
    )


#: Flags whose value is prose a human needs to read before it is published.
_GH_BODY_FLAGS = ("--body", "-b", "--title", "-t", "--comment")


def _preview_github(arguments: dict) -> list[str]:
    """Show the text a gh call would publish, before it publishes it.

    The same rule ``write_file`` gets, for a sharper reason. A file you can fix;
    a comment posted under your name on somebody else's pull request you cannot
    un-post. Approving text you have not seen is not approval.
    """
    argv = _argv(arguments.get("args"), "gh")
    if isinstance(argv, str):
        return []

    lines: list[str] = []
    for index, item in enumerate(argv):
        flag, _, inline = item.partition("=")
        if flag not in _GH_BODY_FLAGS:
            continue
        value = inline or (argv[index + 1] if index + 1 < len(argv) else "")
        if not value:
            continue
        lines.append(f"{flag}:")
        lines.extend(_elide(value.splitlines()))
    return lines


#: name -> what else to show at the gate. Absent means the one-liner is enough.
_PREVIEWS: dict[str, Callable[[dict], list[str]]] = {
    "write_file": _preview_write,
    "edit_file": _preview_edit,
    "github": _preview_github,
}


def preview_call(name: str, arguments: dict) -> list[str]:
    """Extra lines for the confirmation prompt, beyond ``describe_call``.

    Never raises. This runs *before* the handler, on arguments a model supplied
    and nothing has validated — the hazard ``as_text`` was written for, where an
    unchecked ``.splitlines()`` could take down a run over a preview. A gate that
    crashes is worse than one that shows less.
    """
    builder = _PREVIEWS.get(name)
    if builder is None:
        return []
    try:
        return builder(arguments)
    except Exception:  # pragma: no cover - defensive, and deliberately broad
        return ["(could not preview these arguments)"]


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
    if name in ("git", "github"):
        # The normalised argv, so what is shown is what will run — a stripped
        # leading "git" or a coerced number must not make the two differ.
        program = "git" if name == "git" else "gh"
        argv = _argv(arguments.get("args"), program)
        if isinstance(argv, str):
            return f"{program} (bad arguments)"
        return f"run: {program} " + " ".join(
            _quote_for_display(item) for item in argv
        )
    return f"{name}({', '.join(f'{k}={v!r}' for k, v in arguments.items())})"


def _quote_for_display(item: str) -> str:
    """Quote an argv element only when it would otherwise read as two.

    Display only — nothing here is handed to a shell, because these tools do not
    use one. A commit message has to look like one argument at the gate, or
    `commit -m Fix the parser` reads as four files.
    """
    return f'"{item}"' if (not item or " " in item or '"' in item) else item
