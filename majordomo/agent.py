"""The agent loop — the one place a model decides what happens next.

Everything else in Majordomo is a **fixed pipeline**: the code chooses every
step and the model only produces text. Workers fetch, the router sizes, the
coordinator fuses, in that order, every time. This is different in kind — the
model picks the next action from a tool set and keeps going until it is done.

    ask the model
      ├─ it wants tools  → run them (asking first where it matters) → repeat
      └─ it answers      → stop

That is the whole loop. Two things keep it from being dangerous:

**A turn cap.** A confused model will happily read the same file forever. The
cap is not a performance concern, it is the difference between a wrong answer
and an unbounded one.

**A gate on anything that changes the world.** Reads run freely; writes and
commands are shown and confirmed. A declined action returns a *result* saying
so, not an abort — the model can then try something else, which is usually what
you want. Aborting on a "no" teaches nothing and loses the work so far.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from majordomo import tools
from majordomo.config import Config

#: How many model calls one task may take before we stop and report.
MAX_TURNS = 24

SYSTEM = (
    "You are Majordomo's coding agent, working in a project directory on the "
    "user's machine.\n\n"
    "Work in small steps and use the tools rather than guessing. Read a file "
    "before you edit it, so the edit matches what is actually there. Prefer "
    "edit_file over write_file for an existing file — a full rewrite discards "
    "anything you did not know was in it.\n\n"
    "Use the git tool for the repository you are in, and the github tool for "
    "anything on GitHub — including a repository you do not have locally, which "
    "it reaches with --repo owner/name. Do not search the disk for a repo that "
    "lives on GitHub, and do not fall back to run_command for git: it has to "
    "ask permission even for `git status`. Pass their arguments as a list: "
    "[\"log\", \"--oneline\", \"-5\"].\n\n"
    "Do not guess an identifier you were not given. If you need a repository's "
    "owner, or an issue number, find it with the github tool — "
    "[\"search\", \"repos\", \"<name>\"] — or stop and ask. A plausible guess "
    "that turns out to be somebody else's repository wastes the turn and "
    "reports on the wrong thing.\n\n"
    "Three things can happen when you call a tool, and they need different "
    "responses:\n"
    "- It runs. Reading is free — files, listings, searches, git status, git "
    "log, diff --stat, branch listings, and reading pull requests and issues. "
    "Batch these; nobody is interrupted by them.\n"
    "- It is shown to the user for approval first. Writes, commands, commits, "
    "pushes, and anything posted to GitHub, along with git commands that print "
    "file contents. If the user declines one, do not try the same thing again — "
    "find another way or explain why you cannot.\n"
    "- It is refused outright, and the result says so. Credential files, "
    "force-pushing, `gh auth`. A refusal will not change on a second attempt, "
    "so do not rephrase it: stop, and tell the user what you were trying to do "
    "and why it needs them.\n\n"
    "A tool result beginning with ERROR is information, not a dead end. Read it "
    "and adjust.\n\n"
    "When the task is done, say what you changed in a sentence or two. Do not "
    "list every file you looked at."
)


@dataclass
class Step:
    """One tool call and what came of it. The record of what actually happened."""

    name: str
    arguments: dict
    result: str
    approved: bool = True
    #: The gate fired for this call — the predicate said it changes the world.
    #:
    #: Recorded rather than re-derived. ``approved`` cannot stand in for it:
    #: that defaults True for calls nobody was asked about, so every ``read_file``
    #: would read as a change. And asking ``tools`` again afterwards would re-run
    #: a predicate on arguments that may be malformed, outside the ``try`` that
    #: protected it the first time — and has no answer at all for the
    #: unknown-tool and ``__malformed__`` steps, which have no ``Tool`` behind
    #: them.
    gated: bool = False


@dataclass
class Outcome:
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    #: Set when the loop stopped for a reason other than the model finishing.
    stopped_because: str = ""

    @property
    def changed_anything(self) -> bool:
        """Did anything authorised to change the world actually run?

        Note the wording. A gated call that ran and *failed* — ``git commit``
        with nothing staged — still counts, because this reads the gate's
        decision rather than the outcome. Parsing exit codes to sharpen that
        would mean a detector per tool, which is how this project has repeatedly
        got itself into trouble.

        Derived from ``Step.gated`` rather than a list of tool names. The list
        was a second, independent encoding of write-ness, and it would have
        reported "nothing changed" after a ``git commit`` — a rule stated
        correctly in one place and not applied to its sibling, which is the
        defect shape this codebase keeps producing. One predicate decides, once,
        and the answer is recorded.
        """
        return any(step.gated and step.approved for step in self.steps)


def last_result(outcome) -> str:
    """What the agent found before it stopped, if it never got to say it.

    An agent that runs a command and then loses its next model call still *has*
    the output — it is sitting in the last step. Printing only ``answer`` threw
    that away and reported a failure instead, which is backwards: the tool had
    done the work and the summary was the only part missing.

    Only the last step, and only its result. Earlier steps were inputs to a plan
    that never finished; the newest one is what the task was actually reaching
    for. Nothing is inferred about whether it *is* the answer — it is labelled
    as what it is, so you can judge.
    """
    for step in reversed(outcome.steps):
        if step.approved and step.result and not step.result.startswith("ERROR"):
            return step.result
    return ""


def always_allow(_name: str, _arguments: dict) -> bool:
    """A confirmer that approves everything. For tests, and read-only runs."""
    return True


def run(
    task: str,
    config: Config,
    root: Path | str = ".",
    confirm=always_allow,
    write=print,
    max_turns: int = MAX_TURNS,
) -> Outcome:
    """Work on ``task`` until the model is done, the cap is hit, or it fails.

    Args:
        confirm: called as ``confirm(name, arguments)`` before any tool marked
            ``needs_confirmation``. Returning False declines that one call.
        write:   progress output. Every tool call is announced *before* it runs,
            so a long task is legible while it happens rather than afterwards.

    Never raises on a model failure — an ``LLMError`` ends the loop with what
    was done so far recorded, because a half-finished task you can see beats an
    exception that discards the trail.
    """
    from majordomo.llm import LLMError, complete_with_tools

    project = Path(root).resolve()
    # Falls back to the chat model, which is what this used before the roles
    # were split — so an unset agent_model behaves exactly as it always did.
    model = config.brain.agent_model or config.brain.chat_model
    schemas = tools.schemas()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Project directory: {project}\n\n{task}"},
    ]
    outcome = Outcome()

    for turn in range(max_turns):
        try:
            reply = complete_with_tools(messages, config.brain, model, schemas)
        except LLMError as exc:
            outcome.stopped_because = f"the model call failed: {exc}"
            return outcome

        if not reply.tool_calls:
            outcome.answer = reply.text.strip()
            return outcome

        # Any prose alongside tool calls is the model narrating its plan. Worth
        # showing — it is the only window into why it is doing what it does.
        if reply.text.strip():
            write(reply.text.strip())

        messages.append(reply.raw)

        for call in reply.tool_calls:
            step = _run_one(call, project, confirm, write)
            outcome.steps.append(step)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": step.result,
                }
            )

    outcome.stopped_because = (
        f"reached the {max_turns}-step limit without finishing — the task may "
        f"be too large, or it may be going in circles"
    )
    return outcome


def _run_one(call, project: Path, confirm, write) -> Step:
    """Execute one tool call, gating it if it changes anything."""
    tool = tools.TOOLS.get(call.name)
    if tool is None:
        # Not fatal: hand it back so the model can pick a real one.
        return Step(
            call.name,
            call.arguments,
            f"ERROR: no tool called {call.name!r}. Available: "
            f"{', '.join(sorted(tools.TOOLS))}",
        )

    if "__malformed__" in call.arguments:
        return Step(
            call.name,
            call.arguments,
            "ERROR: the arguments were not valid JSON. Send them again as a "
            "JSON object.",
        )

    # Before the gate: a call that cannot run must not interrupt anyone to ask
    # about it. A real session showed `git (bad arguments)` / `allow this?` /
    # `git (bad arguments)` — a prompt approving nothing, spending the only
    # thing the gate has, which is being worth reading.
    unusable = tool.unusable(call.arguments)
    if unusable:
        return Step(call.name, call.arguments, unusable)

    described = tools.describe_call(call.name, call.arguments)

    # Asked once, of this specific call, and the answer carried into the Step.
    # Some tools are a read or a write depending on their arguments — `git log`
    # against `git commit` — so this is a question about the call, not the tool.
    gated = tool.requires_approval(call.arguments)

    if gated:
        if not confirm(call.name, call.arguments):
            # A refusal is a result, not an abort. The model can adapt; ending
            # the run would throw away everything done so far.
            return Step(
                call.name,
                call.arguments,
                "The user declined this action. Do not retry it — find another "
                "way, or explain why you cannot.",
                approved=False,
                gated=True,
            )
    # Announced once the gate is passed, whichever branch got us here. This
    # used to sit in an `else`, so harmless reads were announced and approved
    # writes and shell commands were not — under `--yes` the agent rewrote files
    # and ran commands in complete silence, the exact inverse of the intent.
    write(f"  · {described}")

    handler = tools.HANDLERS[call.name]
    try:
        result = handler(project, **call.arguments)
    except TypeError as exc:
        # Wrong or missing arguments for the tool's signature.
        result = f"ERROR: bad arguments for {call.name}: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        result = f"ERROR: {call.name} failed: {exc}"

    return Step(call.name, call.arguments, result, gated=gated)
