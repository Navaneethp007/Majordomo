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
    "Writes and commands are shown to the user for approval before they run. "
    "If one is declined, do not try the same thing again — either find another "
    "way or explain why you cannot.\n\n"
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


@dataclass
class Outcome:
    answer: str = ""
    steps: list[Step] = field(default_factory=list)
    #: Set when the loop stopped for a reason other than the model finishing.
    stopped_because: str = ""

    @property
    def changed_anything(self) -> bool:
        return any(
            step.approved and step.name in ("write_file", "edit_file", "run_command")
            for step in self.steps
        )


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

    described = tools.describe_call(call.name, call.arguments)

    if tool.needs_confirmation:
        if not confirm(call.name, call.arguments):
            # A refusal is a result, not an abort. The model can adapt; ending
            # the run would throw away everything done so far.
            return Step(
                call.name,
                call.arguments,
                "The user declined this action. Do not retry it — find another "
                "way, or explain why you cannot.",
                approved=False,
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

    return Step(call.name, call.arguments, result)
