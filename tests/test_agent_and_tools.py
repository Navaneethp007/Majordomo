"""Tests for the agent loop and its tools.

Most of these are about the safety model rather than the happy path: what the
agent is *refused*, what it must ask before doing, and what stops it going in
circles. Those are the parts where being wrong is expensive.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from majordomo import agent, tools
from majordomo import config as config_module
from majordomo.llm import LLMError, Reply, ToolCall

CFG = config_module.build(config_module.DEFAULTS)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "app.py").write_text('"""An app."""\nVALUE = 1\n', encoding="utf-8")
    (tmp_path / "notes.md").write_text("# Notes\nsomething\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.py").write_text("DEEP = True\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.pyc").write_text("x", encoding="utf-8")
    return tmp_path


def call(name, **arguments):
    return ToolCall(id="c1", name=name, arguments=arguments)


def replies(*items):
    """A complete_with_tools that returns each item in turn."""
    return mock.Mock(side_effect=list(items))


def tool_reply(*calls):
    return Reply(text="", tool_calls=list(calls), raw={"role": "assistant"})


def final(text):
    return Reply(text=text, tool_calls=[], raw={"role": "assistant", "content": text})


# ---------------------------------------------------------------------------
# Path confinement — the property that matters most
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "escape",
    [
        "../outside.txt",
        "../../../../etc/passwd",
        "sub/../../outside.txt",
        # Absolute on both: a leading slash is root-relative on Windows too, so
        # this leaves the project directory either way.
        "/etc/shadow",
    ],
)
def test_paths_outside_the_project_are_refused(project, escape):
    """Resolution happens before the check because that is the only order that
    works — traversal and symlinks both look ordinary until resolved."""
    with pytest.raises(tools.OutsideProject):
        tools.resolve(project, escape)


@pytest.mark.skipif(os.name != "nt", reason="a drive letter is only absolute on Windows")
def test_a_drive_letter_path_is_refused_on_windows(project):
    """Split out because what this asserts is "an absolute path escapes", and
    the spelling of *absolute* is platform-specific.

    `C:/Windows/...` leaves the project on Windows and is an ordinary relative
    directory called `C:` on Linux, where it stays inside and is correctly
    allowed. Parametrised together, the case claimed to be about confinement and
    was really about path syntax — the kind of thing that only shows up on the
    second platform.
    """
    with pytest.raises(tools.OutsideProject):
        tools.resolve(project, "C:/Windows/System32/config/SAM")


def test_paths_inside_the_project_resolve(project):
    assert tools.resolve(project, "app.py").name == "app.py"
    assert tools.resolve(project, "sub/deep.py").name == "deep.py"
    assert tools.resolve(project, "sub/../app.py").name == "app.py"


def test_every_tool_refuses_an_escaping_path(project):
    assert "ERROR" in tools.read_file(project, path="../secret")
    assert "ERROR" in tools.list_files(project, directory="../")
    assert "ERROR" in tools.grep(project, pattern="x", directory="../")
    assert "ERROR" in tools.write_file(project, path="../evil.txt", content="x")
    assert "ERROR" in tools.edit_file(project, path="../evil.txt", old="a", new="b")


# ---------------------------------------------------------------------------
# The read tools
# ---------------------------------------------------------------------------


def test_read_file_numbers_lines(project):
    out = tools.read_file(project, path="app.py")
    assert "1  " in out and "An app." in out


def test_read_file_reports_a_miss_rather_than_raising(project):
    assert "ERROR: no file" in tools.read_file(project, path="nope.py")


def test_list_files_is_relative_and_skips_noise(project):
    out = tools.list_files(project, directory=".")
    assert "app.py" in out and "sub/deep.py" in out
    assert "__pycache__" not in out          # never worth the model's context


def test_list_files_honours_a_pattern(project):
    out = tools.list_files(project, directory=".", pattern="*.md")
    assert "notes.md" in out and "app.py" not in out


def test_grep_reports_file_and_line(project):
    out = tools.grep(project, pattern="VALUE")
    assert "app.py:2:" in out


def test_grep_reports_a_bad_regex_rather_than_raising(project):
    assert "ERROR: bad regular expression" in tools.grep(project, pattern="[unclosed")


def test_a_long_result_is_truncated_and_says_so(project):
    (project / "big.txt").write_text("x" * 40_000, encoding="utf-8")
    out = tools.read_file(project, path="big.txt")
    assert len(out) < 40_000
    assert "truncated" in out


# ---------------------------------------------------------------------------
# The write tools
# ---------------------------------------------------------------------------


def test_write_file_creates_and_reports(project):
    result = tools.write_file(project, path="new.txt", content="hello")
    assert "Created" in result
    assert (project / "new.txt").read_text(encoding="utf-8") == "hello"


def test_edit_file_replaces_exactly_once(project):
    assert "Edited" in tools.edit_file(project, path="app.py", old="VALUE = 1", new="VALUE = 2")
    assert "VALUE = 2" in (project / "app.py").read_text(encoding="utf-8")


def test_an_ambiguous_edit_is_refused(project):
    """Replacing the first of several is how an edit silently lands on the
    wrong line."""
    (project / "twice.py").write_text("x = 1\nx = 1\n", encoding="utf-8")
    result = tools.edit_file(project, path="twice.py", old="x = 1", new="x = 2")

    assert "appears 2 times" in result
    assert (project / "twice.py").read_text(encoding="utf-8") == "x = 1\nx = 1\n"


def test_an_edit_that_matches_nothing_is_refused(project):
    assert "does not appear" in tools.edit_file(
        project, path="app.py", old="NOT THERE", new="x"
    )


def test_run_command_returns_output_and_exit_code(project):
    out = tools.run_command(project, command=f'"{sys.executable}" -c "print(chr(104)+chr(105))"' )
    assert "exit code 0" in out and "hi" in out


def test_a_failing_command_reports_rather_than_raising(project):
    out = tools.run_command(project, command=f'"{sys.executable}" -c "import sys; sys.exit(3)"' )
    assert "exit code 3" in out


# ---------------------------------------------------------------------------
# The confirmation gate
# ---------------------------------------------------------------------------


def test_reads_are_not_gated(project):
    asked = []

    def confirm(name, args):
        asked.append(name)
        return True

    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(tool_reply(call("read_file", path="app.py")), final("done")),
    ):
        agent.run("look", CFG, root=project, confirm=confirm, write=lambda *_: None)

    assert asked == []          # nothing to approve for a read


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("write_file", {"path": "new.txt", "content": "x"}),
        ("edit_file", {"path": "app.py", "old": "VALUE = 1", "new": "VALUE = 2"}),
        ("run_command", {"command": "echo hi"}),
    ],
)
def test_anything_that_changes_the_world_is_gated(project, name, arguments):
    asked = []

    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(tool_reply(call(name, **arguments)), final("done")),
    ):
        agent.run(
            "change it",
            CFG,
            root=project,
            confirm=lambda n, a: asked.append(n) or True,
            write=lambda *_: None,
        )

    assert asked == [name]


def test_a_declined_write_does_not_happen(project):
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(
            tool_reply(call("write_file", path="new.txt", content="x")),
            final("understood"),
        ),
    ):
        outcome = agent.run(
            "write it",
            CFG,
            root=project,
            confirm=lambda n, a: False,
            write=lambda *_: None,
        )

    assert not (project / "new.txt").exists()
    assert outcome.steps[0].approved is False
    assert outcome.changed_anything is False


def test_a_decline_returns_a_result_rather_than_aborting(project):
    """The model can then adapt. Aborting would throw away the work so far."""
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(
            tool_reply(call("write_file", path="new.txt", content="x")),
            final("I will not retry that."),
        ),
    ) as called:
        outcome = agent.run(
            "write it", CFG, root=project, confirm=lambda n, a: False, write=lambda *_: None
        )

    assert called.call_count == 2                       # the loop continued
    assert "declined" in outcome.steps[0].result
    assert outcome.answer == "I will not retry that."


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def test_the_loop_runs_tools_then_answers(project):
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(
            tool_reply(call("list_files", directory=".")),
            tool_reply(call("read_file", path="app.py")),
            final("VALUE is 1."),
        ),
    ):
        outcome = agent.run("what is VALUE", CFG, root=project, write=lambda *_: None)

    assert [s.name for s in outcome.steps] == ["list_files", "read_file"]
    assert outcome.answer == "VALUE is 1."
    assert outcome.stopped_because == ""


def test_the_turn_cap_stops_a_model_going_in_circles(project):
    """Not a performance concern — the difference between a wrong answer and an
    unbounded one."""
    forever = mock.Mock(return_value=tool_reply(call("read_file", path="app.py")))

    with mock.patch("majordomo.llm.complete_with_tools", forever):
        outcome = agent.run(
            "loop", CFG, root=project, write=lambda *_: None, max_turns=4
        )

    assert forever.call_count == 4
    assert "4-step limit" in outcome.stopped_because


def test_a_model_failure_keeps_what_was_done(project):
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(tool_reply(call("read_file", path="app.py")), LLMError("503")),
    ):
        outcome = agent.run("read it", CFG, root=project, write=lambda *_: None)

    assert len(outcome.steps) == 1              # the trail survives
    assert "503" in outcome.stopped_because


def test_an_unknown_tool_is_reported_to_the_model(project):
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(tool_reply(call("delete_everything", path="/")), final("ok")),
    ):
        outcome = agent.run("do it", CFG, root=project, write=lambda *_: None)

    assert "no tool called" in outcome.steps[0].result
    assert "read_file" in outcome.steps[0].result      # says what does exist


def test_malformed_arguments_are_reported_not_executed(project):
    bad = ToolCall(id="c1", name="write_file", arguments={"__malformed__": "{oops"})

    with mock.patch(
        "majordomo.llm.complete_with_tools", replies(tool_reply(bad), final("ok"))
    ):
        outcome = agent.run("do it", CFG, root=project, write=lambda *_: None)

    assert "not valid JSON" in outcome.steps[0].result


def test_wrong_arguments_for_a_tool_are_reported(project):
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(tool_reply(call("read_file", wrong="x")), final("ok")),
    ):
        outcome = agent.run("do it", CFG, root=project, write=lambda *_: None)

    # read_file tolerates unknown kwargs, so it reports the missing path instead
    assert "ERROR" in outcome.steps[0].result


def test_narration_alongside_tool_calls_is_shown(project):
    """The only window into why it is doing what it does."""
    said = []
    narrated = Reply(
        text="I will look at app.py first.",
        tool_calls=[call("read_file", path="app.py")],
        raw={"role": "assistant"},
    )

    with mock.patch(
        "majordomo.llm.complete_with_tools", replies(narrated, final("done"))
    ):
        agent.run("look", CFG, root=project, write=said.append)

    assert any("look at app.py first" in line for line in said)


def test_changed_anything_tracks_only_approved_changes(project):
    """Reads the gate's recorded decision, not a list of tool names.

    The name list was a second, independent encoding of write-ness and would
    have said "nothing changed" after a `git commit` — so what is asserted here
    is the property, and `gated` is what carries it.
    """
    outcome = agent.Outcome(
        steps=[
            agent.Step("read_file", {}, "ok"),                       # never gated
            agent.Step("write_file", {}, "declined", approved=False, gated=True),
        ]
    )
    assert outcome.changed_anything is False

    outcome.steps.append(agent.Step("edit_file", {}, "Edited x", gated=True))
    assert outcome.changed_anything is True


def test_changed_anything_reads_the_gate_not_the_tool_name(project):
    """An approved ungated call is not a change, whatever it is called.

    The complement of the above: `approved` defaults True for calls nobody was
    asked about, so it cannot stand in for `gated`.
    """
    outcome = agent.Outcome(steps=[agent.Step("write_file", {}, "ok")])
    assert outcome.changed_anything is False   # gated defaults False


def test_an_approved_gated_call_counts_even_when_it_failed(project):
    """The property is "something authorised to change the world ran", not
    "the world changed" — a commit with nothing staged still counts."""
    outcome = agent.Outcome(
        steps=[agent.Step("run_command", {}, "exit code 1", gated=True)]
    )
    assert outcome.changed_anything is True


def test_the_loop_records_what_the_gate_decided(project):
    """End to end, rather than hand-built Steps: the flag must come from the
    real gate, or `changed_anything` is asserting about a field nobody sets."""
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(
            tool_reply(call("read_file", path="app.py")),
            tool_reply(call("write_file", path="new.txt", content="x")),
            final("done"),
        ),
    ):
        outcome = agent.run(
            "do it", CFG, root=project, confirm=agent.always_allow, write=lambda *_: None
        )

    assert [step.gated for step in outcome.steps] == [False, True]
    assert outcome.changed_anything is True


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def test_every_tool_has_a_handler():
    assert set(tools.TOOLS) == set(tools.HANDLERS)


def test_schemas_are_the_shape_the_api_wants():
    for schema in tools.schemas():
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["name"] and function["description"]
        assert function["parameters"]["type"] == "object"


def test_only_world_changing_tools_need_confirmation():
    """A statement about the *predicate*, not a fixed set.

    This used to assert the gated set was exactly
    {write_file, edit_file, run_command}. That was right while gating was a
    per-tool boolean; now some tools are gated by their arguments, so the
    closed-world version would break on every addition while checking nothing
    useful. What matters is unchanged: reads are free, and everything that can
    change the world is either always gated or gated by a predicate.
    """
    free = {name for name, tool in tools.TOOLS.items() if not tool.needs_confirmation}
    assert free == {"read_file", "list_files", "grep"}

    for name, tool in tools.TOOLS.items():
        if name in free:
            # Free means free for every call — no predicate can sneak a gate in.
            assert tool.gate is None, name
            assert tool.requires_approval({}) is False, name
        else:
            # Gated means a human is asked, either always or by the predicate.
            assert tool.needs_confirmation, name


def test_a_tool_with_no_gate_is_always_gated():
    """The three original write tools keep their all-or-nothing behaviour."""
    for name in ("write_file", "edit_file", "run_command"):
        tool = tools.TOOLS[name]
        assert tool.gate is None
        assert tool.requires_approval({}) is True
        assert tool.requires_approval({"anything": "at all"}) is True


def test_a_gate_cannot_ungate_a_tool_that_needs_no_confirmation():
    """`requires_approval` short-circuits on the flag, so a predicate on a read
    tool could never *add* a gate — and, more importantly, a truthy predicate
    could never be mistaken for "always gate". That inversion is why this is a
    method rather than a widened `needs_confirmation` field."""
    free = tools.Tool(
        name="x", description="d", parameters={}, gate=lambda _a: True
    )
    assert free.requires_approval({}) is False


def test_describe_call_shows_what_you_are_approving():
    assert "run: pytest" in tools.describe_call("run_command", {"command": "pytest"})
    assert "app.py" in tools.describe_call(
        "write_file", {"path": "app.py", "content": "x" * 50}
    )


# ---------------------------------------------------------------------------
# The credential denylist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        ".env.production",
        "config/.env",
        ".ssh/id_rsa",
        ".ssh/id_ed25519",
        ".aws/credentials",
        ".gnupg/secring.gpg",
        ".kube/config",
        "certs/server.pem",
        "keys/private.key",
        "app.p12",
        "deploy.pfx",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "secrets.json",
        "service-account.json",
        "secrets.yaml",
        "credentials.toml",
    ],
)
def test_credential_files_are_refused_even_inside_the_project(project, path):
    """Confinement is not enough when the root is a home directory — and a
    project's own .env is exactly as leakable as one in ~."""
    target = project / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("SECRET=hunter2", encoding="utf-8")

    with pytest.raises(tools.Sensitive):
        tools.resolve(project, path)

    result = tools.read_file(project, path=path)
    assert "ERROR" in result and "hunter2" not in result


@pytest.mark.parametrize(
    "path",
    ["app.py", "environment.py", "keyboard.py", "notes.md", "sub/deep.py", "envoy.txt"],
)
def test_ordinary_files_are_not_caught_by_the_denylist(project, path):
    """A denylist that eats environment.py or keyboard.py would be worse than
    none — you would turn it off."""
    target = project / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x = 1", encoding="utf-8")

    assert tools.resolve(project, path).name == Path(path).name


def test_a_credential_file_cannot_be_written_either(project):
    """Writing .env is how a well-meaning agent overwrites your real keys."""
    assert "ERROR" in tools.write_file(project, path=".env", content="KEY=stolen")
    assert not (project / ".env").exists()


def test_grep_does_not_leak_credentials_line_by_line(project):
    """grep prints matching *lines*, so refusing the directory is not enough."""
    (project / ".env").write_text("OPENROUTER_API_KEY=sk-secret123\n", encoding="utf-8")
    (project / "readme.md").write_text("set OPENROUTER_API_KEY first\n", encoding="utf-8")

    out = tools.grep(project, pattern="OPENROUTER_API_KEY")

    assert "readme.md" in out            # the ordinary mention still found
    assert "sk-secret123" not in out     # the value never printed
    assert ".env" not in out


def test_the_refusal_tells_the_model_what_to_do_instead(project):
    (project / ".env").write_text("KEY=x", encoding="utf-8")
    result = tools.read_file(project, path=".env")

    # Otherwise it hunts for another way in — run_command would find one.
    assert "ask the user" in result


def test_sensitive_and_outside_are_both_caught_as_refused():
    assert issubclass(tools.Sensitive, tools.Refused)
    assert issubclass(tools.OutsideProject, tools.Refused)


# ---------------------------------------------------------------------------
# Arguments that are not the type the schema promised
# ---------------------------------------------------------------------------


def test_describing_a_call_never_raises_on_a_list(project):
    """This runs *before* the tool handler gets a chance to refuse, so it has
    to survive anything the model sends."""
    described = tools.describe_call("write_file", {"path": "a.py", "content": ["x", "y"]})
    assert "a.py" in described

    described = tools.describe_call("edit_file", {"path": "a.py", "old": ["x"], "new": "y"})
    assert "a.py" in described


def test_a_list_content_is_refused_with_a_usable_message(project):
    result = tools.write_file(project, path="new.py", content=["line one", "line two"])

    assert "must be a string" in result
    assert "list" in result
    assert not (project / "new.py").exists()


@pytest.mark.parametrize("field", ["old", "new"])
def test_a_list_edit_argument_is_refused(project, field):
    arguments = {"path": "app.py", "old": "VALUE = 1", "new": "VALUE = 2"}
    arguments[field] = ["VALUE = 1"]

    result = tools.edit_file(project, **arguments)

    assert f"`{field}` must be a string" in result
    assert "VALUE = 1" in (project / "app.py").read_text(encoding="utf-8")


def test_a_malformed_argument_does_not_end_the_run(project):
    """The gate runs outside `_run_one`'s try, so an unchecked splitlines()
    took down the whole run and discarded its trail — over a preview."""
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(
            tool_reply(call("write_file", path="a.py", content=["one", "two"])),
            final("I will send it as a string."),
        ),
    ):
        outcome = agent.run(
            "write it",
            CFG,
            root=project,
            confirm=lambda n, a: True,
            write=lambda *_: None,
        )

    assert "must be a string" in outcome.steps[0].result
    assert outcome.answer == "I will send it as a string."


# ---------------------------------------------------------------------------
# Which model the agent uses
# ---------------------------------------------------------------------------


def model_used(config, project):
    with mock.patch(
        "majordomo.llm.complete_with_tools", replies(final("done"))
    ) as called:
        agent.run("task", config, root=project, write=lambda *_: None)
    return called.call_args[0][2]


def test_the_agent_uses_its_own_model(project):
    """Split from chat_model because the roles want different things: chat wants
    a model that talks like a colleague, the agent wants one that writes code
    and calls tools — and the agent's latency compounds over many turns."""
    config = config_module.build(
        {**config_module.DEFAULTS,
         "brain": {"chat_model": "talker", "agent_model": "coder"}}
    )
    assert model_used(config, project) == "coder"


def test_an_unset_agent_model_falls_back_to_chat(project):
    """So an existing config with no agent_model behaves exactly as before."""
    config = config_module.build(
        {**config_module.DEFAULTS,
         "brain": {"chat_model": "talker", "agent_model": ""}}
    )
    assert model_used(config, project) == "talker"


# ---------------------------------------------------------------------------
# Confinement is checked on what a walk yields, not just where it starts
# ---------------------------------------------------------------------------


@pytest.fixture
def with_neighbour(project):
    """A project with a secret sitting beside it, outside the root."""
    (project.parent / "PRIVATE.txt").write_text("SECRET_TOKEN=abc123", encoding="utf-8")
    return project


def test_a_glob_cannot_walk_out_of_the_project(with_neighbour):
    """`resolve` gates the directory argument; the model-supplied glob went
    straight to rglob, which walks upward happily. No symlink required, and
    list_files is ungated so nothing prompted."""
    out = tools.list_files(with_neighbour, directory=".", pattern="../*")

    assert "PRIVATE" not in out


def test_grep_cannot_print_lines_from_outside_the_project(with_neighbour):
    """The worse half: grep prints matching *lines*, so this shipped the file's
    contents to a model provider with no prompt at all."""
    out = tools.grep(with_neighbour, pattern="SECRET_TOKEN", glob="../*")

    assert "abc123" not in out
    assert "PRIVATE" not in out


def test_ordinary_globs_still_work(project):
    assert "sub/deep.py" in tools.list_files(project, directory=".", pattern="**/*.py")
    assert "app.py:2:" in tools.grep(project, pattern="VALUE", glob="*.py")


def test_within_accepts_inside_and_refuses_outside(project):
    assert tools.within(project, project / "app.py")
    assert tools.within(project, project / "sub" / "deep.py")
    assert not tools.within(project, project.parent / "PRIVATE.txt")
    assert not tools.within(project, project.parent)


# ---------------------------------------------------------------------------
# Approved actions are announced
# ---------------------------------------------------------------------------


def announced(project, name, arguments):
    said = []
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(tool_reply(call(name, **arguments)), final("done")),
    ):
        agent.run("go", CFG, root=project, confirm=agent.always_allow, write=said.append)
    return "\n".join(said)


def test_an_approved_write_is_announced(project):
    """The announcement sat in the `else` of the confirmation branch, so reads
    were announced and writes were not — under --yes the agent rewrote files in
    complete silence, the inverse of the intent."""
    assert "new.txt" in announced(project, "write_file", {"path": "new.txt", "content": "x"})


def test_an_approved_command_is_announced(project):
    assert "echo hi" in announced(project, "run_command", {"command": "echo hi"})


def test_an_approved_edit_is_announced(project):
    out = announced(
        project, "edit_file", {"path": "app.py", "old": "VALUE = 1", "new": "VALUE = 2"}
    )
    assert "app.py" in out


def test_reads_are_still_announced(project):
    assert "app.py" in announced(project, "read_file", {"path": "app.py"})


def test_a_declined_action_is_not_announced_as_if_it_ran(project):
    said = []
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(tool_reply(call("write_file", path="new.txt", content="x")), final("ok")),
    ):
        agent.run("go", CFG, root=project, confirm=lambda n, a: False, write=said.append)

    assert not any("new.txt" in line for line in said)


def test_a_list_command_is_refused_like_its_siblings(project):
    result = tools.run_command(project, command=["echo", "hi"])
    assert "must be a string" in result


# ---------------------------------------------------------------------------
# Work already done is not thrown away to report a failure
# ---------------------------------------------------------------------------


def test_the_last_result_survives_a_failed_model_call(project):
    """The exact shape of a real failure: the agent runs a command, gets its
    output, and then loses its next model call. The output is right there."""
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(
            tool_reply(call("run_command", command=f'"{sys.executable}" -c "print(42)"' )),
            LLMError("provider rate-limited"),
        ),
    ):
        outcome = agent.run(
            "count them", CFG, root=project,
            confirm=agent.always_allow, write=lambda *_: None,
        )

    assert outcome.answer == ""                       # it never got to summarise
    assert "42" in agent.last_result(outcome)         # but the answer exists


def test_an_error_result_is_not_offered_as_an_answer(project):
    """A refusal is not a finding."""
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(tool_reply(call("read_file", path="nope.py")), LLMError("down")),
    ):
        outcome = agent.run("read it", CFG, root=project, write=lambda *_: None)

    assert agent.last_result(outcome) == ""


def test_a_declined_step_is_not_offered_as_an_answer(project):
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(
            tool_reply(call("write_file", path="x.txt", content="x")),
            LLMError("down"),
        ),
    ):
        outcome = agent.run(
            "write it", CFG, root=project,
            confirm=lambda n, a: False, write=lambda *_: None,
        )

    assert agent.last_result(outcome) == ""


def test_the_newest_useful_result_wins(project):
    outcome = agent.Outcome(
        steps=[
            agent.Step("read_file", {}, "older content"),
            agent.Step("run_command", {}, "exit code 0\nstdout:\n412"),
        ]
    )
    assert "412" in agent.last_result(outcome)


def test_nothing_to_salvage_from_an_empty_run():
    assert agent.last_result(agent.Outcome()) == ""


# ---------------------------------------------------------------------------
# was_truncated — the fact, not an inference from length
# ---------------------------------------------------------------------------

def test_was_truncated_is_exact_at_the_boundary():
    """`MAX_RESULT_CHARS <= len(body)` was arithmetic about a side effect, and
    claimed a truncation that never happened for a body of exactly the limit."""
    exact = "x" * tools.MAX_RESULT_CHARS
    assert tools._truncate(exact) == exact
    assert tools.was_truncated(exact) is False

    over = "x" * (tools.MAX_RESULT_CHARS + 1)
    assert tools.was_truncated(tools._truncate(over)) is True

    assert tools.was_truncated("short") is False


def test_a_truncated_read_is_detectable_by_its_caller(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("y" * (tools.MAX_RESULT_CHARS * 2), encoding="utf-8")

    body = tools.read_file(tmp_path, "big.txt")
    assert tools.was_truncated(body)


# ---------------------------------------------------------------------------
# Decoding process output
#
# Measured rather than reasoned about, because the obvious answer is wrong. For
# one em dash on Windows:
#
#   git            emits e2 80 94   — UTF-8, whatever the console code page
#   a python child emits 97         — cp1252, from its own locale
#
# So forcing UTF-8 breaks native commands and keeping the locale codec mangles
# git. UTF-8 being self-validating is the way out: try the checkable codec
# first, fall back to the permissive one.
# ---------------------------------------------------------------------------

EM_DASH = "—"


def test_utf8_output_decodes_as_utf8():
    assert tools._decode_output(EM_DASH.encode("utf-8")) == EM_DASH


def test_locale_encoded_output_falls_back():
    """cp1252's em dash is a single byte and not valid UTF-8 at all, so the
    strict attempt fails cleanly and the fallback gets it right."""
    import locale

    raw = EM_DASH.encode(locale.getpreferredencoding(False))
    assert tools._decode_output(raw) == EM_DASH


def test_undecodable_output_never_raises():
    """A tool result must always be a string the model can read."""
    assert isinstance(tools._decode_output(b"\xff\xfe\x00broken"), str)


def test_run_command_keeps_a_native_commands_non_ascii(project):
    """This mojibaked before: `text=True` decoded with the locale codec, which
    was right here and wrong for git — and the suite had no case either way."""
    out = tools.run_command(project, f'"{sys.executable}" -c "print(chr(0x2014))"')
    assert EM_DASH in out


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_format_result_accepts_bytes_or_text(stream):
    """`_run` captures text, `run_command` captures bytes; one formatter."""
    for payload in (EM_DASH, EM_DASH.encode("utf-8")):
        done = subprocess.CompletedProcess(
            args=["x"],
            returncode=0,
            **{stream: payload, "stdout" if stream == "stderr" else "stderr": None},
        )
        assert EM_DASH in tools._format_result(done)


# ---------------------------------------------------------------------------
# The git tool
#
# Run against a real repository, because a mock of git tests a mock of git —
# and `run_command`'s tests and scaffold's already shell out, so this is the
# house style rather than a new risk.
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    """A real git repository, isolated from the developer's own git config.

    The isolation is not optional. Without it the suite inherits whatever is in
    ~/.gitconfig — aliases, hooks, `init.defaultBranch`, `core.autocrlf` — and,
    worst of all, `commit.gpgsign = true`, which blocks on a passphrase prompt
    and would hang the whole run with no indication why. Same reasoning as
    `isolated_home` in conftest; it was simply missing for git.
    """
    work = tmp_path / "work"
    work.mkdir()
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"),
        "GIT_CONFIG_SYSTEM": str(tmp_path / "gitconfig-system"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }

    def run(*argv):
        done = subprocess.run(
            ["git", *argv], cwd=work, env=env, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr
        return done

    run("init", "-q")
    run("config", "user.name", "Test")
    run("config", "user.email", "test@example.invalid")
    (work / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    run("add", "app.py")
    run("commit", "-qm", "Add app — with an em dash")

    # The tool shells out itself, so the env has to be in place for *it* too.
    for key, value in env.items():
        if key.startswith("GIT_"):
            os.environ[key] = value
    yield work
    for key in list(os.environ):
        if key.startswith("GIT_CONFIG") or key == "GIT_TERMINAL_PROMPT":
            del os.environ[key]


# --- classification --------------------------------------------------------


@pytest.mark.parametrize(
    "args,gated",
    [
        (["status"], False),
        (["log", "--oneline"], False),
        (["diff", "--stat"], False),
        (["diff", "--name-only"], False),
        (["branch"], False),
        (["branch", "-a"], False),
        (["remote", "-v"], False),
        (["rev-parse", "HEAD"], False),
        # Content-printing: as revealing as read_file, so it asks.
        (["diff"], True),
        (["show", "HEAD"], True),
        (["log", "-p"], True),
        (["log", "--patch"], True),
        # Writes.
        (["commit", "-m", "x"], True),
        (["add", "."], True),
        (["checkout", "main"], True),
        (["branch", "-D", "old"], True),
        (["tag", "v1"], True),
        (["push"], True),
        # Unknown and malformed default to asking.
        (["bisect", "start"], True),
        (["some-future-subcommand"], True),
        ([], True),
    ],
)
def test_git_gating_is_per_call(args, gated):
    assert tools.TOOLS["git"].requires_approval({"args": args}) is gated


def test_a_read_only_git_call_runs_without_asking(repo):
    """The whole point of the classification: no prompt for a look."""
    asked = []
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(tool_reply(call("git", args=["status", "--short"])), final("clean")),
    ):
        outcome = agent.run(
            "what is the state of the repo?",
            CFG,
            root=repo,
            confirm=lambda n, a: asked.append(n) or True,
            write=lambda *_: None,
        )

    assert asked == []                       # nothing was gated
    assert outcome.changed_anything is False
    assert "exit code 0" in outcome.steps[0].result


def test_a_git_commit_is_gated_and_counts_as_a_change(repo):
    (repo / "new.txt").write_text("x", encoding="utf-8")
    asked = []
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(
            tool_reply(call("git", args=["add", "new.txt"])),
            tool_reply(call("git", args=["commit", "-m", "Add new"])),
            final("committed"),
        ),
    ):
        outcome = agent.run(
            "commit it",
            CFG,
            root=repo,
            confirm=lambda n, a: asked.append(n) or True,
            write=lambda *_: None,
        )

    assert asked == ["git", "git"]
    assert outcome.changed_anything is True


# --- the refusals ----------------------------------------------------------


def test_git_cannot_print_a_credential_file(repo):
    """The hole this tool nearly opened, and the test that matters most.

    `read_file` refuses `.env` outright because a read means the contents reach
    a model provider and a leaked key cannot be un-leaked. `git show HEAD:.env`
    prints the same bytes. A free `git show` would have been a brand-new ungated
    path to exactly what the denylist exists to stop.
    """
    secret = repo / ".env"
    secret.write_text("OPENROUTER_API_KEY=sk-or-realkey\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", ".env"], cwd=repo, capture_output=True)
    subprocess.run(
        ["git", "commit", "-qm", "oops"], cwd=repo, capture_output=True
    )

    for args in (
        ["show", "HEAD:.env"],
        ["log", "-p", "--", ".env"],
        ["diff", "HEAD~1", "--", ".env"],
        ["cat-file", "-p", "HEAD:.env"],
    ):
        result = tools.git(repo, args=args)
        assert result.startswith("ERROR:"), args
        assert "sk-or-realkey" not in result, args
        assert "credential" in result, args


def test_the_credential_refusal_says_what_to_do_instead(repo):
    result = tools.git(repo, args=["show", "HEAD:.env"])
    assert "ask the user" in result.lower()


@pytest.mark.parametrize(
    "flag",
    ["-C", "--git-dir=/elsewhere/.git", "--work-tree=/elsewhere", "-c", "--exec-path=/x"],
)
def test_git_refuses_options_before_the_subcommand(repo, flag):
    """One rule instead of a list: nothing before the subcommand.

    `-C` and `--git-dir` redirect git out of the project entirely, and `-c` is
    arbitrary config injection — `git -c core.fsmonitor='sh -c …' status` is
    command execution through a subcommand classified as a *read*.
    """
    result = tools.git(repo, args=[flag, "status"])
    assert result.startswith("ERROR:")
    assert "before the subcommand" in result


def test_a_config_injection_cannot_ride_along_on_a_read(repo):
    result = tools.git(repo, args=["-c", "core.fsmonitor=echo pwned", "status"])
    assert result.startswith("ERROR:")


@pytest.mark.parametrize("force", ["--force", "-f", "-qf", "-fu"])
def test_force_pushing_is_refused(repo, force):
    result = tools.git(repo, args=["push", force, "origin", "main"])
    assert result.startswith("ERROR:")
    assert "every time" in result            # finality, or the model retries
    assert "--force-with-lease" in result    # and the alternative


@pytest.mark.parametrize(
    "lease", ["--force-with-lease", "--force-with-lease=main", "--force-if-includes"]
)
def test_the_lease_forms_are_allowed(repo, lease):
    """The safe form. Refusing it would push the model toward plain --force,
    which is the opposite of the intent — and a substring test for '--force'
    would have caught it, which is the over-eager detector pre-loaded."""
    assert not _refused(tools.git(repo, args=["push", lease, "origin", "main"]))
    assert tools.TOOLS["git"].requires_approval({"args": ["push", lease]}) is True


@pytest.mark.parametrize("args", [["clean", "-f"], ["branch", "-f", "x"], ["tag", "-f", "v1"]])
def test_local_f_flags_are_gated_not_refused(repo, args):
    """Scoped to push: -f means something harmless-ish and local elsewhere."""
    assert tools.TOOLS["git"].requires_approval({"args": args}) is True
    assert "refused" not in tools.git(repo, args=args)


def _refused(result: str) -> bool:
    """Our refusal, as opposed to git's own non-zero exit."""
    return result.startswith("ERROR:") and "refused" in result


# --- argument shape, decoding, and the shell ------------------------------


def test_git_argument_shape_errors_name_the_shape(repo):
    result = tools.git(repo, args="log --oneline")
    assert result.startswith("ERROR:")
    assert '["log", "--oneline"]' in result   # shows the shape it wants


def test_git_never_splits_a_string_itself(repo):
    """shlex.split would re-guess the quoting the list exists to remove."""
    assert "exit code" not in tools.git(repo, args="status")


def test_git_strips_a_leading_git(repo):
    """Models send it about half the time; erroring would be pedantry."""
    assert "exit code 0" in tools.git(repo, args=["git", "status", "--short"])


def test_git_coerces_a_number(repo):
    """str(5) has exactly one reading, unlike joining a list of lines."""
    assert "exit code 0" in tools.git(repo, args=["log", "-n", 1, "--oneline"])


def test_git_with_no_arguments_asks_what_to_do(repo):
    result = tools.git(repo, args=[])
    assert result.startswith("ERROR:")
    assert "what should git do" in result


def test_git_output_keeps_its_non_ascii(repo):
    """git emits UTF-8 whatever the console code page. Decoded with the locale
    codec — the old behaviour — this came back as three junk characters."""
    result = tools.git(repo, args=["log", "-1", "--format=%s"])
    assert "—" in result        # the em dash from the fixture's commit
    assert "�" not in result


def test_git_takes_no_shell(repo):
    """argv, not a command line: the chaining never reaches a shell."""
    result = tools.git(repo, args=["log", "&&", "echo", "pwned"])
    assert "pwned" not in result
    assert "exit code 0" not in result     # git itself rejects the pathspec


def test_git_is_announced_with_the_argv_that_will_run(repo):
    """The gate must show the normalised argv, or a stripped "git" and a
    coerced number make the display and the command differ."""
    described = tools.describe_call("git", {"args": ["git", "commit", "-m", "Fix the parser"]})
    assert described == 'run: git commit -m "Fix the parser"'


def test_a_repository_above_the_project_root_is_refused(tmp_path):
    """`cwd=root` is not "the repo is the root" — git walks upward. This is
    reachable because chat.run_agent passes no root at all, so it defaults to
    the mj process's working directory."""
    outer = tmp_path / "outer"
    (outer / "inner").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=outer, capture_output=True)

    result = tools.git(outer / "inner", args=["status"])
    assert result.startswith("ERROR:")
    assert "outside the project directory" in result


def test_not_being_a_repository_is_left_to_git(tmp_path):
    """git's own message is clearer than anything we would write, and `git init`
    has a real use here."""
    result = tools.git(tmp_path, args=["status"])
    assert not result.startswith("ERROR:")
    assert "not a git repository" in result.lower()


# ---------------------------------------------------------------------------
# Previewing a call at the gate
#
# These moved here with `_edit_preview` when it left `cli` — a preview is a
# property of the call, not of the terminal, and `cli` kept only the
# indentation. Moved rather than aliased: an alias is the drift this project
# keeps warning about.
# ---------------------------------------------------------------------------


def test_an_append_shows_the_appended_lines():
    """The bug this function exists for: showing the first N lines of each side
    made an append look like a no-op, because both sides start the same."""
    old = "def add(a, b):\n    return a + b\n"
    new = old + "\ndef subtract(a, b):\n    return a - b\n"

    preview = "\n".join(tools._edit_preview(old, new))

    assert "+ def subtract(a, b):" in preview
    assert "unchanged line(s)" in preview          # the shared prefix, summarised
    assert "- def add(a, b):" not in preview       # not re-shown as a removal


def test_a_change_in_the_middle_shows_both_sides():
    old = "a\nb\nc\n"
    new = "a\nB\nc\n"

    preview = tools._edit_preview(old, new)

    assert "- b" in preview and "+ B" in preview
    assert preview[0] == "  1 unchanged line(s)"
    assert preview[-1] == "  1 unchanged line(s)"


def test_a_long_change_says_how_much_was_elided():
    """write_file already said '… N more lines'; edit_file silently truncated."""
    preview = "\n".join(tools._edit_preview("x\n", "\n".join(str(i) for i in range(40))))

    assert "more line(s)" in preview
    shown = [line for line in preview.splitlines() if line.startswith("+ ")]
    assert len(shown) == tools.EDIT_PREVIEW_LINES + 1      # +1 for the elision note


def test_a_whitespace_only_edit_says_so_rather_than_printing_nothing():
    assert "no visible change" in "\n".join(tools._edit_preview("a\nb\n", "a\nb"))


# ---------------------------------------------------------------------------
# The github tool
#
# `gh` is never run for real — the conftest guard makes that impossible, and
# the reason is in its docstring: a test reaching real gh does not spend a quota,
# it writes to a third party under the developer's identity. `_run` is the seam.
# ---------------------------------------------------------------------------


def fake_run(stdout="", returncode=0, stderr="", record=None):
    def run(argv, cwd, timeout):
        if record is not None:
            record.append((list(argv), timeout))
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    return run


@pytest.mark.parametrize(
    "args,gated",
    [
        (["pr", "view", "12"], False),
        (["pr", "view", "12", "--comments"], False),
        (["pr", "list"], False),
        (["pr", "diff", "12"], False),
        (["issue", "view", "7"], False),
        (["issue", "list"], False),
        (["repo", "view"], False),
        # Outward facing: other people see these.
        (["pr", "create", "--title", "x", "--body", "y"], True),
        (["pr", "comment", "12", "-b", "hi"], True),
        (["pr", "merge", "12"], True),
        (["pr", "close", "12"], True),
        (["issue", "create", "--title", "x"], True),
        (["issue", "comment", "7", "-b", "hi"], True),
        # Unknown defaults to asking.
        (["project", "item-add"], True),
        ([], True),
    ],
)
def test_gh_gating_is_per_call(args, gated):
    assert tools.TOOLS["github"].requires_approval({"args": args}) is gated


@pytest.mark.parametrize("noun", ["auth", "secret", "ssh-key", "gpg-key", "config"])
def test_gh_credential_nouns_are_refused(tmp_path, noun, monkeypatch):
    """`auth` would print or change a credential through a model, and the rest
    manage secrets and keys — never something to do by proxy.

    `api` is deliberately *not* here any more: see the gh-api tests below.
    """
    monkeypatch.setattr(tools, "_run", fake_run(stdout="gho_secrettoken"))

    result = tools.github(tmp_path, args=[noun, "token"])
    assert result.startswith("ERROR:")
    assert "every time" in result
    assert "gho_secrettoken" not in result     # never ran, so never printed


def test_a_gh_read_runs_and_returns_the_output(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(tools, "_run", fake_run(stdout="Three comments…", record=seen))

    result = tools.github(tmp_path, args=["pr", "view", "12", "--comments"])

    assert "Three comments" in result
    assert seen[0][0] == ["gh", "pr", "view", "12", "--comments"]
    assert seen[0][1] == tools.NETWORK_TIMEOUT    # not the local 120s


def test_a_missing_gh_is_distinct_from_a_logged_out_one(tmp_path, monkeypatch):
    """One message for both sends the model round in circles fixing the wrong
    thing — the lesson read_file's per-failure messages were written for."""
    def absent(argv, cwd, timeout):
        raise tools.ExecutableMissing("gh is not on PATH")

    monkeypatch.setattr(tools, "_run", absent)
    missing = tools.github(tmp_path, args=["pr", "list"])
    assert "not on PATH" in missing
    assert "cli.github.com" in missing

    monkeypatch.setattr(
        tools,
        "_run",
        fake_run(returncode=4, stderr="gh auth login required"),
    )
    logged_out = tools.github(tmp_path, args=["pr", "list"])
    assert "gh auth login" in logged_out
    assert "not on PATH" not in logged_out


def test_a_timed_out_post_says_the_outcome_is_unknown(tmp_path, monkeypatch):
    """The one error here that is NOT safe to retry.

    Every other ERROR string means "that did not happen, try again". A timed-out
    `pr create` may well have succeeded, and a retry opens a second pull
    request — so the vocabulary needs a word for "unknown", which it did not have.
    """
    def slow(argv, cwd, timeout):
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(tools, "_run", slow)

    result = tools.github(tmp_path, args=["pr", "create", "--title", "x"])
    assert "UNKNOWN" in result
    assert "Do not retry" in result


def test_a_gh_post_shows_its_body_before_approval():
    """Approving text you cannot see is not approval — and unlike a file, a
    comment under your name cannot be un-posted."""
    body = "This looks wrong because…\nSecond line.\nThird line."
    lines = tools.preview_call(
        "github",
        {"args": ["pr", "create", "--title", "Fix the fuser", "--body", body]},
    )
    joined = "\n".join(lines)

    assert "Fix the fuser" in joined
    assert "This looks wrong because" in joined
    assert "Third line." in joined


def test_a_long_gh_body_is_elided_with_a_count():
    body = "\n".join(f"line {i}" for i in range(40))
    joined = "\n".join(tools.preview_call("github", {"args": ["pr", "create", "-b", body]}))
    assert "more line(s)" in joined


def test_an_inline_body_flag_is_previewed_too():
    joined = "\n".join(
        tools.preview_call("github", {"args": ["pr", "comment", "--body=inline form"]})
    )
    assert "inline form" in joined


def test_previewing_junk_arguments_never_raises():
    """This runs before the handler, on arguments nothing has validated. A gate
    that crashes is worse than one that shows less."""
    for arguments in ({"args": None}, {"args": "a string"}, {"args": [1, 2]}, {}):
        assert isinstance(tools.preview_call("github", arguments), list)


def test_gh_is_announced_with_the_normalised_argv():
    described = tools.describe_call("github", {"args": ["gh", "pr", "comment", "-b", "two words"]})
    assert described == 'run: gh pr comment -b "two words"'


@pytest.mark.skipif(
    shutil.which("gh") is None,
    reason="with no gh to resolve, _resolve_exe refuses before the guard can fire",
)
def test_the_suite_cannot_reach_real_gh_even_by_accident(tmp_path):
    """The belt to the stub's braces.

    A test that forgets to patch `_run` must fail loudly rather than quietly
    opening a pull request. The guard raises past `github`'s `except OSError`
    deliberately — a refusal the tool could catch and report as a string would
    look like an ordinary failure, and the point is that it cannot be mistaken
    for one.
    """
    with pytest.raises(RuntimeError, match="real `gh`"):
        tools.github(tmp_path, args=["pr", "list"])


def test_git_is_not_blocked_by_the_gh_guard(repo):
    """Only gh is refused. Real git is how the git tool is tested."""
    assert "exit code 0" in tools.git(repo, args=["status", "--short"])


# ---------------------------------------------------------------------------
# Two holes the first version of this classification left open
#
# Both found by review, not by the tests above — which is the point worth
# recording: those tests were complete with respect to the threat list I had
# written them from, and the threat list was short. These are written from the
# other direction, by asking what *else* could reach the same outcome.
# ---------------------------------------------------------------------------


@pytest.fixture
def repo_with_secret(repo):
    """A repository whose history contains a committed credential file."""
    (repo / ".env").write_text("OPENROUTER_API_KEY=sk-SUPERSECRET\n", encoding="utf-8")
    subprocess.run(["git", "add", "-f", ".env"], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "oops"], cwd=repo, capture_output=True)
    return repo


def test_an_object_id_cannot_be_used_to_dodge_the_denylist(repo_with_secret):
    """The hole: `cat-file blob <sha>` names no path, so the path check cannot
    see it. Two free calls — ls-tree for the sha, cat-file for the blob — and
    the key was in a provider's logs with nobody asked.

    The rule that holds is not "does it print file contents" (`blame` does, and
    is free, because `read_file` is free too and the path is checkable). It is
    whether what gets printed can be *reviewed* before it runs. A path can; an
    object id cannot.
    """
    listing = tools.git(repo_with_secret, args=["ls-tree", "HEAD"])
    sha = next(l for l in listing.splitlines() if ".env" in l).split()[2]

    for args in (
        ["cat-file", "blob", sha],
        ["cat-file", "-p", sha],
        ["cat-file", "-t", sha],
        ["diff-tree", sha],
    ):
        assert tools.TOOLS["git"].requires_approval({"args": args}) is True, args


def test_the_protection_does_not_depend_on_one_spelling(repo_with_secret):
    """`cat-file -p` was gated by coincidence — `-p` is in the patch-flag set,
    which exists for `log -p`. The `blob` spelling takes no flag at all, so the
    one form was protected and the obvious adjacent one was not."""
    gated = tools.TOOLS["git"].requires_approval
    assert gated({"args": ["cat-file", "-p", "HEAD"]}) is True
    assert gated({"args": ["cat-file", "blob", "HEAD"]}) is True


def test_listing_object_ids_stays_free_because_it_leads_nowhere(repo):
    """ls-tree is the sha oracle, and that is fine once nothing free resolves a
    sha to content — it leaks names, exactly as the free `list_files` does."""
    assert tools.TOOLS["git"].requires_approval({"args": ["ls-tree", "HEAD"]}) is False
    assert tools.TOOLS["git"].requires_approval({"args": ["rev-parse", "HEAD"]}) is False


def test_a_bare_stash_is_a_write_and_never_runs_unasked(repo):
    """`git stash` with no argument is `git stash push`, not `stash list`.

    It was classified as a listing read, so the agent could silently revert
    uncommitted work — and the inversion was total: `stash pop`, the *recovery*,
    asked permission while the destruction did not.
    """
    (repo / "app.py").write_text("VALUE = 999  # unsaved\n", encoding="utf-8")
    assert tools.TOOLS["git"].requires_approval({"args": ["stash"]}) is True

    # The work is still there, because nothing ran it.
    assert "999" in (repo / "app.py").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "args,gated",
    [
        (["stash"], True),            # = stash push
        (["stash", "push"], True),
        (["stash", "pop"], True),
        (["stash", "drop"], True),
        (["stash", "clear"], True),
        (["stash", "list"], False),   # harmless, and was gated — backwards
        (["stash", "show"], True),    # prints a patch
    ],
)
def test_stash_is_classified_per_subcommand(args, gated):
    assert tools.TOOLS["git"].requires_approval({"args": args}) is gated


@pytest.mark.parametrize("lister", [["branch"], ["tag"], ["remote"], ["config", "--list"]])
def test_the_other_listers_really_do_list_when_bare(lister):
    """The rule `_GIT_LISTERS` states is true for every remaining member —
    which is why `stash` had to leave rather than the rule being rewritten."""
    assert tools.TOOLS["git"].requires_approval({"args": lister}) is False


def test_a_ci_log_is_not_a_free_read():
    """`gh run view --log` dumps a CI log, and CI logs routinely contain tokens
    that the web UI masked and the raw output does not. No local denylist can
    reach remote content, so the classification is the only control."""
    gated = tools.TOOLS["github"].requires_approval
    assert gated({"args": ["run", "view", "--log"]}) is True
    assert gated({"args": ["run", "view", "123"]}) is True
    assert gated({"args": ["run", "list"]}) is False


def test_a_pr_diff_stays_free_as_a_stated_exception():
    """The one deliberate exception to "content-printing reads are gated": it is
    the diff of the change you were asked about, the user already has it in a
    browser, and gating it makes "review this PR" a prompt per file."""
    assert tools.TOOLS["github"].requires_approval({"args": ["pr", "diff", "12"]}) is False


def test_the_toplevel_check_costs_one_process_per_root(repo, monkeypatch):
    """It runs before *every* git call, so uncached it made `git status` two
    processes."""
    calls = []
    real = tools._run

    def counting(argv, cwd, timeout):
        calls.append(list(argv))
        return real(argv, cwd, timeout)

    monkeypatch.setattr(tools, "_run", counting)
    tools._TOPLEVEL_CACHE.clear()

    tools.git(repo, args=["status", "--short"])
    tools.git(repo, args=["log", "--oneline", "-1"])

    rev_parses = [c for c in calls if "rev-parse" in c]
    assert len(rev_parses) == 1          # not one per call


# ---------------------------------------------------------------------------
# Order was part of the policy, and nothing said so
#
# The third defect of this shape in one classifier. `stash list` was added as a
# read *above* the patch-flag gate — while its own comment explained why it
# could not go through the general path — so `git stash list -p` dumped a
# stashed diff for free. The fix was structural: the unconditional gates and
# the allowlist are now separate functions, so a new read cannot be placed
# above a gate even by accident.
# ---------------------------------------------------------------------------


def test_a_patch_flag_gates_whatever_it_is_attached_to():
    """The unconditional rule, asserted against every read that takes -p.

    Written as a sweep rather than a list of known cases, because the three
    defects here were all "the case nobody enumerated".
    """
    gated = tools.TOOLS["git"].requires_approval
    for base in (
        ["log"], ["stash", "list"], ["diff", "--stat"], ["show", "--stat"],
        ["whatchanged"], ["blame"], ["status"],
    ):
        assert gated({"args": base + ["-p"]}) is True, base
        assert gated({"args": base + ["--patch"]}) is True, base


def test_stash_list_is_free_but_its_patch_is_not(repo_with_secret):
    """The reachable leak: a stash holding a credential change, dumped by -p."""
    subprocess.run(["git", "stash", "push", "-u"], cwd=repo_with_secret, capture_output=True)

    gated = tools.TOOLS["git"].requires_approval
    assert gated({"args": ["stash", "list"]}) is False
    assert gated({"args": ["stash", "list", "-p"]}) is True


def test_the_allowlist_cannot_outvote_an_unconditional_gate():
    """Structural, not positional.

    `_git_is_read` can only answer "is this a known read"; the gates live in
    `_git_needs_approval` above it. So even a subcommand the allowlist says yes
    to is still gated when a patch flag is present — which is what the previous
    arrangement got wrong by letting a special case run first.
    """
    assert tools._git_is_read("stash", set(), ["list"]) is True        # allowlisted
    assert tools._git_needs_approval(["stash", "list", "-p"]) is True  # still gated


# ---------------------------------------------------------------------------
# What a real session exposed
#
# From a transcript: the gate asked permission for `git (bad arguments)` and
# then ran nothing, and the agent went hunting the home directory for a
# repository that lives on GitHub. The second was the worse one and it was not
# the model's fault — the tool's own description said "in this repository", so
# it was told the github tool could not do what was being asked.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arguments",
    [{"args": "status"}, {"args": None}, {"args": {}}, {"command": "git status"}],
)
def test_an_impossible_call_does_not_ask_permission(project, arguments):
    """A prompt that cannot lead anywhere spends the only thing the gate has,
    which is being worth reading."""
    asked = []
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(tool_reply(ToolCall(id="c1", name="git", arguments=arguments)), final("ok")),
    ):
        outcome = agent.run(
            "look at the repo",
            CFG,
            root=project,
            confirm=lambda n, a: asked.append(n) or True,
            write=lambda *_: None,
        )

    assert asked == []                                   # nobody was interrupted
    assert outcome.steps[0].result.startswith("ERROR:")  # and the model was told why
    assert outcome.changed_anything is False


def test_the_rejection_and_the_handler_agree_on_what_is_usable():
    """Skipping the prompt is only safe while the call truly cannot run, so both
    sides go through the same `_argv`."""
    tool = tools.TOOLS["git"]
    for arguments in ({"args": "status"}, {"args": []}, {"args": None}):
        assert tool.unusable(arguments).startswith("ERROR:")
    assert tool.unusable({"args": ["status"]}) == ""


def test_a_validator_that_raises_does_not_take_down_the_gate():
    exploding = tools.Tool(
        name="x", description="d", parameters={}, needs_confirmation=True,
        reject=lambda _a: 1 / 0,
    )
    assert exploding.unusable({}) == ""


def test_the_github_tool_says_it_can_reach_any_repository():
    """It said "in this repository", so asked about a repo on GitHub the agent
    searched the local disk — doing exactly what it had been told."""
    description = tools.TOOLS["github"].description
    assert "--repo" in description
    assert "any" in description
    assert "do not go searching the local disk" in description.lower()


def test_finding_an_unknown_owner_is_a_free_read():
    """The alternative to guessing one, so it has to be cheap enough to use."""
    assert tools.TOOLS["github"].requires_approval({"args": ["search", "repos", "x"]}) is False


def test_the_agent_prompt_routes_github_work_to_the_github_tool():
    assert "--repo owner/name" in agent.SYSTEM
    assert "Do not search the disk" in agent.SYSTEM
    assert "Do not guess an identifier" in agent.SYSTEM


# ---------------------------------------------------------------------------
# A refusal is also a call that cannot run
#
# `unusable`'s docstring said "why this call cannot run" and was wired only to
# the shape errors, so the five outright refusals still prompted. The force-push
# one was the worst: asked to authorise a force push, agreed, then told no —
# which does not merely waste the prompt, it misrepresents what the tool does.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,args",
    [
        ("git", ["show", "HEAD:.env"]),
        ("git", ["log", "-p", "--", ".env"]),
        ("git", ["push", "--force"]),
        ("git", ["push", "-qf", "origin", "main"]),
        ("git", ["-c", "core.fsmonitor=echo", "status"]),
        ("git", ["-C", "/elsewhere", "status"]),
        ("github", ["api", "user"]),
        ("github", ["auth", "token"]),
        ("github", ["secret", "list"]),
    ],
)
def test_a_refusal_never_asks_permission_first(project, name, args):
    asked = []
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(tool_reply(call(name, args=args)), final("ok")),
    ):
        outcome = agent.run(
            "do it",
            CFG,
            root=project,
            confirm=lambda n, a: asked.append(n) or True,
            write=lambda *_: None,
        )

    assert asked == [], f"{name} {args} asked permission for a refusal"
    assert outcome.steps[0].result.startswith("ERROR:")
    assert outcome.changed_anything is False


def test_what_is_still_gated_is_still_gated(project):
    """The complement: the pre-gate refusals must not have swallowed the gate.
    `--force-with-lease` is the sharp case — allowed, and therefore asked."""
    asked = []
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(
            tool_reply(call("git", args=["push", "--force-with-lease"])), final("ok")
        ),
    ):
        agent.run(
            "push it",
            CFG,
            root=project,
            confirm=lambda n, a: asked.append(n) or False,
            write=lambda *_: None,
        )

    assert asked == ["git"]


def test_the_gate_and_the_handler_agree_on_every_refusal(project):
    """Skipping a prompt is only safe while the call truly cannot execute, so
    both sides go through the same function."""
    for args in (
        ["show", "HEAD:.env"], ["push", "--force"], ["-c", "x=y", "status"], "a string",
    ):
        from_gate = tools.TOOLS["git"].unusable({"args": args})
        from_handler = tools.git(project, args=args)
        assert bool(from_gate) is from_handler.startswith("ERROR:"), args
        if from_gate:
            assert from_gate == from_handler, args


def test_the_foreign_repo_check_stays_after_the_gate(tmp_path):
    """The one refusal that cannot move: it shells out to `git rev-parse`, and
    running git before the user has agreed to anything is a different trade from
    reading the argv."""
    outer = tmp_path / "outer"
    (outer / "inner").mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=outer, capture_output=True)
    tools._TOPLEVEL_CACHE.clear()

    # Nothing in the argv is wrong, so the pre-gate check passes it...
    assert tools.TOOLS["git"].unusable({"args": ["status"]}) == ""
    # ...and the handler refuses it once it has looked.
    assert "outside the project directory" in tools.git(outer / "inner", args=["status"])


# --- the noun/verb slot ----------------------------------------------------


def test_a_flag_value_cannot_occupy_the_subcommand_slot():
    """`_gh_pair` took the first two non-flag words wherever they sat, so

        ["--repo", "pr", "--template", "view", "issue", "create"]

    classified as `pr view` while the command was `issue create`. Nothing ran,
    because gh rejects that form — but that left the safety resting on an
    external CLI's parser rather than on this classifier, and the description
    actively teaches --repo.
    """
    sneaky = ["--repo", "pr", "--template", "view", "issue", "create"]
    assert tools._gh_pair(sneaky) == ("", "")
    assert tools.TOOLS["github"].requires_approval({"args": sneaky}) is True


@pytest.mark.parametrize(
    "args",
    [
        ["pr", "view", "1", "--repo", "o/n"],
        ["issue", "list", "--repo", "o/n"],
        ["search", "repos", "x"],
        ["pr", "list"],
    ],
)
def test_the_legitimate_forms_still_read_as_reads(args):
    assert tools.TOOLS["github"].requires_approval({"args": args}) is False


def test_the_description_names_exactly_the_searches_that_are_free():
    """It said "search" generally while `search code` and `search commits` are
    gated. Gating is the safe direction, so this was a doc-vs-code mismatch
    rather than a hole — the model would simply be surprised once, which is a
    wasted turn."""
    description = tools.TOOLS["github"].description
    assert "search repos/prs/issues" in description

    gated = tools.TOOLS["github"].requires_approval
    for free in ("repos", "prs", "issues"):
        assert gated({"args": ["search", free, "x"]}) is False, free
    for asks in ("code", "commits"):
        assert gated({"args": ["search", asks, "x"]}) is True, asks


# ---------------------------------------------------------------------------
# gh api — classified, not refused
#
# It was refused on the grounds that it is "an arbitrary HTTP client carrying
# the user's token". That argument does not hold: every gh subcommand carries
# the token, `pr view` as much as `api`, and nothing here can read it either
# way — it belongs to a CLI the user authenticated themselves.
#
# The real distinction is narrower. `api` is the one subcommand whose
# read/write-ness is not in its name, which makes it harder to classify rather
# than impossible. Refusing it also cost something real: reading the files of a
# repository you do not have locally.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["api", "repos/open-gsd/gsd-core"],
        ["api", "repos/open-gsd/gsd-core/readme"],
        ["api", "repos/open-gsd/gsd-core/contents/README.md"],
        ["api", "repos/o/n/git/trees/HEAD?recursive=1"],
        ["api", "--method", "GET", "repos/o/n"],
        ["api", "-X", "GET", "repos/o/n"],
        # --method GET wins over fields: gh then sends them as a query string.
        ["api", "--method", "GET", "-f", "per_page=5", "repos/o/n/issues"],
        ["api", "-X", "HEAD", "repos/o/n"],
    ],
)
def test_reading_through_gh_api_is_free(args):
    assert tools.TOOLS["github"].requires_approval({"args": args}) is False


@pytest.mark.parametrize(
    "args",
    [
        ["api", "-X", "POST", "repos/o/n/issues"],
        ["api", "--method", "PATCH", "repos/o/n"],
        ["api", "--method=DELETE", "repos/o/n"],
        ["api", "-X", "PUT", "repos/o/n/contents/x"],
        # No -X at all, but fields: gh switches to POST by itself, which its own
        # --help states. Absence of a method is NOT enough to call it a read.
        ["api", "repos/o/n/issues", "-f", "title=hi"],
        ["api", "repos/o/n/issues", "--field", "title=hi"],
        ["api", "repos/o/n/issues", "-F", "body=@file"],
        ["api", "repos/o/n/issues", "--raw-field", "title=hi"],
        ["api", "repos/o/n/issues", "--input", "body.json"],
        # The attached form gh also accepts.
        ["api", "repos/o/n/issues", "-ftitle=hi"],
    ],
)
def test_writing_through_gh_api_is_gated(args):
    assert tools.TOOLS["github"].requires_approval({"args": args}) is True


@pytest.mark.parametrize(
    "args",
    [
        ["api", "graphql", "-f", "query=query{viewer{login}}"],
        ["api", "graphql", "--method", "GET", "-f", "query=mutation{...}"],
        ["api", "graphql"],
    ],
)
def test_graphql_is_always_gated_because_the_verb_is_in_the_body(args):
    """The honest exception. A GraphQL mutation lives in the query body, not in
    a flag, so no amount of argv inspection can tell a read from a write — and a
    classifier that cannot state its own rule is what this project keeps getting
    wrong. `--method GET` does not rescue it either."""
    assert tools.TOOLS["github"].requires_approval({"args": args}) is True


def test_an_unknown_method_is_treated_as_a_write():
    """Unknown falls to gated, like every other unrecognised thing here."""
    gated = tools.TOOLS["github"].requires_approval
    assert gated({"args": ["api", "-X", "FROBNICATE", "repos/o/n"]}) is True
    assert gated({"args": ["api", "-X"]}) is True          # method flag, no value


def test_gh_api_is_no_longer_refused(tmp_path, monkeypatch):
    """It must actually run now, not just classify as a read."""
    seen = []
    monkeypatch.setattr(tools, "_run", fake_run(stdout='{"name":"gsd-core"}', record=seen))

    result = tools.github(tmp_path, args=["api", "repos/open-gsd/gsd-core"])

    assert not result.startswith("ERROR:")
    assert "gsd-core" in result
    assert seen[0][0] == ["gh", "api", "repos/open-gsd/gsd-core"]


def test_a_remote_repo_can_now_be_read_without_cloning(tmp_path, monkeypatch):
    """The gap the refusal created, and the reason this was worth changing."""
    monkeypatch.setattr(tools, "_run", fake_run(stdout="# GSD Core\n\nGit. Ship. Done."))
    gated = tools.TOOLS["github"].requires_approval

    for args in (
        ["repo", "view", "open-gsd/gsd-core"],
        ["api", "repos/open-gsd/gsd-core/readme"],
        ["api", "repos/open-gsd/gsd-core/git/trees/HEAD?recursive=1"],
    ):
        assert gated({"args": args}) is False, args
        assert "GSD Core" in tools.github(tmp_path, args=args)


# ---------------------------------------------------------------------------
# gh api — the spellings, enumerated
#
# The first classifier detected dangerous flags instead of allowlisting safe
# ones, and `gh api -XDELETE repos/owner/repo` went through as a *read*: pflag
# accepts an attached value, so `-XDELETE` is `-X DELETE`, matched nothing, and
# fell out the bottom. A repository deleted with no prompt, and `-XPOST
# …/comments` posting under the user's name past the one gate that refuses
# outward posts even under --yes.
#
# It is now an allowlist, so the question stops being "did I think of this
# spelling" — but the spellings are enumerated anyway, because a classifier
# that replaced a refusal is the only thing left between the model and the whole
# API, and this list is what that costs.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["api", "-XDELETE", "repos/owner/repo"],      # the one that got through
        ["api", "-XPOST", "repos/o/n/issues/1/comments"],
        ["api", "-XPUT", "repos/o/n/subscription"],
        ["api", "-XPATCH", "repos/o/n/issues/1"],
        ["api", "-X=DELETE", "repos/owner/repo"],
        ["api", "-X", "DELETE", "repos/owner/repo"],
        ["api", "--method=DELETE", "repos/owner/repo"],
        ["api", "-sXDELETE", "repos/owner/repo"],     # clustered short flags
        ["api", "-X"],                                # flag with no value
        ["api", "--method"],
        ["api", "--frobnicate", "repos/o/n"],         # a flag nobody vouched for
    ],
)
def test_every_write_spelling_of_gh_api_is_gated(args):
    assert tools.TOOLS["github"].requires_approval({"args": args}) is True


@pytest.mark.parametrize(
    "args",
    [
        ["api", "repos/o/n"],
        ["api", "-XGET", "repos/o/n"],
        ["api", "-X", "GET", "repos/o/n"],
        ["api", "-XHEAD", "repos/o/n"],
        # --method GET wins over fields: gh then sends them as a query string,
        # which is documented. A single-pass version got this wrong by returning
        # on the field before it had read the method.
        ["api", "--method", "GET", "-f", "per_page=5", "repos/o/n/issues"],
        ["api", "-H", "Accept: application/vnd.github+json", "repos/o/n"],
        ["api", "--paginate", "--jq", ".[].name", "repos/o/n/issues"],
        ["api", "--cache", "1h", "repos/o/n"],
    ],
)
def test_every_read_spelling_stays_free(args):
    assert tools.TOOLS["github"].requires_approval({"args": args}) is False


def test_an_unrecognised_flag_is_gated_rather_than_guessed_at():
    """The property that makes the -X spellings stop mattering: this is an
    allowlist, so a flag nobody enumerated is a flag nobody vouched for."""
    gated = tools.TOOLS["github"].requires_approval
    for invented in ("--nextyears-flag", "-Z", "--write-everything"):
        assert gated({"args": ["api", invented, "repos/o/n"]}) is True, invented


# --- credential files through the API -------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "repos/o/n/contents/.env",
        "repos/o/n/contents/.env.production",
        "repos/o/n/contents/.ssh/id_rsa",
        "repos/o/n/contents/config/credentials.yml",
        "repos/o/n/contents/certs/server.pem",
        "repos/o/n/contents/.env?ref=main",            # query string stripped
    ],
)
def test_a_credential_file_is_refused_through_the_api_too(tmp_path, path, monkeypatch):
    """`read_file` refuses it, `git show HEAD:.env` refuses it, and this route
    returned it base64-encoded for free. The denylist had two doors covered and
    a third standing open.

    Refused rather than gated, matching the other two: there is no answer to
    "shall I send your private key to a model provider?" that should be yes.
    """
    monkeypatch.setattr(tools, "_run", fake_run(stdout="BASE64SECRET"))

    result = tools.github(tmp_path, args=["api", path])
    assert result.startswith("ERROR:")
    assert "credential" in result
    assert "BASE64SECRET" not in result          # never ran


def test_the_api_refusal_does_not_ask_first(project, monkeypatch):
    asked = []
    with mock.patch(
        "majordomo.llm.complete_with_tools",
        replies(
            tool_reply(call("github", args=["api", "repos/o/n/contents/.env"])),
            final("ok"),
        ),
    ):
        agent.run(
            "read it", CFG, root=project,
            confirm=lambda n, a: asked.append(n) or True, write=lambda *_: None,
        )
    assert asked == []


@pytest.mark.parametrize(
    "path",
    ["repos/o/n/contents/src/main.go", "repos/o/n/readme", "repos/o/n/contents/README.md"],
)
def test_ordinary_files_still_read_freely(tmp_path, path, monkeypatch):
    monkeypatch.setattr(tools, "_run", fake_run(stdout="# GSD Core"))
    assert tools.TOOLS["github"].requires_approval({"args": ["api", path]}) is False
    assert "GSD Core" in tools.github(tmp_path, args=["api", path])


# ---------------------------------------------------------------------------
# The asset has three routes, not one
#
# The rule was settled locally — not "does it print file contents" but "can
# what it prints be reviewed before it runs" — and then applied to subcommand
# names only. So the same two-step came back over HTTP:
#
#     gh api repos/o/n/git/trees/HEAD?recursive=1   paths and blob shas
#     gh api repos/o/n/git/blobs/<sha>              the file, by sha
#
# which is `ls-tree` -> `cat-file blob` with a different transport. Three fixes
# had each been correct for the route they were written against, while the asset
# was reachable by path, by object id, and by archive.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "repos/o/n/git/blobs/15f649cedea151e2c781808571d0bbb7f57b2ad6",
        "repos/o/n/tarball/main",
        "repos/o/n/zipball/main",
        "repos/o/n/tarball",
        # The raw endpoint `gh run view --log` wraps. Gating the wrapper and
        # leaving the endpoint free was the same mistake twice in one feature.
        "repos/o/n/actions/runs/123/logs",
        "repos/o/n/actions/jobs/456/logs",
    ],
)
def test_content_with_no_reviewable_address_is_gated(path):
    assert tools.TOOLS["github"].requires_approval({"args": ["api", path]}) is True


def test_the_sha_oracle_stays_free_because_it_leads_nowhere():
    """`git/trees` returns names and shas, no content — exactly as `ls-tree` is
    free locally. An oracle is harmless once nothing free resolves a sha."""
    gated = tools.TOOLS["github"].requires_approval
    assert gated({"args": ["api", "repos/o/n/git/trees/HEAD?recursive=1"]}) is False
    assert gated({"args": ["api", "repos/o/n/git/refs/heads/main"]}) is False


def test_the_blob_route_is_the_cat_file_hole_over_http():
    """Stated as the equivalence, so the next person fixing one route sees the
    other two."""
    assert tools.TOOLS["git"].requires_approval({"args": ["cat-file", "blob", "abc"]}) is True
    assert tools.TOOLS["github"].requires_approval(
        {"args": ["api", "repos/o/n/git/blobs/abc"]}
    ) is True


@pytest.mark.parametrize(
    "path",
    [
        "repos/o/n/contents/%2Eenv",            # GitHub decodes this to .env
        "repos/o/n/contents/%2Essh/id_rsa",
        "repos/o/n/contents/config%2Fcredentials.yml",
    ],
)
def test_percent_encoding_does_not_smuggle_a_credential_path(tmp_path, path, monkeypatch):
    """The check read the literal `%2Eenv`; GitHub reads `.env`. The only
    encoding layer in play, since the rest of these paths arrive literal."""
    monkeypatch.setattr(tools, "_run", fake_run(stdout="BASE64SECRET"))

    result = tools.github(tmp_path, args=["api", path])
    assert result.startswith("ERROR:")
    assert "BASE64SECRET" not in result


@pytest.mark.parametrize(
    "path",
    ["repos/o/n", "repos/o/n/issues", "repos/o/n/readme", "repos/o/n/contents/src/main.go"],
)
def test_ordinary_endpoints_are_untouched(path):
    """The whole point of unrefusing `gh api` — summarising a remote repo
    without cloning it must still work."""
    assert tools.TOOLS["github"].requires_approval({"args": ["api", path]}) is False


@pytest.mark.parametrize(
    "encoded",
    [
        ".env",
        "%2Eenv",          # one layer
        "%2E%65nv",        # one layer, split differently
        "%252Eenv",        # two layers — a single unquote left this free
        "%25%32%45env",
        "%25252Eenv",      # three
    ],
)
def test_no_amount_of_encoding_smuggles_a_credential_path(tmp_path, encoded, monkeypatch):
    """A single decode handled one layer and compared the second literally.

    Whether GitHub itself decodes twice is not establishable without a live API
    call, so the fix removes the question rather than betting on the answer:
    decode to a fixed point. `unquote` is idempotent once no `%` remains, so the
    loop terminates on its own.
    """
    monkeypatch.setattr(tools, "_run", fake_run(stdout="BASE64SECRET"))

    result = tools.github(tmp_path, args=["api", f"repos/o/n/contents/{encoded}"])
    assert result.startswith("ERROR:"), encoded
    assert "BASE64SECRET" not in result


def test_decoding_terminates_and_leaves_ordinary_paths_alone():
    assert tools._fully_unquoted("%25252Eenv") == ".env"
    assert tools._fully_unquoted("src/main.go") == "src/main.go"
    assert tools._fully_unquoted("") == ""
    # A string that never stops producing % would hit the cap rather than spin.
    assert isinstance(tools._fully_unquoted("%25" * 50), str)
