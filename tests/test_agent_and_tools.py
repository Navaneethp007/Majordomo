"""Tests for the agent loop and its tools.

Most of these are about the safety model rather than the happy path: what the
agent is *refused*, what it must ask before doing, and what stops it going in
circles. Those are the parts where being wrong is expensive.
"""
from __future__ import annotations

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
        "C:/Windows/System32/config/SAM",
        "/etc/shadow",
    ],
)
def test_paths_outside_the_project_are_refused(project, escape):
    """Resolution happens before the check because that is the only order that
    works — traversal and symlinks both look ordinary until resolved."""
    with pytest.raises(tools.OutsideProject):
        tools.resolve(project, escape)


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
    out = tools.run_command(project, command="python -c \"print('hi')\"")
    assert "exit code 0" in out and "hi" in out


def test_a_failing_command_reports_rather_than_raising(project):
    out = tools.run_command(project, command="python -c \"import sys; sys.exit(3)\"")
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
    outcome = agent.Outcome(
        steps=[
            agent.Step("read_file", {}, "ok"),
            agent.Step("write_file", {}, "declined", approved=False),
        ]
    )
    assert outcome.changed_anything is False

    outcome.steps.append(agent.Step("edit_file", {}, "Edited x"))
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
    gated = {name for name, tool in tools.TOOLS.items() if tool.needs_confirmation}
    assert gated == {"write_file", "edit_file", "run_command"}


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
