"""Tests for the assistant subcommands: ask, activity, remember, start.

These cover the CLI's own decisions — what exits non-zero, what degrades to a
warning, what reaches the model — rather than the modules underneath, which are
tested in test_memory / test_activity / test_chat_and_context.
"""
from __future__ import annotations

from unittest import mock

import pytest

from majordomo import activity, cli, memory
from majordomo.llm import LLMError, MissingApiKey


def run(*argv):
    cli.main(list(argv))


# ---------------------------------------------------------------------------
# mj remember
# ---------------------------------------------------------------------------


def test_remember_writes_and_lists(capsys):
    run("remember", "Builds", "for", "Windows", "first", "--type", "preference")
    assert "Remembered as" in capsys.readouterr().out

    run("remember")
    out = capsys.readouterr().out
    assert "preference" in out and "Builds for Windows first" in out


def test_remember_joins_multiple_words():
    run("remember", "Prefers", "concise", "answers")
    assert memory.read_all()[0].description == "Prefers concise answers"


def test_a_near_duplicate_is_refused_when_nobody_can_answer(capsys):
    """Piped or scripted, there is no one to ask, so it must not write blind."""
    run("remember", "Builds for Windows first, cross-platform later")
    with pytest.raises(SystemExit) as exit_info:
        run("remember", "He builds for Windows first; cross-platform later")

    assert exit_info.value.code == 1
    assert "--update" in capsys.readouterr().err
    assert len(memory.read_all()) == 1


def duplicate_answering(reply, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda _: reply)


def test_a_near_duplicate_offers_to_update_it(monkeypatch):
    """The paraphrase that started this: 'med-dark roast' vs 'medium-dark
    roasted' shared only two literal words and sailed through."""
    run("remember", "I like med-dark roast coffee")
    duplicate_answering("u", monkeypatch)

    run("remember", "I prefer my coffee medium-dark roasted")

    remaining = memory.read_all()
    assert len(remaining) == 1
    assert remaining[0].description == "I prefer my coffee medium-dark roasted"


def test_you_can_keep_both(monkeypatch):
    run("remember", "I like med-dark roast coffee")
    duplicate_answering("k", monkeypatch)

    run("remember", "I prefer my coffee medium-dark roasted")
    assert len(memory.read_all()) == 2


def test_cancelling_writes_nothing(monkeypatch):
    run("remember", "I like med-dark roast coffee")
    duplicate_answering("c", monkeypatch)

    with pytest.raises(SystemExit):
        run("remember", "I prefer my coffee medium-dark roasted")
    assert len(memory.read_all()) == 1


def test_a_different_fact_on_the_same_topic_is_not_flagged():
    """The reason this asks instead of refusing: overlap cannot tell a
    restatement from a contradiction, so the check must not be a verdict."""
    run("remember", "Builds for Windows first")
    run("remember", "Tests on Linux CI")

    assert len(memory.read_all()) == 2


def test_force_keeps_both(capsys):
    run("remember", "Builds for Windows first, cross-platform later")
    run("remember", "He builds for Windows first; cross-platform later", "--force")
    assert len(memory.read_all()) == 2


def test_update_replaces_in_place():
    run("remember", "Builds for Windows first")
    name = memory.read_all()[0].name
    run("remember", "Now builds for Linux first", "--update", name)

    remaining = memory.read_all()
    assert len(remaining) == 1
    assert remaining[0].description == "Now builds for Linux first"


def test_forget_removes_it(capsys):
    run("remember", "A passing thought")
    name = memory.read_all()[0].name

    run("remember", "--forget", name)
    assert "Forgot" in capsys.readouterr().out
    assert memory.read_all() == []


def test_update_with_an_unknown_name_exits_one(capsys):
    """A typo would otherwise create the duplicate this flag exists to avoid."""
    run("remember", "Builds for Windows first")
    with pytest.raises(SystemExit) as exit_info:
        run("remember", "Now builds for Linux", "--update", "typo-in-the-name")

    assert exit_info.value.code == 1
    assert "no memory named" in capsys.readouterr().err
    assert len(memory.read_all()) == 1


def test_update_with_no_text_exits_one(capsys):
    """Silently listing here would look like the update succeeded."""
    run("remember", "Builds for Windows first")
    name = memory.read_all()[0].name

    with pytest.raises(SystemExit) as exit_info:
        run("remember", "--update", name)

    assert exit_info.value.code == 1
    assert "needs the replacement text" in capsys.readouterr().err


def test_forget_an_unknown_name_exits_one(capsys):
    with pytest.raises(SystemExit) as exit_info:
        run("remember", "--forget", "never-existed")

    assert exit_info.value.code == 1
    assert "no memory named" in capsys.readouterr().err


def test_a_refused_write_exits_one(capsys):
    with pytest.raises(SystemExit) as exit_info:
        run("remember", "my key is sk-abcdefghijklmnop1234567890")

    assert exit_info.value.code == 1
    assert "credential" in capsys.readouterr().err


def test_empty_store_says_how_to_start(capsys):
    run("remember")
    assert "Nothing remembered yet" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# mj activity
# ---------------------------------------------------------------------------


def seed():
    activity.append_events(
        [
            activity.ActivityEvent(
                "u1", "pr", "2026-08-28T10:00:00Z", "nav/majordomo", "A pull request", "u1"
            )
        ]
    )


def test_activity_prints_the_cache(capsys):
    seed()
    run("activity")
    out = capsys.readouterr().out
    assert "A pull request" in out
    assert "Cache newest entry: 2026-08-28" in out


def test_days_zero_is_not_swallowed_by_the_default(capsys):
    """`args.days or config...` makes 0 mean 90 — a legitimate 'just today'."""
    activity.append_events(
        [
            activity.ActivityEvent(
                "old", "pr", "2026-01-02T10:00:00Z", "nav/majordomo", "Ancient", "old"
            )
        ]
    )
    run("activity", "--days", "0")
    assert "No recorded activity in the last 0 days" in capsys.readouterr().out


def test_a_partial_refresh_reports_both_the_count_and_the_gap(capsys):
    with mock.patch.object(
        activity,
        "refresh",
        return_value=activity.RefreshResult(fetched=2, added=2, error="commits: 403"),
    ):
        run("activity", "--refresh")

    err = capsys.readouterr().err
    assert "Fetched 2, 2 new." in err      # what we did get
    assert "403" in err                    # and that it is incomplete


def test_activity_on_an_empty_cache_suggests_refresh(capsys):
    run("activity")
    out = capsys.readouterr().out
    assert "No recorded activity" in out
    assert "--refresh" in out


def test_refresh_failure_warns_but_still_prints(capsys):
    """A GitHub outage costs freshness, not the answer."""
    seed()
    with mock.patch.object(
        activity, "refresh", return_value=activity.RefreshResult(error="no token")
    ):
        run("activity", "--refresh")

    captured = capsys.readouterr()
    assert "could not refresh" in captured.err
    assert "A pull request" in captured.out


def test_refresh_reports_what_it_added(capsys):
    with mock.patch.object(
        activity, "refresh", return_value=activity.RefreshResult(fetched=7, added=3)
    ):
        run("activity", "--refresh")

    assert "Fetched 7, 3 new." in capsys.readouterr().err


# ---------------------------------------------------------------------------
# mj ask
# ---------------------------------------------------------------------------


def test_ask_sends_context_and_prints_the_reply(capsys):
    run("remember", "Builds for Windows first")
    seed()

    with mock.patch("majordomo.llm.complete", return_value="You built things.") as called:
        run("ask", "what", "have", "I", "been", "doing?", "--no-refresh")

    messages = called.call_args[0][0]
    assert messages[0]["role"] == "system"
    # the question is last, and the context precedes it
    assert messages[-1]["content"].rstrip().endswith("what have I been doing?")
    assert "Windows first" in messages[-1]["content"]
    assert "A pull request" in messages[-1]["content"]
    assert "You built things." in capsys.readouterr().out


def test_ask_uses_the_chat_model_not_the_fuser():
    with mock.patch("majordomo.llm.complete", return_value="ok") as called:
        run("ask", "anything", "--no-refresh")

    from majordomo import config as config_module

    assert called.call_args[0][2] == config_module.build(
        config_module.DEFAULTS
    ).brain.chat_model


def test_ask_works_with_no_context_at_all(capsys):
    """General chat is a byproduct of the same path and must not need context."""
    with mock.patch("majordomo.llm.complete", return_value="A light roast is milder."):
        run("ask", "difference", "between", "light", "and", "dark", "roast?", "--no-refresh")

    assert "milder" in capsys.readouterr().out


def test_ask_exits_one_on_a_missing_key(capsys):
    with mock.patch("majordomo.llm.complete", side_effect=MissingApiKey("set OPENROUTER_API_KEY")):
        with pytest.raises(SystemExit) as exit_info:
            run("ask", "anything", "--no-refresh")

    assert exit_info.value.code == 1
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_ask_exits_one_on_a_model_failure(capsys):
    with mock.patch("majordomo.llm.complete", side_effect=LLMError("503")):
        with pytest.raises(SystemExit) as exit_info:
            run("ask", "anything", "--no-refresh")

    assert exit_info.value.code == 1
    assert "503" in capsys.readouterr().err


def test_ask_refreshes_a_stale_cache_by_default():
    with mock.patch.object(activity, "is_stale", return_value=True), mock.patch.object(
        activity, "refresh", return_value=activity.RefreshResult()
    ) as refreshed, mock.patch("majordomo.llm.complete", return_value="ok"):
        run("ask", "anything")

    assert refreshed.called


def test_no_refresh_leaves_the_cache_alone():
    with mock.patch.object(activity, "refresh") as refreshed, mock.patch(
        "majordomo.llm.complete", return_value="ok"
    ):
        run("ask", "anything", "--no-refresh")

    assert not refreshed.called


def test_explain_reports_what_was_loaded(capsys):
    run("remember", "Builds for Windows first")
    with mock.patch("majordomo.llm.complete", return_value="ok"):
        run("ask", "windows", "--no-refresh", "--explain")

    assert "memories indexed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# mj start
# ---------------------------------------------------------------------------


def test_start_dry_run_creates_nothing(tmp_path, capsys, monkeypatch):
    config = tmp_path / "config.yml"
    config.write_text(f"scaffold:\n  root: {tmp_path.as_posix()}/projects\n", encoding="utf-8")

    run("--config", str(config), "start", "a", "JSON", "differ", "--dry-run")

    assert "Would create" in capsys.readouterr().out
    assert not (tmp_path / "projects").exists()


def test_start_exits_one_when_the_target_is_occupied(tmp_path, capsys):
    projects = tmp_path / "projects" / "a-json-differ"
    projects.mkdir(parents=True)
    (projects / "existing.txt").write_text("x", encoding="utf-8")

    config = tmp_path / "config.yml"
    config.write_text(f"scaffold:\n  root: {tmp_path.as_posix()}/projects\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        run("--config", str(config), "start", "a", "JSON", "differ")

    assert exit_info.value.code == 1
    assert "not empty" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# mj do / mj review
# ---------------------------------------------------------------------------


def test_do_turns_a_missing_key_into_an_error_not_a_traceback(capsys, tmp_path):
    """Found by running it: agent.run swallows LLMError so a half-finished task
    keeps its trail, but MissingApiKey went straight through as a traceback."""
    from majordomo.llm import MissingApiKey

    with mock.patch(
        "majordomo.agent.run", side_effect=MissingApiKey("Set the OPENROUTER_API_KEY")
    ):
        with pytest.raises(SystemExit) as exit_info:
            run("do", "something", "-C", str(tmp_path))

    assert exit_info.value.code == 1
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_do_with_no_task_exits_one(capsys, tmp_path):
    with pytest.raises(SystemExit):
        run("do", "   ", "-C", str(tmp_path))
    assert "do what?" in capsys.readouterr().err


def test_do_reports_declined_actions(capsys, tmp_path):
    from majordomo import agent

    outcome = agent.Outcome(
        answer="I did not change it.",
        steps=[agent.Step("write_file", {}, "declined", approved=False)],
    )
    with mock.patch("majordomo.agent.run", return_value=outcome):
        run("do", "write", "something", "-C", str(tmp_path))

    captured = capsys.readouterr()
    assert "I did not change it." in captured.out
    assert "1 action(s) declined" in captured.err


def test_yes_warns_that_it_skips_every_prompt(capsys, tmp_path):
    from majordomo import agent

    with mock.patch("majordomo.agent.run", return_value=agent.Outcome(answer="ok")) as ran:
        run("do", "something", "--yes", "-C", str(tmp_path))

    assert ran.call_args.kwargs["confirm"] is agent.always_allow
    assert "approves every write" in capsys.readouterr().err


def test_review_opens_claude_code_in_the_repo(tmp_path):
    """Majordomo cannot read code; it opens the reviewer you already have."""
    with mock.patch("majordomo.resume.spawn_detached") as spawned:
        run("review", str(tmp_path))

    argv, kwargs = spawned.call_args
    assert argv[0] == ["claude", "/code-review"]
    assert kwargs["cwd"] == str(tmp_path.resolve())


def test_review_of_a_missing_directory_exits_one(capsys, tmp_path):
    with pytest.raises(SystemExit):
        run("review", str(tmp_path / "nope"))
    assert "no directory" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The edit preview
# ---------------------------------------------------------------------------


def test_an_append_shows_the_appended_lines():
    """The bug this function exists for: showing the first N lines of each side
    made an append look like a no-op, because both sides start the same."""
    old = "def add(a, b):\n    return a + b\n"
    new = old + "\ndef subtract(a, b):\n    return a - b\n"

    preview = "\n".join(cli._edit_preview(old, new))

    assert "+ def subtract(a, b):" in preview
    assert "unchanged line(s)" in preview          # the shared prefix, summarised
    assert "- def add(a, b):" not in preview       # not re-shown as a removal


def test_a_change_in_the_middle_shows_both_sides():
    old = "a\nb\nc\n"
    new = "a\nB\nc\n"

    preview = cli._edit_preview(old, new)

    assert "- b" in preview and "+ B" in preview
    assert preview[0] == "  1 unchanged line(s)"
    assert preview[-1] == "  1 unchanged line(s)"


def test_a_long_change_says_how_much_was_elided():
    """write_file already said '… N more lines'; edit_file silently truncated."""
    preview = "\n".join(cli._edit_preview("x\n", "\n".join(str(i) for i in range(40))))

    assert "more line(s)" in preview
    shown = [line for line in preview.splitlines() if line.startswith("+ ")]
    assert len(shown) == cli.EDIT_PREVIEW_LINES + 1      # +1 for the elision note


def test_a_whitespace_only_edit_says_so_rather_than_printing_nothing():
    assert "no visible change" in "\n".join(cli._edit_preview("a\nb\n", "a\nb"))


def test_the_preview_reaches_the_prompt(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda _: "n")

    cli.confirm_action(
        "edit_file",
        {"path": "adder.py", "old": "def add():\n    pass\n",
         "new": "def add():\n    pass\n\ndef sub():\n    pass\n"},
    )

    assert "+ def sub():" in capsys.readouterr().out


def test_do_refuses_a_directory_that_does_not_exist(capsys, tmp_path):
    """write_file creates parents, so an unvalidated -C typo would silently
    build a whole tree in the wrong place rather than erroring."""
    with pytest.raises(SystemExit) as exit_info:
        run("do", "say hello", "-C", str(tmp_path / "typo"))

    assert exit_info.value.code == 1
    assert "no directory" in capsys.readouterr().err


def test_do_refuses_a_file_as_the_project_root(capsys, tmp_path):
    target = tmp_path / "notes.md"
    target.write_text("x", encoding="utf-8")

    with pytest.raises(SystemExit):
        run("do", "say hello", "-C", str(target))
    assert "no directory" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# mj mic
# ---------------------------------------------------------------------------


def measured(**overrides):
    base = {
        "threshold": 200.0, "min": 10.0, "p10": 30.0, "p50": 400.0,
        "p90": 900.0, "max": 2000.0, "quiet_fraction": 0.3,
        "longest_quiet_ms": 600, "stops_at_ms": 1800,
    }
    base.update(overrides)
    return base


def test_mic_says_nothing_is_wrong_when_nothing_is(capsys):
    with mock.patch("majordomo.asr.measure", return_value=measured()):
        run("mic")
    assert "healthy" in capsys.readouterr().out


def test_mic_spots_a_threshold_above_your_voice(capsys):
    """The reported symptom: recording stopped after exactly the trailing pause
    however you spoke, because nothing ever cleared the threshold."""
    with mock.patch("majordomo.asr.measure", return_value=measured(threshold=1500.0)):
        run("mic")

    out = capsys.readouterr().out
    assert "cut you off" in out
    assert "silence_rms: 164" in out          # between the pauses and the speech


def test_mic_spots_a_threshold_below_the_room(capsys):
    with mock.patch("majordomo.asr.measure", return_value=measured(threshold=5.0)):
        run("mic")

    out = capsys.readouterr().out
    assert "never stop" in out
    assert "silence_rms:" in out


def test_mic_does_not_invent_a_number_when_nobody_spoke(capsys):
    """Suggesting a threshold from room tone alone is how you get told to set
    one below your own noise floor."""
    flat = measured(p10=30.0, p50=35.0, p90=45.0, max=90.0, threshold=250.0)
    with mock.patch("majordomo.asr.measure", return_value=flat):
        run("mic")

    out = capsys.readouterr().out
    assert "Nothing here looks like speech" in out
    assert "silence_rms:" not in out


def test_mic_reports_a_missing_microphone_and_exits_one(capsys):
    from majordomo import asr

    with mock.patch("majordomo.asr.measure", side_effect=asr.MicrophoneUnavailable("no mic")):
        with pytest.raises(SystemExit) as exit_info:
            run("mic")

    assert exit_info.value.code == 1
    assert "no mic" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# mj config
# ---------------------------------------------------------------------------


def test_config_shows_every_role_and_what_it_is_for(capsys):
    """The models were always config, but nothing said so — they were only
    visible by reading config.py."""
    run("config")
    out = capsys.readouterr().out

    for role in ("worker", "fuser", "reducer", "chat", "agent", "fallback"):
        assert role in out
    assert "write the spoken briefing" in out      # not just the role names
    assert "https://openrouter.ai/api/v1" in out


def test_config_says_whether_the_key_is_set(capsys, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    run("config")
    assert "NOT SET" in capsys.readouterr().out


def test_config_init_writes_the_annotated_example(capsys, tmp_path):
    target = tmp_path / "nested" / "config.yml"

    run("--config", str(target), "config", "--init")

    assert "fuser_model" in target.read_text(encoding="utf-8")
    assert "Wrote" in capsys.readouterr().out


def test_config_init_refuses_to_clobber(capsys, tmp_path):
    target = tmp_path / "config.yml"
    target.write_text("brain:\n  chat_model: mine\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exit_info:
        run("--config", str(target), "config", "--init")

    assert exit_info.value.code == 1
    assert "already exists" in capsys.readouterr().err
    assert "mine" in target.read_text(encoding="utf-8")


def test_config_init_force_overwrites(tmp_path):
    target = tmp_path / "config.yml"
    target.write_text("brain:\n  chat_model: mine\n", encoding="utf-8")

    run("--config", str(target), "config", "--init", "--force")

    assert "fuser_model" in target.read_text(encoding="utf-8")


def test_what_init_writes_is_loadable(tmp_path):
    """An example that does not parse is worse than no example."""
    from majordomo import config as config_module

    target = tmp_path / "config.yml"
    run("--config", str(target), "config", "--init")

    assert config_module.load(str(target)).brain.fuser_model


def test_a_non_interactive_run_says_so_instead_of_declining_everything(capsys, monkeypatch):
    """Piped, nobody can answer, so every gated call was declined and the agent
    spent its whole turn budget being told no — for no visible reason."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)

    with pytest.raises(SystemExit) as exit_info:
        cli.confirm_action("write_file", {"path": "a.txt", "content": "x"})

    assert exit_info.value.code == 1
    assert "--yes" in capsys.readouterr().err


def test_do_says_so_when_the_agent_says_nothing(capsys, tmp_path):
    """No answer, no error, exit 0 — indistinguishable from a task that needed
    no output. On stdout rather than stderr now, because it is part of the one
    report that is both shown and stored."""
    from majordomo import agent

    with mock.patch("majordomo.agent.run", return_value=agent.Outcome()):
        run("do", "something", "-C", str(tmp_path))

    assert "without saying anything" in capsys.readouterr().out


def test_ask_prints_the_command_instead_of_offering(capsys):
    """One-shot, so there is nobody to ask — and a raw marker line is not
    something the user was ever meant to see."""
    with mock.patch("majordomo.llm.complete", return_value="NEEDS_AGENT: review the repo"):
        run("ask", "review", "this", "repo", "--no-refresh")

    out = capsys.readouterr().out
    assert 'mj do "review the repo"' in out
    assert "NEEDS_AGENT" not in out


def test_ask_replaces_leaked_tool_markup(capsys):
    with mock.patch("majordomo.llm.complete", return_value="<tool_call>ls -la</tool_call>"):
        run("ask", "what", "is", "in", "that", "folder", "--no-refresh")

    out = capsys.readouterr().out
    assert "tool_call" not in out
    assert "mj do" in out


def test_ask_leaves_an_ordinary_answer_alone(capsys):
    with mock.patch("majordomo.llm.complete", return_value="Coffee is a matter of taste."):
        run("ask", "coffee?", "--no-refresh")

    out = capsys.readouterr().out
    assert "Coffee is a matter of taste." in out
    assert "mj do" not in out


@pytest.mark.parametrize(
    "task,expected",
    [
        ('find the "TODO" markers', 'mj do "find the \\"TODO\\" markers"'),
        ("it's here", 'mj do "it\'s here"'),
        ("plain task", 'mj do "plain task"'),
    ],
)
def test_the_printed_command_can_actually_be_pasted(task, expected, capsys):
    """Interpolated into double quotes, any task containing one broke:
    `mj do "find the "TODO" markers"` is three arguments, not one."""
    with mock.patch(
        "majordomo.llm.complete", return_value=f"NEEDS_AGENT: {task}"
    ):
        run("ask", "do", "it", "--no-refresh")

    assert expected in capsys.readouterr().out


def test_ask_does_not_advertise_a_chat_it_is_not_in(capsys):
    """NO_TOOLS_HERE used to say "without leaving this chat" — but `mj ask` is
    one-shot, and two different instructions in one output is worse than none."""
    with mock.patch("majordomo.llm.complete", return_value="<tool_call>ls</tool_call>"):
        run("ask", "what", "is", "in", "there", "--no-refresh")

    out = capsys.readouterr().out
    assert "/agent" not in out
    assert out.count("mj do") == 1


def test_ask_keeps_the_prose_that_came_with_a_leaked_block(capsys):
    reply = "Here is what I can tell you.\n\n<tool_call>ls</tool_call>"
    with mock.patch("majordomo.llm.complete", return_value=reply):
        run("ask", "look", "--no-refresh")

    out = capsys.readouterr().out
    assert "Here is what I can tell you." in out
    assert "tool_call" not in out


def test_do_prints_what_the_agent_found_when_it_stops_early(capsys, tmp_path):
    """Printing only `answer` discarded work the tool had already done — a
    command was run, its output captured, and then a failure reported instead."""
    from majordomo import agent

    outcome = agent.Outcome(
        steps=[agent.Step("run_command", {}, "exit code 0\nstdout:\n412\n268\n93")],
        stopped_because="the model call failed: rate-limited",
    )
    with mock.patch("majordomo.agent.run", return_value=outcome):
        run("do", "count the lines", "-C", str(tmp_path))

    captured = capsys.readouterr()
    assert "412" in captured.out                  # you get the numbers
    assert "rate-limited" in captured.err         # and still hear why it stopped


def test_do_prefers_a_real_answer_over_a_salvaged_result(capsys, tmp_path):
    from majordomo import agent

    outcome = agent.Outcome(
        answer="There are 773 lines in total.",
        steps=[agent.Step("run_command", {}, "raw tool output")],
    )
    with mock.patch("majordomo.agent.run", return_value=outcome):
        run("do", "count", "-C", str(tmp_path))

    out = capsys.readouterr().out
    assert "773 lines" in out
    assert "raw tool output" not in out


def test_the_quoting_caveat_appears_only_when_it_applies(capsys):
    """A caveat on every command would be ignored by the time it mattered."""
    with mock.patch("majordomo.llm.complete", return_value="NEEDS_AGENT: plain task"):
        run("ask", "do", "it", "--no-refresh")
    assert "cmd.exe" not in capsys.readouterr().out

    with mock.patch(
        "majordomo.llm.complete", return_value='NEEDS_AGENT: find the "TODO" markers'
    ):
        run("ask", "do", "it", "--no-refresh")
    assert "cmd.exe" in capsys.readouterr().out


def test_single_quotes_are_never_used(capsys):
    """Correct in bash and PowerShell, and wrong in cmd.exe, which has no
    single-quote syntax and passes them through literally."""
    from majordomo import cli as cli_mod

    quoted = cli_mod._shell_quote('find the "TODO" markers')
    assert not quoted.startswith("'")
    assert quoted.startswith('"') and quoted.endswith('"')


def test_the_stop_reason_is_not_said_twice(capsys, tmp_path):
    """With nothing to salvage the report *is* the stop message, and printing
    it on both streams reads as two separate problems."""
    from majordomo import agent

    outcome = agent.Outcome(stopped_because="reached the 24-step limit")
    with mock.patch("majordomo.agent.run", return_value=outcome):
        run("do", "loop", "-C", str(tmp_path))

    captured = capsys.readouterr()
    assert "24-step limit" in captured.out
    assert "24-step limit" not in captured.err


def test_the_stop_reason_still_reaches_stderr_when_there_is_other_output(capsys, tmp_path):
    """The exit status is 0 either way, so stderr is how a script notices."""
    from majordomo import agent

    outcome = agent.Outcome(
        steps=[agent.Step("run_command", {}, "412")],
        stopped_because="the model call failed",
    )
    with mock.patch("majordomo.agent.run", return_value=outcome):
        run("do", "count", "-C", str(tmp_path))

    captured = capsys.readouterr()
    assert "412" in captured.out
    assert "the model call failed" in captured.err


def test_ask_keeps_prose_written_before_the_marker(capsys):
    """Fixed in chat.py a round earlier; cli.py never used strip_needs_agent,
    so the two paths disagreed in the same way a second time."""
    reply = "Here is some context.\n\nNEEDS_AGENT: list the api folder"
    with mock.patch("majordomo.llm.complete", return_value=reply):
        run("ask", "look", "--no-refresh")

    out = capsys.readouterr().out
    assert "Here is some context." in out
    assert "NEEDS_AGENT" not in out
    assert "mj do" in out


def test_an_interrupt_is_not_an_answer(monkeypatch):
    """Swallowing Ctrl+C in `ask` collapsed "declined" and "interrupted" into
    one value, and a loop then declined one item and asked about the next."""
    from majordomo import cli as cli_mod

    monkeypatch.setattr("builtins.input", mock.Mock(side_effect=KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        cli_mod.ask("well? ")


def test_end_of_input_still_is_an_answer(monkeypatch, capsys):
    """Nobody is there, which every caller treats as a refusal."""
    from majordomo import cli as cli_mod

    monkeypatch.setattr("builtins.input", mock.Mock(side_effect=EOFError))
    assert cli_mod.ask("well? ") == ""
