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


def test_remember_refuses_a_near_duplicate(capsys):
    run("remember", "Builds for Windows first, cross-platform later")
    with pytest.raises(SystemExit) as exit_info:
        run("remember", "He builds for Windows first; cross-platform later")

    assert exit_info.value.code == 1
    assert "already covers" in capsys.readouterr().err


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
