"""Tests for the context block, the chat loop, and scaffolding.

The single most important assertion in this file is that the prompt *prefix* is
byte-identical across turns. Nothing breaks when it isn't — you simply pay full
price on every message, forever, with no error to notice. So it is asserted
directly rather than left to review.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest import mock

import pytest

from majordomo import activity, chat, context, memory, prompts, scaffold
from majordomo import config as config_module

CFG = config_module.build(config_module.DEFAULTS)
NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


@pytest.fixture
def populated():
    """Two memories and two activity events in the isolated home."""
    memory.write_memory(
        memory.MemoryCandidate(
            description="Builds for Windows first, cross-platform later",
            body="Native Windows is the target; other platforms are phase two.",
            type="preference",
        )
    )
    memory.write_memory(
        memory.MemoryCandidate(description="Prefers concise answers", type="preference")
    )
    activity.append_events(
        [
            activity.ActivityEvent(
                "u1", "pr", "2026-08-28T10:00:00Z", "nav/majordomo", "Speak only when needed", "u1"
            ),
            activity.ActivityEvent(
                "u2", "commit", "2026-08-27T09:00:00Z", "nav/majordomo", "Add the worker", "u2"
            ),
        ]
    )


# ---------------------------------------------------------------------------
# The context block
# ---------------------------------------------------------------------------


def test_context_includes_memory_and_activity(populated):
    text = context.build(CFG, query="windows", now=NOW).render()
    assert "What I know about you" in text
    assert "Windows first" in text
    assert "Your recent GitHub activity" in text
    assert "Speak only when needed" in text


def test_context_loads_bodies_only_for_relevant_memories(populated):
    ctx = context.build(CFG, query="windows cross-platform", now=NOW)
    loaded = [name for name, _ in ctx.memory_bodies]
    assert any("windows" in name for name in loaded)
    # the concise-answers memory has no body distinct from its description
    assert len(loaded) == 1


def test_context_with_no_query_takes_newest_first(populated):
    ctx = context.build(CFG, query=None, now=NOW)
    assert ctx.memory_index  # index is always present
    assert len(ctx.memory_bodies) <= CFG.memory.max_bodies_loaded


def test_context_is_empty_when_nothing_is_known():
    assert context.build(CFG, query="anything", now=NOW).render().strip() == (
        "## Your recent GitHub activity\n\nNo recorded GitHub activity."
    )


def test_context_respects_memory_disabled(populated):
    cfg = config_module.build({**config_module.DEFAULTS, "memory": {"enabled": False}})
    text = context.build(cfg, query="windows", now=NOW).render()
    assert "What I know about you" not in text
    assert "Your recent GitHub activity" in text


def test_context_drops_the_index_own_heading(populated):
    """INDEX.md carries a title for a human; nested under a heading it is noise."""
    text = context.build(CFG, query="windows", now=NOW).render()
    assert "# Majordomo memory" not in text


def test_context_states_coverage_as_a_date_not_a_clock(populated):
    """A clock reading in the prefix changes the bytes on every call."""
    text = context.build(CFG, query="windows", now=NOW).render()
    assert "cache covers up to 2026-08-28" in text


# ---------------------------------------------------------------------------
# The caching guarantee
# ---------------------------------------------------------------------------


def test_ask_prompt_puts_the_question_last(populated):
    messages = prompts.build_ask_prompt(
        context.build(CFG, query="q", now=NOW).render(), "WHAT DID I DO?"
    )
    assert messages[-1]["content"].rstrip().endswith("WHAT DID I DO?")


def test_ask_prefix_is_byte_identical_across_calls(populated):
    """Nothing volatile may leak above the question, or nothing ever caches."""
    first = context.build(CFG, query="q", now=None).render()
    time.sleep(1.05)                    # a second passes; the bytes must not
    second = context.build(CFG, query="q", now=None).render()
    assert first == second


def test_chat_system_prompt_is_frozen_across_turns(populated):
    session = chat.new_session(CFG)
    original = session.system

    for i in range(6):
        session.turns.append(chat.Turn("user", f"q{i}"))
        session.turns.append(chat.Turn("assistant", f"a{i}"))
        assert session.messages()[0]["content"] == original

    assert session.system == original


def test_chat_messages_are_system_then_turns_in_order():
    session = chat.Session(system="SYS")
    session.turns = [chat.Turn("user", "one"), chat.Turn("assistant", "two")]
    assert [m["role"] for m in session.messages()] == ["system", "user", "assistant"]
    assert session.messages()[1]["content"] == "one"


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


def _long_session(turns=28, size=200):
    session = chat.Session(system="SYS")
    for i in range(turns // 2):
        session.turns.append(chat.Turn("user", f"q{i} " + "x" * size))
        session.turns.append(chat.Turn("assistant", f"a{i} " + "y" * size))
    return session


def test_needs_compaction_tracks_the_threshold():
    small = config_module.build(
        {**config_module.DEFAULTS,
         "brain": {**config_module.DEFAULTS["brain"], "chat_compact_threshold_tokens": 500}}
    )
    assert chat.needs_compaction(_long_session(), small) is True
    assert chat.needs_compaction(chat.Session(system="SYS"), small) is False


def test_compaction_folds_old_turns_and_keeps_recent_verbatim():
    session = _long_session()
    last_before = session.turns[-1].content

    with mock.patch("majordomo.llm.complete", return_value="Agreed to build X."):
        assert chat.compact(session, CFG) is True

    assert len(session.turns) == chat.KEEP_RECENT_TURNS + 1
    assert session.turns[0].content.startswith("[Earlier in this conversation")
    assert "Agreed to build X." in session.turns[0].content
    assert session.turns[-1].content == last_before
    assert session.compactions == 1


def test_compaction_declines_on_a_short_conversation():
    session = chat.Session(system="SYS", turns=[chat.Turn("user", "hi")])
    assert chat.compact(session, CFG) is False


def test_a_failed_summary_leaves_the_conversation_intact():
    """Dropping turns because a summary call failed loses what you came for."""
    from majordomo.llm import LLMError

    session = _long_session()
    before = list(session.turns)

    with mock.patch("majordomo.llm.complete", side_effect=LLMError("503")):
        assert chat.compact(session, CFG) is False

    assert session.turns == before


def test_an_empty_summary_is_not_treated_as_a_fold():
    session = _long_session()
    with mock.patch("majordomo.llm.complete", return_value="   "):
        assert chat.compact(session, CFG) is False
    assert len(session.turns) == 28


# ---------------------------------------------------------------------------
# Sending
# ---------------------------------------------------------------------------


def test_send_appends_both_turns():
    session = chat.Session(system="SYS")
    with mock.patch("majordomo.llm.complete", return_value="An answer."):
        assert chat.send(session, "A question.", CFG) == "An answer."

    assert [t.role for t in session.turns] == ["user", "assistant"]
    assert session.turns[1].content == "An answer."


def test_a_failed_send_removes_the_unanswered_turn():
    """A retry must not stack two copies of the same message into history."""
    from majordomo.llm import LLMError

    session = chat.Session(system="SYS")
    with mock.patch("majordomo.llm.complete", side_effect=LLMError("503")):
        with pytest.raises(chat.ChatFailed):
            chat.send(session, "A question.", CFG)

    assert session.turns == []


def test_an_empty_reply_is_labelled_rather_than_stored_blank():
    session = chat.Session(system="SYS")
    with mock.patch("majordomo.llm.complete", return_value=None):
        assert chat.send(session, "q", CFG) == "(empty response)"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_transcript_round_trip(tmp_path):
    session = chat.Session(system="SYS", path=tmp_path / "c.jsonl")
    session.turns = [chat.Turn("user", "one"), chat.Turn("assistant", "two")]
    chat.save(session)

    restored = chat.load_turns(session.path)
    assert [t.content for t in restored] == ["one", "two"]


def test_loading_a_missing_or_broken_transcript_never_raises(tmp_path):
    assert chat.load_turns(tmp_path / "nope.jsonl") == []

    broken = tmp_path / "b.jsonl"
    broken.write_text('{"role":"user","content":"kept"}\nnot json\n{"role":"nonsense"}\n',
                      encoding="utf-8")
    assert [t.content for t in chat.load_turns(broken)] == ["kept"]


def test_save_is_silent_when_there_is_nowhere_to_save():
    chat.save(chat.Session(system="SYS"))     # path is None; must not raise


def test_transcript_is_readable_prose():
    session = chat.Session(system="SYS")
    session.turns = [chat.Turn("user", "an idea"), chat.Turn("assistant", "a reply")]
    assert session.transcript() == "Me: an idea\n\nMajordomo: a reply"


# ---------------------------------------------------------------------------
# Memory proposals
# ---------------------------------------------------------------------------


def test_propose_parses_the_line_format():
    reply = "preference | Prefers concise answers\nproject | Majordomo is his assistant"
    with mock.patch("majordomo.llm.complete", return_value=reply):
        candidates = chat.propose("transcript", CFG)

    assert [c.type for c in candidates] == ["preference", "project"]
    assert candidates[0].description == "Prefers concise answers"


def test_propose_understands_nothing():
    with mock.patch("majordomo.llm.complete", return_value="NOTHING"):
        assert chat.propose("transcript", CFG) == []


def test_propose_survives_a_model_failure():
    from majordomo.llm import LLMError

    with mock.patch("majordomo.llm.complete", side_effect=LLMError("503")):
        assert chat.propose("transcript", CFG) == []


def test_propose_skips_unparseable_lines_and_coerces_bad_types():
    reply = "no pipe here\n- nonsense | A real one\n| \n"
    with mock.patch("majordomo.llm.complete", return_value=reply):
        candidates = chat.propose("transcript", CFG)

    assert len(candidates) == 1
    assert candidates[0].type == "user"          # unknown type falls back
    assert candidates[0].description == "A real one"


# ---------------------------------------------------------------------------
# Scaffolding
# ---------------------------------------------------------------------------


def test_plan_refuses_a_non_empty_directory(tmp_path):
    cfg = config_module.build({**config_module.DEFAULTS, "scaffold": {"root": str(tmp_path)}})
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "file.txt").write_text("x", encoding="utf-8")

    with pytest.raises(scaffold.ScaffoldError, match="not empty"):
        scaffold.plan("Taken", "brief", cfg)


def test_plan_allows_an_empty_existing_directory(tmp_path):
    cfg = config_module.build({**config_module.DEFAULTS, "scaffold": {"root": str(tmp_path)}})
    (tmp_path / "empty").mkdir()
    assert scaffold.plan("Empty", "brief", cfg).path.name == "empty"


def test_dry_run_touches_nothing(tmp_path, capsys):
    cfg = config_module.build({**config_module.DEFAULTS, "scaffold": {"root": str(tmp_path)}})
    assert scaffold.start("A JSON differ", cfg, dry_run=True) is None
    assert not (tmp_path / "a-json-differ").exists()
    assert "Would create" in capsys.readouterr().out


def test_create_writes_the_files_and_inits_git(tmp_path):
    cfg = config_module.build({**config_module.DEFAULTS, "scaffold": {"root": str(tmp_path)}})
    target = scaffold.plan("A JSON differ", "THE BRIEF", cfg)

    path, warnings = scaffold.create(target)

    assert (path / "README.md").is_file()
    assert (path / "BRIEF.md").read_text(encoding="utf-8") == "THE BRIEF"
    assert (path / ".git").is_dir()
    assert warnings == []


def test_a_failed_git_init_warns_and_still_hands_off(tmp_path):
    """Raising here abandons a usable project AND makes every retry fail.

    By the time git runs, the directory and the brief exist — so an exception
    leaves the path non-empty, which `plan` then rejects forever.
    """
    cfg = config_module.build({**config_module.DEFAULTS, "scaffold": {"root": str(tmp_path)}})
    said = []

    failed = mock.Mock(returncode=1, stderr="git: command not found", stdout="")
    with mock.patch("subprocess.run", return_value=failed), mock.patch(
        "majordomo.resume.spawn_detached"
    ) as spawned:
        path = scaffold.start("A JSON differ", cfg, write=said.append)

    assert path is not None
    assert (path / "BRIEF.md").is_file()
    assert spawned.called                                   # Claude Code still opened
    assert any("git init" in line for line in said)         # and we said so


def test_a_target_that_is_a_file_gets_a_sentence_not_a_traceback(tmp_path):
    """`path.exists()` is true for a file; `iterdir()` then raises."""
    cfg = config_module.build({**config_module.DEFAULTS, "scaffold": {"root": str(tmp_path)}})
    (tmp_path / "a-json-differ").write_text("i am a file", encoding="utf-8")

    with pytest.raises(scaffold.ScaffoldError, match="exists as a file"):
        scaffold.plan("A JSON differ", "brief", cfg)


def test_start_launches_claude_in_the_new_directory(tmp_path):
    cfg = config_module.build({**config_module.DEFAULTS, "scaffold": {"root": str(tmp_path)}})

    with mock.patch("majordomo.resume.spawn_detached") as spawned:
        path = scaffold.start("A JSON differ", cfg, write=lambda *_: None)

    argv, kwargs = spawned.call_args
    assert argv[0][0] == "claude"
    assert kwargs["cwd"] == str(path)


def test_brief_carries_the_transcript_and_the_context(populated):
    ctx = context.build(CFG, query="idea", now=NOW).render()
    brief = scaffold.build_brief("A JSON differ", "Me: build it\n\nMajordomo: ok", ctx)

    assert brief.startswith("# A JSON differ")
    assert "How we got here" in brief
    assert "Me: build it" in brief
    assert "Windows first" in brief


def test_brief_omits_empty_sections():
    brief = scaffold.build_brief("A JSON differ", "", "")
    assert "How we got here" not in brief
    assert "Context on who" not in brief


def test_slug_is_filesystem_safe():
    assert scaffold.slug("A CLI that diffs two JSON files!") == "a-cli-that-diffs-two-json-files"
    assert scaffold.slug("???") == "project"
