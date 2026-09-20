"""Tests for the context block, the chat loop, and scaffolding.

The single most important assertion in this file is that the prompt *prefix* is
byte-identical across turns. Nothing breaks when it isn't — you simply pay full
price on every message, forever, with no error to notice. So it is asserted
directly rather than left to review.
"""
from __future__ import annotations

import io
import time
from datetime import datetime, timezone
from unittest import mock

import pytest

from majordomo import activity, chat, context, memory, prompts, scaffold
from majordomo import session as session_mod
from majordomo import config as config_module
from majordomo.paths import chats_dir

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
    session.log = [chat.Turn("user", "one"), chat.Turn("assistant", "two")]
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


# ---------------------------------------------------------------------------
# /clear and session selection
# ---------------------------------------------------------------------------

def test_clear_empties_the_turns_but_keeps_the_frozen_prefix():
    """Rebuilding the prefix would reread memory and activity and produce
    different bytes — which is exactly the caching guarantee context.py exists
    to hold. Clearing is about the turns."""
    session = chat.Session(system="SYS", turns=[chat.Turn("user", "one")], compactions=2)
    before = session.system
    old_path = session.path

    chat.clear(session)

    assert session.turns == []
    assert session.compactions == 0
    assert session.system == before
    assert session.path != old_path          # a new transcript, not an overwrite


def test_clear_starts_a_new_transcript_file(tmp_path):
    session = chat.Session(system="SYS", path=tmp_path / "old.jsonl")
    session.log = [chat.Turn("user", "kept")]
    chat.save(session)

    chat.clear(session)
    chat.save(session)

    assert chat.load_turns(tmp_path / "old.jsonl")[0].content == "kept"
    assert chat.load_turns(session.path) == []


def test_saved_sessions_is_oldest_first_and_never_raises():
    assert chat.saved_sessions() == []        # no chats dir yet

    for stamp in ("20260901-100000", "20260903-100000", "20260902-100000"):
        path = chats_dir() / f"{stamp}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"role":"user","content":"x"}\n', encoding="utf-8")

    assert [p.stem for p in chat.saved_sessions()] == [
        "20260901-100000", "20260902-100000", "20260903-100000"
    ]
    assert chat.latest_transcript().stem == "20260903-100000"


def _saved(stem, turns=1):
    path = chats_dir() / f"{stem}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join('{"role":"user","content":"a question"}\n' for _ in range(turns)),
        encoding="utf-8",
    )
    return path


def test_find_session_matches_a_unique_prefix():
    _saved("20260901-100000")
    _saved("20260903-120000")

    assert chat.find_session("20260901").stem == "20260901-100000"
    assert chat.find_session("20260903-12").stem == "20260903-120000"


def test_find_session_refuses_an_ambiguous_prefix():
    """Returning the first match would silently open the wrong conversation."""
    _saved("20260901-100000")
    _saved("20260901-110000")

    assert chat.find_session("20260901") is None
    assert chat.match_count("20260901") == 2


def test_find_session_on_no_match():
    assert chat.find_session("nope") is None
    assert chat.match_count("nope") == 0


def test_describe_session_shows_the_opening_question():
    path = _saved("20260901-100000", turns=3)
    line = chat.describe_session(path)

    assert "20260901-100000" in line
    assert "3 turns" in line
    assert "a question" in line


def test_new_session_can_open_a_named_transcript(populated):
    path = _saved("20260901-100000", turns=2)
    session = chat.new_session(CFG, transcript=path)

    assert len(session.turns) == 2
    assert session.path == path


def test_new_session_without_resume_starts_empty(populated):
    _saved("20260901-100000", turns=5)
    assert chat.new_session(CFG).turns == []


# ---------------------------------------------------------------------------
# Compaction: the failures found by actually driving it
# ---------------------------------------------------------------------------


def sized(role: str, tokens: int) -> chat.Turn:
    return chat.Turn(role, "x" * (tokens * 4))


def small_threshold(tokens: int):
    return config_module.build(
        {**config_module.DEFAULTS, "brain": {"chat_compact_threshold_tokens": tokens}}
    )


def test_few_but_huge_turns_can_still_compact():
    """The bug: compaction *triggered* on tokens but was *guarded* on turn
    count, so four pasted files crossed the threshold, hit a `<= 10 turns`
    guard, and could never compact. The conversation just grew until the
    provider rejected it."""
    config = small_threshold(2_000)
    session = chat.Session(
        system="SYS", turns=[sized("user", 1_500), sized("assistant", 1_500),
                             sized("user", 1_500), sized("assistant", 1_500)]
    )
    assert chat.needs_compaction(session, config)

    with mock.patch("majordomo.llm.complete", return_value="notes"):
        assert chat.compact(session, config) is True

    assert chat.estimate_tokens(session) < 4_000


def test_the_current_exchange_is_never_folded_away():
    """Compaction that eats the question produces a model answering something
    nobody asked."""
    config = small_threshold(100)
    session = chat.Session(
        system="SYS",
        turns=[sized("user", 900), sized("assistant", 900),
               chat.Turn("user", "the actual question")],
    )

    assert session_mod._recent_to_keep(session, config) >= chat.MIN_RECENT_TURNS

    with mock.patch("majordomo.llm.complete", return_value="notes"):
        chat.compact(session, config)

    assert session.turns[-1].content == "the actual question"


def test_an_existing_summary_is_carried_not_resummarised():
    """Re-summarising the summary is a telephone game: measured live, a fact
    stated in turn one survived the first compaction and was gone by the
    third."""
    config = small_threshold(500)
    session = chat.Session(
        system="SYS",
        turns=[
            chat.Turn("user", f"{chat.SUMMARY_MARKER}\n\nCat named Pilot, a tabby."),
            sized("assistant", 2_000),
            sized("user", 2_000),
            sized("assistant", 2_000),
        ],
    )

    with mock.patch("majordomo.llm.complete", return_value="notes") as called:
        chat.compact(session, config)

    prompt = called.call_args[0][0][0]["content"]
    assert "Cat named Pilot" in prompt
    assert "existing notes" in prompt.lower()
    # and it is presented as established fact, not as more conversation to fold
    assert "Reproduce every fact" in prompt


def test_nothing_new_to_fold_leaves_the_notes_alone():
    config = small_threshold(1)
    session = chat.Session(
        system="SYS",
        turns=[
            chat.Turn("user", f"{chat.SUMMARY_MARKER}\n\nNotes."),
            chat.Turn("assistant", "a"),
            chat.Turn("user", "b"),
        ],
    )

    with mock.patch("majordomo.llm.complete") as called:
        assert chat.compact(session, config) is False
    called.assert_not_called()


def test_the_summary_prompt_asks_for_facts_about_the_user():
    """It used to ask only for 'decisions and constraints', so a stated fact —
    the thing a personal assistant exists to retain — was correctly dropped."""
    config = small_threshold(10)
    session = chat.Session(
        system="SYS", turns=[sized("user", 100), sized("assistant", 100),
                             chat.Turn("user", "q"), chat.Turn("assistant", "a")]
    )

    with mock.patch("majordomo.llm.complete", return_value="notes") as called:
        chat.compact(session, config)

    prompt = called.call_args[0][0][0]["content"].lower()
    assert "about themselves" in prompt
    assert "anything they said to remember" in prompt


def test_a_failed_compaction_is_recorded_rather_than_discarded():
    """send() used to call compact() and ignore the result, so the one problem
    that compounds every turn was the one nothing reported."""
    config = small_threshold(10)
    session = chat.Session(system="SYS", turns=[])

    with mock.patch("majordomo.llm.complete", side_effect=["", "reply"]):
        chat.send(session, "x" * 400, config)

    assert session.compaction_failed is True


def test_a_successful_compaction_clears_the_flag():
    config = small_threshold(10)
    session = chat.Session(system="SYS", turns=[sized("user", 50), sized("assistant", 50),
                                                sized("user", 50), sized("assistant", 50)])
    session.compaction_failed = True

    with mock.patch("majordomo.llm.complete", side_effect=["notes", "reply"]):
        chat.send(session, "next", config)

    assert session.compaction_failed is False


def test_clear_resets_the_failure_flag():
    session = chat.Session(system="SYS", turns=[chat.Turn("user", "x")])
    session.compaction_failed = True
    session.compactions = 3

    chat.clear(session)

    assert session.compaction_failed is False
    assert session.compactions == 0
    assert session.system == "SYS"      # the frozen prefix survives


def test_compaction_does_not_destroy_the_saved_transcript(tmp_path):
    """save() used to write `turns`, which compaction shrinks — so folding a
    pasted file out of context also overwrote it on disk. Losing it from
    context is the feature; losing it from disk was data loss."""
    config = small_threshold(10)
    session = chat.Session(system="SYS", path=tmp_path / "c.jsonl")

    with mock.patch("majordomo.llm.complete", side_effect=["reply one", "notes", "reply two"]):
        chat.send(session, "a very long pasted file " * 50, config)
        chat.save(session)
        chat.send(session, "and a follow-up", config)
        chat.save(session)

    assert session.compactions == 1                          # it really did compact
    assert session.turns[0].content.startswith(chat.SUMMARY_MARKER)   # context folded
    assert "a very long pasted file" in session.log[0].content        # record intact

    saved = chat.load_turns(session.path)
    assert any("a very long pasted file" in t.content for t in saved)


def test_a_failed_call_writes_nothing_to_the_log():
    """The user turn is rolled out of context on failure; the record must not
    keep a question that was never answered."""
    from majordomo.llm import LLMError

    config = small_threshold(100_000)
    session = chat.Session(system="SYS")

    with mock.patch("majordomo.llm.complete", side_effect=LLMError("503")):
        with pytest.raises(chat.ChatFailed):
            chat.send(session, "unanswered", config)

    assert session.log == []
    assert session.turns == []


def test_resuming_restores_both_the_context_and_the_record(tmp_path, monkeypatch):
    session = chat.Session(system="SYS", path=tmp_path / "c.jsonl")
    session.log = [chat.Turn("user", "one"), chat.Turn("assistant", "two")]
    chat.save(session)

    resumed = chat.new_session(CFG, transcript=session.path)

    assert [t.content for t in resumed.turns] == ["one", "two"]
    assert [t.content for t in resumed.log] == ["one", "two"]


# ---------------------------------------------------------------------------
# /agent inside a conversation
# ---------------------------------------------------------------------------


def fake_terminal(said=None, answer=""):
    """A console that records what was written and gives one canned answer.

    Replaces `mock.patch("builtins.input")` throughout. That the prompt is now
    an injected callable rather than a global is the whole point of `Terminal`:
    the test says what it wants instead of reaching into the interpreter.
    """
    said = [] if said is None else said
    return chat.Terminal(
        write=said.append,
        ask=lambda _question: answer,
        confirm=lambda _name, _arguments: answer.lower() in ("y", "yes"),
    )


def handle(line, session, config=CFG, answer=""):
    written = []
    ended = chat._handle_command(line, session, config, fake_terminal(written, answer))
    return ended, "\n".join(written)


def test_a_missing_key_does_not_kill_the_repl(tmp_path):
    """MissingApiKey is a bare Exception, not an LLMError, so agent.run's
    handler missed it and it propagated out through _handle_command — taking
    the conversation with it. cmd_do guards the identical call."""
    from majordomo.llm import MissingApiKey

    session = chat.Session(system="SYS", path=tmp_path / "c.jsonl")
    session.turns = [chat.Turn("user", "earlier")]
    session.log = [chat.Turn("user", "earlier")]

    with mock.patch("majordomo.agent.run", side_effect=MissingApiKey("Set OPENROUTER_API_KEY")):
        ended, written = handle("/agent do something", session)

    assert ended is False                        # back to the prompt
    assert "OPENROUTER_API_KEY" in written
    assert session.turns[0].content == "earlier"  # conversation intact


def test_agent_turns_reach_the_saved_transcript(tmp_path):
    """save() writes `log`, so appending to `turns` alone silently dropped the
    agent's work from the transcript."""
    from majordomo import agent as agent_mod

    session = chat.Session(system="SYS", path=tmp_path / "c.jsonl")
    outcome = agent_mod.Outcome(answer="Added the docstring.")

    with mock.patch("majordomo.agent.run", return_value=outcome):
        handle("/agent add a docstring", session)

    saved = [t.content for t in chat.load_turns(session.path)]
    assert any("add a docstring" in c for c in saved)
    assert "Added the docstring." in saved
    assert [t.content for t in session.turns] == saved   # context and record agree


def test_an_agent_that_stopped_early_is_still_recorded(tmp_path):
    from majordomo import agent as agent_mod

    session = chat.Session(system="SYS", path=tmp_path / "c.jsonl")
    outcome = agent_mod.Outcome(stopped_because="reached the 24-step limit")

    with mock.patch("majordomo.agent.run", return_value=outcome):
        _, written = handle("/agent loop forever", session)

    assert "24-step limit" in written
    assert "24-step limit" in chat.load_turns(session.path)[-1].content


def test_quitting_without_typing_writes_no_transcript(tmp_path, monkeypatch):
    """An empty transcript still sorts newest by mtime, so opening the REPL and
    quitting made *that* the latest session — and --resume then restored nothing
    over yesterday's conversation."""
    monkeypatch.setattr(session_mod, "chats_dir", lambda: tmp_path)
    yesterday = tmp_path / "20260901-120000.jsonl"
    yesterday.write_text('{"role": "user", "content": "real work", "at": ""}\n', encoding="utf-8")

    session = chat.Session(system="SYS", path=tmp_path / "20260902-120000.jsonl")
    with mock.patch.object(chat, "new_session", return_value=session):
        with mock.patch("majordomo.keys.read_line", side_effect=EOFError):
            chat.run(CFG)

    assert not session.path.exists()
    assert chat.latest_transcript() == yesterday


def test_a_conversation_is_still_saved_on_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "chats_dir", lambda: tmp_path)
    session = chat.Session(system="SYS", path=tmp_path / "c.jsonl")
    session.log = [chat.Turn("user", "something real")]

    with mock.patch.object(chat, "new_session", return_value=session):
        with mock.patch("majordomo.keys.read_line", side_effect=EOFError):
            with mock.patch.object(chat, "_offer_memories"):
                chat.run(CFG)

    assert "something real" in chat.load_turns(session.path)[0].content


def test_a_compacted_conversation_is_still_saved(tmp_path, monkeypatch):
    """Guarding on `turns` rather than `log` would skip saving a long
    conversation that had just been folded down."""
    monkeypatch.setattr(session_mod, "chats_dir", lambda: tmp_path)
    session = chat.Session(system="SYS", path=tmp_path / "c.jsonl")
    session.log = [chat.Turn("user", "hours of work")]
    session.turns = []                       # as if compaction emptied it

    with mock.patch.object(chat, "new_session", return_value=session):
        with mock.patch("majordomo.keys.read_line", side_effect=EOFError):
            with mock.patch.object(chat, "_offer_memories"):
                chat.run(CFG)

    assert session.path.exists()


def test_resume_skips_an_empty_transcript(tmp_path, monkeypatch):
    """`run` no longer writes empty ones, but any already on disk would still
    sort newest and shadow real work — and resuming nothing is worse than
    reaching one file further back."""
    monkeypatch.setattr(session_mod, "chats_dir", lambda: tmp_path)
    real = tmp_path / "20260901-120000.jsonl"
    real.write_text('{"role": "user", "content": "real", "at": ""}\n', encoding="utf-8")
    (tmp_path / "20260902-120000.jsonl").write_text("", encoding="utf-8")

    assert chat.latest_transcript() == real


def test_an_empty_transcript_is_ignored_not_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "chats_dir", lambda: tmp_path)
    (tmp_path / "20260901-120000.jsonl").write_text("x", encoding="utf-8")
    empty = tmp_path / "20260902-120000.jsonl"
    empty.write_text("", encoding="utf-8")

    chat.latest_transcript()

    assert empty.exists()                       # yours to remove, not ours
    assert len(chat.saved_sessions()) == 2      # --list still shows both


def test_no_transcripts_at_all_resumes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "chats_dir", lambda: tmp_path)
    assert chat.latest_transcript() is None


# ---------------------------------------------------------------------------
# The REPL survives a missing key, and the record outlives compaction
# ---------------------------------------------------------------------------


def test_a_missing_key_does_not_kill_an_ordinary_message():
    """cmd_do and /agent were both guarded; `send` — every message you type —
    was not, so the first one with no key killed the REPL and the exit-path
    save() never ran."""
    from majordomo.llm import MissingApiKey

    session = chat.Session(system="SYS", turns=[chat.Turn("user", "earlier")])

    with mock.patch("majordomo.llm.complete", side_effect=MissingApiKey("Set OPENROUTER_API_KEY")):
        with pytest.raises(chat.ChatFailed) as exc_info:
            chat.send(session, "hello", CFG)

    assert "OPENROUTER_API_KEY" in str(exc_info.value)     # still fixable
    assert [t.content for t in session.turns] == ["earlier"]   # rolled back


def test_a_missing_key_during_compaction_is_not_fatal():
    from majordomo.llm import MissingApiKey

    config = small_threshold(10)
    session = chat.Session(
        system="SYS",
        turns=[sized("user", 50), sized("assistant", 50), sized("user", 50),
               sized("assistant", 50)],
    )

    with mock.patch("majordomo.llm.complete", side_effect=MissingApiKey("no key")):
        assert chat.compact(session, config) is False

    assert len(session.turns) == 4          # nothing lost


def test_the_transcript_is_what_was_said_not_the_summary():
    """It feeds /remember and /build — the two moments a session decides what
    to keep permanently. A summary is the wrong input at exactly that point."""
    session = chat.Session(system="SYS")
    session.log = [chat.Turn("user", "I prefer dark roast"),
                   chat.Turn("assistant", "Noted.")]
    session.turns = [chat.Turn("user", f"{chat.SUMMARY_MARKER}\n\nlikes coffee")]

    text = session.transcript()

    assert "I prefer dark roast" in text
    assert chat.SUMMARY_MARKER not in text


def test_a_session_with_no_log_still_has_a_transcript():
    """Constructed by hand, as much of the test suite does."""
    session = chat.Session(system="SYS", turns=[chat.Turn("user", "hi")])
    assert "hi" in session.transcript()


def test_resuming_a_long_conversation_compacts_before_the_first_message(tmp_path, monkeypatch):
    """The file is the record and is never compacted, so a resumed chat arrives
    at full length. Folding it on the first message means re-sending the whole
    history once, at full price, before deciding it was too long."""
    monkeypatch.setattr(session_mod, "chats_dir", lambda: tmp_path)
    path = tmp_path / "c.jsonl"
    big = chat.Session(system="SYS", path=path)
    big.log = [sized("user", 400), sized("assistant", 400)] * 6
    chat.save(big)

    config = small_threshold(2_000)
    with mock.patch("majordomo.llm.complete", return_value="notes"):
        resumed = chat.new_session(config, transcript=path)

    assert resumed.compactions == 1
    assert chat.estimate_tokens(resumed) < 4_800
    assert len(resumed.log) == 12          # the record is untouched


def test_resuming_a_short_conversation_does_not_compact(tmp_path, monkeypatch):
    monkeypatch.setattr(session_mod, "chats_dir", lambda: tmp_path)
    path = tmp_path / "c.jsonl"
    small = chat.Session(system="SYS", path=path)
    small.log = [chat.Turn("user", "hi"), chat.Turn("assistant", "hello")]
    chat.save(small)

    with mock.patch("majordomo.llm.complete") as called:
        resumed = chat.new_session(CFG, transcript=path)

    called.assert_not_called()
    assert len(resumed.turns) == 2


# ---------------------------------------------------------------------------
# Handing a request to the agent
# ---------------------------------------------------------------------------


def test_the_marker_is_parsed_structurally():
    """The same discipline as `needs_you`: decided in Python from a marker the
    model was told to emit, never inferred from how a sentence reads."""
    assert prompts.needs_agent("NEEDS_AGENT: review the raad-whatsapp folder") == (
        "review the raad-whatsapp folder"
    )
    assert prompts.needs_agent("  NEEDS_AGENT:  trim it  ") == "trim it"
    assert prompts.needs_agent("Sure, here is the answer.") is None


def test_the_task_may_sit_on_the_next_line():
    """`NEEDS_AGENT:\n<task>` returned None, and a None means no offer fires and
    the raw reply prints — putting the protocol token on screen, which is the
    one outcome it exists to prevent."""
    assert prompts.needs_agent("NEEDS_AGENT:\nreview the api folder") == (
        "review the api folder"
    )
    assert prompts.needs_agent("NEEDS_AGENT:\n\n  look inside  ") == "look inside"
    assert prompts.needs_agent("NEEDS_AGENT:\n```\nread it\n```") == "read it"


def test_a_bare_marker_is_still_a_hand_off():
    """Empty, not None: the caller falls back to what was asked for. None would
    print the marker."""
    assert prompts.needs_agent("NEEDS_AGENT:") == ""


def test_an_ordinary_answer_costs_one_membership_test():
    assert prompts.needs_agent("a long ordinary reply\nover several lines") is None


@pytest.mark.parametrize(
    "leaked",
    [
        "<dots_function_call>\ninvoke name bash",
        '<tool_call>{"name": "ls"}</tool_call>',
        "<function=read_file>{}</function>",
        '{"tool_calls": [{"function": {"name": "ls"}}]}',
    ],
)
def test_leaked_tool_call_markup_is_recognised(leaked):
    """Observed with dots-3, which wrote out a <dots_function_call> block
    complete with a shell command when asked to look inside a folder. Raw XML in
    the terminal reads as though something ran. Nothing did."""
    assert prompts.looks_like_a_tool_call(leaked)


@pytest.mark.parametrize(
    "ordinary",
    [
        "Use `ls -la` to list the files.",
        "The function call syntax varies by provider.",
        "I would call the function `add(a, b)` here.",
    ],
)
def test_ordinary_prose_about_functions_is_not_mistaken_for_one(ordinary):
    assert not prompts.looks_like_a_tool_call(ordinary)


def offer(reply, asked="look at that folder", answer="n", seeded=False):
    said = []
    session = chat.Session(system="SYS")
    if seeded:
        # As `send` leaves it: the raw reply already appended, which is what
        # `_restate_last_answer` exists to correct.
        session.turns = [chat.Turn("assistant", reply)]
        session.log = list(session.turns)
    took_over = chat._offer_agent(
        session, CFG, reply, asked, fake_terminal(said, answer)
    )
    return took_over, "\n".join(said), session


def test_an_ordinary_reply_is_left_alone():
    took_over, said, _ = offer("Coffee is a matter of taste.")
    assert took_over is False
    assert said == ""


def test_the_marker_produces_an_offer():
    took_over, said, _ = offer("NEEDS_AGENT: review the raad-whatsapp folder")

    assert took_over is True                       # chat does not print the marker
    assert "needs the agent" in said
    assert "review the raad-whatsapp folder" in said


def test_declining_runs_nothing_and_says_how():
    with mock.patch("majordomo.chat.run_agent") as ran:
        took_over, said, _ = offer("NEEDS_AGENT: delete everything", answer="n")

    ran.assert_not_called()
    assert took_over is True
    assert "/agent" in said


def test_accepting_runs_exactly_what_agent_would():
    """The offer exists to *be* the command, so it must not be a second path."""
    with mock.patch("majordomo.chat.run_agent") as ran:
        offer("NEEDS_AGENT: add a docstring to render.py", answer="y")

    assert ran.call_args[0][2] == "add a docstring to render.py"


def test_leaked_markup_is_replaced_not_printed():
    took_over, said, _ = offer("<dots_function_call>\ninvoke name bash\nls -la")

    assert took_over is True
    assert "dots_function_call" not in said        # never reaches the terminal
    assert "cannot read files" in said


def test_leaked_markup_falls_back_to_what_you_asked_for():
    """There is no marker to read a task from, so the user's own sentence is
    the best available description of the job."""
    with mock.patch("majordomo.chat.run_agent") as ran:
        offer("<tool_call>ls</tool_call>", asked="check the config folder", answer="y")

    assert ran.call_args[0][2] == "check the config folder"


def test_a_refused_prompt_declines_rather_than_running():
    """No answer at all — a closed stdin, a Ctrl+C — is a refusal."""
    session = chat.Session(system="SYS")
    with mock.patch("majordomo.chat.run_agent") as ran:
        chat._offer_agent(
            session, CFG, "NEEDS_AGENT: x", "x", chat.HEADLESS
        )

    ran.assert_not_called()


def test_agent_and_the_offer_share_one_runner():
    """Two copies would eventually stop agreeing about what the command does."""
    import inspect

    source = inspect.getsource(chat._handle_command)
    assert "run_agent(session, config, argument, terminal)" in source


# ---------------------------------------------------------------------------
# The detector must not eat correct answers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "discussion",
    [
        'OpenAI returns a "tool_calls" array in the message object.',
        'The Anthropic API uses <invoke name="get_weather"> in its examples.',
        "You parse tool_calls from choices[0].message.",
        "A <tool_call> block is what providers emit.",
        "The <function=name> syntax varies between providers.",
        "Majordomo defines its tools in tools.py, then passes them as `tools`.",
        # Fenced: quoted precisely because it is an illustration. Allowing a
        # fence *prefix* did nothing — the token still starts its own line
        # inside the block — so the answer was truncated at the fence.
        "A tool call looks like this:\n\n```\n<tool_call>\n{}\n```\n\nThat is the shape.",
        "Example:\n```json\n\"tool_calls\": []}\n```",
    ],
)
def test_talking_about_tool_calls_is_not_emitting_one(discussion):
    """A substring net over the reply flagged correct answers — and this project
    is itself an LLM tool, so "how do tool calls work?" is a question actually
    worth asking. An emitted call opens its own line; a mention sits inside a
    sentence."""
    assert not prompts.looks_like_a_tool_call(discussion)


@pytest.mark.parametrize(
    "emitted",
    [
        "<dots_function_call>\ninvoke name bash",
        '<tool_call>{"name": "ls"}</tool_call>',
        "<function=read_file>{}</function>",
        '{"tool_calls": [{"function": {"name": "ls"}}]}',
        "Here is what I found.\n\n<tool_call>ls -la</tool_call>",
    ],
)
def test_an_emitted_call_is_still_caught(emitted):
    assert prompts.looks_like_a_tool_call(emitted)


def test_stripping_keeps_the_prose_around_the_block():
    """A model that wrote three good paragraphs and one stray block should lose
    the block, not the paragraphs."""
    reply = "First point.\n\nSecond point.\n\n<tool_call>{\"name\": \"ls\"}</tool_call>"

    kept = prompts.strip_tool_call(reply)

    assert "First point." in kept and "Second point." in kept
    assert "tool_call" not in kept


def test_stripping_an_ordinary_reply_changes_nothing():
    assert prompts.strip_tool_call("Just an answer.") == "Just an answer."


# ---------------------------------------------------------------------------
# What is stored is what you were shown
# ---------------------------------------------------------------------------


def stored_after(reply, answer="n"):
    session = chat.Session(system="SYS")
    session.turns = [chat.Turn("user", "look at that folder"),
                     chat.Turn("assistant", reply)]
    session.log = list(session.turns)
    with mock.patch("majordomo.chat.run_agent"):
        chat._offer_agent(
            session, CFG, reply, "look at that folder", fake_terminal(answer=answer)
        )
    return session


def test_a_bare_marker_is_not_left_in_the_conversation():
    """`send` stores the reply the moment it arrives, so suppressing it on
    screen alone left the marker replayed to the model next turn, restored by
    --resume, and handed to the memory proposer at exit."""
    session = stored_after("NEEDS_AGENT: review the raad-whatsapp folder")

    assert prompts.NEEDS_AGENT_MARKER not in session.turns[-1].content
    assert prompts.NEEDS_AGENT_MARKER not in session.log[-1].content
    assert "needs the agent" in session.turns[-1].content


def test_leaked_markup_is_not_left_in_the_conversation():
    session = stored_after("Some prose.\n\n<tool_call>ls</tool_call>")

    assert "tool_call" not in session.turns[-1].content
    assert "tool_call" not in session.log[-1].content
    assert "Some prose." in session.turns[-1].content      # the answer survives


def test_the_context_and_the_record_agree():
    session = stored_after("NEEDS_AGENT: do a thing")
    assert session.turns[-1].content == session.log[-1].content


def test_declining_still_leaves_you_the_answer():
    """Decline the offer and the prose that came with the block must remain."""
    said = []
    session = chat.Session(system="SYS")
    session.turns = [chat.Turn("assistant", "x")]
    session.log = list(session.turns)

    chat._offer_agent(
        session, CFG,
        "Three good paragraphs.\n\n<tool_call>ls</tool_call>",
        "look", fake_terminal(said, "n"),
    )

    assert "Three good paragraphs." in "\n".join(said)


# ---------------------------------------------------------------------------
# The wait
# ---------------------------------------------------------------------------


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


def test_nothing_is_printed_when_output_is_not_a_terminal():
    """Redirected, a session would fill with timer frames."""
    plain = io.StringIO()

    with chat.Waiting(stream=plain):
        time.sleep(chat.Waiting.INTERVAL * 3)

    assert plain.getvalue() == ""


def test_the_timer_shows_seconds_not_just_activity():
    """Latency here ranges 2–31s on the same prompt. A spinner looks identical
    at both; the number is what tells you it is stuck."""
    tty = FakeTTY()

    with mock.patch("majordomo.render.supports_ansi", return_value=True):
        with chat.Waiting(stream=tty):
            time.sleep(chat.Waiting.INTERVAL * 3)

    frames = tty.getvalue()
    assert "thinking" in frames
    assert "s" in frames


def test_the_line_is_cleared_when_the_reply_lands():
    """Residue on the prompt line is worse than no timer at all."""
    tty = FakeTTY()

    with mock.patch("majordomo.render.supports_ansi", return_value=True):
        with chat.Waiting(stream=tty):
            time.sleep(chat.Waiting.INTERVAL * 2)

    last = tty.getvalue().split("\r")[-1]
    assert last.strip() == ""


def test_a_closed_stream_does_not_take_the_conversation_down():
    """A progress indicator is never worth that."""
    tty = FakeTTY()
    tty.close()

    with mock.patch("majordomo.render.supports_ansi", return_value=True):
        with chat.Waiting(stream=tty):
            time.sleep(chat.Waiting.INTERVAL * 2)


def test_the_thread_stops_with_the_block():
    import threading

    before = threading.active_count()
    with mock.patch("majordomo.render.supports_ansi", return_value=True):
        with chat.Waiting(stream=FakeTTY()):
            time.sleep(chat.Waiting.INTERVAL)
    time.sleep(chat.Waiting.INTERVAL * 2)

    assert threading.active_count() <= before


# ---------------------------------------------------------------------------
# "Remember this"
# ---------------------------------------------------------------------------


def test_the_remember_marker_is_parsed_structurally():
    """Whether a sentence "sounds like" the user asked to be remembered is not
    something to branch on."""
    reply = "Noted.\n\nREMEMBER: He prefers medium-dark roast coffee."

    assert prompts.wants_remembered(reply) == "He prefers medium-dark roast coffee."
    assert prompts.wants_remembered("just an answer") is None


def test_the_marker_line_never_reaches_the_reader():
    reply = "Noted, dark roast it is.\n\nREMEMBER: He prefers dark roast."
    assert prompts.strip_remember(reply) == "Noted, dark roast it is."


def test_only_chat_is_told_about_the_protocol():
    """`mj ask` is one-shot — nobody to confirm to, so a marker would print."""
    assert prompts.REMEMBER_MARKER in prompts.build_chat_system_prompt("")
    assert prompts.REMEMBER_MARKER not in prompts.build_ask_prompt("", "q")[0]["content"]


def remembering(reply, answer="y"):
    """The two steps `run` takes: strip the marker, then ask about it.

    Separate on purpose — the marker must come out before anything is shown
    or stored, and the question must come after, or you approve a memory
    without having seen what produced it.
    """
    said = []
    cleaned, fact = chat._proposed_memory(reply)
    if fact:
        chat._save_memory(fact, CFG, fake_terminal(said, answer))
    return cleaned, "\n".join(said)


def test_saying_remember_this_offers_and_writes():
    """It used to get a friendly "noted" and nothing on disk."""
    cleaned, said = remembering("Noted.\n\nREMEMBER: He prefers dark roast coffee.")

    assert "dark roast coffee" in said
    assert [m.description for m in memory.read_all()] == [
        "He prefers dark roast coffee."
    ]
    assert cleaned == "Noted."


def test_declining_writes_nothing():
    remembering("Noted.\n\nREMEMBER: He prefers dark roast.", answer="n")
    assert memory.read_all() == []


def test_the_marker_is_stripped_even_when_declined():
    """It was never meant to be read, and the caller stores what comes back."""
    cleaned, _ = remembering("Noted.\n\nREMEMBER: something", answer="n")
    assert prompts.REMEMBER_MARKER not in cleaned


def test_an_ordinary_reply_is_returned_untouched():
    cleaned, said = remembering("Coffee is a matter of taste.")
    assert cleaned == "Coffee is a matter of taste."
    assert said == ""


def test_a_refused_memory_says_why_and_does_not_crash():
    """A credential in the fact is refused by write_memory, and that has to
    land as a line rather than a traceback in the REPL."""
    _cleaned, said = remembering("Sure.\n\nREMEMBER: his key is sk-abcdefghijklmnop1234")

    assert "not saved" in said
    assert memory.read_all() == []


def test_remember_with_an_argument_writes_directly():
    """The explicit path, for when you already know what you want kept."""
    session = chat.Session(system="SYS")

    chat._handle_command(
        "/remember He builds for Windows first", session, CFG, fake_terminal(answer="y")
    )

    assert [m.description for m in memory.read_all()] == ["He builds for Windows first"]


def test_remember_with_no_argument_still_proposes():
    session = chat.Session(system="SYS", turns=[chat.Turn("user", "hi")])
    with mock.patch.object(chat, "_offer_memories") as proposed:
        chat._handle_command("/remember", session, CFG, lambda *_: None)
    proposed.assert_called_once()


def test_memory_disabled_is_reported_not_silently_skipped():
    config = config_module.build(
        {**config_module.DEFAULTS, "memory": {"enabled": False}}
    )
    said = []
    chat._save_memory("a fact", config, fake_terminal(said))

    assert "disabled" in "\n".join(said)
    assert memory.read_all() == []


def test_prose_before_the_marker_is_kept():
    """The leaked-tool-call branch already did this. The marker branch did not,
    so anything written before it vanished from the screen, `turns`, `log` and
    the memory proposer — even when the offer was declined."""
    reply = "Sure, I can point you at that.\n\nNEEDS_AGENT: list the api folder"
    took_over, said, session = offer(reply)

    assert took_over is True
    assert "Sure, I can point you at that." in said
    assert prompts.NEEDS_AGENT_MARKER not in said


def test_the_kept_prose_is_what_gets_stored():
    reply = "Here is some context.\n\nNEEDS_AGENT: do the thing"
    _took, _said, session = offer(reply, seeded=True)

    assert "Here is some context." in session.turns[-1].content
    assert prompts.NEEDS_AGENT_MARKER not in session.log[-1].content


def test_both_branches_strip_their_own_marker():
    """Two paths doing the same job differently is how one of them stays wrong."""
    assert prompts.strip_needs_agent("prose\n\nNEEDS_AGENT: x") == "prose"
    assert prompts.strip_tool_call("prose\n\n<tool_call>x</tool_call>") == "prose"


def test_a_reply_that_is_only_a_remember_line_stores_something():
    """It stripped to "" and an empty assistant turn was stored, replayed and
    resumed as a blank. `send` guards the same case."""
    cleaned, _said = remembering("REMEMBER: He prefers dark roast.")
    assert cleaned.strip() != ""


# ---------------------------------------------------------------------------
# Every command branch, run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    ["/help", "/context", "/clear", "/remember", "/remember a fact", "/agent",
     "/agent do a thing", "/build", "/build mytool", "/voice", "/exit", "/nonsense"],
)
def test_every_command_branch_survives_being_run(command, tmp_path):
    """`/build` referenced a parameter that had been renamed out from under it,
    and 976 tests passed because none of them ran it. A NameError there unwinds
    out of the REPL and the exit-path save never happens, so the conversation
    is gone. This is the cheapest thing that would have caught it."""
    session = chat.Session(
        system="SYS",
        turns=[chat.Turn("user", "x"), chat.Turn("assistant", "y")],
        path=tmp_path / "c.jsonl",
    )
    session.log = list(session.turns)

    with mock.patch("majordomo.scaffold.from_chat"), \
         mock.patch("majordomo.chat.run_agent"), \
         mock.patch("majordomo.chat.propose", return_value=[]), \
         mock.patch("majordomo.chat._listen", return_value=None):
        chat._handle_command(command, session, CFG, fake_terminal(answer="n"))


def test_clear_reports_without_a_file_too():
    """`save` already tolerates a session with no path; naming the file
    afterwards did not."""
    said = []
    session = chat.Session(system="SYS", turns=[chat.Turn("user", "x")], path=None)

    chat._handle_command("/clear", session, CFG, fake_terminal(said))

    assert "Saved 1 turns." in "\n".join(said)


def test_build_hands_the_terminal_to_the_scaffolder():
    session = chat.Session(system="SYS", turns=[chat.Turn("user", "an idea")])
    said = []

    with mock.patch("majordomo.scaffold.from_chat") as built:
        chat._handle_command("/build mytool", session, CFG, fake_terminal(said))

    assert built.call_args.kwargs["write"] is not None


# ---------------------------------------------------------------------------
# One clock
# ---------------------------------------------------------------------------


def test_both_layers_stamp_turns_from_the_same_clock():
    """Harmless while both were byte-identical — and the moment either is made
    injectable for testing compaction, the terminal layer and the session model
    start stamping from different clocks and ordering stops being reliable."""
    assert chat._now is session_mod._now


# ---------------------------------------------------------------------------
# Ctrl+C means stop, not "no"
# ---------------------------------------------------------------------------


def interrupting(said=None):
    said = [] if said is None else said

    def ask(_question):
        raise KeyboardInterrupt

    return chat.Terminal(write=said.append, ask=ask), said


def test_an_interrupt_stops_the_memory_review_rather_than_declining_one():
    from majordomo import memory as memory_mod

    terminal, said = interrupting()
    session = chat.Session(system="SYS", turns=[chat.Turn("user", "hi")])
    candidates = [memory_mod.MemoryCandidate(description=f"fact {n}") for n in range(3)]

    with mock.patch("majordomo.chat.propose", return_value=candidates):
        chat._offer_memories(session, CFG, terminal)

    shown = [line for line in said if "fact" in line]
    assert len(shown) == 1               # it stopped, it did not move on
    assert memory.read_all() == []


def test_an_interrupt_at_the_agent_offer_keeps_the_conversation():
    """Ending a conversation over a change of mind about one task would be a
    worse answer than declining it."""
    terminal, _said = interrupting()
    session = chat.Session(system="SYS", turns=[chat.Turn("assistant", "x")])
    session.log = list(session.turns)

    with mock.patch("majordomo.chat.run_agent") as ran:
        handled = chat._offer_agent(
            session, CFG, "NEEDS_AGENT: a task", "a task", terminal
        )

    assert handled is True
    ran.assert_not_called()


# ---------------------------------------------------------------------------
# The memory question comes after the answer
# ---------------------------------------------------------------------------


def test_the_marker_is_split_out_without_asking_anything():
    """Extraction is pure and separate from the asking: the marker must leave
    the text before anything is shown or stored, and the question must come
    after, or you approve a memory without seeing what produced it."""
    cleaned, fact = chat._proposed_memory("Noted.\n\nREMEMBER: He prefers dark roast.")

    assert cleaned == "Noted."
    assert fact == "He prefers dark roast."


def test_an_ordinary_reply_proposes_nothing():
    cleaned, fact = chat._proposed_memory("Coffee is a matter of taste.")
    assert cleaned == "Coffee is a matter of taste."
    assert fact is None


def test_a_reply_that_is_only_the_marker_still_stores_something():
    cleaned, fact = chat._proposed_memory("REMEMBER: He prefers dark roast.")
    assert cleaned.strip() != ""
    assert fact == "He prefers dark roast."
