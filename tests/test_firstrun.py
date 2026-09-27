"""Tests for the first-run flow.

Two things are being guarded here beyond "does it work".

**It must run exactly once, and must not greet an existing install.** The
trigger is a marker file rather than "does ``~/.majordomo`` exist", because that
directory appears from any first write — the hook creates it to log a failure,
``config --init`` creates it to hold the config. Someone whose first act was
``mj install-hooks`` would otherwise never be introduced at all.

**It must not touch anything real.** The flow installs Claude Code hooks and
Windows scheduled tasks when you say yes, so every test here answers no unless
it is specifically about the yes path, and the ``isolated_claude_home`` fixture
in ``conftest`` is the belt to this file's braces.
"""
from __future__ import annotations

import pytest

from majordomo import config as config_mod
from majordomo import firstrun, paths


def loaded():
    """The config `cli` would hand in. Loaded here, not inside `run`."""
    return config_mod.load(None)


def answers(*values):
    """A fake `ask` that replies with each value in turn, then "" forever.

    Running dry rather than raising is deliberate: "" is what `cli.ask` returns
    when nobody is there, so a flow that asks one question more than a test
    expected declines it instead of erroring, which is the production behaviour.
    """
    queue = list(values)

    def ask(_question: str) -> str:
        return queue.pop(0) if queue else ""

    return ask


def silent(_text: str = "") -> None:
    pass


def collected(lines):
    return lambda text="": lines.append(text)


# ---------------------------------------------------------------------------
# needed()
# ---------------------------------------------------------------------------

def test_a_fresh_home_needs_setup():
    assert firstrun.needed() is True


def test_the_marker_means_we_have_met():
    paths.ensure_home()
    firstrun.marker_path().write_text("x", encoding="utf-8")
    assert firstrun.needed() is False


@pytest.mark.parametrize("evidence", ["config.yml", "state.jsonl"])
def test_a_lived_in_home_is_not_a_first_run(evidence):
    """Everyone already using this has a home and no marker. Greeting them with
    setup would be a worse first impression than the menu was."""
    home = paths.ensure_home()
    (home / evidence).write_text("x", encoding="utf-8")

    assert firstrun.needed() is False
    # And it records the conclusion, so the check is one stat call next time.
    assert firstrun.marker_path().exists()


@pytest.mark.parametrize("evidence", ["memory", "chats"])
def test_a_lived_in_home_is_not_a_first_run_for_directories(evidence):
    home = paths.ensure_home()
    (home / evidence).mkdir()
    assert firstrun.needed() is False


def test_a_key_already_in_dot_env_counts_as_set_up():
    """The README has always said to put your key in ~/.majordomo/.env.

    Doing exactly what the documentation says was the one way to be greeted with
    "first run — three questions" after configuring the thing.
    """
    home = paths.ensure_home()
    (home / ".env").write_text("OPENROUTER_API_KEY=x\n", encoding="utf-8")
    assert firstrun.needed() is False


def test_the_hook_creating_the_home_does_not_count_as_set_up():
    """The failure the marker exists to prevent.

    `hook.log_error` mkdirs the home to write a traceback. If existence were the
    signal, one logged hook failure would mean you are never introduced.
    """
    paths.ensure_home()  # the directory, and nothing in it
    assert firstrun.needed() is True


# ---------------------------------------------------------------------------
# run() — the marker
# ---------------------------------------------------------------------------

def test_declining_everything_still_counts_as_meeting(monkeypatch):
    """"Have we met" is answered by a run, not by what was agreed to."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    report = firstrun.run(config=loaded(), write=silent, ask=answers())

    assert firstrun.marker_path().exists()
    assert firstrun.needed() is False
    assert report.did == []
    assert "hooks" in report.skipped
    assert "trigger" in report.skipped


def test_setup_asks_nothing_twice(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    firstrun.run(config=loaded(), write=silent, ask=answers())
    assert firstrun.needed() is False


# ---------------------------------------------------------------------------
# run() — the key
# ---------------------------------------------------------------------------

def test_a_key_lands_on_disk_and_in_the_environment(monkeypatch):
    """Both, or the very first turn fails as though the key were wrong.

    `main` loads the .env before dispatch, so this process never re-reads it.
    A key written to disk but absent from os.environ produces a MissingApiKey on
    the first message — indistinguishable, to the user, from a bad key.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from majordomo import dotenv

    report = firstrun.run(config=loaded(), write=silent, ask=answers("sk-or-typed"))

    assert "key" in report.did
    assert "OPENROUTER_API_KEY=sk-or-typed" in dotenv.env_path().read_text(encoding="utf-8")
    import os

    assert os.environ["OPENROUTER_API_KEY"] == "sk-or-typed"


def test_an_existing_key_is_left_alone(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-already-here")
    from majordomo import dotenv

    report = firstrun.run(config=loaded(), write=silent, ask=answers())

    assert any("already set" in s for s in report.skipped)
    assert not dotenv.env_path().exists()


def test_writing_a_key_preserves_other_lines(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from majordomo import dotenv

    paths.ensure_home()
    dotenv.env_path().write_text("NVIDIA_API_KEY=keep-me\n", encoding="utf-8")

    firstrun.run(config=loaded(), write=silent, ask=answers("sk-or-new"))

    body = dotenv.env_path().read_text(encoding="utf-8")
    assert "NVIDIA_API_KEY=keep-me" in body
    assert "OPENROUTER_API_KEY=sk-or-new" in body


def test_setup_states_which_models_it_chose(monkeypatch):
    """Stated, not asked.

    "Which model should write your spoken briefing?" is unanswerable by someone
    who installed this a minute ago, and asking it rebuilds the thing one front
    door removed. One line and a pointer is the whole intervention.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    lines = []
    firstrun.run(config=loaded(), write=collected(lines), ask=answers("sk-or-x"))
    text = "\n".join(lines)

    assert "six model roles" in text
    assert "mj config" in text
    # And it is a statement, not a question.
    assert "?" not in text.split("six model roles")[0].splitlines()[-1]


def test_the_ollama_path_warns_that_the_agent_needs_tool_calling(monkeypatch):
    """One model across six roles is a guess, and `agent_model` is the role it is
    most likely wrong for — plenty of small local models cannot call tools at
    all. Silence there means finding out mid-task."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    lines = []
    firstrun.run(
        config=loaded(),
        write=collected(lines),
        ask=answers("", "y"),
        probe=lambda: "llama3.2",
    )
    text = "\n".join(lines)

    assert "tool-calling" in text
    assert "brain.agent_model" in text


def test_the_key_is_never_echoed_back(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    lines = []
    firstrun.run(config=loaded(), write=collected(lines), ask=answers("sk-or-secret"))
    assert not any("sk-or-secret" in line for line in lines)


# ---------------------------------------------------------------------------
# run() — Ollama
# ---------------------------------------------------------------------------

def test_ollama_is_not_probed_when_a_key_is_given(monkeypatch):
    """The probe is off the hot path. Anyone who pastes a key never waits on a
    connection to a server they do not run."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    firstrun.run(config=loaded(), write=silent,
        ask=answers("sk-or-typed"),
        probe=lambda: pytest.fail("should not probe after a key"),
    )


def test_declining_the_key_offers_a_local_model(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    from majordomo import config as config_mod

    report = firstrun.run(config=loaded(), write=silent,
        ask=answers("", "y"),  # no key, yes to Ollama
        probe=lambda: "qwen2.5:14b",
    )

    assert "ollama" in report.did

    config = config_mod.load(None)
    assert config.brain.base_url == "http://localhost:11434/v1"
    assert config.brain.chat_model == "qwen2.5:14b"
    # The whole point: empty means no key, which `llm` now honours. A *named*
    # variable would raise before the first request.
    assert config.brain.api_key_env == ""


def test_choosing_ollama_keeps_every_other_section(monkeypatch, tmp_path):
    """The one that mattered.

    This wrote config.yml wholesale, so `mj setup --force` on an existing config
    deleted `voice`, `sources`, everything — including
    `voice.asr_function_id`, which the example config says outright is tedious to
    track down. Reachable by the exact sequence someone runs to *add* a key.

    Worth being precise about the kind of mistake: `_write_env` in the same file
    already refuses to overwrite an existing key and explains why. The rule was
    written down and then not applied to its sibling twenty lines away.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import yaml

    paths.ensure_home()
    target = paths.default_config_path()
    target.write_text(
        yaml.safe_dump(
            {
                "brain": {"chat_model": "mine/chosen", "timeout": 42.0},
                "voice": {"asr_function_id": "hard-to-find-id", "enabled": True},
                "sources": {"github": {"enabled": True}},
            }
        ),
        encoding="utf-8",
    )

    report = firstrun.run(
        config=loaded(), write=silent, ask=answers("", "y"), probe=lambda: "llama3.2"
    )
    assert "ollama" in report.did

    after = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert after["voice"]["asr_function_id"] == "hard-to-find-id"
    assert after["sources"]["github"]["enabled"] is True
    # brain is repointed, and keys it does not set survive too.
    assert after["brain"]["base_url"] == "http://localhost:11434/v1"
    assert after["brain"]["chat_model"] == "llama3.2"
    assert after["brain"]["timeout"] == 42.0


@pytest.mark.parametrize(
    "body",
    [
        "brain:\nvoice:\n  asr_function_id: keep-me\n",      # bare key -> None
        "brain:\n\nvoice:\n  asr_function_id: keep-me\n",
        "# brain:\n#   chat_model: x\nbrain:\nvoice:\n  asr_function_id: keep-me\n",
    ],
)
def test_a_brain_section_with_nothing_under_it_still_merges(monkeypatch, body):
    """`brain:` with no value parses to **None**, not to a missing key.

    `config.load` accepts that file happily, so it is a working config — and it
    is what you get by commenting the block out, which the annotated example
    invites. `{**None}` raises, so the merge died, the broad `except` turned it
    into "could not write the config", and the config was left unchanged.

    The top-level guard one line above handles this exact shape; the check was
    simply not repeated a level down.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import yaml

    paths.ensure_home()
    target = paths.default_config_path()
    target.write_text(body, encoding="utf-8")

    firstrun._configure_ollama("llama3.2")

    after = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert after["brain"]["chat_model"] == "llama3.2"
    assert after["voice"]["asr_function_id"] == "keep-me"


def test_a_failed_merge_leaves_no_orphan_backup(monkeypatch):
    """The backup used to be written first, so anything failing in between left a
    `.bak` beside an unchanged config — a file claiming something was replaced
    when nothing was."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    paths.ensure_home()
    target = paths.default_config_path()
    target.write_text("brain:\n  timeout: 42.0\n", encoding="utf-8")

    import yaml

    def boom(*_args, **_kwargs):
        raise RuntimeError("dump failed")

    monkeypatch.setattr(yaml, "safe_dump", boom)

    with pytest.raises(RuntimeError):
        firstrun._configure_ollama("llama3.2")

    assert list(paths.majordomo_home().glob("*.bak")) == []
    assert "timeout: 42.0" in target.read_text(encoding="utf-8")


def test_rewriting_the_config_leaves_a_backup(monkeypatch):
    """A merge cannot keep comments, so the original has to survive somewhere."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    paths.ensure_home()
    paths.default_config_path().write_text(
        "# a comment I wrote\nbrain:\n  timeout: 42.0\n", encoding="utf-8"
    )

    backup = firstrun._configure_ollama("llama3.2")

    assert backup is not None and backup.is_file()
    assert "a comment I wrote" in backup.read_text(encoding="utf-8")


def test_a_fresh_config_needs_no_backup():
    assert firstrun._configure_ollama("llama3.2") is None
    assert paths.default_config_path().is_file()


def test_ollama_honours_an_explicit_config_path(monkeypatch, tmp_path):
    """`--config` must mean the same thing here as everywhere else.

    It did not: the door loaded the named file, setup read and wrote the
    *default* one, and the session then ran on the named one. One flag, two
    meanings, in a single invocation.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    elsewhere = tmp_path / "other.yml"

    firstrun.run(
        config=loaded(),
        write=silent,
        ask=answers("", "y"),
        probe=lambda: "llama3.2",
        config_path=elsewhere,
    )

    assert elsewhere.is_file()
    assert not paths.default_config_path().exists()


def test_declining_ollama_leaves_no_config(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    firstrun.run(config=loaded(), write=silent, ask=answers("", "n"), probe=lambda: "llama3.2")
    assert not paths.default_config_path().exists()


def test_no_ollama_found_says_so_and_moves_on(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    report = firstrun.run(config=loaded(), write=silent, ask=answers(""), probe=lambda: None)
    assert "model" in report.skipped


def test_the_probe_survives_a_raising_transport():
    """Catches bare Exception on purpose.

    The suite's network guard raises RuntimeError, which an httpx-specific
    except clause would let escape — and a probe that can take down the front
    door is worse than one that reports nothing.
    """

    class Boom:
        @staticmethod
        def get(*_args, **_kwargs):
            raise RuntimeError("no network in tests")

    assert firstrun.probe_ollama(client=Boom()) is None


def test_the_probe_reads_the_first_model_name():
    class Fake:
        @staticmethod
        def get(*_args, **_kwargs):
            class R:
                @staticmethod
                def json():
                    return {"models": [{"name": "llama3.2"}, {"name": "other"}]}

            return R()

    assert firstrun.probe_ollama(client=Fake()) == "llama3.2"


def test_the_probe_handles_an_empty_model_list():
    class Fake:
        @staticmethod
        def get(*_args, **_kwargs):
            class R:
                @staticmethod
                def json():
                    return {"models": []}

            return R()

    assert firstrun.probe_ollama(client=Fake()) is None


def test_the_real_probe_finds_nothing_under_the_network_guard():
    """Belt and braces: the default path must not raise either."""
    assert firstrun.probe_ollama() is None


# ---------------------------------------------------------------------------
# run() — the two outward-facing wirings
# ---------------------------------------------------------------------------

def test_hooks_are_installed_only_when_asked(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    settings = tmp_path / "settings.json"

    report = firstrun.run(config=loaded(), write=silent,
        ask=answers("y", "n"),  # yes hooks, no trigger
        settings_path=settings,
    )

    assert "hooks" in report.did
    assert settings.exists()


def test_a_blank_answer_declines(monkeypatch, tmp_path):
    """A stray Enter must never authorise a change outside this project."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    settings = tmp_path / "settings.json"

    report = firstrun.run(config=loaded(), write=silent, ask=answers("", ""), settings_path=settings)

    assert "hooks" in report.skipped
    assert not settings.exists()


def test_a_failing_installer_is_reported_and_does_not_abort(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    from majordomo import install as install_mod

    def boom(**_kwargs):
        raise OSError("settings file is read-only")

    monkeypatch.setattr(install_mod, "install", boom)

    report = firstrun.run(config=loaded(), write=silent, ask=answers("y", "n"))

    assert "hooks" in report.failed
    # Still finished: the trigger question was reached and the marker written.
    assert "trigger" in report.skipped
    assert firstrun.marker_path().exists()


def test_the_trigger_is_skipped_off_windows(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-x")
    # Patched through the named predicate, never `os.name` itself: that is
    # the same module object pathlib reads, and setting it to "posix" on
    # Windows makes Path() raise inside pytest's own reporting.
    monkeypatch.setattr(firstrun, "_is_windows", lambda: False)

    report = firstrun.run(config=loaded(), write=silent, ask=answers("n", "y"))

    assert any("not Windows" in s for s in report.skipped)
