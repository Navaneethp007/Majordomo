"""Tests for majordomo.config — no file I/O beyond tmp_path, no network."""
from __future__ import annotations

from pathlib import Path

import pytest

from majordomo import config as config_module
from majordomo.config import Config, ConfigFileNotFound


# ---------------------------------------------------------------------------
# Loading with no config file at all
# ---------------------------------------------------------------------------

def test_load_missing_default_path_uses_defaults(tmp_path, monkeypatch):
    """Majordomo must run with zero setup — every knob has a default."""
    monkeypatch.setattr(config_module, "default_config_path", lambda: tmp_path / "nope.yml")

    cfg = config_module.load(None)

    assert isinstance(cfg, Config)
    assert cfg.brain.provider == "openrouter"
    assert cfg.voice.provider == "nvidia"
    assert cfg.router.size_threshold_tokens > 0
    assert cfg.sources.github.enabled is True


def test_load_explicit_missing_path_raises(tmp_path):
    """An explicitly-named config that isn't there is a typo, not a default."""
    with pytest.raises(ConfigFileNotFound):
        config_module.load(str(tmp_path / "absent.yml"))


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def test_file_values_win_over_defaults(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text(
        "brain:\n"
        "  fuser_model: some/model-v2\n"
        "router:\n"
        "  size_threshold_tokens: 42\n",
        encoding="utf-8",
    )

    cfg = config_module.load(str(path))

    assert cfg.brain.fuser_model == "some/model-v2"
    assert cfg.router.size_threshold_tokens == 42


def test_merge_is_deep_not_shallow(tmp_path):
    """Setting one key in a section must not wipe that section's siblings.

    A shallow merge would leave brain.base_url unset the moment the user
    overrides brain.fuser_model — the exact bug that makes config feel cursed.
    """
    path = tmp_path / "config.yml"
    path.write_text("brain:\n  fuser_model: only/this\n", encoding="utf-8")

    cfg = config_module.load(str(path))

    assert cfg.brain.fuser_model == "only/this"
    assert cfg.brain.base_url == config_module.DEFAULTS["brain"]["base_url"]
    assert cfg.brain.api_key_env == "OPENROUTER_API_KEY"


def test_empty_file_is_all_defaults(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("", encoding="utf-8")

    cfg = config_module.load(str(path))

    assert cfg.brain.provider == "openrouter"


def test_unknown_keys_are_ignored_not_fatal(tmp_path):
    """A stale key from an older version must not brick the whole app."""
    path = tmp_path / "config.yml"
    path.write_text("brain:\n  wat: 1\nnonsense: true\n", encoding="utf-8")

    cfg = config_module.load(str(path))

    assert cfg.brain.provider == "openrouter"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

def test_numeric_strings_are_coerced(tmp_path):
    """YAML quoting accidents shouldn't produce a str where an int is compared."""
    path = tmp_path / "config.yml"
    path.write_text("router:\n  size_threshold_tokens: '4096'\n", encoding="utf-8")

    cfg = config_module.load(str(path))

    assert cfg.router.size_threshold_tokens == 4096
    assert isinstance(cfg.router.size_threshold_tokens, int)


def test_config_sections_are_frozen():
    cfg = config_module.build(config_module.DEFAULTS)
    with pytest.raises(Exception):
        cfg.router.size_threshold_tokens = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Empty sections — the shape a plausible hand-edit produces
# ---------------------------------------------------------------------------

def test_empty_section_does_not_crash(tmp_path):
    """`brain:` with nothing indented under it parses to {"brain": None}. A
    .get() default only applies when the key is ABSENT, so this produced an
    AttributeError traceback — and _load_config catches only ConfigFileNotFound."""
    path = tmp_path / "config.yml"
    path.write_text("brain:\nvoice:\nrouter:\nsources:\n", encoding="utf-8")

    cfg = config_module.load(str(path))

    assert cfg.brain.provider == "openrouter"
    assert cfg.voice.provider == "nvidia"
    assert cfg.router.size_threshold_tokens > 0
    assert cfg.sources.github.enabled is True


def test_empty_nested_section_does_not_crash(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("sources:\n  github:\n  sessions:\n", encoding="utf-8")

    cfg = config_module.load(str(path))

    assert cfg.sources.github.token_env == "MAJORDOMO_GH_TOKEN"
    assert cfg.sources.sessions.stale_after_hours == 72


def test_build_tolerates_none_sections_directly():
    cfg = config_module.build({"brain": None, "voice": None, "sources": None, "router": None})
    assert cfg.brain.provider == "openrouter"


def test_repeat_after_minutes_is_configurable(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("voice:\n  repeat_after_minutes: 15\n", encoding="utf-8")
    assert config_module.load(str(path)).voice.repeat_after_minutes == 15


# ---------------------------------------------------------------------------
# The shipped example
# ---------------------------------------------------------------------------


EXAMPLE = Path(config_module.__file__).resolve().parent / "config.example.yml"


def test_the_example_config_exists_where_the_cli_looks_for_it():
    """Beside the module, not at the repo root — a wheel install has no repo
    root, and `mj config --init` has to work for someone who ran pip install."""
    assert EXAMPLE.is_file()


def test_the_example_loads_and_agrees_with_the_defaults():
    """A stale example is worse than none: it teaches keys that do nothing."""
    from_example = config_module.load(str(EXAMPLE))
    from_defaults = config_module.build(config_module.DEFAULTS)

    assert from_example.brain == from_defaults.brain
    assert from_example.router == from_defaults.router
    assert from_example.memory == from_defaults.memory
    assert from_example.scaffold == from_defaults.scaffold


def test_the_example_documents_every_brain_role():
    """The roles are the thing someone cloning this has to choose between, and
    they are invisible without an explanation of what each is optimising for."""
    text = EXAMPLE.read_text(encoding="utf-8")

    for role in ("worker_model", "fuser_model", "reducer_model",
                 "chat_model", "agent_model", "fallback_model"):
        assert role in text, f"{role} is not in the example"


def test_the_example_names_no_secrets():
    """It names environment *variables*, never values, so it stays safe to
    commit and safe to paste into an issue."""
    text = EXAMPLE.read_text(encoding="utf-8")

    assert "sk-" not in text
    assert "ghp_" not in text
    assert "API_KEY:" not in text        # a key named as a value, not an env var
