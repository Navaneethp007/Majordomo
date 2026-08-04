"""Tests for majordomo.config — no file I/O beyond tmp_path, no network."""
from __future__ import annotations

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
    assert cfg.voice.provider == "elevenlabs"
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
