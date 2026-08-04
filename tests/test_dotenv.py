"""Tests for the .env loader."""
from __future__ import annotations

import os

from majordomo import dotenv


def write(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parses_simple_pairs():
    assert dotenv.parse("A=1\nB=two") == {"A": "1", "B": "two"}


def test_ignores_blanks_and_comments():
    assert dotenv.parse("# a note\n\nA=1\n   \n# another\nB=2") == {"A": "1", "B": "2"}


def test_strips_export_prefix():
    """People paste `export KEY=…` straight out of shell instructions."""
    assert dotenv.parse("export OPENROUTER_API_KEY=sk-or-1") == {"OPENROUTER_API_KEY": "sk-or-1"}


def test_strips_matching_quotes():
    """A quoted key reaching the API verbatim would fail auth invisibly."""
    assert dotenv.parse("A='sk-1'\nB=\"sk-2\"") == {"A": "sk-1", "B": "sk-2"}


def test_keeps_unmatched_quotes():
    assert dotenv.parse("A='sk-1") == {"A": "'sk-1"}


def test_value_may_contain_equals():
    """Base64 and JWT-ish keys end in '=' padding all the time."""
    assert dotenv.parse("A=abc=def==") == {"A": "abc=def=="}


def test_ignores_lines_without_an_equals():
    assert dotenv.parse("nonsense\nA=1") == {"A": "1"}


def test_ignores_empty_key():
    assert dotenv.parse("=novalue\nA=1") == {"A": "1"}


def test_surrounding_whitespace_is_trimmed():
    assert dotenv.parse("  A = 1  ") == {"A": "1"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_sets_missing_vars(tmp_path, monkeypatch):
    monkeypatch.delenv("MJ_TEST_KEY", raising=False)
    applied = dotenv.load(write(tmp_path, "MJ_TEST_KEY=from-file"))

    assert applied == ["MJ_TEST_KEY"]
    assert os.environ["MJ_TEST_KEY"] == "from-file"


def test_real_env_var_always_wins(tmp_path, monkeypatch):
    """So setx still works, CI still works, and a one-off shell override still
    shadows the file for a single run."""
    monkeypatch.setenv("MJ_TEST_KEY", "from-environment")
    applied = dotenv.load(write(tmp_path, "MJ_TEST_KEY=from-file"))

    assert applied == []
    assert os.environ["MJ_TEST_KEY"] == "from-environment"


def test_empty_env_var_is_treated_as_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MJ_TEST_KEY", "")
    dotenv.load(write(tmp_path, "MJ_TEST_KEY=from-file"))
    assert os.environ["MJ_TEST_KEY"] == "from-file"


def test_missing_file_is_not_an_error(tmp_path):
    assert dotenv.load(tmp_path / "absent.env") == []


def test_bom_prefixed_file_still_loads(tmp_path, monkeypatch):
    monkeypatch.delenv("MJ_TEST_KEY", raising=False)
    dotenv.load(write(tmp_path, "﻿MJ_TEST_KEY=ok"))
    assert os.environ["MJ_TEST_KEY"] == "ok"


def test_broken_file_cannot_stop_a_briefing(tmp_path, monkeypatch):
    monkeypatch.delenv("MJ_TEST_KEY", raising=False)
    applied = dotenv.load(write(tmp_path, "\x00garbage\nMJ_TEST_KEY=ok"))
    assert "MJ_TEST_KEY" in applied


def test_default_path_is_beside_the_config(tmp_path, monkeypatch):
    monkeypatch.setenv("MAJORDOMO_HOME", str(tmp_path))
    assert dotenv.env_path() == tmp_path / ".env"
