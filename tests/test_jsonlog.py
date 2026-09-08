"""Tests for the shared half of an append-only JSONL log.

The selection logic lives in ``state`` and ``activity`` and differs between
them. What is here is the rewrite, which is the part where being wrong corrupts
a file rather than merely keeping the wrong rows.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from unittest import mock

import pytest

from majordomo import activity, jsonlog, state


@dataclass
class Row:
    value: str

    def to_json(self) -> dict:
        return {"value": self.value}


def test_rewrite_replaces_the_file_with_exactly_these_rows(tmp_path):
    target = tmp_path / "log.jsonl"
    target.write_text('{"value": "old"}\n{"value": "older"}\n', encoding="utf-8")

    jsonlog.rewrite(target, [Row("kept")])

    lines = target.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [{"value": "kept"}]


def test_the_temporary_file_sits_beside_the_target(tmp_path):
    """os.replace is only atomic within a filesystem, so the temp file cannot
    live in the system temp directory."""
    target = tmp_path / "log.jsonl"
    seen = {}

    real_replace = jsonlog.os.replace

    def spy(src, dst):
        seen["src"] = str(src)
        return real_replace(src, dst)

    with mock.patch.object(jsonlog.os, "replace", spy):
        jsonlog.rewrite(target, [Row("x")])

    assert seen["src"].startswith(str(tmp_path))


def test_no_temporary_file_is_left_behind(tmp_path):
    """A stray `.compacting` file beside the log is how the next run gets
    confused about which one is real."""
    target = tmp_path / "log.jsonl"
    jsonlog.rewrite(target, [Row("x")])

    assert not list(tmp_path.glob("*.compacting"))
    assert target.is_file()


def test_rewriting_to_nothing_leaves_an_empty_file(tmp_path):
    target = tmp_path / "log.jsonl"
    target.write_text('{"value": "old"}\n', encoding="utf-8")

    jsonlog.rewrite(target, [])

    assert target.read_text(encoding="utf-8") == ""


def test_unicode_survives_the_round_trip(tmp_path):
    target = tmp_path / "log.jsonl"
    jsonlog.rewrite(target, [Row("café ☕")])

    assert "café ☕" in target.read_text(encoding="utf-8")


def test_is_large_only_fires_past_the_threshold(tmp_path):
    target = tmp_path / "log.jsonl"
    target.write_text("x" * 100, encoding="utf-8")

    assert jsonlog.is_large(target, max_bytes=50)
    assert not jsonlog.is_large(target, max_bytes=1000)


def test_is_large_never_raises_on_a_missing_file(tmp_path):
    """Housekeeping paths must not fail because they could not tell."""
    assert jsonlog.is_large(tmp_path / "absent.jsonl") is False


def test_both_logs_share_one_threshold():
    """Two copies of a constant is one copy too many — they would drift."""
    assert state.COMPACT_OVER_BYTES is jsonlog.COMPACT_OVER_BYTES
    assert activity.COMPACT_OVER_BYTES is jsonlog.COMPACT_OVER_BYTES


@pytest.mark.parametrize("module", [state, activity])
def test_neither_module_still_rewrites_by_hand(module):
    """A fix to the atomic write has to reach both, which it cannot if either
    keeps its own copy."""
    import inspect

    source = inspect.getsource(module.compact)
    assert "jsonlog.rewrite" in source
    assert "os.replace" not in source
