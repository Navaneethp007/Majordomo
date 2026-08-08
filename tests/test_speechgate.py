"""Tests for the speech gate — the thing that stops the briefing becoming noise.

Two firing patterns motivate all of this:

- A cold boot runs OnLogon immediately and OnBoot a minute later.
  ``MultipleInstancesPolicy`` is per-task and cannot dedupe across them.
- Kernel-Power 107 fires on every modern-standby resume, many times a day.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from majordomo import speechgate
from majordomo.models import Briefing, NeedsYouItem, SourceReport

NOW = datetime(2026, 8, 8, 9, 0, tzinfo=timezone.utc)

BLOCKED = NeedsYouItem(kind="session_blocked", title="majordomo (vscode)", detail="d", source="sessions")
REVIEW = NeedsYouItem(kind="review_request", title="repo#12", detail="d", source="github")


# ---------------------------------------------------------------------------
# fingerprint — must track the situation, not the wording
# ---------------------------------------------------------------------------

def test_rewording_does_not_count_as_a_change():
    """The model rephrases every run. Hashing the prose would make identical
    circumstances look novel each time and defeat the whole gate."""
    a = speechgate.fingerprint(Briefing("Two PRs need review.", [REVIEW]))
    b = speechgate.fingerprint(Briefing("You have two reviews waiting!", [REVIEW]))
    assert a == b


def test_different_items_change_the_fingerprint():
    a = speechgate.fingerprint(Briefing("x", [REVIEW]))
    b = speechgate.fingerprint(Briefing("x", [REVIEW, BLOCKED]))
    assert a != b


def test_item_order_does_not_change_the_fingerprint():
    a = speechgate.fingerprint(Briefing("x", [REVIEW, BLOCKED]))
    b = speechgate.fingerprint(Briefing("x", [BLOCKED, REVIEW]))
    assert a == b


def test_a_source_going_down_is_a_change():
    """'GitHub token expired' must not hash to the same thing as 'all quiet'."""
    quiet = speechgate.fingerprint(Briefing("x", []), [SourceReport("github", True, "ok")])
    broken = speechgate.fingerprint(Briefing("x", []), [SourceReport.failed("github", "401")])
    assert quiet != broken


# ---------------------------------------------------------------------------
# is_repeat
# ---------------------------------------------------------------------------

def test_nothing_spoken_yet_is_not_a_repeat(tmp_path):
    assert speechgate.is_repeat("abc", 120, now=NOW, path=tmp_path / "none.json") is False


def test_same_situation_within_cooldown_is_a_repeat(tmp_path):
    """The blocked-since-9am case: don't read it out on every lid open."""
    marker = tmp_path / "last.json"
    speechgate.record("abc", now=NOW, path=marker)

    assert speechgate.is_repeat("abc", 120, now=NOW + timedelta(minutes=5), path=marker) is True


def test_cold_boot_double_fire_is_suppressed(tmp_path):
    """OnLogon then OnBoot a minute later — the second must stay silent."""
    marker = tmp_path / "last.json"
    speechgate.record("abc", now=NOW, path=marker)

    assert speechgate.is_repeat("abc", 120, now=NOW + timedelta(minutes=1), path=marker) is True


def test_same_situation_after_cooldown_speaks_again(tmp_path):
    marker = tmp_path / "last.json"
    speechgate.record("abc", now=NOW, path=marker)

    assert speechgate.is_repeat("abc", 120, now=NOW + timedelta(hours=3), path=marker) is False


def test_a_changed_situation_speaks_immediately(tmp_path):
    """Cooldown must never swallow news."""
    marker = tmp_path / "last.json"
    speechgate.record("abc", now=NOW, path=marker)

    assert speechgate.is_repeat("xyz", 120, now=NOW + timedelta(seconds=5), path=marker) is False


def test_corrupt_marker_does_not_silence_forever(tmp_path):
    """A bad marker file must fail toward speaking, not toward permanent silence."""
    marker = tmp_path / "last.json"
    marker.write_text("{ not json", encoding="utf-8")

    assert speechgate.is_repeat("abc", 120, now=NOW, path=marker) is False


def test_record_survives_an_unwritable_path(tmp_path):
    """Failing to record costs one duplicate; raising would cost the briefing."""
    speechgate.record("abc", now=NOW, path=tmp_path / "nope" / "deep" / "x.json")


def test_round_trip(tmp_path):
    marker = tmp_path / "last.json"
    speechgate.record("fp-1", now=NOW, path=marker)
    fp, at = speechgate.read_last(marker)

    assert fp == "fp-1"
    assert at == NOW
