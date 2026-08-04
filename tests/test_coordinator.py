"""Tests for routing + fusing. The model is mocked throughout."""
from __future__ import annotations

import pytest

from majordomo import config as config_module, coordinator
from majordomo.llm import LLMError, MissingApiKey
from majordomo.models import NeedsYouItem, SourceReport


def make_config(threshold=8000):
    data = config_module._deep_merge(
        config_module.DEFAULTS, {"router": {"size_threshold_tokens": threshold}}
    )
    return config_module.build(data)


def report(summary="two PRs waiting", source="github", ok=True, items=None):
    return SourceReport(source=source, ok=ok, summary=summary, items=items or [])


class Recorder:
    """Stands in for llm.complete, recording each call and replying in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, brain, model):
        self.calls.append({"messages": messages, "model": model})
        reply = self.replies.pop(0) if self.replies else "ok"
        if isinstance(reply, Exception):
            raise reply
        return reply


class RoleRecorder(Recorder):
    """Replies by *which* prompt it was handed, so a test doesn't have to
    predict how many chunks a payload will split into."""

    def __call__(self, messages, brain, model):
        self.calls.append({"messages": messages, "model": model})
        text = str(messages)
        if "Merge them" in text:
            self.merges = getattr(self, "merges", 0) + 1
            return "merged summary"
        return "found something"

    @property
    def reduce_calls(self) -> int:
        return len(self.calls) - getattr(self, "merges", 0)


# ---------------------------------------------------------------------------
# chunk
# ---------------------------------------------------------------------------

def test_chunk_splits_on_line_boundaries():
    """A PR entry cut in half mid-line would be reduced into nonsense."""
    text = "\n".join(f"  - repo#{i}: a pull request title" for i in range(2000))
    chunks = coordinator.chunk(text, size=1000)

    assert len(chunks) > 1
    for piece in chunks:
        assert not piece.startswith(" - repo#") or piece.endswith("title")
    assert "\n".join(chunks) == text


def test_chunk_empty_is_empty():
    assert coordinator.chunk("") == []


def test_chunk_short_text_is_one_piece():
    assert coordinator.chunk("one line") == ["one line"]


# ---------------------------------------------------------------------------
# route_report — the fork
# ---------------------------------------------------------------------------

def test_small_source_takes_one_cheap_call(monkeypatch):
    recorder = Recorder(["nothing urgent"])
    monkeypatch.setattr(coordinator, "complete", recorder)

    routed = coordinator.route_report(report(), make_config())

    assert routed.path == "cheap"
    assert len(recorder.calls) == 1
    assert recorder.calls[0]["model"] == make_config().brain.worker_model


def test_oversized_source_escalates_and_fans_out(monkeypatch):
    """The observable fork: one call per chunk, then a merge."""
    recorder = RoleRecorder([])
    monkeypatch.setattr(coordinator, "complete", recorder)
    monkeypatch.setattr(coordinator, "CHUNK_CHARS", 500)

    big = "\n".join(f"  - repo#{i:03d}: a pull request title" for i in range(60))
    routed = coordinator.route_report(report(summary=big), make_config(threshold=10))

    assert routed.path == "escalated"
    assert recorder.reduce_calls > 1, "escalation must actually fan out"
    assert recorder.merges == 1, "the chunks must be folded back into one summary"
    assert routed.summary == "merged summary"
    assert all(c["model"] == make_config().brain.reducer_model for c in recorder.calls)


def test_escalation_reason_names_the_threshold(monkeypatch):
    monkeypatch.setattr(coordinator, "complete", Recorder(["a", "b"]))
    routed = coordinator.route_report(report(summary="x" * 500), make_config(threshold=10))
    assert "exceeds threshold 10" in routed.route_reason


def test_reducer_drops_chunks_that_report_nothing(monkeypatch):
    recorder = Recorder(["NOTHING", "NOTHING", "NOTHING"])
    monkeypatch.setattr(coordinator, "complete", recorder)
    monkeypatch.setattr(coordinator, "CHUNK_CHARS", 200)

    routed = coordinator.route_report(report(summary="x " * 400), make_config(threshold=10))

    assert "Nothing in github needs a decision" in routed.summary
    # No merge call — there was nothing to merge. (The reduce prompt itself
    # contains the word NOTHING, so identify the merge by its own wording.)
    assert all("Merge them" not in str(c["messages"]) for c in recorder.calls)


def test_truncation_is_stated_not_silent(monkeypatch):
    """Never drop a source silently — log what was truncated (spec §9)."""
    monkeypatch.setattr(coordinator, "complete", Recorder(["x"] * 20))
    monkeypatch.setattr(coordinator, "CHUNK_CHARS", 100)
    monkeypatch.setattr(coordinator, "MAX_CHUNKS", 2)

    big = "\n".join("a line of text here" for _ in range(200))
    routed = coordinator.route_report(report(summary=big), make_config(threshold=10))

    assert "truncated" in routed.route_reason


def test_error_stub_is_passed_through_unrouted(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a failed source has nothing to reason about")

    monkeypatch.setattr(coordinator, "complete", boom)
    stub = SourceReport.failed("github", "bad token")

    assert coordinator.route_report(stub, make_config()) is stub


# ---------------------------------------------------------------------------
# Degradation
# ---------------------------------------------------------------------------

def test_llm_failure_falls_back_to_raw_text(monkeypatch):
    monkeypatch.setattr(coordinator, "complete", Recorder([LLMError("503")]))

    routed = coordinator.route_report(report(summary="the raw list"), make_config())

    assert routed.ok
    assert "the raw list" in routed.summary
    assert "model failed" in routed.route_reason


def test_llm_failure_fallback_states_truncation(monkeypatch):
    monkeypatch.setattr(coordinator, "complete", Recorder([LLMError("503")]))

    routed = coordinator.route_report(report(summary="y" * 5000), make_config())

    assert "truncated" in routed.summary
    assert "truncated it" in routed.route_reason


def test_missing_key_propagates_rather_than_repeating(monkeypatch):
    """It's the same failure for every source — the CLI should stop once, not
    print the identical missing-key warning five times."""
    monkeypatch.setattr(coordinator, "complete", Recorder([MissingApiKey("no key")]))

    with pytest.raises(MissingApiKey):
        coordinator.route_report(report(), make_config())


def test_needs_you_items_survive_routing(monkeypatch):
    monkeypatch.setattr(coordinator, "complete", Recorder(["summary"]))
    item = NeedsYouItem(kind="review_request", title="repo#1", detail="d", source="github")

    routed = coordinator.route_report(report(items=[item]), make_config())

    assert routed.items == [item]


# ---------------------------------------------------------------------------
# fuse
# ---------------------------------------------------------------------------

def test_fuse_makes_exactly_one_call(monkeypatch):
    recorder = Recorder(["Two PRs need your review; nothing is blocked."])
    monkeypatch.setattr(coordinator, "complete", recorder)

    briefing = coordinator.fuse([report(), report(source="sessions")], make_config())

    assert len(recorder.calls) == 1
    assert recorder.calls[0]["model"] == make_config().brain.fuser_model
    assert briefing.briefing_text.startswith("Two PRs")


def test_fuse_collects_needs_you_from_every_source(monkeypatch):
    monkeypatch.setattr(coordinator, "complete", Recorder(["briefing"]))
    a = NeedsYouItem(kind="review_request", title="a", detail="", source="github")
    b = NeedsYouItem(kind="session_blocked", title="b", detail="", source="sessions")

    briefing = coordinator.fuse(
        [report(items=[a]), report(source="sessions", items=[b])], make_config()
    )

    assert briefing.needs_you == [a, b]


def test_fuse_degrades_to_raw_list_when_the_brain_dies(monkeypatch):
    """The Voicelog pattern — you still get the information, just unfused."""
    monkeypatch.setattr(coordinator, "complete", Recorder([LLMError("down")]))

    briefing = coordinator.fuse(
        [report(summary="two PRs"), report(source="sessions", summary="one blocked")],
        make_config(),
    )

    assert "two PRs" in briefing.briefing_text
    assert "one blocked" in briefing.briefing_text


def test_fuse_degrades_when_the_key_is_missing(monkeypatch):
    monkeypatch.setattr(coordinator, "complete", Recorder([MissingApiKey("no key")]))
    briefing = coordinator.fuse([report(summary="two PRs")], make_config())
    assert "two PRs" in briefing.briefing_text


def test_fuse_marks_unavailable_sources_in_the_fallback(monkeypatch):
    monkeypatch.setattr(coordinator, "complete", Recorder([LLMError("down")]))

    briefing = coordinator.fuse([SourceReport.failed("github", "401")], make_config())

    assert "unavailable" in briefing.briefing_text


def test_fuse_prompt_flags_unavailable_sources(monkeypatch):
    recorder = Recorder(["briefing"])
    monkeypatch.setattr(coordinator, "complete", recorder)

    coordinator.fuse([SourceReport.failed("github", "401")], make_config())

    assert "UNAVAILABLE" in str(recorder.calls[0]["messages"])


def test_fuse_with_no_reports(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("nothing to fuse means no call")

    monkeypatch.setattr(coordinator, "complete", boom)
    assert coordinator.fuse([], make_config()).briefing_text == "Nothing to report."


def test_empty_model_reply_falls_back(monkeypatch):
    """Free-tier models return empty strings more often than you'd like."""
    monkeypatch.setattr(coordinator, "complete", Recorder(["   "]))

    briefing = coordinator.fuse([report(summary="two PRs")], make_config())

    assert "two PRs" in briefing.briefing_text
