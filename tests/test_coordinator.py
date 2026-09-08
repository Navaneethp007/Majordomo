"""Tests for routing + fusing. The model is mocked throughout."""
from __future__ import annotations

import pytest
from unittest import mock

from majordomo import config as config_module, coordinator
from majordomo.llm import LLMError, MissingApiKey
from majordomo.models import NeedsYouItem, SourceReport

CFG = config_module.build(config_module.DEFAULTS)


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


# ---------------------------------------------------------------------------
# Null content — free-tier models return `content: null` on filtered completions
# ---------------------------------------------------------------------------

def test_null_content_does_not_crash_fuse(monkeypatch):
    """AttributeError is not LLMError, so a null answer sailed past every
    `except (LLMError, MissingApiKey)` and took down the whole briefing."""
    monkeypatch.setattr(coordinator, "complete", Recorder([None]))

    briefing = coordinator.fuse([report(summary="two PRs")], make_config())

    assert "two PRs" in briefing.briefing_text


def test_null_content_does_not_crash_the_cheap_path(monkeypatch):
    monkeypatch.setattr(coordinator, "complete", Recorder([None]))
    routed = coordinator.route_report(report(summary="raw text"), make_config())
    assert routed.summary == "raw text"


def test_null_content_does_not_crash_the_escalated_path(monkeypatch):
    monkeypatch.setattr(coordinator, "complete", Recorder([None, None, None, None, None]))
    monkeypatch.setattr(coordinator, "CHUNK_CHARS", 200)

    routed = coordinator.route_report(report(summary="x " * 400), make_config(threshold=10))

    assert routed.ok


# ---------------------------------------------------------------------------
# Grounding — the fuser must not invent obligations
# ---------------------------------------------------------------------------

def test_fuse_prompt_carries_the_needs_you_list(monkeypatch):
    """The original bug: needs_you was computed and then never shown to the
    model writing the briefing, so it re-derived actionability from prose and
    got it wrong."""
    recorder = Recorder(["briefing"])
    monkeypatch.setattr(coordinator, "complete", recorder)
    item = NeedsYouItem(
        kind="review_request", title="majordomo#12",
        detail="Your review is requested.", source="github",
    )

    coordinator.fuse([report(items=[item])], make_config())

    prompt = str(recorder.calls[0]["messages"])
    assert "majordomo#12" in prompt
    assert "Your review is requested." in prompt
    assert "DECISIONS" in prompt


def test_fuse_prompt_states_plainly_when_nothing_needs_you(monkeypatch):
    """With an empty list the model announced three obligations. It must be told
    the list is empty AND complete, or it pads the silence with fiction."""
    recorder = Recorder(["briefing"])
    monkeypatch.setattr(coordinator, "complete", recorder)

    coordinator.fuse([report(items=[])], make_config())

    prompt = str(recorder.calls[0]["messages"])
    assert "(nothing)" in prompt
    assert "Do NOT invent tasks" in prompt


def test_fuse_prompt_separates_decisions_from_context(monkeypatch):
    recorder = Recorder(["briefing"])
    monkeypatch.setattr(coordinator, "complete", recorder)

    coordinator.fuse([report(summary="some background")], make_config())

    prompt = str(recorder.calls[0]["messages"])
    assert prompt.index("DECISIONS") < prompt.index("CONTEXT (background only)")
    assert "background only" in prompt


def test_fuse_prompt_warns_about_the_two_specific_misreads(monkeypatch):
    """Both of these produced real fabrications: an active session's topic read
    as a decision, and a participating notification read as a review request."""
    recorder = Recorder(["briefing"])
    monkeypatch.setattr(coordinator, "complete", recorder)

    coordinator.fuse([report()], make_config())
    prompt = str(recorder.calls[0]["messages"])

    assert "still working needs nothing" in prompt
    assert "not a review request" in prompt


def test_fuse_prompt_stays_silent_about_healthy_sources(monkeypatch):
    """'No sources were unavailable' was padding — the model inverting a
    conditional instruction. Don't give it the conditional when it doesn't apply."""
    recorder = Recorder(["briefing"])
    monkeypatch.setattr(coordinator, "complete", recorder)

    coordinator.fuse([report(ok=True)], make_config())

    assert "could not be reached" not in str(recorder.calls[0]["messages"])


def test_fuse_prompt_names_an_unavailable_source(monkeypatch):
    recorder = Recorder(["briefing"])
    monkeypatch.setattr(coordinator, "complete", recorder)

    coordinator.fuse([SourceReport.failed("github", "401")], make_config())

    prompt = str(recorder.calls[0]["messages"])
    assert "could not be reached" in prompt
    assert "github" in prompt


def test_pre_summarised_source_skips_the_model(monkeypatch):
    """The sessions worker writes prose deterministically. Re-summarising it
    cost a call and destroyed the detail — 'two sessions, one on the payments[]
    question' came back as 'You were reviewing GitHub'."""
    def boom(*a, **k):
        raise AssertionError("a pre-summarised source must not be sent to the model")

    monkeypatch.setattr(coordinator, "complete", boom)
    r = SourceReport(
        source="sessions", ok=True,
        summary="NO ACTION NEEDED — 2 running: fe-raad-erp (vscode) — payments[] shape",
        pre_summarised=True,
    )

    routed = coordinator.route_report(r, make_config())

    assert routed.summary == r.summary, "the detail must survive verbatim"
    assert "no model call" in routed.route_reason


def test_pre_summarised_source_still_escalates_when_huge(monkeypatch):
    """Skipping the cheap call must not disable size escalation — a genuinely
    enormous payload still needs reducing."""
    recorder = RoleRecorder([])
    monkeypatch.setattr(coordinator, "complete", recorder)
    monkeypatch.setattr(coordinator, "CHUNK_CHARS", 500)

    big = "\n".join(f"  - session {i:03d} doing something" for i in range(60))
    routed = coordinator.route_report(
        SourceReport("sessions", True, big, pre_summarised=True), make_config(threshold=10)
    )

    assert routed.path == "escalated"
    assert recorder.reduce_calls > 1


# ---------------------------------------------------------------------------
# Context items must never become decisions
# ---------------------------------------------------------------------------

def test_context_items_never_reach_the_decisions_block(monkeypatch):
    """Gmail is a digest. If its items could land in DECISIONS, the briefing
    would start asserting that you must read your email."""
    from majordomo.models import ContextItem

    recorder = Recorder(["briefing"])
    monkeypatch.setattr(coordinator, "complete", recorder)
    mail = ContextItem(kind="unread_mail", title="Priya: Invoice rounding",
                       detail="Unread.", source="gmail")

    coordinator.fuse(
        [SourceReport("gmail", True, "NO ACTION NEEDED — 1 unread", context_items=[mail])],
        make_config(),
    )

    prompt = str(recorder.calls[0]["messages"])
    # Split on the block header, not the word — "DECISIONS" also appears in the
    # instructions above it.
    decisions = prompt.split("DECISIONS (complete list")[1].split("CONTEXT (background only)")[0]
    assert "(nothing)" in decisions
    assert "Invoice rounding" not in decisions
    # ...and the mail is still present as background.
    assert "1 unread" in prompt.split("CONTEXT (background only)")[1]


def test_context_items_survive_routing(monkeypatch):
    from majordomo.models import ContextItem

    monkeypatch.setattr(coordinator, "complete", Recorder(["x"]))
    mail = ContextItem(kind="unread_mail", title="t", detail="d", source="gmail")

    routed = coordinator.route_report(
        SourceReport("gmail", True, "digest", context_items=[mail], pre_summarised=True),
        make_config(),
    )
    assert routed.context_items == [mail]


def test_fuse_collects_context_from_every_source(monkeypatch):
    from majordomo.models import ContextItem

    monkeypatch.setattr(coordinator, "complete", Recorder(["briefing"]))
    a = ContextItem(kind="unread_mail", title="a", detail="", source="gmail")

    briefing = coordinator.fuse(
        [report(), SourceReport("gmail", True, "digest", context_items=[a])], make_config()
    )
    assert briefing.context == [a]


def test_context_items_survive_every_routing_path(monkeypatch):
    """Only the pre_summarised branch carried them. Lowering the router
    threshold — which the config comment invites — silently emptied the mail
    list from both the CLI and the panel, with no error anywhere."""
    from majordomo.models import ContextItem

    mail = ContextItem(kind="unread_mail", title="Priya: Invoice", detail="d", source="gmail")

    def source(**kw):
        return SourceReport("gmail", True, "a digest line\nanother line",
                            context_items=[mail], **kw)

    # cheap summarize
    monkeypatch.setattr(coordinator, "complete", Recorder(["summary"]))
    assert coordinator.route_report(source(), make_config()).context_items == [mail]

    # escalated
    monkeypatch.setattr(coordinator, "complete", RoleRecorder([]))
    monkeypatch.setattr(coordinator, "CHUNK_CHARS", 20)
    assert coordinator.route_report(source(), make_config(threshold=1)).context_items == [mail]

    # LLM failure fallback
    monkeypatch.setattr(coordinator, "complete", Recorder([LLMError("503")]))
    assert coordinator.route_report(source(), make_config()).context_items == [mail]

    # pre-summarised
    monkeypatch.setattr(coordinator, "complete", Recorder(["x"]))
    assert coordinator.route_report(
        source(pre_summarised=True), make_config()
    ).context_items == [mail]


# ---------------------------------------------------------------------------
# The fuser output guard
# ---------------------------------------------------------------------------

def test_a_reasoning_dump_falls_back_instead_of_being_spoken():
    """A *successful* call returning nonsense must degrade like a failed one.

    On 2026-09-01 a reasoning model answered the fuse prompt with nine
    paragraphs of deliberation. No `except LLMError` caught it — the call
    succeeded — so the pipeline printed it and handed it to TTS.
    """
    reports = [SourceReport("github", True, "Two PRs await review.")]
    dump = "Here's a thinking process:\n\n" + ("1. Analyse the request. " * 300)

    with mock.patch.object(coordinator, "complete", return_value=dump):
        briefing = coordinator.fuse(reports, CFG)

    assert briefing.briefing_text == coordinator._raw_briefing(reports)
    assert "looks like reasoning" in briefing.note
    assert str(coordinator.MAX_BRIEFING_CHARS) in briefing.note


def test_an_ordinary_briefing_passes_through_untouched():
    reports = [SourceReport("github", True, "Two PRs await review.")]

    with mock.patch.object(
        coordinator, "complete", return_value="Two pull requests are waiting on you."
    ):
        briefing = coordinator.fuse(reports, CFG)

    assert briefing.briefing_text == "Two pull requests are waiting on you."
    assert briefing.note == ""


def test_a_long_but_legitimate_briefing_is_not_rejected():
    """The guard is a nonsense detector, not a style rule."""
    reports = [SourceReport("github", True, "x")]
    wordy = "A" * (coordinator.MAX_BRIEFING_CHARS - 1)

    with mock.patch.object(coordinator, "complete", return_value=wordy):
        assert coordinator.fuse(reports, CFG).briefing_text == wordy


def test_an_empty_reply_is_reported_not_silently_blank():
    reports = [SourceReport("github", True, "Two PRs await review.")]

    with mock.patch.object(coordinator, "complete", return_value=""):
        briefing = coordinator.fuse(reports, CFG)

    assert briefing.briefing_text == coordinator._raw_briefing(reports)
    assert "returned nothing" in briefing.note


def test_a_failed_call_records_why_it_degraded():
    reports = [SourceReport("github", True, "Two PRs await review.")]

    with mock.patch.object(coordinator, "complete", side_effect=LLMError("503")):
        briefing = coordinator.fuse(reports, CFG)

    assert "503" in briefing.note


def test_explain_surfaces_a_degraded_fuse():
    """The fusing step belongs to no source, so it has nowhere else to report."""
    from majordomo import brief

    reports = [SourceReport("github", True, "x")]
    dump = "y" * 5000

    with mock.patch.object(coordinator, "complete", return_value=dump):
        result = brief.BriefResult(
            briefing=coordinator.fuse(reports, CFG), reports=reports
        )

    assert "fuse" in result.explain()
    assert "degraded" in result.explain()
