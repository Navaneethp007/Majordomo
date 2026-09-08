"""The routing + fusing half of the pipeline.

``route_report`` puts one worker's raw report through the gate — cheap summarize
or escalated reducer agent. ``fuse`` takes every routed report and produces the
single briefing.

Degradation is the design constraint throughout, following Voicelog's pattern:
an LLM failure is *recoverable* and must never cost you the briefing. If the
fuser dies you still get the per-source rundown; if a reducer dies that one
source falls back to a truncated cheap summary. What we never do is drop a
source silently — a truncation is always stated.
"""
from __future__ import annotations

from majordomo import prompts, router
from majordomo.config import Config
from majordomo.llm import LLMError, MissingApiKey, complete
from majordomo.models import Briefing, NeedsYouItem, SourceReport

#: Characters per chunk when reducing an oversized source. Comfortably inside a
#: small free-tier context window with room for the prompt around it.
CHUNK_CHARS = 12_000

#: Cap on chunks. Past this we truncate — and say we truncated.
MAX_CHUNKS = 6


def chunk(text: str, size: int | None = None) -> list[str]:
    """Split on line boundaries so a PR entry is never cut in half.

    ``size`` resolves at call time rather than as a default argument, so the
    module constant is genuinely overridable — bound as a default it would have
    frozen at import and silently ignored every later change.
    """
    if size is None:
        size = CHUNK_CHARS
    if not text:
        return []

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for line in text.split("\n"):
        if current and length + len(line) + 1 > size:
            chunks.append("\n".join(current))
            current, length = [], 0
        current.append(line)
        length += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def reduce_source(report: SourceReport, config: Config) -> tuple[str, str]:
    """The escalation agent: reason an oversized source down before fusing.

    Returns ``(summary, note)`` where note records any truncation.

    Raises:
        LLMError / MissingApiKey: the caller decides how to degrade.
    """
    pieces = chunk(report.summary)
    note = ""
    if len(pieces) > MAX_CHUNKS:
        dropped = len(pieces) - MAX_CHUNKS
        pieces = pieces[:MAX_CHUNKS]
        note = f"truncated: {dropped} of {dropped + MAX_CHUNKS} chunks not read"

    extracts: list[str] = []
    for index, piece in enumerate(pieces, start=1):
        reply = complete(
            prompts.build_reduce_prompt(report.source, piece, index, len(pieces)),
            config.brain,
            config.brain.reducer_model,
        )
        cleaned = (reply or "").strip()
        if cleaned and cleaned.upper() != "NOTHING":
            extracts.append(cleaned)

    if not extracts:
        return f"Nothing in {report.source} needs a decision.", note

    merged = complete(
        prompts.build_reduce_merge_prompt(report.source, "\n".join(extracts)),
        config.brain,
        config.brain.reducer_model,
    )
    return (merged or "").strip(), note


def summarize_source(report: SourceReport, config: Config) -> str:
    """The cheap path: one call, no fan-out."""
    reply = complete(
        prompts.build_summarize_prompt(report.source, report.summary),
        config.brain,
        config.brain.worker_model,
    )
    return (reply or "").strip()


def route_report(report: SourceReport, config: Config) -> SourceReport:
    """Put one report through the gate and reason it down. Never raises."""
    if not report.ok:
        return report  # an error stub has nothing to reason about

    decision = router.decide(report.summary, config.router)

    try:
        if decision.path == "escalate":
            summary, note = reduce_source(report, config)
            reason = decision.reason + (f"; {note}" if note else "")
            return SourceReport(
                source=report.source,
                ok=True,
                summary=summary or report.summary,
                items=report.items,
                context_items=report.context_items,
                path="escalated",
                route_reason=reason,
            )

        if report.pre_summarised:
            # Already human-readable. A model pass here is a call spent to lose
            # information — nothing to gain when the worker wrote prose on purpose.
            return SourceReport(
                source=report.source,
                ok=True,
                summary=report.summary,
                items=report.items,
                context_items=report.context_items,
                path="cheap",
                route_reason="already summarised by its worker; no model call",
                pre_summarised=True,
            )

        summary = summarize_source(report, config)
        return SourceReport(
            source=report.source,
            ok=True,
            summary=summary or report.summary,
            items=report.items,
            context_items=report.context_items,
            path="cheap",
            route_reason=decision.reason,
        )

    except MissingApiKey:
        # Unrecoverable and identical for every source — let the CLI stop once
        # rather than reporting the same missing key five times.
        raise
    except LLMError as exc:
        # Never drop the source. Fall back to its raw text, truncated, and say
        # so — an unexplained gap in a briefing is worse than a stated one.
        raw = report.summary
        truncated = len(raw) > 2_000
        return SourceReport(
            source=report.source,
            ok=True,
            summary=raw[:2_000] + ("… (truncated)" if truncated else ""),
            items=report.items,
            context_items=report.context_items,
            path="error",
            route_reason=f"model failed ({exc}); fell back to raw text"
            + (" and truncated it" if truncated else ""),
        )


#: A briefing that runs past this many characters did not come back as the four
#: sentences it was asked for. Four long sentences are ~600 chars, so this is
#: generous — it is a nonsense detector, not a style rule.
MAX_BRIEFING_CHARS = 1_200


def is_plausible_briefing(text: str) -> bool:
    """Does this look like the four sentences of prose we asked for?

    On 2026-09-01 a reasoning model answered the fuse prompt with "Here's a
    thinking process:" and nine paragraphs of deliberation. The call *succeeded*,
    so no ``except LLMError`` caught it, and the pipeline printed it and handed
    it to TTS — which would have read the model's private reasoning aloud, for
    over a minute, on a path where nobody is watching.

    Length is the check because it is the sturdy one. Sniffing for markers
    (``<think>``, "thinking process") is model-specific and rots the moment a
    model words it differently; "far longer than four sentences" holds whoever
    is serving the request.
    """
    return len(text) <= MAX_BRIEFING_CHARS


def _raw_briefing(reports: list[SourceReport]) -> str:
    """The un-fused fallback, when the brain is unavailable entirely."""
    lines = []
    for report in reports:
        label = report.source if report.ok else f"{report.source} (unavailable)"
        lines.append(f"{label}: {report.summary}")
    return "\n".join(lines) if lines else "Nothing to report."


def fuse(reports: list[SourceReport], config: Config) -> Briefing:
    """Fuse every source into one briefing. Never raises on a model failure."""
    needs_you: list[NeedsYouItem] = [item for report in reports for item in report.items]
    context = [item for report in reports for item in report.context_items]

    if not reports:
        return Briefing(briefing_text="Nothing to report.", needs_you=[])

    try:
        # `or ""` matches reduce_source and summarize_source. llm.complete now
        # normalises a null content itself, so this is belt-and-braces — but
        # fuse being the one caller without the guard is exactly how a null
        # answer turned into an AttributeError that no `except LLMError`
        # anywhere up the stack could catch.
        text = (
            complete(
                prompts.build_fuse_prompt(reports),
                config.brain,
                config.brain.fuser_model,
            )
            or ""
        ).strip()
    except (LLMError, MissingApiKey) as exc:
        # The Voicelog degradation pattern: you still get the information, just
        # unfused. This is the whole reason the raw summaries are kept around.
        return Briefing(
            briefing_text=_raw_briefing(reports),
            needs_you=needs_you,
            context=context,
            note=f"fuser failed ({exc}); fell back to the raw per-source list",
        )

    if not text:
        return Briefing(
            briefing_text=_raw_briefing(reports),
            needs_you=needs_you,
            context=context,
            note="fuser returned nothing; fell back to the raw per-source list",
        )

    if not is_plausible_briefing(text):
        # A *successful* call that returned nonsense. Same fallback as a failed
        # one, because the outcome for the listener is the same — except this
        # path is worse if unhandled: the text is spoken rather than erroring.
        return Briefing(
            briefing_text=_raw_briefing(reports),
            needs_you=needs_you,
            context=context,
            note=(
                f"fuser returned {len(text)} chars for a four-sentence briefing "
                f"(limit {MAX_BRIEFING_CHARS}) — looks like reasoning, not prose; "
                f"fell back to the raw per-source list"
            ),
        )

    return Briefing(briefing_text=text, needs_you=needs_you, context=context)
