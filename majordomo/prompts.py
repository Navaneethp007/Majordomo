"""Prompt assembly. Pure string building — no I/O, so it is trivially testable."""
from __future__ import annotations

from majordomo.models import SourceReport

_VOICE = (
    "You are a majordomo: the head of household staff who runs the workers and "
    "briefs the master. You are brief, plain-spoken and never breathless. You "
    "never pad, never congratulate, and never invent items that were not in the "
    "data you were given."
)


def build_summarize_prompt(source: str, payload: str) -> list[dict]:
    """The cheap path: one shot, one source, no judgement beyond compression."""
    return [
        {"role": "system", "content": _VOICE},
        {
            "role": "user",
            "content": (
                f"Here is the raw {source} data. In at most three sentences, say what "
                f"actually needs a decision from me. If nothing does, say so in one "
                f"sentence. Do not list everything — judge.\n\n{payload}"
            ),
        },
    ]


def build_reduce_prompt(source: str, chunk: str, part: int, total: int) -> list[dict]:
    """The escalated path: reduce one chunk of an oversized source.

    Each chunk is reasoned about on its own terms rather than merely truncated —
    that reasoning is the entire justification for spending the extra calls.
    """
    return [
        {"role": "system", "content": _VOICE},
        {
            "role": "user",
            "content": (
                f"This is part {part} of {total} of a large {source} payload — too big "
                f"to read at once. From THIS part only, extract just the items that "
                f"plausibly need a decision from me, one per line, with enough context "
                f"to identify each. Discard the rest. If nothing in this part matters, "
                f"reply exactly: NOTHING\n\n{chunk}"
            ),
        },
    ]


def build_reduce_merge_prompt(source: str, extracts: str) -> list[dict]:
    """Fold the per-chunk extracts into one judged summary of the source."""
    return [
        {"role": "system", "content": _VOICE},
        {
            "role": "user",
            "content": (
                f"These are the notable items pulled from a large {source} payload, "
                f"read in parts. Merge them into at most four sentences describing what "
                f"needs a decision from me, worst first. Drop duplicates.\n\n{extracts}"
            ),
        },
    ]


def build_fuse_prompt(reports: list[SourceReport]) -> list[dict]:
    """The coordinator: every source in, one briefing out."""
    blocks = []
    for report in reports:
        header = f"## {report.source}"
        if not report.ok:
            header += " (UNAVAILABLE)"
        blocks.append(f"{header}\n{report.summary}")
    body = "\n\n".join(blocks)

    return [
        {"role": "system", "content": _VOICE},
        {
            "role": "user",
            "content": (
                "Below are reports from each of my sources. Write me one short spoken "
                "briefing — the kind you'd give someone who just sat back down at their "
                "desk. Lead with whatever is blocking me. Mention any source marked "
                "UNAVAILABLE in a half-sentence, then move on. Four sentences at most. "
                "Plain prose only: no headings, no bullet points, no markdown.\n\n"
                f"{body}"
            ),
        },
    ]
