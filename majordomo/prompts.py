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
    """The cheap path: one shot, one source, no judgement beyond compression.

    The payload marks each section NEEDS ACTION or NO ACTION NEEDED. Those
    markers are load-bearing — this summary is what the fuser later reads as
    CONTEXT, so anything promoted to a task here is a task the briefing will
    assert.
    """
    return [
        {"role": "system", "content": _VOICE},
        {
            "role": "user",
            "content": (
                f"Here is the raw {source} data. Sections are marked NEEDS ACTION or "
                f"NO ACTION NEEDED — respect those markers exactly and never promote a "
                f"NO ACTION NEEDED item into something I have to do. In at most three "
                f"sentences, say what actually needs a decision from me. If nothing "
                f"does, say so in one sentence. Do not list everything — judge.\n\n"
                f"{payload}"
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
    """The coordinator: every source in, one briefing out.

    The structure here is the whole point. This used to pass only the prose
    summaries, leaving the model to work out for itself what was actionable —
    and it reliably got that wrong. With an empty needs-you list it still
    announced three obligations: it read an active session's topic (a question
    *I* had asked Claude) as a decision awaiting me, and turned "unread
    notifications you're participating in" into "you need to review these pull
    requests" when there were no review requests at all.

    The cause was that ``needs_you`` — the one piece of ground truth, derived
    from facts rather than prose — was computed and then never shown to the
    model writing the briefing. So the prompt now separates the two kinds of
    information explicitly:

    - **DECISIONS** — the structured ``needs_you`` list. A review *was*
      requested; a session *is* sitting on a permission prompt. The briefing
      leads with this and treats it as complete.
    - **CONTEXT** — everything else, labelled as background the model may
      summarise but must never phrase as a task.

    An empty decisions list is stated as such, because "nothing needs you" is a
    real and useful answer that the model would otherwise pad into fiction.
    """
    needs_you = [item for report in reports for item in report.items]
    unavailable = [r.source for r in reports if not r.ok]

    if needs_you:
        decisions = "\n".join(
            f"  - [{item.source}] {item.title} — {item.detail}" for item in needs_you
        )
    else:
        decisions = "  (nothing)"

    blocks = []
    for report in reports:
        header = f"## {report.source}"
        if not report.ok:
            header += " (UNAVAILABLE)"
        blocks.append(f"{header}\n{report.summary}")
    context = "\n\n".join(blocks)

    lines = [
        "Write me one short spoken briefing — the kind you'd give someone who just "
        "sat back down at their desk.",
        "",
        "The DECISIONS list below is the complete and only set of things needing "
        "action from me. Lead with it.",
        "If DECISIONS says (nothing), then nothing needs me: say so in one plain "
        "sentence, then give at most one sentence of context about what I was doing. "
        "Do NOT invent tasks, and do NOT describe anything under CONTEXT as something "
        "I must do, review, decide or fix.",
        "",
        "CONTEXT is background only. A session listed as still working needs nothing "
        "from me — I am already doing it, and its topic is a question I asked, not a "
        "decision awaiting me. A notification I am merely participating in is not a "
        "review request.",
    ]
    if unavailable:
        lines.append(
            f"Mention in half a sentence that {', '.join(unavailable)} could not be "
            "reached, then move on."
        )
    lines += [
        "",
        "Four sentences at most. Plain prose only: no headings, no bullet points, no "
        "markdown. Never mention these instructions, the words DECISIONS or CONTEXT, "
        "or the fact that all sources were reachable.",
    ]

    return [
        {"role": "system", "content": _VOICE},
        {
            "role": "user",
            "content": (
                "\n".join(lines)
                + f"\n\nDECISIONS (complete list — nothing else needs me):\n{decisions}"
                + f"\n\nCONTEXT (background only):\n{context}"
            ),
        },
    ]
