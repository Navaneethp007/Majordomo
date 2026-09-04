"""Prompt assembly. Pure string building — no I/O, so it is trivially testable."""
from __future__ import annotations

from majordomo.models import SourceReport

_VOICE = (
    "You are a majordomo: the head of household staff who runs the workers and "
    "briefs the master. You are brief, plain-spoken and never breathless. You "
    "never pad, never congratulate, and never invent items that were not in the "
    "data you were given."
)


#: The assistant voice, as distinct from the briefing voice above. The briefing
#: is read aloud to someone who just sat down; this is a conversation with
#: someone who is working. Different job, different register.
#:
#: The two cases below are separated deliberately, and the separation was
#: learned the hard way. An earlier version had one rule — "where the context
#: does not answer it, say so rather than guessing" — written about *activity*.
#: The model applied it to everything, so "who is the president of India" came
#: back opening with "I don't have that in my cache". A guardrail scoped wider
#: than the risk it guards against makes the assistant useless at the ordinary
#: half of its job.
_ASSISTANT = (
    "You are Majordomo, a personal assistant to a developer. Below you have "
    "what you have learned about him over time and his recent GitHub "
    "activity.\n\n"
    "Answer what was actually asked, and lead with the answer rather than "
    "working up to it.\n\n"
    "**Questions about him, his projects, or his work.** Use the context below "
    "and say what you are drawing on. Never invent an activity, a repository, "
    "or a fact about him that is not there — a fabricated commit is far worse "
    "than saying the cache does not go back that far. This is the only place "
    "that restriction applies.\n\n"
    "**Everything else — general knowledge, opinions, explanations, advice.** "
    "Just answer, the way any capable assistant would. Do not preface it by "
    "explaining what your context does or does not contain; he knows what you "
    "can see, and it is not interesting. Give a real answer with a real "
    "opinion. Only add a caveat where you are genuinely unsure of a specific "
    "figure — a price, a measurement, a date — and then keep it to a clause, "
    "not a paragraph.\n\n"
    "Where the two overlap, use both: his own work is often the most useful "
    "thing to reason from."
)


def build_ask_prompt(context_text: str, question: str) -> list[dict]:
    """One-shot question against everything known.

    The ordering here is load-bearing and not cosmetic — see ``context.py``.
    Everything stable comes first and the question is last, so the prefix is
    byte-identical across calls and caches. Putting the question anywhere but
    the end means paying full price on every request.
    """
    content = f"{context_text}\n\n---\n\n{question}" if context_text else question
    return [
        {"role": "system", "content": _ASSISTANT},
        {"role": "user", "content": content},
    ]


def build_chat_system_prompt(context_text: str) -> str:
    """The frozen prefix for an interactive session.

    Returned as one system string rather than a message list because the caller
    holds the growing turn list and must be able to keep this part unchanged.
    """
    if not context_text:
        return _ASSISTANT
    return f"{_ASSISTANT}\n\n---\n\n{context_text}"


def build_memory_proposal_prompt(transcript: str) -> list[dict]:
    """Ask what, if anything, is worth remembering from a conversation.

    Note what this prompt does *not* do: it never writes. It returns candidates
    that a human accepts or drops. A model that could write to memory unattended
    would eventually record something wrong about you, and a wrong memory is
    replayed into every future conversation that matches it.
    """
    return [
        {"role": "system", "content": _ASSISTANT},
        {
            "role": "user",
            "content": (
                "Below is a conversation we just had. Identify anything worth "
                "remembering about me for future conversations — durable "
                "preferences, facts about my projects, how I like to work.\n\n"
                "Rules:\n"
                "- Only things that will still be true in six months.\n"
                "- Nothing already recorded in a repo, git history, or code.\n"
                "- Never credentials, keys, or passwords.\n"
                "- If nothing qualifies, reply with exactly: NOTHING\n\n"
                "Format each on its own line as:\n"
                "type | one-line description\n"
                "where type is one of: user, preference, project, reference\n\n"
                f"{transcript}"
            ),
        },
    ]


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
