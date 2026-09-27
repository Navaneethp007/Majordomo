"""Prompt assembly. Pure string building — no I/O, so it is trivially testable."""
from __future__ import annotations
import re
from dataclasses import dataclass

from majordomo.models import SourceReport

_VOICE = (
    "You are a majordomo: the head of household staff who runs the workers and "
    "briefs the master. You are brief, plain-spoken and never breathless. You "
    "never pad, never congratulate, and never invent items that were not in the "
    "data you were given."
)


#: What the model emits instead of answering when the question needs the disk.
#: Parsed in Python by ``needs_agent`` — the caller then offers to hand it to
#: the agent, which is the only thing here that can actually read a file.
NEEDS_AGENT_MARKER = "NEEDS_AGENT:"

#: The assistant voice, as distinct from the briefing voice above. The briefing
#: is read aloud to someone who just sat down; this is a conversation with
#: someone who is working. Different job, different register.
#:
#: The two cases below are separated deliberately, and the separation was
#: learned the hard way. An earlier version had one rule — "where the context
#: does not answer it, say so rather than guessing" — written about *activity*.
#: The model applied it to everything, so "who is the president of India" came
#: back opening with "I don't have that in my cache". A guardrail scoped wider
#: than the risk it guards against makes the assistant useless at the ordinary half of its job.
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

#: Appended for chat and `mj ask`, which have no tools. Not for the memory
#: proposal, which has its own strict output format and no reason to be told
#: about a protocol it must never use.
#:
#: The ordering inside matters and was measured. With context loaded, an earlier
#: version lost to the "questions about him use the context below" rule: asked
#: to look in a folder, the model reported the folder was absent from its
#: context — true, irrelevant, and not what was asked. Saying which rule wins is
#: what stopped that.
_NO_TOOLS = (
    "\n\n**You cannot read files, list directories, or run commands.** There is "
    "no tool here and no access to his machine.\n\n"
    "When answering would require looking at a file, a directory or a "
    "repository, your **entire reply** is this one line:\n\n"
    f"    {NEEDS_AGENT_MARKER} <the task, phrased for a coding agent>\n\n"
    "No preamble, no apology, no code fence, nothing after it. Something else "
    "acts on that line and hands the work to an agent that genuinely can look. "
    "Never offer to have files pasted to you instead, and never emit a tool "
    "call or anything shaped like one — it does not run, and it reads as though "
    "something happened when nothing did.\n\n"
    "Where he asks you to **open something on disk**, this outranks the rule "
    "about his context above: do not answer such a request by reporting that "
    "the file or folder is absent from your context — true, and not what was "
    "asked.\n\n"
    "That is a narrow exception, not a general one. "
    "\"What have I been doing in <repo>?\" is a question about his activity and "
    "you answer it from context as usual. Only a request to *look inside* "
    "something — read this file, list that directory, check what is in there — "
    "goes to the agent.\n\n"
    "Example:\n"
    "  Him: what is in the api folder?\n"
    f"  You: {NEEDS_AGENT_MARKER} list and summarise the contents of the api folder"
    "\n\n"
    "Only for requests that really need the filesystem. A question you can "
    "answer from context or general knowledge is not one of those."
)


#: What the model emits when you ask it to keep something. Parsed in Python by
#: ``wants_remembered``, then offered — never written unattended. A wrong memory
#: replays into every future conversation that matches it, which is why
#: ``build_memory_proposal_prompt`` never writes either.
REMEMBER_MARKER = "REMEMBER:"

#: Appended alongside ``_NO_TOOLS`` for chat. Not for `mj ask`: one-shot, so
#: there is nobody to confirm to, and a marker nobody acts on would print.
_REMEMBER = (
    "\n\nWhen he asks you to remember something — \"remember this\", \"keep that "
    "in mind\", \"note that I prefer X\" — end your reply with a line of its "
    "own:\n\n"
    f"    {REMEMBER_MARKER} <the fact, in one sentence, written about him>\n\n"
    "Answer him normally first; this line goes last. Something else acts on it "
    "and asks him before anything is saved, so do not claim you have saved "
    "anything.\n\n"
    "Only for facts that will still be true in six months — preferences, how he "
    "works, what his projects are. Not a passing detail of the conversation you "
    "are already having, and never a credential."
)


def wants_remembered(reply: str) -> str | None:
    """The fact the model wants kept, if it asked for one.

    Structural, like ``needs_agent``: a marker the model was told to emit,
    parsed here. Whether a sentence "sounds like" the user asked to be
    remembered is not something to branch on.
    """
    if REMEMBER_MARKER not in reply:
        return None
    _, _, rest = reply.partition(REMEMBER_MARKER)
    for line in rest.splitlines():
        fact = line.strip().strip("`").strip()
        if fact:
            return fact
    return None


def strip_remember(reply: str) -> str:
    """The reply without the marker line, which was never meant to be read."""
    if REMEMBER_MARKER not in reply:
        return reply
    head, _, _ = reply.partition(REMEMBER_MARKER)
    return head.rstrip()


def needs_agent(reply: str) -> str | None:
    """The task the model wants handed to the agent, if it asked for one.

    A **structural** signal, not prose-sniffing. The same discipline as
    ``needs_you`` in the briefing: the decision is made in Python from a marker
    the model was told to emit, never inferred from how a sentence reads. "It
    sounds like it is refusing" is not something to branch on.

    Three outcomes, and the middle one matters: ``None`` for an ordinary
    answer (the common path, costing one ``in``), the task where there is
    one, and ``""`` where the marker appeared but carried nothing — still a
    hand-off, because the alternative is printing the marker.
    """
    if NEEDS_AGENT_MARKER not in reply:
        return None

    _, _, rest = reply.partition(NEEDS_AGENT_MARKER)
    # The task may sit on the marker's line or the one after it. Reading only
    # the marker's own line returned None for `NEEDS_AGENT:\n<task>`, and a
    # None here means no offer fires and the raw reply prints — putting the
    # protocol token on screen, which is the single outcome it exists to
    # prevent. Blank lines are skipped for the same reason.
    for line in rest.splitlines():
        task = line.strip().strip("`").strip()
        if task:
            return task

    # The marker was there and carried nothing. Still a hand-off: the caller
    # falls back to what the user asked for. Returning None would print the
    # marker.
    return ""


def strip_needs_agent(reply: str) -> str:
    """The reply without the marker line, keeping anything written before it.

    The sibling path for a leaked tool call already did this — kept the
    paragraphs, lost the block. This one did not, so prose written before the
    marker was dropped from the screen, from ``turns``, from ``log`` and from
    the memory proposer, even when the offer was declined. Two paths doing the
    same job disagreed, and the one with no stripping was the original.
    """
    if NEEDS_AGENT_MARKER not in reply:
        return reply
    head, _, _ = reply.partition(NEEDS_AGENT_MARKER)
    return head.rstrip()


#: A tool call the model *emitted*, as opposed to one it is talking about.
#:
#: The distinction is the whole point. A substring net over the reply flagged
#: ordinary answers — and this project is itself an LLM tool, so "how do tool
#: calls work?" is a question you will actually ask:
#:
#:     The Anthropic API uses <invoke name="get_weather"> in its examples.
#:
#: Two rules, and both were learned by getting it wrong:
#:
#: 1. **The opening token must start a line.** A mention sits inside a sentence.
#: 2. **It must not be inside a code fence.** This is the one that took two
#:    attempts. Allowing an optional fence *prefix* did nothing, because the
#:    token still starts its own line within the block — so "a tool call looks
#:    like this: ```<tool_call>…```" was flagged and the answer truncated at the
#:    fence. Fenced content is quoted precisely because it is an illustration.
#:
#: False negatives are cheap — the text simply shows as written. A false
#: positive throws away a correct answer.
_EMITTED_CALL = re.compile(
    r"^[^\S\n]*"
    r"(?:<\|?[a-z_]*(?:function|tool)[a-z_]*(?:_call)?[|>=\s]"
    r"|<invoke\s+name\s*="
    r"|<function\s*="
    r"|\{\s*\"tool_calls\"\s*:)",
    re.IGNORECASE | re.MULTILINE,
)

_FENCE_LINE = re.compile(r"^[^\S\n]*```", re.MULTILINE)


def _outside_fences(text: str) -> str:
    """``text`` with fenced blocks blanked out, offsets preserved.

    Blanked rather than removed so a match position in the result is a valid
    position in the original — ``strip_tool_call`` cuts the real string at an
    index found here.
    """
    out = list(text)
    inside = False
    start = 0
    for line in text.splitlines(keepends=True):
        end = start + len(line)
        fenced = _FENCE_LINE.match(line) is not None
        if inside or fenced:
            for i in range(start, end):
                if out[i] != "\n":
                    out[i] = " "
        if fenced:
            inside = not inside
        start = end
    return "".join(out)


def _emitted_call_at(text: str):
    """The match for a call the model actually emitted, or None."""
    return _EMITTED_CALL.search(_outside_fences(text))


def looks_like_a_tool_call(text: str) -> bool:
    """Did the model write out a tool call instead of answering?

    Chat and `mj ask` deliberately have no tools — `/agent` and `mj do` are the
    paths that do. Asked to read a file, a model may still produce a block of
    provider-specific call syntax, which lands raw in the terminal and reads as
    though something ran. Nothing did.

    Observed with dots-3, which emitted a ``<dots_function_call>`` block
    complete with a shell command when asked to look inside a folder.
    """
    return _emitted_call_at(text) is not None


def strip_tool_call(text: str) -> str:
    """The reply with an emitted call removed, keeping everything else.

    Replacing the whole message loses real content: a model that wrote three
    good paragraphs and one stray ``<function_call>`` block should cost you the
    block, not the paragraphs. Drops from the first line that opens a call to
    the end, because everything after it is the call's arguments and closing
    tags rather than prose.
    """
    match = _emitted_call_at(text)
    if match is None:
        return text.strip()
    return text[: match.start()].strip()


#: Shown in place of that block. Says what happened and what to do instead —
#: the request itself is reasonable, it was simply made of the wrong command.
NO_TOOLS_HERE = (
    "I started to call a tool, but this is a plain conversation — I cannot "
    "read files or run commands here, so nothing happened."
)


@dataclass(frozen=True)
class AgentHandoff:
    """A reply that wants the agent, already parsed.

    Attributes:
        task: What to hand over. Never empty — see ``classify_reply``.
        leaked: The model wrote a tool call instead of using the marker.
        kept: The prose to show first, with the marker or call removed. May be
            empty, when the reply was nothing but the marker.
    """

    task: str
    leaked: bool
    kept: str


def classify_reply(reply: str, asked_for: str) -> AgentHandoff | None:
    """Does this reply want the agent? ``None`` when it does not.

    There are two ways in. The model is *told* to emit ``NEEDS_AGENT: <task>``
    when a question genuinely needs the disk — a structural signal, parsed here
    rather than inferred from prose, the same discipline as ``needs_you``. And
    when it ignores that and writes out a tool call instead, that is the same
    request wearing the wrong syntax, so it counts too.

    ── WHY THIS IS ONE FUNCTION ─────────────────────────────────────────────
    ``chat._offer_agent`` and ``cli.cmd_ask`` both need this, and they had a
    copy each. The copies disagreed twice: first about whether a leaked tool
    call keeps the prose that came with it, then about whether the marker branch
    does. Both times the branch that was already right stayed right and its
    sibling stayed wrong, because nothing made them one algorithm.

    What the two callers genuinely do differ on is what happens *next* — chat
    asks permission, ``mj ask`` prints the command, because a one-shot has
    nobody to ask. That difference is real and stays with them. The parse is not
    a difference, so it lives here.
    """
    task = needs_agent(reply)
    leaked = task is None and looks_like_a_tool_call(reply)
    if task is None and not leaked:
        return None

    # An empty task means the marker arrived carrying nothing. It is still a
    # hand-off — the alternative is printing the protocol token at the user —
    # so fall back to what was actually asked for.
    if not task:
        task = asked_for

    # Keep whatever prose came with it, on either branch. A model that wrote
    # three good paragraphs and one stray marker should lose the marker, not the
    # paragraphs, and declining the offer must not cost you the answer.
    kept = strip_tool_call(reply) if leaked else strip_needs_agent(reply)

    return AgentHandoff(task=task, leaked=leaked, kept=kept)


def build_ask_prompt(context_text: str, question: str) -> list[dict]:
    """One-shot question against everything known.

    The ordering here is load-bearing and not cosmetic — see ``context.py``.
    Everything stable comes first and the question is last, so the prefix is
    byte-identical across calls and caches. Putting the question anywhere but
    the end means paying full price on every request.
    """
    content = f"{context_text}\n\n---\n\n{question}" if context_text else question
    return [
        {"role": "system", "content": _ASSISTANT + _NO_TOOLS},
        {"role": "user", "content": content},
    ]


def build_chat_system_prompt(context_text: str) -> str:
    """The frozen prefix for an interactive session.

    Returned as one system string rather than a message list because the caller
    holds the growing turn list and must be able to keep this part unchanged.
    """
    system = _ASSISTANT + _NO_TOOLS + _REMEMBER
    if not context_text:
        return system
    return f"{system}\n\n---\n\n{context_text}"


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
