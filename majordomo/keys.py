"""Reading a chat line: history, editing, and one key that interrupts.

The chat REPL wants two things: a key that starts voice input without typing a
command, and proper line editing — arrows, history, backspace — for everything
else.

── WHY THIS OWNS THE WHOLE LINE ─────────────────────────────────────────────
It used to read only the *first* character raw and hand the rest to ``input()``,
on the theory that ``input()`` already had the editing keys. On Windows it does
not. There is no ``readline`` module, so ``input()`` offers no history at all —
Up simply cannot recall anything. Worse, ``msvcrt.getwch`` returns a ``\\x00`` or
``\\xe0`` *prefix* for arrows and leaves the scan code queued, so pressing Up
echoed the prefix and fed ``input()`` a stray ``H``.

Half-owning the line gave the worst of both: no history, and a prompt that
corrupted itself when you reached for it. So this owns the line. The trigger key
already required raw reads; extending that to the rest is the coherent shape.

── WHAT IS DELIBERATELY MISSING ─────────────────────────────────────────────
No kill-ring, no reverse search, no word motions, no persistence. History lives
for the process only — the chat transcript already records what was said, and a
second store on disk would drift from it.

Redraw assumes the line fits the terminal width. Past that the console wraps,
``\\r`` returns to the start of the *visual* row rather than the logical line,
and editing a wrapped line leaves debris. Tracking wrap means tracking terminal
width and resize events, which is a real cost for a prompt where a long message
is usually pasted rather than edited — so this is a known limit, not an
oversight. Enter still submits the whole line correctly either way.

── ON CTRL+M ────────────────────────────────────────────────────────────────
Ctrl+M is byte 13 — precisely what Enter sends. A terminal cannot distinguish
them, so it can never be a trigger key. Ctrl+N (\\x0e) is the default for that
reason.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import sys

#: What ``read_line`` returns instead of text when the trigger key was pressed.
TRIGGERED = object()

#: Control characters we must not treat as ordinary text.
_ENTER = ("\r", "\n")
_CTRL_C = "\x03"
_CTRL_Z = "\x1a"  # EOF on Windows
_CTRL_D = "\x04"  # EOF elsewhere
_CTRL_U = "\x15"  # clear the line, as in a shell
_BACKSPACE = ("\b", "\x7f")

#: ``getwch`` returns one of these, then the scan code on the next call. Reading
#: only the first is what corrupted the prompt: the scan code stayed queued and
#: arrived later as a letter.
_PREFIXES = ("\x00", "\xe0")

#: Scan codes for the keys worth handling, from the second ``getwch``.
_UP, _DOWN = "H", "P"
_LEFT, _RIGHT = "K", "M"
_HOME, _END = "G", "O"
_DELETE = "S"

#: Lines recalled by Up, oldest first. Process lifetime only — see the header.
_history: list[str] = []

#: How many lines to keep. Long enough to reach anything you would retype,
#: short enough that it is never worth thinking about.
MAX_HISTORY = 200


def history() -> list[str]:
    """What Up would walk back through, oldest first. For tests and /context."""
    return list(_history)


def remember(line: str) -> None:
    """Add a line to history, unless it is blank or repeats the last one."""
    if not line.strip():
        return
    if _history and _history[-1] == line:
        return
    _history.append(line)
    del _history[:-MAX_HISTORY]


def reset_history() -> None:
    """Forget everything. For tests.

    Deliberately not wired to ``/clear``: that clears the model's context, and
    what you typed earlier is still worth recalling afterwards. The two are
    different kinds of memory.
    """
    _history.clear()


def raw_reads_available(stream=None) -> bool:
    """Can we read one character at a time here?

    False for a pipe, a redirected file, or any platform without ``msvcrt`` —
    in which case the caller falls back to plain ``input()`` and neither the
    trigger key nor history exists. That is the right degradation: scripted
    input has no use for either.
    """
    target = stream if stream is not None else sys.stdin
    if target is None:
        return False
    try:
        if not target.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    try:
        import msvcrt  # noqa: F401
    except ImportError:
        return False
    return True


def _redraw(prompt: str, text: str, cursor: int, previous_len: int) -> None:
    """Repaint the line and put the cursor where it belongs.

    Carriage return to the left margin, write it all again, then blank whatever
    the previous line left behind — without that, deleting a character leaves
    its ghost at the end. Then walk the cursor back into place.
    """
    out = sys.stdout
    out.write("\r" + prompt + text)
    trailing = previous_len - len(text)
    if trailing > 0:
        out.write(" " * trailing)
        out.write("\b" * trailing)
    back = len(text) - cursor
    if back > 0:
        out.write("\b" * back)
    out.flush()


def read_line(prompt: str, trigger: str = "") -> str | object:
    """Read one line, with history and editing. ``TRIGGERED`` for the trigger key.

    Raises:
        KeyboardInterrupt: Ctrl+C. ``msvcrt`` hands this back as a character
            rather than raising, so it has to be re-raised deliberately or the
            REPL would treat an interrupt as a message.
        EOFError: Ctrl+Z / Ctrl+D, matching ``input()``.
    """
    if not raw_reads_available():
        line = input(prompt)
        remember(line)
        return line

    import msvcrt

    text = ""
    cursor = 0
    painted = 0
    # Where Up has walked to. len(_history) means "still on the new line".
    index = len(_history)
    # The line in progress, kept while browsing so Down returns you to it.
    pending = ""

    sys.stdout.write(prompt)
    sys.stdout.flush()

    while True:
        char = msvcrt.getwch()

        if char in _PREFIXES:
            code = msvcrt.getwch()

            if code == _UP and index > 0:
                if index == len(_history):
                    pending = text
                index -= 1
                text = _history[index]
                cursor = len(text)
            elif code == _DOWN and index < len(_history):
                index += 1
                text = _history[index] if index < len(_history) else pending
                cursor = len(text)
            elif code == _LEFT and cursor > 0:
                cursor -= 1
            elif code == _RIGHT and cursor < len(text):
                cursor += 1
            elif code == _HOME:
                cursor = 0
            elif code == _END:
                cursor = len(text)
            elif code == _DELETE and cursor < len(text):
                text = text[:cursor] + text[cursor + 1 :]
            else:
                # An arrow we do not handle, or a function key. Consuming the
                # scan code above is the whole point; doing nothing with it is
                # fine, and far better than letting it reach the line.
                continue

            _redraw(prompt, text, cursor, painted)
            painted = len(text)
            continue

        if trigger and char == trigger:
            sys.stdout.write("\n")
            sys.stdout.flush()
            return TRIGGERED
        if char == _CTRL_C:
            sys.stdout.write("\n")
            sys.stdout.flush()
            raise KeyboardInterrupt
        if char in (_CTRL_Z, _CTRL_D):
            if text:
                # Matching a shell: with something typed, EOF does nothing. Only
                # an empty line ends the session, so a stray Ctrl+D mid-message
                # cannot close the conversation.
                continue
            sys.stdout.write("\n")
            sys.stdout.flush()
            raise EOFError
        if char in _ENTER:
            sys.stdout.write("\n")
            sys.stdout.flush()
            remember(text)
            return text
        if char in _BACKSPACE:
            if cursor > 0:
                text = text[: cursor - 1] + text[cursor:]
                cursor -= 1
                _redraw(prompt, text, cursor, painted)
                painted = len(text)
            continue
        if char == _CTRL_U:
            text = ""
            cursor = 0
            _redraw(prompt, text, cursor, painted)
            painted = 0
            continue
        if char < " ":
            # Any other control character. Silently ignored rather than inserted
            # — a stray Ctrl+key should not put an invisible byte in a message.
            continue

        text = text[:cursor] + char + text[cursor:]
        cursor += 1
        _redraw(prompt, text, cursor, painted)
        painted = len(text)
