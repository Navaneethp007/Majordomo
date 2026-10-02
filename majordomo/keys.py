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

Wrapping *is* handled, after a spell when it was not. The note here used to
call it an acceptable limit on the grounds that "a long message is usually
pasted rather than edited" — which was simply wrong: a typed prompt runs past
eighty columns constantly, and every keystroke after that repainted the prompt
and the first row's worth of text underneath the line. See ``_redraw``.

Terminal *resizing* mid-line is still not handled. The width is read fresh on
each keystroke, so it corrects itself on the next one.

── ON CTRL+M ────────────────────────────────────────────────────────────────
Ctrl+M is byte 13 — precisely what Enter sends. A terminal cannot distinguish
them, so it can never be a trigger key. Ctrl+N (\\x0e) is the default for that
reason.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import shutil
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


def _ansi_available(stream=None) -> bool:
    """Will the cursor codes render, or land as visible garbage?

    Also the thing that *enables* them: on Windows, virtual-terminal processing
    is off per console until something calls ``SetConsoleMode``, which
    ``render.supports_ansi`` does. Nothing had — ``read_line`` runs before the
    first reply is rendered, so the very first line you typed was the one at
    risk of showing raw escapes.
    """
    from majordomo import render

    return render.supports_ansi(stream)


def _terminal_width(stream=None) -> int:
    """How many columns the line has before it wraps. Never zero."""
    try:
        return max(1, shutil.get_terminal_size(fallback=(80, 24)).columns)
    except (OSError, ValueError):
        return 80


def _terminal_height(stream=None) -> int:
    """How many rows there are to repaint within. Never zero."""
    try:
        return max(1, shutil.get_terminal_size(fallback=(80, 24)).lines)
    except (OSError, ValueError):
        return 24


def _redraw(prompt: str, text: str, cursor: int, row: int, stream=None) -> int:
    """Repaint the whole line and park the cursor. Returns its new row.

    ``\\r`` returns to the start of the *visual* row, not of the line. That is
    fine until the line wraps, and then it is the bug you see: the cursor is on
    the second row, so every keystroke repainted the prompt and the first row's
    worth of text *underneath* the line, over and over. The old note here called
    that an acceptable limit because "a long message is usually pasted rather
    than edited" — which was wrong. A typed prompt runs past eighty columns all
    the time.

    So the row is tracked. Go up to the line's first row, clear everything from
    there down, write it again, and walk the cursor back to where it belongs.
    ``row`` is where the cursor was left last time; the return value is where it
    is now, and the caller carries it between keystrokes.
    """
    out = stream if stream is not None else sys.stdout

    if not _ansi_available(out):
        # No cursor control: repaint one row and pad the tail, which is what
        # this did before wrapping was handled. Wrong past the margin, but
        # wrong quietly rather than spraying escape codes at you.
        out.write("\r" + prompt + text + " ")
        back = len(text) - cursor
        if back > 0:
            out.write("\b" * (back + 1))
        out.flush()
        return 0

    width = _terminal_width(out)

    if row:
        out.write(f"\x1b[{row}A")
    # To the left margin, then erase from here to the end of the screen. That
    # is what removes a row the line no longer needs — the old space-padding
    # trick only ever cleared the tail of one row.
    out.write("\r\x1b[0J")
    out.write(prompt + text)

    end = len(prompt) + len(text)
    if end and end % width == 0:
        # The text ends exactly at the margin. Terminals differ on whether the
        # cursor has wrapped yet; writing one space forces the question, so the
        # arithmetic below is true either way.
        out.write(" ")

    end_row = end // width
    target_row, target_column = divmod(len(prompt) + cursor, width)

    if end_row > target_row:
        out.write(f"\x1b[{end_row - target_row}A")
    out.write("\r")
    if target_column:
        out.write(f"\x1b[{target_column}C")

    out.flush()
    return target_row


def _finish_line(prompt: str, text: str, row: int, stream=None) -> None:
    """Move past the end of a possibly-wrapped line and start a new one.

    A bare newline from wherever the cursor happens to sit leaves the tail of a
    wrapped line above it and the next prompt written over the middle of it.
    """
    out = stream if stream is not None else sys.stdout
    if not _ansi_available(out):
        out.write("\n")
        out.flush()
        return

    width = _terminal_width(out)
    end_row = (len(prompt) + len(text)) // width
    if end_row > row:
        out.write(f"\x1b[{end_row - row}B")
    out.write("\n")
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
    row = 0
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

            row = _redraw(prompt, text, cursor, row)
            continue

        if trigger and char == trigger:
            _finish_line(prompt, text, row)
            return TRIGGERED
        if char == _CTRL_C:
            _finish_line(prompt, text, row)
            raise KeyboardInterrupt
        if char in (_CTRL_Z, _CTRL_D):
            if text:
                # Matching a shell: with something typed, EOF does nothing. Only
                # an empty line ends the session, so a stray Ctrl+D mid-message
                # cannot close the conversation.
                continue
            _finish_line(prompt, text, row)
            raise EOFError
        if char in _ENTER:
            _finish_line(prompt, text, row)
            remember(text)
            return text
        if char in _BACKSPACE:
            if cursor > 0:
                text = text[: cursor - 1] + text[cursor:]
                cursor -= 1
                row = _redraw(prompt, text, cursor, row)
            continue
        if char == _CTRL_U:
            text = ""
            cursor = 0
            row = _redraw(prompt, text, cursor, row)
            continue
        if char < " ":
            # Any other control character. Silently ignored rather than inserted
            # — a stray Ctrl+key should not put an invisible byte in a message.
            continue

        text = text[:cursor] + char + text[cursor:]
        cursor += 1
        row = _redraw(prompt, text, cursor, row)


#: Escape, for cancelling a picker. Ctrl+C does too, by raising.
_ESCAPE = "\x1b"

#: How many options to show at once, at most. A longer list scrolls rather than
#: repainting a screenful.
#:
#: This is a ceiling, not the answer — see ``_visible_count``. A list taller than
#: the *terminal* cannot be repainted at all, because the rows we would move back
#: up to have already scrolled away, so the real limit is whichever of the two is
#: smaller. Naming that hazard here and then not measuring the terminal was how
#: the first version of this shipped.
VISIBLE_OPTIONS = 10


def _visible_count(stream=None) -> int:
    """How many options actually fit. At least one.

    Two rows go to the prompt and the hint, so the window is the terminal's
    height minus those — clamped to ``VISIBLE_OPTIONS`` because ten is already
    more than anyone scans, and clamped to one because a terminal reporting two
    rows must still show something rather than dividing down to nothing.

    Reachable in an IDE panel dragged small or a tmux split, not in a normal
    window — which is exactly why it needs measuring rather than assuming.
    """
    return max(1, min(VISIBLE_OPTIONS, _terminal_height(stream) - 2))


def choose(prompt: str, options: list[str], stream=None) -> int | None:
    """Pick one of ``options`` with the arrow keys. Returns its index, or None.

    ── WHY THIS LIVES WITH THE LINE EDITOR ──────────────────────────────────
    It is the same job: own the raw input, repaint rows, put the cursor back.
    ``read_line`` already learned the hard parts — that ``getwch`` returns a
    prefix *and* a scan code for an arrow, and that ``\r`` returns to the start
    of the visual row rather than the line — so a picker written anywhere else
    would have to learn them again.

    Degrades the same way too. Without raw reads — a pipe, a redirect, any
    platform without ``msvcrt`` — there are no arrow keys to press, so it prints
    a numbered list and reads a number. Scripted input has no use for a cursor.

    Returns:
        The chosen index, or ``None`` if the user cancelled with Escape.

    Raises:
        KeyboardInterrupt: Ctrl+C, re-raised deliberately so a caller can tell
            "I changed my mind" from "I picked nothing" — the distinction
            ``cli.ask`` was fixed to preserve.
    """
    if not options:
        return None

    out = stream if stream is not None else sys.stdout

    if not raw_reads_available():
        return _choose_by_number(prompt, options, out)

    import msvcrt

    cursor = 0
    top = 0
    painted = 0

    while True:
        # Measured every repaint, not once: a terminal can be resized mid-pick,
        # and the window is what the move-up arithmetic depends on.
        visible = _visible_count(out)
        top = _window_top(cursor, top, len(options), visible)
        painted = _paint_options(prompt, options, cursor, top, painted, visible, out)

        char = msvcrt.getwch()

        if char in _PREFIXES:
            code = msvcrt.getwch()
            if code == _UP:
                cursor = (cursor - 1) % len(options)
            elif code == _DOWN:
                cursor = (cursor + 1) % len(options)
            elif code == _HOME:
                cursor = 0
            elif code == _END:
                cursor = len(options) - 1
            continue

        if char in _ENTER:
            _finish_options(painted, out)
            return cursor
        if char == _ESCAPE:
            _finish_options(painted, out)
            return None
        if char == _CTRL_C:
            _finish_options(painted, out)
            raise KeyboardInterrupt
        if char in (_CTRL_Z, _CTRL_D):
            _finish_options(painted, out)
            return None


def _window_top(cursor: int, top: int, count: int, visible: int) -> int:
    """Which option is at the top, so the cursor stays on screen."""
    if count <= visible:
        return 0
    top = min(top, cursor)
    top = max(top, cursor - visible + 1)
    return max(0, min(top, count - visible))


def _paint_options(prompt, options, cursor, top, painted, visible, out) -> int:
    """Repaint the list. Returns how many rows it occupies.

    ``painted`` is how many rows went down last time, which is what lets the
    cursor get back to the top. The same bookkeeping ``_redraw`` needed once the
    line could wrap, and for the same reason: there is no way to ask a terminal
    where it is.
    """
    window = options[top : top + visible]

    if not _ansi_available(out):
        # No cursor control, so no repainting: print the list once per keypress
        # and let it scroll. Ugly, and only reachable on a terminal that cannot
        # do what every terminal since 1979 can.
        out.write(prompt + "\n")
        for index, text in enumerate(window):
            out.write(("> " if top + index == cursor else "  ") + text + "\n")
        out.flush()
        return 0

    if painted:
        out.write(f"\x1b[{painted}A")
    out.write("\r\x1b[0J")

    width = _terminal_width(out)
    out.write(prompt[: width - 1] + "\n")
    for index, text in enumerate(window):
        selected = top + index == cursor
        marker = "> " if selected else "  "
        # Reverse video for the selected row, and a marker as well rather than
        # instead: a highlight alone is invisible in a terminal whose theme
        # ignores it, and on a copy-pasted screenshot.
        body = (marker + text)[: width - 1]
        out.write(f"\x1b[7m{body}\x1b[0m\n" if selected else body + "\n")

    # ASCII only. This writes straight to the stream rather than through
    # `cli.safe_print`, so there is nothing to catch a UnicodeEncodeError — and
    # a default Windows console is cp1252, which has no arrow glyphs. "↑↓" would
    # crash the picker on exactly the platform this is written for. (The chat
    # prompt's "›" survives only because cp1252 happens to include it.)
    hint = "  up/down to move, Enter to choose, Esc to cancel"
    if len(options) > visible:
        hint = f"  {cursor + 1}/{len(options)} -" + hint[1:]
    out.write(hint[: width - 1])
    out.flush()

    # prompt + options + hint, and the hint has no newline, so the cursor sits
    # on the last row rather than below it.
    return 1 + len(window)


def _finish_options(painted: int, out) -> None:
    """Leave the list on screen and move past it."""
    out.write("\n")
    out.flush()


def _choose_by_number(prompt: str, options: list[str], out) -> int | None:
    """The fallback: a numbered list and one number."""
    out.write(prompt + "\n")
    for index, text in enumerate(options, start=1):
        out.write(f"  {index:>2}. {text}\n")
    out.flush()

    try:
        answer = input("  number (Enter to cancel): ").strip()
    except EOFError:
        return None
    if not answer:
        return None
    try:
        chosen = int(answer)
    except ValueError:
        return None
    return chosen - 1 if 1 <= chosen <= len(options) else None
