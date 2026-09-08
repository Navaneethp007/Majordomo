"""Markdown into something a terminal actually shows.

Models write markdown whatever you ask of them. In a chat window that lands as
literal ``**Blue Tokai**`` and ``### Notes``, which reads worse than plain prose
would have — the markup is noise the moment nothing renders it.

Two paths, chosen by whether anything is watching:

- **A terminal** gets ANSI: bold stays bold, headings stand out, bullets become
  bullets.
- **Anything else** — a pipe, a file, ``pythonw`` with no console — gets the
  markers *stripped*. Escape codes in a redirected file are worse than the
  asterisks they replaced.

Never applied to text bound for TTS. ``tts.speech_text`` already strips markdown
for the speech path, and feeding it ANSI would have it read escape codes aloud.

Deliberately hand-rolled rather than pulling in ``rich``: this is sixty lines,
and the dependency list is two packages on purpose.
"""
from __future__ import annotations

import os
import re
import sys

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
RESET = "\033[0m"

_FENCE = re.compile(r"^\s*```")
_HEADING = re.compile(r"^(\s*)#{1,6}\s+(.*)$")
_BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]*)\)")
#: Italic emphasis: *paired* asterisks around a short run of text on one line.
#:
#: Only ever removes markers that come as a pair. The previous version deleted
#: any asterisk that merely *looked* like a marker, which meant a command
#: written without backticks — as models routinely do — came out changed:
#:
#:     'Run rm *.pyc to clean'   ->  'Run rm .pyc to clean'
#:     'def f(*a, **k)'          ->  'def f(a, k)'
#:     'Use **kwargs and *args'  ->  'Use kwargs and args'
#:
#: Holding code spans out of the substitution fixed the backticked case and left
#: this one, which is the same failure: output that looks right and is wrong.
#: A lone asterisk is now left exactly as written — a stray marker on screen is
#: a cosmetic annoyance, and a silently altered command is not.
#:
#: Asterisks only. `some_function`, `__init__` and `MAX_TOKENS` are identifiers,
#: and underscore emphasis is rare enough in model output that eating them would
#: cost far more than it saves.
#: The character classes exclude ``*`` as well as word characters, so a marker
#: can never be half of a ``**`` that bold did not consume. Without that,
#: ``glob **/*.py`` paired the second asterisk of ``**`` with the one in ``/*``
#: and came out as ``glob */.py``.
_STRAY = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]{1,200}?)(?<!\s)\*(?![\w*])")


def bullet_char(stream=None) -> str:
    """'•' where the console can encode it, '-' where it cannot.

    Which consoles can is not obvious: cp1252 has a bullet at 0x95 and handles
    it fine, while the older OEM pages a Windows console can still land on —
    cp437, cp850 — and plain ascii do not. ``safe_print`` would degrade those to
    '?', and a list of question marks is worse than a list of hyphens, so ask
    the stream rather than guessing from the platform.
    """
    target = stream if stream is not None else sys.stdout
    encoding = getattr(target, "encoding", None) or "ascii"
    try:
        "•".encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return "-"
    return "•"


def supports_ansi(stream=None) -> bool:
    """Will escape codes render, or land as visible garbage?

    Windows 10+ can do ANSI but does not enable it for a console by default —
    the mode is set per handle, which is what the ctypes call below does. If
    that fails we answer no rather than guess, because unrendered escape codes
    are worse than the markdown they were meant to replace.
    """
    target = stream if stream is not None else sys.stdout
    if target is None:
        return False
    try:
        if not target.isatty():
            return False
    except (AttributeError, ValueError):
        return False

    # Honoured by convention across CLI tools; cheap to respect.
    if os.environ.get("NO_COLOR"):
        return False

    if sys.platform != "win32":
        return True

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # -11 = STD_OUTPUT_HANDLE. Mode 7 is the default 3 plus 0x0004,
        # ENABLE_VIRTUAL_TERMINAL_PROCESSING.
        return bool(kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7))
    except Exception:
        return False


def render(text: str, ansi: bool | None = None, bullet: str | None = None) -> str:
    """Markdown to terminal text. ``ansi=None`` decides from stdout."""
    if ansi is None:
        ansi = supports_ansi()
    if bullet is None:
        bullet = bullet_char()

    out: list[str] = []
    in_fence = False

    for line in text.splitlines():
        if _FENCE.match(line):
            # Drop the fence itself; what is inside is code and must survive
            # untouched — an asterisk in a shell glob is not emphasis.
            in_fence = not in_fence
            continue
        if in_fence:
            out.append(f"{DIM}{line}{RESET}" if ansi else line)
            continue

        heading = _HEADING.match(line)
        if heading:
            indent, body = heading.groups()
            body = _inline(body, ansi)
            out.append(f"{indent}{BOLD}{body}{RESET}" if ansi else f"{indent}{body}")
            continue

        listed = _BULLET.match(line)
        if listed:
            indent, body = listed.groups()
            out.append(f"{indent}{bullet} {_inline(body, ansi)}")
            continue

        out.append(_inline(line, ansi))

    return "\n".join(out)


def _inline(text: str, ansi: bool) -> str:
    """Bold, inline code and links, within a single line.

    Code spans are held out of every other substitution rather than unwrapped
    first. Unwrapping first left the markers indistinguishable: ``rm *.pyc``
    inside backticks became ``rm .pyc`` — a command that silently does
    something else, on the display path of every answer, with nothing to
    indicate it. Content inside backticks is quoted precisely because it is not
    markup, so the fix is to never let the markup pass see it.
    """
    pieces: list[str] = []
    last = 0
    for match in _CODE.finditer(text):
        pieces.append(_markup(text[last : match.start()], ansi))
        code = match.group(1)
        pieces.append(f"{CYAN}{code}{RESET}" if ansi else code)
        last = match.end()
    pieces.append(_markup(text[last:], ansi))
    return "".join(pieces)


def _markup(text: str, ansi: bool) -> str:
    """Emphasis and links, on a stretch known to be outside any code span."""
    text = _LINK.sub(r"\1 (\2)", text)
    text = _BOLD.sub(rf"{BOLD}\1{RESET}" if ansi else r"\1", text)
    # Italics are unwrapped rather than rendered: terminal support is patchy
    # enough that a stray marker is the likelier outcome than slanted text.
    # The backreference keeps the text and drops only a *matched pair*.
    return _STRAY.sub(r"\1", text)
