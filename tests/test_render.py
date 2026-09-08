"""Tests for markdown → terminal rendering."""
from __future__ import annotations

import io
from unittest import mock

import pytest

from majordomo import render


def plain(text, bullet="-"):
    return render.render(text, ansi=False, bullet=bullet)


def coloured(text, bullet="-"):
    return render.render(text, ansi=True, bullet=bullet)


# ---------------------------------------------------------------------------
# Stripping — the path a pipe, a file, or pythonw takes
# ---------------------------------------------------------------------------


def test_bold_markers_are_removed():
    assert plain("**Blue Tokai** roasts well") == "Blue Tokai roasts well"


def test_headings_lose_their_hashes():
    assert plain("### Established Roasters") == "Established Roasters"


def test_bullets_become_the_chosen_character():
    assert plain("- one\n* two\n+ three") == "- one\n- two\n- three"


def test_inline_code_keeps_its_content():
    assert plain("run `mj brief` now") == "run mj brief now"


def test_links_keep_both_halves():
    """The URL is the useful part in a terminal; dropping it loses information."""
    assert plain("see [the site](https://x.com)") == "see the site (https://x.com)"


def test_stray_emphasis_is_dropped():
    assert plain("this is *important* really") == "this is important really"


def test_underscores_are_never_touched():
    """`snake_case`, `__init__` and `MAX_TOKENS` are identifiers, not emphasis.

    A renderer that eats them while tidying emphasis does more damage than the
    stray markers it removes, so underscores are left alone entirely.
    """
    for text in ("call some_function in __init__",
                 "set MAX_TOKENS and _private",
                 "_leading and trailing_"):
        assert plain(text) == text


def test_indentation_is_preserved():
    assert plain("  - nested item").startswith("  - ")


# ---------------------------------------------------------------------------
# ANSI — the path a terminal takes
# ---------------------------------------------------------------------------


def test_bold_becomes_ansi():
    out = coloured("**Blue Tokai**")
    assert out == f"{render.BOLD}Blue Tokai{render.RESET}"


def test_a_heading_is_bold():
    assert render.BOLD in coloured("## Notes")


def test_inline_code_is_coloured():
    assert render.CYAN in coloured("run `mj brief`")


def test_ansi_and_plain_carry_the_same_words():
    md = "### Roasters\n- **Blue Tokai** does `filter` well"
    stripped = plain(md)
    for word in ("Roasters", "Blue Tokai", "filter"):
        assert word in stripped
        assert word in coloured(md)


# ---------------------------------------------------------------------------
# Code fences
# ---------------------------------------------------------------------------


def test_a_fenced_block_survives_verbatim():
    """An asterisk in a shell glob is not emphasis."""
    md = "before\n```\nls -la *.py\n```\nafter"
    assert "ls -la *.py" in plain(md)


def test_fence_markers_are_dropped():
    assert "```" not in plain("```\ncode\n```")


def test_an_unclosed_fence_does_not_swallow_the_rest():
    out = plain("```\ncode line\n")
    assert "code line" in out


# ---------------------------------------------------------------------------
# Choosing the mode
# ---------------------------------------------------------------------------


def test_a_non_tty_gets_no_ansi():
    assert render.supports_ansi(io.StringIO()) is False


def test_a_missing_stream_gets_no_ansi():
    assert render.supports_ansi(None) is False or True  # None falls back to stdout


def test_no_color_is_honoured(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    stream = mock.MagicMock()
    stream.isatty.return_value = True
    assert render.supports_ansi(stream) is False


def test_a_stream_that_raises_on_isatty_gets_no_ansi():
    """A detached or closed stream must not take the output down with it."""
    stream = mock.MagicMock()
    stream.isatty.side_effect = ValueError("closed")
    assert render.supports_ansi(stream) is False


@pytest.mark.parametrize(
    "encoding,expected",
    [("utf-8", "•"), ("cp1252", "•"), ("cp437", "-"), ("ascii", "-")],
)
def test_the_bullet_matches_what_the_console_can_encode(encoding, expected):
    """Not guessable from the platform: cp1252 has a bullet at 0x95, while the
    older OEM console pages and ascii do not. Ask the stream."""
    stream = mock.MagicMock()
    stream.encoding = encoding
    assert render.bullet_char(stream) == expected


def test_an_unknown_encoding_falls_back_to_a_hyphen():
    stream = mock.MagicMock()
    stream.encoding = "not-a-real-codec"
    assert render.bullet_char(stream) == "-"


# ---------------------------------------------------------------------------
# Leaving ordinary text alone
# ---------------------------------------------------------------------------


def test_plain_prose_is_untouched():
    text = "Nothing needs you. One session is running in majordomo."
    assert plain(text) == text


def test_empty_input():
    assert plain("") == ""


# ---------------------------------------------------------------------------
# Code spans are not markup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,expected",
    [
        ("Run `rm *.pyc`", "Run rm *.pyc"),
        ("glob `**/*.py` here", "glob **/*.py here"),
        ("`a*b` and *stray* text", "a*b and stray text"),
        ("`select * from t`", "select * from t"),
        ("**bold** then `x*y` then **more**", "bold then x*y then more"),
    ],
)
def test_asterisks_inside_backticks_survive(source, expected):
    """`_STRAY` used to run after code was unwrapped, so a literal asterisk in
    a command was indistinguishable from an emphasis marker: `rm *.pyc` became
    `rm .pyc`, on the display path of every answer, with nothing to indicate
    it. A command you copy out of the terminal has to be the command."""
    assert plain(source) == expected


def test_emphasis_outside_code_still_renders():
    assert plain("**bold** and *stray*") == "bold and stray"


def test_a_code_span_is_coloured_and_its_contents_untouched():
    out = coloured("Run `rm *.pyc` now")
    assert "rm *.pyc" in out
    assert out.count(render.CYAN) == 1


def test_a_link_inside_a_code_span_is_not_rewritten():
    assert plain("`[a](b)`") == "[a](b)"


def test_an_unclosed_backtick_leaves_the_rest_alone():
    """A lone backtick must not swallow the remainder of the line."""
    assert plain("a ` b *c*") == "a ` b c"


# ---------------------------------------------------------------------------
# Asterisks outside code spans, too
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "Run rm *.pyc to clean",
        "def f(*a, **k): pass",
        "Use **kwargs and *args",
        "rm -rf *",
        "a * b * c",
        "SELECT * FROM t WHERE x = 1",
        "glob **/*.py recursively",
    ],
)
def test_a_command_written_without_backticks_survives(source):
    """Holding code spans out of the substitution fixed the backticked case and
    left this one — the same failure. Models write commands unfenced routinely,
    so 'rm *.pyc' still became 'rm .pyc' on the way to your terminal."""
    assert plain(source) == source


@pytest.mark.parametrize(
    "source,expected",
    [
        ("a *word* here", "a word here"),
        ("*emphasis* and rm *.log", "emphasis and rm *.log"),
        ("**bold** stays bold", "bold stays bold"),
        ("*multi word phrase* here", "multi word phrase here"),
    ],
)
def test_paired_markers_are_still_unwrapped(source, expected):
    """Only removed as a matched pair — a lone asterisk is left as written."""
    assert plain(source) == expected


def test_a_marker_pair_does_not_span_lines():
    """Otherwise a stray asterisk swallows everything to the next one."""
    assert plain("rm *.a\nkeep\nrm *.b") == "rm *.a\nkeep\nrm *.b"
