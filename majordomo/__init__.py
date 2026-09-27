"""Majordomo — the head of household staff who runs the workers and briefs you."""

#: Single source of truth for the version. ``pyproject.toml`` reads this
#: attribute rather than carrying its own literal, so the two cannot disagree.
__version__ = "0.1.0"

#: The distribution name, which is **not** the name of this package. `majordomo`
#: was taken on PyPI, and `mj` turned out better anyway: the word you install and
#: the word you type are the same one.
DISTRIBUTION = "mj"


def install_hint(extra: str) -> str:
    """How to add an optional extra, for whichever way this was installed.

    Lives here, once, because four modules need to say it — ``asr``,
    ``tts``, ``documents``, ``tray`` — and each had its own copy naming
    ``pip install majordomo[...]``. Both halves of that are now wrong: the
    distribution is ``mj``, and the recommended install is ``uv tool``, which
    puts the package in an isolated environment where a plain ``pip install``
    lands somewhere else entirely and appears to do nothing.

    Both forms are named because we cannot tell from in here which one applies,
    and guessing wrong wastes the reader's time on the one line that was supposed
    to save it.
    """
    spec = f'"{DISTRIBUTION}[{extra}]"'
    return f"uv tool install --force {spec}   (or, in a venv: pip install {spec})"
