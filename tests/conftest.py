"""Shared test fixtures.

The one thing here is a guarantee: **no test may write to any directory the
developer actually uses.** That means ``~/.majordomo`` and ``~/.claude``.

This is not hypothetical. The hook swallows its failures into
``~/.majordomo/error.log``, and several tests deliberately provoke failures —
so before this fixture existed, running the suite quietly appended tracebacks
with pytest paths in them to the developer's own log, where they'd later look
like a real bug in a real session.

``~/.claude`` was the same hole, still open. ``install.install()`` defaults to
``paths.claude_settings_path()``, and the only thing keeping the suite off the
real file was that every case in ``test_install.py`` passes ``settings_path=``
by hand and nothing drives ``install-hooks`` through ``cli.main``. Both are
conventions, not guarantees — one test answering "yes" to a hooks prompt would
have rewritten the developer's live Claude Code configuration.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Point MAJORDOMO_HOME at a per-test directory, for every test."""
    home = tmp_path / "majordomo-home"
    home.mkdir()
    monkeypatch.setenv("MAJORDOMO_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def isolated_claude_home(tmp_path, monkeypatch):
    """Point CLAUDE_CONFIG_DIR at a per-test directory, for every test.

    The companion to ``isolated_home``, for the *other* directory this project
    writes to. ``paths.claude_settings_path()`` honours this variable, so the
    default target of ``install.install()`` becomes a temporary file and a test
    that installs hooks for real cannot touch the developer's own settings.
    """
    home = tmp_path / "claude-home"
    home.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home))
    return home


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch, request):
    """**No test may reach the network.** Anything that tries fails loudly.

    Also not hypothetical. A patch written as ``mock.patch.object(llm.httpx,
    "post", ...)`` stopped intercepting when the call moved behind a pooled
    client, and the test quietly started making real requests against the
    developer's own account — spending a daily free-tier allowance to assert
    something about backoff. The failure looked like a 401 from the code under
    test, which is the worst way to find out.

    A test that genuinely wants the network can ask for it with
    ``@pytest.mark.network``. Nothing does today.
    """
    if "network" in request.keywords:
        return

    import httpx

    def refuse(*_args, **_kwargs):
        raise RuntimeError(
            "this test tried to make a real HTTP request — its mock is not "
            "intercepting. Patch the seam the code actually calls."
        )

    for name in ("request", "send"):
        monkeypatch.setattr(httpx.Client, name, refuse, raising=False)
    for name in ("get", "post", "put", "delete", "patch", "request", "stream"):
        monkeypatch.setattr(httpx, name, refuse, raising=False)


@pytest.fixture(autouse=True)
def no_real_gh(monkeypatch, request):
    r"""**No test may run the real `gh`.** Anything that tries fails loudly.

    The sibling of ``no_real_network``, and the stakes are strictly higher. That
    fixture exists because a stale mock quietly spent the developer's own API
    allowance. A test that reaches real ``gh`` does not spend an allowance — it
    **writes to a third party under the developer's identity**: opens a real pull
    request, comments on a real issue, perhaps on somebody else's repository.
    There is no quota draining to notice, and the *success* case is the bad case.

    Deliberately not a blanket ``subprocess`` block. The git tool's tests run
    real git in a temporary repository, and ``run_command``'s have always shelled
    out — both are safe and both are better than mocking. Only ``gh`` is refused.

    The check is on the **resolved executable**, not on ``argv[0]``. ``_run``
    passes what ``shutil.which`` returned, so on this machine that is
    ``C:\Program Files\GitHub CLI\gh.exe`` and a test for ``argv[0] == "gh"``
    would miss every real call. Patch the seam the code actually calls.

    ``@pytest.mark.gh`` opts out, for symmetry with ``@pytest.mark.network``.
    Nothing uses it.
    """
    if "gh" in request.keywords:
        return

    import subprocess
    from pathlib import Path

    real_run, real_popen = subprocess.run, subprocess.Popen

    def names(command) -> list[str]:
        if isinstance(command, (str, bytes)):
            # shell=True: a whole command line. Checked loosely, because
            # `run_command` legitimately builds strings and a shell command
            # mentioning gh is rare enough that a false positive is cheap.
            text = command.decode() if isinstance(command, bytes) else command
            return [text.split()[0]] if text.split() else []
        return [str(part) for part in (command or [])][:1]

    def is_gh(command) -> bool:
        return any(Path(name).stem.lower() == "gh" for name in names(command))

    def guard(original):
        def checked(command, *args, **kwargs):
            if is_gh(command):
                raise RuntimeError(
                    "this test tried to run the real `gh` — it would act on "
                    "GitHub as you. Patch majordomo.tools._run instead."
                )
            return original(command, *args, **kwargs)

        return checked

    monkeypatch.setattr(subprocess, "run", guard(real_run))
    monkeypatch.setattr(subprocess, "Popen", guard(real_popen))
