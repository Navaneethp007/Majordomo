"""Shared test fixtures.

The one thing here is a guarantee: **no test may write to the real
``~/.majordomo``.**

This is not hypothetical. The hook swallows its failures into
``~/.majordomo/error.log``, and several tests deliberately provoke failures —
so before this fixture existed, running the suite quietly appended tracebacks
with pytest paths in them to the developer's own log, where they'd later look
like a real bug in a real session.
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
