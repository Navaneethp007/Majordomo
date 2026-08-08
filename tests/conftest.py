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
