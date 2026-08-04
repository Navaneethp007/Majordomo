"""Tests for the local panel — real server on a real loopback port, mocked pipeline.

localhost is not a trust boundary on a shared machine: any page you happen to
have open could POST to /api/resume and start launching editors. So the token
checks below are the point of this module.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from majordomo import brief, panel
from majordomo import config as config_module
from majordomo.models import Briefing, NeedsYouItem, Session

CFG = config_module.build(config_module.DEFAULTS)

ITEM = NeedsYouItem(
    kind="session_blocked",
    title="majordomo (vscode)",
    detail="Waiting for your approval.",
    source="sessions",
    action="abc-123",
)


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(
        brief, "run",
        lambda c, s=None: brief.BriefResult(briefing=Briefing("All quiet.", [ITEM])),
    )
    handle = panel.serve(CFG)
    yield handle
    handle.stop()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.read().decode("utf-8")


def post(url):
    request = urllib.request.Request(url, method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, response.read().decode("utf-8")


def token_of(handle) -> str:
    return handle.url.split("t=")[1]


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

def test_binds_to_loopback_only(server):
    assert server.server.server_address[0] == "127.0.0.1"


def test_root_serves_the_page(server):
    status, body = get(server.url)
    assert status == 200
    assert "<title>Majordomo</title>" in body


def test_brief_endpoint_returns_the_briefing(server):
    status, body = get(f"http://127.0.0.1:{server.server.server_address[1]}/api/brief?t={token_of(server)}")
    payload = json.loads(body)

    assert status == 200
    assert payload["briefing_text"] == "All quiet."
    assert payload["needs_you"][0]["title"] == "majordomo (vscode)"


# ---------------------------------------------------------------------------
# The token — the only thing between a stray tab and launching processes
# ---------------------------------------------------------------------------

def test_brief_without_a_token_is_forbidden(server):
    port = server.server.server_address[1]
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(f"http://127.0.0.1:{port}/api/brief")
    assert exc.value.code == 403


def test_brief_with_a_wrong_token_is_forbidden(server):
    port = server.server.server_address[1]
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(f"http://127.0.0.1:{port}/api/brief?t=guessed")
    assert exc.value.code == 403


def test_resume_without_a_token_launches_nothing(server, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("an unauthenticated request must never launch anything")

    monkeypatch.setattr(panel.resume_mod, "launch", boom)
    port = server.server.server_address[1]

    with pytest.raises(urllib.error.HTTPError) as exc:
        post(f"http://127.0.0.1:{port}/api/resume?id=abc-123")
    assert exc.value.code == 403


def test_each_run_gets_a_fresh_token(monkeypatch):
    monkeypatch.setattr(brief, "run", lambda c, s=None: brief.BriefResult(briefing=Briefing("x")))
    a, b = panel.serve(CFG), panel.serve(CFG)
    try:
        assert token_of(a) != token_of(b)
    finally:
        a.stop()
        b.stop()


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def test_resume_launches_the_matching_session(server, monkeypatch):
    launched = []
    monkeypatch.setattr(panel.resume_mod, "launch", lambda cmd: launched.append(cmd))
    monkeypatch.setattr(
        panel, "fold",
        lambda events, stale_after_hours=72: [
            Session("abc-123", "vscode", "c:/x", "blocked", "2026-08-04T09:00:00+00:00")
        ],
    )
    port = server.server.server_address[1]

    status, body = post(f"http://127.0.0.1:{port}/api/resume?t={token_of(server)}&id=abc-123")

    assert status == 200
    assert json.loads(body)["ok"] is True
    assert launched[0].uri.endswith("session=abc-123")


def test_resume_unknown_session_is_404(server, monkeypatch):
    monkeypatch.setattr(panel, "fold", lambda events, stale_after_hours=72: [])
    port = server.server.server_address[1]

    with pytest.raises(urllib.error.HTTPError) as exc:
        post(f"http://127.0.0.1:{port}/api/resume?t={token_of(server)}&id=nope")
    assert exc.value.code == 404


def test_unknown_path_is_404(server):
    port = server.server.server_address[1]
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(f"http://127.0.0.1:{port}/api/nonsense?t={token_of(server)}")
    assert exc.value.code == 404
