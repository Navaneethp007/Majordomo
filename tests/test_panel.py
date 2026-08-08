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


# ---------------------------------------------------------------------------
# Concurrency and error surfacing
# ---------------------------------------------------------------------------

def test_server_is_threaded(server):
    """/api/brief runs the whole fetch->route->fuse with a 120s LLM timeout. On
    a single-threaded server that held the only request slot, so any second
    request hung the tab."""
    from http.server import ThreadingHTTPServer

    assert isinstance(server.server, ThreadingHTTPServer)


def test_a_slow_brief_does_not_block_other_requests(monkeypatch):
    """The concrete symptom: click Re-brief, then the page can't load anything."""
    import threading
    import time

    release = threading.Event()

    def slow(config, state_file=None):
        release.wait(timeout=10)
        return brief.BriefResult(briefing=Briefing("done"))

    monkeypatch.setattr(brief, "run", slow)
    handle = panel.serve(CFG)
    try:
        port = handle.server.server_address[1]
        slow_call = threading.Thread(
            target=lambda: get(f"http://127.0.0.1:{port}/api/brief?t={token_of(handle)}")
        )
        slow_call.start()
        time.sleep(0.3)  # let the slow request take a slot

        status, _ = get(handle.url)  # must still be served
        assert status == 200
    finally:
        release.set()
        slow_call.join(timeout=10)
        handle.stop()


def test_missing_api_key_returns_an_error_not_a_dead_handler(monkeypatch):
    """brief.run raises MissingApiKey by contract. Uncaught, the handler thread
    died, no response arrived, and the page sat on 'Loading…' with no reason."""
    from majordomo.llm import MissingApiKey

    def boom(config, state_file=None):
        raise MissingApiKey("Set the OPENROUTER_API_KEY environment variable")

    monkeypatch.setattr(brief, "run", boom)
    handle = panel.serve(CFG)
    try:
        port = handle.server.server_address[1]
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(f"http://127.0.0.1:{port}/api/brief?t={token_of(handle)}")
        assert exc.value.code == 500
        assert "OPENROUTER_API_KEY" in exc.value.read().decode()
    finally:
        handle.stop()


def test_an_unexpected_error_also_returns_json(monkeypatch):
    def boom(config, state_file=None):
        raise RuntimeError("something odd")

    monkeypatch.setattr(brief, "run", boom)
    handle = panel.serve(CFG)
    try:
        port = handle.server.server_address[1]
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(f"http://127.0.0.1:{port}/api/brief?t={token_of(handle)}")
        assert exc.value.code == 500
        assert "something odd" in exc.value.read().decode()
    finally:
        handle.stop()


def test_the_page_renders_errors_rather_than_hanging(server):
    """The client half: without this the page shows 'Loading…' forever."""
    _, body = get(server.url)
    assert "res.ok" in body and "Briefing failed" in body
