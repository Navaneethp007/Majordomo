"""Tests for the GitHub worker — HTTP fully mocked, no network, no real gh."""
from __future__ import annotations

import subprocess
from unittest import mock

import httpx
import pytest

from majordomo import config as config_module
from majordomo.workers import github
from majordomo.workers.github import GitHubError

CFG = config_module.build(config_module.DEFAULTS)

SEARCH_REVIEWS = {
    "items": [
        {
            "number": 12,
            "title": "Add the escalation gate",
            "repository_url": "https://api.github.com/repos/nav/majordomo",
            "html_url": "https://github.com/nav/majordomo/pull/12",
        }
    ]
}
SEARCH_MINE = {"items": [{"number": 9, "title": "Vendor the TTS adapter",
                          "repository_url": "https://api.github.com/repos/nav/voicelog",
                          "html_url": "https://github.com/nav/voicelog/pull/9"}]}
NOTIFICATIONS = [
    {"repository": {"full_name": "nav/familiar"},
     "subject": {"type": "PullRequest", "title": "Fix the statusline"}}
]


def fake_client(responses):
    """A stand-in httpx.Client whose GETs return canned payloads in order."""
    calls = []

    class FakeResponse:
        def __init__(self, payload, status=200):
            self._payload = payload
            self.status_code = status
            self.is_success = 200 <= status < 300
            self.text = "error body"

        def json(self):
            if isinstance(self._payload, Exception):
                raise self._payload
            return self._payload

    class FakeClient:
        def __init__(self, *a, **k):
            self.init_kwargs = k

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            calls.append((url, params))
            item = responses[len(calls) - 1]
            if isinstance(item, tuple):
                return FakeResponse(item[0], item[1])
            return FakeResponse(item)

    return FakeClient, calls


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------

def test_gh_token_is_preferred(monkeypatch):
    monkeypatch.setattr(github, "token_from_gh", lambda: "gho_fromgh")
    assert github.resolve_token(CFG.sources.github, env={"MAJORDOMO_GH_TOKEN": "env"}) == "gho_fromgh"


def test_env_var_is_the_fallback(monkeypatch):
    monkeypatch.setattr(github, "token_from_gh", lambda: None)
    assert github.resolve_token(CFG.sources.github, env={"MAJORDOMO_GH_TOKEN": "ghp_env"}) == "ghp_env"


def test_no_token_anywhere_raises_with_instructions(monkeypatch):
    monkeypatch.setattr(github, "token_from_gh", lambda: None)
    with pytest.raises(GitHubError) as exc:
        github.resolve_token(CFG.sources.github, env={})
    assert "gh auth login" in str(exc.value)
    assert "MAJORDOMO_GH_TOKEN" in str(exc.value)


def test_gh_absent_returns_none(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("gh not found")

    monkeypatch.setattr(subprocess, "run", boom)
    assert github.token_from_gh() is None


def test_gh_logged_out_returns_none(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="not logged in"),
    )
    assert github.token_from_gh() is None


def test_gh_timeout_returns_none(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("gh", 10)

    monkeypatch.setattr(subprocess, "run", boom)
    assert github.token_from_gh() is None


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def test_fetch_pulls_all_three_streams(monkeypatch):
    FakeClient, calls = fake_client([SEARCH_REVIEWS, SEARCH_MINE, NOTIFICATIONS])
    monkeypatch.setattr(httpx, "Client", FakeClient)

    raw = github.fetch(CFG.sources.github, "tok")

    assert len(calls) == 3
    assert raw["review_requests"][0]["number"] == 12
    assert raw["my_prs"][0]["number"] == 9
    assert raw["notifications"][0]["subject"]["title"] == "Fix the statusline"


def test_fetch_sends_bearer_token(monkeypatch):
    captured = {}

    class Recorder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, params=None):
            return type("R", (), {"is_success": True, "status_code": 200,
                                  "json": lambda self: {"items": []}, "text": ""})()

    monkeypatch.setattr(httpx, "Client", Recorder)
    github.fetch(CFG.sources.github, "tok-123")

    assert captured["headers"]["Authorization"] == "Bearer tok-123"


def test_fetch_http_error_raises_githuberror(monkeypatch):
    FakeClient, _ = fake_client([({}, 401)])
    monkeypatch.setattr(httpx, "Client", FakeClient)

    with pytest.raises(GitHubError) as exc:
        github.fetch(CFG.sources.github, "bad")
    assert "401" in str(exc.value)


def test_fetch_unparseable_body_raises_githuberror(monkeypatch):
    FakeClient, _ = fake_client([ValueError("not json")])
    monkeypatch.setattr(httpx, "Client", FakeClient)

    with pytest.raises(GitHubError):
        github.fetch(CFG.sources.github, "tok")


# ---------------------------------------------------------------------------
# Shaping
# ---------------------------------------------------------------------------

def test_to_text_names_repo_and_number():
    text = github.to_text(
        {"review_requests": SEARCH_REVIEWS["items"], "my_prs": [], "notifications": []}
    )
    assert "majordomo#12" in text
    assert "Add the escalation gate" in text


def test_to_text_empty_payload_is_empty_string():
    assert github.to_text({"review_requests": [], "my_prs": [], "notifications": []}) == ""


def test_review_requests_become_needs_you_items():
    items = github.direct_items(
        {"review_requests": SEARCH_REVIEWS["items"], "my_prs": [], "notifications": []}
    )
    assert len(items) == 1
    assert items[0].kind == "review_request"
    assert items[0].action == "https://github.com/nav/majordomo/pull/12"


def test_your_own_prs_are_not_needs_you():
    """Your own open PR isn't a decision waiting on you — a review request is."""
    items = github.direct_items(
        {"review_requests": [], "my_prs": SEARCH_MINE["items"], "notifications": []}
    )
    assert items == []


# ---------------------------------------------------------------------------
# run() — the boundary that must never raise
# ---------------------------------------------------------------------------

def test_run_happy_path(monkeypatch):
    FakeClient, _ = fake_client([SEARCH_REVIEWS, SEARCH_MINE, NOTIFICATIONS])
    monkeypatch.setattr(httpx, "Client", FakeClient)

    report = github.run(CFG, token="tok")

    assert report.ok
    assert "majordomo#12" in report.summary


def test_run_returns_error_stub_on_http_failure(monkeypatch):
    FakeClient, _ = fake_client([({}, 500)])
    monkeypatch.setattr(httpx, "Client", FakeClient)

    report = github.run(CFG, token="tok")

    assert report.ok is False
    assert report.path == "error"
    assert "500" in report.error


def test_run_returns_error_stub_on_missing_token(monkeypatch):
    monkeypatch.setattr(github, "token_from_gh", lambda: None)
    monkeypatch.delenv("MAJORDOMO_GH_TOKEN", raising=False)

    report = github.run(CFG)

    assert report.ok is False
    assert "gh auth login" in report.error


def test_run_returns_error_stub_on_network_failure(monkeypatch):
    def boom(*a, **k):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "Client", boom)
    report = github.run(CFG, token="tok")

    assert report.ok is False


def test_run_never_raises_even_on_a_surprise(monkeypatch):
    """The worker contract is absolute: one dead source must not cost the briefing."""
    monkeypatch.setattr(github, "fetch", mock.Mock(side_effect=RuntimeError("surprise")))

    report = github.run(CFG, token="tok")

    assert report.ok is False
    assert "unexpected" in report.error


def test_run_with_nothing_pending_says_so(monkeypatch):
    FakeClient, _ = fake_client([{"items": []}, {"items": []}, []])
    monkeypatch.setattr(httpx, "Client", FakeClient)

    report = github.run(CFG, token="tok")

    assert report.ok
    assert report.items == []
    assert "Nothing waiting" in report.summary


# ---------------------------------------------------------------------------
# gh resolution — a .cmd shim must not read as "not logged in"
# ---------------------------------------------------------------------------

def test_gh_is_resolved_through_which(monkeypatch):
    """A scoop/npm install of gh is a .cmd shim, which subprocess will not run
    without a shell. Passing a bare 'gh' raised OSError, got swallowed to None,
    and told a logged-in user to run `gh auth login`."""
    monkeypatch.setattr(github.shutil, "which", lambda name: r"C:\tools\gh.cmd")
    seen = {}

    def record(argv, **kw):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, 0, stdout="gho_tok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", record)

    assert github.token_from_gh() == "gho_tok"
    assert seen["argv"][0] == r"C:\tools\gh.cmd", "must invoke the resolved path"


def test_gh_not_on_path_returns_none_without_running_anything(monkeypatch):
    monkeypatch.setattr(github.shutil, "which", lambda name: None)

    def boom(*a, **k):
        raise AssertionError("must not spawn anything when gh isn't installed")

    monkeypatch.setattr(subprocess, "run", boom)
    assert github.token_from_gh() is None
