"""The GitHub worker.

Fetches the three things that can actually require a decision — PRs waiting on
your review, your own PRs that came back with changes requested or failing
checks, and unread participating notifications — then reasons over them.

Auth: ``gh auth token`` first, since `gh` is already logged in on a developer
machine and asking someone to mint a PAT for their own laptop is friction for
nothing. Falls back to the configured env var so this still works where `gh`
isn't installed. Requests go over httpx either way, so the network surface stays
one mockable thing.
"""
from __future__ import annotations

import json
import subprocess

import httpx

from majordomo.config import Config, GitHubConfig
from majordomo.models import NeedsYouItem, SourceReport

NAME = "github"

#: Cap what we pull. A pathological account shouldn't turn into an unbounded
#: fetch — and anything past this is what the router's size gate is for.
PER_PAGE = 50


class GitHubError(Exception):
    """A GitHub fetch failed. Caught at the worker boundary, never escapes."""


def token_from_gh() -> str | None:
    """Ask the `gh` CLI for its token. Returns None if gh is absent or logged out."""
    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=10,
            # Windows: gh may be a .cmd shim, so let the shell resolve it.
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    token = result.stdout.strip()
    return token or None


def resolve_token(cfg: GitHubConfig, env: dict[str, str] | None = None) -> str:
    """gh first, configured env var second."""
    import os

    token = token_from_gh()
    if token:
        return token

    source = env if env is not None else os.environ
    token = source.get(cfg.token_env, "").strip()
    if token:
        return token

    raise GitHubError(
        f"no GitHub token — run `gh auth login`, or set {cfg.token_env}"
    )


def _get(client: httpx.Client, url: str, params: dict | None = None) -> object:
    response = client.get(url, params=params)
    if not response.is_success:
        raise GitHubError(f"HTTP {response.status_code}: {response.text[:200]}")
    try:
        return response.json()
    except ValueError as exc:
        raise GitHubError(f"unparseable response from {url}") from exc


def fetch(cfg: GitHubConfig, token: str) -> dict[str, list]:
    """Pull the raw payload. Raises GitHubError; the caller converts to a stub."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    with httpx.Client(base_url=cfg.api_base, headers=headers, timeout=cfg.timeout) as client:
        review_requests = _get(
            client,
            "/search/issues",
            {"q": "is:open is:pr review-requested:@me archived:false", "per_page": PER_PAGE},
        )
        mine = _get(
            client,
            "/search/issues",
            {"q": "is:open is:pr author:@me archived:false", "per_page": PER_PAGE},
        )
        notifications = _get(
            client,
            "/notifications",
            {"participating": "true", "per_page": PER_PAGE},
        )

    def items(payload: object) -> list:
        if isinstance(payload, dict):
            found = payload.get("items")
            return found if isinstance(found, list) else []
        return payload if isinstance(payload, list) else []

    return {
        "review_requests": items(review_requests),
        "my_prs": items(mine),
        "notifications": items(notifications),
    }


def _pr_line(pr: dict) -> str:
    repo = (pr.get("repository_url") or "").rsplit("/", 1)[-1]
    number = pr.get("number", "?")
    title = pr.get("title", "(untitled)")
    return f"{repo}#{number}: {title}"


def to_text(raw: dict[str, list]) -> str:
    """Flatten the payload to the text the router sizes and the model reads."""
    lines: list[str] = []

    if raw["review_requests"]:
        lines.append("PRs awaiting your review:")
        lines += [f"  - {_pr_line(pr)}" for pr in raw["review_requests"]]
    if raw["my_prs"]:
        lines.append("Your open PRs:")
        lines += [f"  - {_pr_line(pr)}" for pr in raw["my_prs"]]
    if raw["notifications"]:
        lines.append("Unread notifications you're participating in:")
        for note in raw["notifications"]:
            subject = note.get("subject") or {}
            repo = (note.get("repository") or {}).get("full_name", "?")
            lines.append(f"  - [{repo}] {subject.get('type', '?')}: {subject.get('title', '')}")

    return "\n".join(lines)


def direct_items(raw: dict[str, list]) -> list[NeedsYouItem]:
    """The needs-you list, derived structurally rather than from model prose.

    A review request is unambiguously a thing that needs you — there is no
    judgement to make, so we don't ask the model to make one and risk it
    dropping an entry it decided was boring.
    """
    return [
        NeedsYouItem(
            kind="review_request",
            title=_pr_line(pr),
            detail="Your review is requested.",
            source=NAME,
            action=pr.get("html_url"),
        )
        for pr in raw["review_requests"]
    ]


def run(config: Config, token: str | None = None) -> SourceReport:
    """Fetch and report. Never raises — a failure becomes an error stub."""
    cfg = config.sources.github
    try:
        resolved = token or resolve_token(cfg)
        raw = fetch(cfg, resolved)
    except GitHubError as exc:
        return SourceReport.failed(NAME, str(exc))
    except Exception as exc:
        return SourceReport.failed(NAME, f"unexpected: {exc}")

    text = to_text(raw)
    if not text:
        return SourceReport(
            source=NAME,
            ok=True,
            summary="Nothing waiting on you in GitHub.",
            items=[],
            path="cheap",
            route_reason="empty payload",
        )

    # The report leaves here *unreasoned*: the router decides whether this goes
    # through a cheap summarize or an escalated reducer, and does that for every
    # source uniformly. Reasoning here too would duplicate that call.
    return SourceReport(
        source=NAME,
        ok=True,
        summary=text,
        items=direct_items(raw),
        path="cheap",
        route_reason="pending routing",
    )
