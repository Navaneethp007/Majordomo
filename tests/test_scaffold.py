"""Tests for what `mj start` puts on disk.

The brief is assembled from your memories and your recent work across every
repository you touch, and it lands in a directory that is `git init`-ed a
moment later. So these are as much about what does *not* leave the machine as
about the scaffold being correct.
"""
from __future__ import annotations

import subprocess
from unittest import mock

from majordomo import config as config_module
from majordomo import scaffold

# ---------------------------------------------------------------------------
# What lands in the new repository
# ---------------------------------------------------------------------------


def test_the_brief_is_not_tracked_by_git(tmp_path):
    """create() runs `git init` immediately after writing BRIEF.md, and the
    brief is context about the *user* — recent work across every repo they
    touch. Untracked, the first `git add .` cannot publish it."""
    target = scaffold.Plan(name="thing", path=tmp_path / "thing", brief="# thing\nsecrets")
    path, _ = scaffold.create(target)

    ignored = (path / ".gitignore").read_text(encoding="utf-8")
    assert "BRIEF.md" in ignored

    result = subprocess.run(
        ["git", "check-ignore", "BRIEF.md"], cwd=path, capture_output=True, text=True
    )
    assert result.returncode == 0, "git does not actually ignore it"


def test_the_gitignore_also_covers_env_and_caches(tmp_path):
    path, _ = scaffold.create(
        scaffold.Plan(name="thing", path=tmp_path / "thing", brief="x")
    )
    ignored = (path / ".gitignore").read_text(encoding="utf-8")

    for pattern in (".env", "__pycache__/", "node_modules/"):
        assert pattern in ignored


def test_the_dry_run_names_every_file_it_would_write(tmp_path):
    """A dry run that omits a file is worse than no dry run."""
    described = scaffold.Plan(
        name="thing", path=tmp_path / "thing", brief="x"
    ).describe()

    assert "README.md" in described
    assert "BRIEF.md" in described
    assert ".gitignore" in described


def test_a_brief_carries_less_activity_than_a_briefing(monkeypatch, tmp_path):
    """Sixty commits about an unrelated project crowd out the idea, and they
    describe work that is nobody's business but the user's."""
    seen = {}

    def fake_build(config, query=None, activity_limit=60, **kwargs):
        seen["limit"] = activity_limit
        return mock.Mock(render=lambda: "ctx")

    monkeypatch.setattr("majordomo.context.build", fake_build)
    config = config_module.build(
        {**config_module.DEFAULTS, "scaffold": {"root": str(tmp_path)}}
    )

    scaffold.start("an idea", config, dry_run=True, write=lambda *_: None)

    assert seen["limit"] == scaffold.BRIEF_ACTIVITY_LIMIT
    assert scaffold.BRIEF_ACTIVITY_LIMIT < 60
