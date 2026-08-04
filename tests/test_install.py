"""Tests for majordomo.install.

`~/.claude/settings.json` is the user's live configuration and is already
occupied — on this machine the Familiar project owns hooks on SessionStart,
PostToolUse, Stop and SessionEnd, plus the statusLine. So the bar here is not
"does it install", it is **"does it leave everything it does not own exactly as
it found it"**.
"""
from __future__ import annotations

import copy
import json

import pytest

from majordomo import install

CLI = "c:/Users/nvps7/majordomo"

# A faithful reduction of the real settings.json on this machine: another tool
# already holds hooks on two of the events we also want, plus a statusLine.
FAMILIAR_HANDLER = {
    "type": "command",
    "command": "node",
    "args": ["C:/Users/nvps7/Familiar/dist/cli.js", "hook", "--event=SessionStart"],
    "timeout": 10,
}

EXISTING = {
    "env": {"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"},
    "hooks": {
        "SessionStart": [{"matcher": "startup|resume", "hooks": [FAMILIAR_HANDLER]}],
        "PostToolUse": [
            {
                "matcher": "Bash",
                "hooks": [
                    {
                        "type": "command",
                        "command": "node",
                        "args": ["C:/Users/nvps7/Familiar/dist/cli.js", "hook", "--event=PostToolUse"],
                        "timeout": 10,
                    }
                ],
            }
        ],
        "SessionEnd": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": "node",
                        "args": ["C:/Users/nvps7/Familiar/dist/cli.js", "hook", "--event=SessionEnd"],
                        "timeout": 10,
                    }
                ]
            }
        ],
    },
    "statusLine": {"type": "command", "command": 'node "C:/Users/nvps7/Familiar/dist/cli.js" statusline'},
    "effortLevel": "medium",
}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("MAJORDOMO_HOME", str(tmp_path / "home"))
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(EXISTING, indent=2), encoding="utf-8")
    return path


def read(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The property that matters most: do no harm
# ---------------------------------------------------------------------------

def test_existing_handlers_survive_install(settings):
    install.install(settings_path=settings, cli_path=CLI)

    after = read(settings)
    for event in ("SessionStart", "PostToolUse", "SessionEnd"):
        surviving = [
            h
            for group in after["hooks"][event]
            for h in group["hooks"]
            if "Familiar" in " ".join(h.get("args", []))
        ]
        assert surviving, f"Familiar's {event} handler was destroyed"


def test_unrelated_settings_are_untouched(settings):
    install.install(settings_path=settings, cli_path=CLI)

    after = read(settings)
    assert after["statusLine"] == EXISTING["statusLine"]
    assert after["env"] == EXISTING["env"]
    assert after["effortLevel"] == "medium"


def test_uninstall_restores_original_shape(settings):
    before = read(settings)

    install.install(settings_path=settings, cli_path=CLI)
    install.uninstall(settings_path=settings, cli_path=CLI)

    assert read(settings) == before


def test_install_backs_up_first(settings):
    result = install.install(settings_path=settings, cli_path=CLI)

    assert result.backup is not None
    assert json.loads(result.backup.read_text(encoding="utf-8")) == EXISTING


# ---------------------------------------------------------------------------
# Does it actually install
# ---------------------------------------------------------------------------

def test_all_five_handlers_are_added(settings):
    install.install(settings_path=settings, cli_path=CLI)
    after = read(settings)

    installed = {
        (event, group.get("matcher"))
        for event, groups in after["hooks"].items()
        for group in groups
        for h in group["hooks"]
        if install._is_ours(h)
    }

    assert installed == {
        ("SessionStart", "startup|resume"),
        ("UserPromptSubmit", None),
        ("Notification", "idle_prompt"),
        ("Notification", "permission_prompt"),
        ("SessionEnd", None),
    }


def test_handlers_carry_event_and_matcher_in_argv(settings):
    """The Notification payload doesn't say which matcher fired, so argv must."""
    install.install(settings_path=settings, cli_path=CLI)
    after = read(settings)

    notif = {
        group["matcher"]: [h for h in group["hooks"] if install._is_ours(h)][0]
        for group in after["hooks"]["Notification"]
    }

    assert "--matcher=idle_prompt" in notif["idle_prompt"]["args"]
    assert "--matcher=permission_prompt" in notif["permission_prompt"]["args"]
    assert "--event=Notification" in notif["idle_prompt"]["args"]


def test_joins_existing_group_with_same_matcher(settings):
    """Our SessionStart matcher is identical to Familiar's, so we must land in
    that same group rather than creating a duplicate group beside it."""
    install.install(settings_path=settings, cli_path=CLI)
    after = read(settings)

    groups = [g for g in after["hooks"]["SessionStart"] if g.get("matcher") == "startup|resume"]
    assert len(groups) == 1
    assert len(groups[0]["hooks"]) == 2


# ---------------------------------------------------------------------------
# Re-running, and edge cases
# ---------------------------------------------------------------------------

def test_reinstall_is_idempotent(settings):
    install.install(settings_path=settings, cli_path=CLI)
    once = read(settings)
    install.install(settings_path=settings, cli_path=CLI)

    assert read(settings) == once


def test_reinstall_upgrades_a_moved_path(settings):
    install.install(settings_path=settings, cli_path="c:/old/location")
    install.install(settings_path=settings, cli_path="c:/new/location")

    after = read(settings)
    ours = [
        h
        for groups in after["hooks"].values()
        for group in groups
        for h in group["hooks"]
        if install._is_ours(h)
    ]
    assert len(ours) == 5, "the stale handlers were stacked instead of replaced"
    assert all("c:/old/location" not in " ".join(h["args"]) for h in ours)


def test_install_into_absent_settings_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MAJORDOMO_HOME", str(tmp_path / "home"))
    path = tmp_path / "settings.json"

    result = install.install(settings_path=path, cli_path=CLI)

    assert result.backup is None
    assert len(read(path)["hooks"]) == 4  # SessionStart, UserPromptSubmit, Notification, SessionEnd


def test_malformed_settings_refuses_rather_than_overwrites(settings):
    """Guessing at a broken settings file would be far worse than declining."""
    settings.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(install.SettingsUnreadable):
        install.install(settings_path=settings, cli_path=CLI)

    assert settings.read_text(encoding="utf-8") == "{ this is not json"


def test_bom_prefixed_settings_still_parses(settings):
    """PowerShell's Out-File writes a BOM; refusing to install over a perfectly
    valid file because of an invisible byte would be baffling."""
    raw = settings.read_text(encoding="utf-8")
    settings.write_text("\ufeff" + raw, encoding="utf-8")

    install.install(settings_path=settings, cli_path=CLI)

    assert "Notification" in read(settings)["hooks"]


def test_uninstall_drops_event_key_it_alone_occupied(settings):
    install.install(settings_path=settings, cli_path=CLI)
    install.uninstall(settings_path=settings, cli_path=CLI)

    hooks = read(settings)["hooks"]
    assert "Notification" not in hooks
    assert "UserPromptSubmit" not in hooks


def test_uninstall_on_absent_file_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("MAJORDOMO_HOME", str(tmp_path / "home"))
    result = install.uninstall(settings_path=tmp_path / "nothing.json", cli_path=CLI)
    assert result.removed == 0


def test_original_dict_is_not_mutated(settings):
    """Sanity check on the merge: it must not alias the parsed structure."""
    snapshot = copy.deepcopy(EXISTING)
    install.install(settings_path=settings, cli_path=CLI)
    assert EXISTING == snapshot
