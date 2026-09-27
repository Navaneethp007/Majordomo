"""Introducing ourselves, once, on the first bare ``mj``.

The front door absorbs setup. Otherwise "install to talking" is four commands
you have to already know about — ``config --init``, edit ``.env``,
``install-hooks``, ``install-trigger`` — which is the same menu problem moved
into the README.

── WHAT DECIDES THAT THIS IS A FIRST RUN ─────────────────────────────────────
An explicit marker file, not "does ``~/.majordomo`` exist". That directory
springs into existence from any first write: ``hook.log_error`` creates it to
record a failure, the hook appends ``state.jsonl`` on every Claude Code prompt,
``config --init`` makes it to hold the config. So someone whose first act is
``mj install-hooks``, followed by a single Claude Code prompt, would have a home
directory before ever running ``mj`` — and would never be introduced at all.

The marker is written even when every question is declined, because the question
it answers is "have we met", not "is anything configured". Declining everything
is a complete answer and must not be asked twice.

── WHY THERE IS NO ``input()`` HERE ──────────────────────────────────────────
``write`` and ``ask`` come in as callables, the same shape ``asr.measure`` and
``scaffold.start`` already use, which keeps every read of stdin inside ``cli``
and makes the whole flow unit-testable with a list-popping fake.

Deliberately *not* ``chat.Terminal``: importing it drags ``session`` → ``context``
→ ``llm`` in at import time, on the one path that now runs for bare ``mj``, and
``Terminal.confirm`` is the tool-approval gate — the wrong shape and the wrong
meaning for a yes-or-no question.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from majordomo import paths

#: Written at the end of setup, whatever was decided. Its presence is the whole
#: of "we have met".
MARKER_NAME = "setup.done"

#: Things that only exist in a home somebody has actually used. Any of them
#: means this is an upgrade, not a first run — see ``needed``.
#:
#: ``.env`` is here because the README has always told people to put their key
#: there, so doing exactly what the documentation says was the one way to get
#: greeted with "first run — three questions" *after* configuring the thing. The
#: outcome was harmless — the key reads as set and the question is skipped — but a
#: false greeting is precisely what this list exists to prevent.
_LIVED_IN = ("config.yml", ".env", "state.jsonl", "memory", "chats")

#: Where a local Ollama listens, and how long we are willing to wait to find
#: out. Short: this runs while someone watches a blank terminal.
OLLAMA_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 0.3


@dataclass
class Report:
    """What setup did. Returned rather than printed, so ``cli`` phrases it.

    Tests assert on this instead of on captured text, which is the same split
    ``trigger.install`` and ``activity.fetch`` already make.
    """

    did: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def marker_path() -> Path:
    return paths.majordomo_home() / MARKER_NAME


def needed() -> bool:
    """Should setup run? Writes the marker and says no for an existing install.

    The escape hatch matters as much as the check. Everyone already using this
    has a home directory and no marker, and greeting them with setup would be a
    worse first impression than the menu was.
    """
    if marker_path().exists():
        return False

    home = paths.majordomo_home()
    if any((home / name).exists() for name in _LIVED_IN):
        # Lived in. Record that we consider ourselves introduced, so this check
        # costs one stat call from now on rather than four.
        _write_marker()
        return False

    return True


def _write_marker() -> None:
    """Record that we have met. Never raises — this must not fail a session."""
    try:
        paths.ensure_home()
        marker_path().write_text(
            "Written by `mj` the first time it ran. Delete this to be asked "
            "again, or run `mj setup --force`.\n",
            encoding="utf-8",
        )
    except OSError:
        # A read-only or full home is a real problem, but not one to surface by
        # refusing to start a conversation. The cost is being asked again.
        pass


def _is_windows() -> bool:
    """Named, rather than an inline ``os.name`` check, so a test can replace it.

    Patching ``os.name`` itself is not an option: it is the same module object
    everywhere, and ``pathlib`` reads it to decide which ``Path`` class to build.
    A test that set it to ``"posix"`` took pytest down mid-report.
    """
    return os.name == "nt"


def _yes(answer: str) -> bool:
    """y/N, with blank meaning no.

    Parsed here rather than in ``cli`` so the default lives in one place: every
    question setup asks changes something outside this project, and a stray
    Enter must never be the thing that authorises it.
    """
    return answer.strip().lower() in ("y", "yes")


def probe_ollama(client=None) -> str | None:
    """A model name from a local Ollama, or ``None``.

    Only reached after someone declines to paste a key, never on the way in. A
    probe on the hot path would cost every user without Ollama a
    connection-refused before their first prompt; after a decline the
    alternative is a dead end, so it earns the round trip.

    Catches bare ``Exception`` on purpose. The test suite's network guard raises
    ``RuntimeError``, which an ``httpx``-specific clause would let escape — and
    every distinguishable failure here has the same answer anyway. The ``client``
    seam exists so that behaviour can be tested rather than assumed.
    """
    try:
        if client is None:
            import httpx

            client = httpx
        response = client.get(f"{OLLAMA_URL}/api/tags", timeout=OLLAMA_TIMEOUT)
        models = response.json().get("models") or []
        for entry in models:
            name = (entry or {}).get("name")
            if name:
                return str(name)
        return None
    except Exception:
        return None


def _write_env(key: str, value: str) -> None:
    """Append ``KEY=value`` to ``~/.majordomo/.env``, leaving the rest alone.

    Never rewrites an existing line for the same key: a setup run that silently
    replaced a working key would be much worse than one that says it is already
    set and moves on, which is what the caller does.
    """
    from majordomo import dotenv

    paths.ensure_home()
    path = dotenv.env_path()
    existing = ""
    if path.is_file():
        existing = path.read_text(encoding="utf-8", errors="replace")
        if existing and not existing.endswith("\n"):
            existing += "\n"
    path.write_text(f"{existing}{key}={value}\n", encoding="utf-8")


def _configure_ollama(model: str, path: Path | None = None) -> Path | None:
    """Point the brain at a local, keyless Ollama. Returns a backup path, if any.

    ── WHY THIS MERGES ──────────────────────────────────────────────────────
    It used to write the file wholesale, which meant ``mj setup --force`` on an
    existing config silently deleted every other section — ``voice``,
    ``sources``, the lot. That included ``voice.asr_function_id``, which the
    example config goes out of its way to say is tedious to find.

    Worth naming the mistake precisely, because it was not an oversight about
    what the code does: ``_write_env`` in this same file already refuses to
    overwrite an existing key and explains why. The rule was understood, written
    down, and then not applied to its sibling twenty lines away. Applying a
    principle in one function and not the next one over is its own failure mode.

    A merge cannot preserve comments — ``yaml`` round-trips them away — so the
    previous file is backed up first and the caller says where it went.
    """
    import yaml

    paths.ensure_home()
    target = path or paths.default_config_path()

    brain = {
        "provider": "ollama",
        "base_url": f"{OLLAMA_URL}/v1",
        # Empty means "no key needed". A *named* variable that is unset is still
        # an error, which is why this cannot be spelled any other way.
        "api_key_env": "",
        "worker_model": model,
        "fuser_model": model,
        "reducer_model": model,
        "chat_model": model,
        "agent_model": model,
        "fallback_model": "",
    }

    existing: dict = {}
    raw = ""
    if target.is_file():
        raw = target.read_text(encoding="utf-8")
        loaded = yaml.safe_load(raw)
        existing = loaded if isinstance(loaded, dict) else {}

    # Only `brain` is replaced, and only the keys above. Every other section is
    # carried through untouched.
    #
    # ``or {}`` rather than ``.get("brain", {})`` because a bare ``brain:`` with
    # nothing under it parses to **None**, not to a missing key — and ``{**None}``
    # raises. ``config.load`` accepts that file happily, so it is a working
    # config, and it is exactly what you get by commenting the block out, which
    # the annotated example invites you to do. The line above guards this same
    # shape at the top level and the check simply was not repeated one level down.
    merged = dict(existing)
    merged["brain"] = {**(existing.get("brain") or {}), **brain}

    body = (
        "# `brain` written by `mj setup`. `mj config --init` writes the annotated\n"
        "# version of this file, which explains every role.\n"
        + yaml.safe_dump(merged, sort_keys=False)
    )

    # Backup *after* the merge succeeds, not before. Written first, any failure
    # in between left an orphan `.bak` beside an unchanged config — a file that
    # says something was replaced when nothing was. Nothing is created unless
    # something is about to change.
    backup: Path | None = None
    if raw:
        # Beside the file it copies, named after it. `paths.backup_path` looked
        # like the seam to reuse, but it hardcodes `settings.<stamp>.json` for
        # Claude Code's settings — so a YAML config landed under a `.json` name,
        # in a shared namespace, where nobody would look for it.
        backup = target.with_name(f"{target.name}.{_stamp()}.bak")
        backup.write_text(raw, encoding="utf-8")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return backup


def _stamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run(*, write, ask, config, config_path=None, settings_path=None, probe=None) -> Report:
    """Introduce ourselves and wire up whatever is agreed to.

    Args:
        write: Where to print. ``cli.safe_print``.
        ask: How to ask. ``cli.ask``. Must return "" when nobody answers.
        config: The already-loaded configuration. Passed in rather than loaded
            here for two reasons. It used to call ``config.load(None)``, which
            raises on a malformed file — breaking the "never raises" promise
            below — and it ignored ``--config`` entirely, so ``mj --config
            other.yml`` on a fresh machine read and wrote the *default* path and
            then ran the session on ``other.yml``. One flag, two meanings, in a
            single invocation.
        config_path: Where to write configuration, matching ``--config``.
            ``None`` means the default location.
        settings_path: Passed through to ``install.install`` so a test can aim it
            somewhere harmless. Production passes nothing.
        probe: Replaces ``probe_ollama``, for tests.

    Returns:
        A ``Report``. Never raises except ``KeyboardInterrupt``, which the caller
        turns into "run `mj setup` to finish it" — an interrupted setup is
        abandoned, not half-applied silently.
    """
    report = Report()
    probe = probe or probe_ollama

    write("Majordomo — first run. Three questions, all skippable.")
    write("")

    key_env = config.brain.api_key_env

    # --- 1. a model to talk to ---------------------------------------------
    if not key_env:
        report.skipped.append("model (config already names a keyless provider)")
    elif os.environ.get(key_env):
        write(f"{key_env} is already set — using it.")
        report.skipped.append(f"key ({key_env} already set)")
    else:
        write(f"An API key for {config.brain.provider}. Paste it, or press Enter")
        write("to skip and pick a model later.")
        typed = ask("  key: ").strip()
        if typed:
            try:
                _write_env(key_env, typed)
            except OSError as exc:
                write(f"  could not write it: {exc}")
                report.failed.append("key")
            else:
                # Written to disk *and* set here. `main` loads the .env before
                # dispatch, so this process never re-reads it — without this the
                # key you just typed would be absent from the environment and
                # the very first turn would fail as though the key were wrong.
                os.environ[key_env] = typed
                write(f"  saved to {_env_display()} (plain text, on this machine only)")
                # Stated, not asked. "Which model should write your spoken
                # briefing?" is unanswerable by someone who installed this a
                # minute ago, and asking it rebuilds the thing the single front
                # door removed — needing to know something before you arrive. One
                # line and a pointer costs nothing and is findable later.
                write(
                    f"  Using {config.brain.provider}'s defaults for all six model "
                    f"roles."
                )
                write("  `mj config` shows them; edit the config file to change any.")
                report.did.append("key")
        else:
            found = probe()
            if found:
                write("")
                write(f"No key — but Ollama is running here, with {found}.")
                write("It needs no key and nothing leaves the machine.")
                if _yes(ask("  Use it? [y/N] ")):
                    try:
                        backup = _configure_ollama(found, config_path)
                    # Broad on purpose: a read-only home, a missing yaml, and a
                    # config file that will not parse all end the same way, and
                    # none of them may take down a session that has not started.
                    except Exception as exc:
                        write(f"  could not write the config: {exc}")
                        report.failed.append("ollama")
                    else:
                        write(f"  pointed every role at {found}.")
                        if backup is not None:
                            write(f"  your previous config is at {backup}")
                            write("  (comments are not carried through a rewrite)")
                        # Said out loud, because one model across six roles is a
                        # guess and two of the roles are the ones it is most
                        # likely to be wrong for. `mj do` and `/agent` need tool
                        # calling, which plenty of small local models simply do
                        # not do — and silently pointing the agent at one means
                        # discovering that mid-task instead of now.
                        write("  Note: `mj do` and /agent need a tool-calling model,")
                        write(f"  which {found} may not be. Set brain.agent_model to a")
                        write("  coder model if the agent misbehaves.")
                        write("  `mj config` shows all six; edit the file to change any.")
                        report.did.append("ollama")
                else:
                    report.skipped.append("model")
            else:
                write("  skipped — `mj config` shows what is missing.")
                report.skipped.append("model")

    # --- 2. session awareness in Claude Code -------------------------------
    write("")
    write("Claude Code hooks let Majordomo see which coding sessions are")
    write("waiting on you. This edits ~/.claude/settings.json (backed up first).")
    if _yes(ask("  Install them? [y/N] ")):
        try:
            from majordomo import install as install_mod

            result = install_mod.install(settings_path=settings_path)
            write(f"  wired into {result.settings_path}")
            report.did.append("hooks")
        except Exception as exc:
            write(f"  could not: {exc}")
            report.failed.append("hooks")
    else:
        report.skipped.append("hooks")

    # --- 3. the spoken briefing on wake ------------------------------------
    write("")
    write("A spoken briefing on login and on wake, via Task Scheduler.")
    if _yes(ask("  Set it up? [y/N] ")):
        if not _is_windows():
            write("  skipped — Task Scheduler is Windows-only.")
            report.skipped.append("trigger (not Windows)")
        else:
            try:
                from majordomo import trigger as trigger_mod

                registered, warnings = trigger_mod.install()
                for name in registered:
                    write(f"  + {name}")
                for warning in warnings:
                    write(f"  ! {warning}")
                report.did.append("trigger")
            except Exception as exc:
                write(f"  could not: {exc}")
                report.failed.append("trigger")
    else:
        report.skipped.append("trigger")

    # The marker goes down whatever happened. "We have met" is true even if the
    # answer to everything was no.
    _write_marker()
    return report


def _env_display() -> str:
    from majordomo import dotenv

    return str(dotenv.env_path())
