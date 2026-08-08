"""Configuration loading.

Shape follows Voicelog's proven approach — a ``DEFAULTS`` dict, a YAML file
merged over it, a frozen dataclass out the other end — with one change: the
merge is **deep**, because Majordomo's config is sectioned (brain / voice /
sources / router) rather than flat. A shallow merge would silently drop
``brain.base_url`` the moment you overrode ``brain.fuser_model``.

Every knob has a default, so Majordomo runs with no config file at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from majordomo.paths import default_config_path


class ConfigFileNotFound(Exception):
    """Raised when an explicitly-supplied config path does not exist."""


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    # The brain: reads raw source data, reasons, writes the briefing.
    "brain": {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        # Per-role models, so the escalation agent can be a different (bigger)
        # model than the fuser without touching code. Verified present on
        # OpenRouter's free tier as of 2026-08; re-check with
        # `curl https://openrouter.ai/api/v1/models` if a call 404s.
        #
        # The fuser writes four sentences of prose — a small fast model is the
        # right shape. The reducer reads oversized payloads and extracts from
        # them, which is the harder job, so it gets the larger model.
        "fuser_model": "google/gemma-4-26b-a4b-it:free",
        "reducer_model": "nvidia/nemotron-3-super-120b-a12b:free",
        "worker_model": "google/gemma-4-26b-a4b-it:free",
        "timeout": 120.0,
        "temperature": 0.3,
    },
    # The voice: speaks the briefing. A separate vendor from the brain by
    # necessity — one to think, one to speak (spec §3). OpenRouter thinks,
    # NVIDIA speaks.
    #
    # To use ElevenLabs instead, three lines change:
    #   provider: elevenlabs
    #   voice_id: pNInz6obpgDQGcFmaJgB     # Adam; must NOT be a Voice Library
    #   model: eleven_flash_v2_5           # voice, which free accounts get 402 on
    #   api_key_env: ELEVENLABS_API_KEY
    "voice": {
        "enabled": True,
        "provider": "nvidia",
        # For Riva this is a voice *name*, not an opaque id like ElevenLabs uses.
        "voice_id": "Magpie-Multilingual.EN-US.Sofia",
        "model": "",  # unused by Riva; the function_id selects the model
        "api_key_env": "NVIDIA_API_KEY",
        "timeout": 90.0,
        # NVCF function id for magpie-tts-multilingual. Riva-only; ignored by
        # every HTTP provider.
        "function_id": "877104f7-e885-42b9-8de8-f6e4c6303969",
        "language": "en-US",
        "sample_rate": 44100,
        # How long an unchanged situation stays quiet under --speak-if-needed.
        # Without this, every modern-standby resume re-reads the same blocked
        # session aloud, and a cold boot says everything twice (OnLogon then
        # OnBoot a minute later).
        "repeat_after_minutes": 120,
    },
    "sources": {
        "github": {
            "enabled": True,
            # Checked only if `gh auth token` is unavailable.
            "token_env": "MAJORDOMO_GH_TOKEN",
            "api_base": "https://api.github.com",
            "timeout": 30.0,
        },
        "gmail": {
            # A digest source: it never contributes needs-you items.
            #
            # Off by default, because nothing provisions credentials for you.
            # Left on, a fresh install fails this source on every run, and the
            # wake trigger would announce that failure aloud every time. Turn it
            # on in ~/.majordomo/config.yml once GMAIL_ADDRESS and
            # GMAIL_APP_PASSWORD are in ~/.majordomo/.env.
            "enabled": False,
            "address_env": "GMAIL_ADDRESS",
            # An app password, not your account password. Requires 2-Step
            # Verification to be on: myaccount.google.com/apppasswords
            "password_env": "GMAIL_APP_PASSWORD",
            # Gmail search syntax. Both halves earn their place: this account
            # has 18,103 unread and `to:me` only narrows that to 17,882, because
            # it matches anything delivered to the address. Recency plus Gmail's
            # own Primary category is what cuts it to ~17/day.
            "query": "is:unread in:inbox category:primary newer_than:1d",
            # A hard cap regardless of what the query is edited to.
            "max_messages": 20,
            "timeout": 30.0,
            # Per-address connect budget. imap.gmail.com resolves to IPv6
            # before IPv4, and on a host with no IPv6 route the default walk
            # costs ~42s. Short budget + IPv4-first makes that ~0.1s.
            "connect_timeout": 5.0,
        },
        "sessions": {
            "enabled": True,
            # Sessions older than this are stale — a machine that slept for a
            # week should not be briefed about last Tuesday's terminal.
            "stale_after_hours": 72,
            # An `active` session that has not made a sound for this long is
            # presumed gone. SessionEnd cannot fire when a process is killed,
            # so without this a crashed window stayed "active" for three days.
            # Stop fires at every turn end, so a working session refreshes this
            # constantly; only a dead one goes quiet.
            "active_timeout_minutes": 90,
        },
    },
    # The escalation gate. Deliberately deterministic: we don't spend a model
    # call deciding whether to use a model (spec §5).
    #
    # 50k rather than the spec's 8k: free-tier context windows are now 256k+,
    # so "it won't fit in one prompt" stopped being the real constraint. What
    # escalation actually buys is *attention* — a model handed 50k tokens of
    # mixed PRs and notifications summarises blandly and drops things, where
    # chunk-then-merge forces it to look at each part. Lower this to see the
    # fork fire on ordinary volume.
    "router": {
        "size_threshold_tokens": 50_000,
    },
}


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrainConfig:
    provider: str
    base_url: str
    api_key_env: str
    fuser_model: str
    reducer_model: str
    worker_model: str
    timeout: float
    temperature: float


@dataclass(frozen=True)
class VoiceConfig:
    enabled: bool
    provider: str
    voice_id: str
    model: str
    api_key_env: str
    timeout: float
    repeat_after_minutes: int = 120
    # Riva-only. Defaulted so the ElevenLabs path never has to know they exist.
    function_id: str = ""
    language: str = "en-US"
    sample_rate: int = 44100


@dataclass(frozen=True)
class GitHubConfig:
    enabled: bool
    token_env: str
    api_base: str
    timeout: float


@dataclass(frozen=True)
class GmailConfig:
    enabled: bool
    address_env: str
    password_env: str
    query: str
    max_messages: int
    timeout: float
    connect_timeout: float = 5.0


@dataclass(frozen=True)
class SessionsConfig:
    enabled: bool
    stale_after_hours: int
    active_timeout_minutes: int = 90


@dataclass(frozen=True)
class SourcesConfig:
    github: GitHubConfig
    gmail: GmailConfig
    sessions: SessionsConfig


@dataclass(frozen=True)
class RouterConfig:
    size_threshold_tokens: int


@dataclass(frozen=True)
class Config:
    brain: BrainConfig
    voice: VoiceConfig
    sources: SourcesConfig
    router: RouterConfig


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` onto ``base``, leaving both untouched."""
    merged = dict(base)
    for key, value in over.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load(path: str | None) -> Config:
    """Load configuration.

    - ``path=None`` → read the default location if present, else pure defaults.
    - ``path`` given → the file must exist; its values merge over the defaults.

    Raises:
        ConfigFileNotFound: an explicitly-named config file is absent.
    """
    if path is None:
        candidate = default_config_path()
        if not candidate.is_file():
            return build(DEFAULTS)
        resolved = candidate
    else:
        from pathlib import Path

        resolved = Path(path)
        if not resolved.is_file():
            raise ConfigFileNotFound(str(resolved))

    with open(resolved, encoding="utf-8") as fh:
        file_data = yaml.safe_load(fh) or {}
    if not isinstance(file_data, dict):
        file_data = {}

    return build(_deep_merge(DEFAULTS, file_data))


def build(data: dict[str, Any]) -> Config:
    """Build a Config from a fully-merged dict.

    Unknown keys are ignored rather than fatal — a stale key left over from an
    older version should not brick the app on upgrade. Scalars are coerced, so a
    YAML quoting accident ('8000') doesn't produce a str where an int is
    compared.
    """
    # `or {}` rather than a .get default: a hand-edited config with a bare
    # `brain:` and nothing indented under it parses to {"brain": None}, and a
    # default only applies when the key is *absent*. Without this, a plausible
    # edit gives the user an AttributeError traceback instead of a config.
    brain = data.get("brain") or {}
    voice = data.get("voice") or {}
    sources = data.get("sources") or {}
    github = sources.get("github") or {}
    gmail = sources.get("gmail") or {}
    sessions = sources.get("sessions") or {}
    router = data.get("router") or {}

    d_brain = DEFAULTS["brain"]
    d_voice = DEFAULTS["voice"]
    d_github = DEFAULTS["sources"]["github"]
    d_gmail = DEFAULTS["sources"]["gmail"]
    d_sessions = DEFAULTS["sources"]["sessions"]
    d_router = DEFAULTS["router"]

    return Config(
        brain=BrainConfig(
            provider=str(brain.get("provider", d_brain["provider"])),
            base_url=str(brain.get("base_url", d_brain["base_url"])).rstrip("/"),
            api_key_env=str(brain.get("api_key_env", d_brain["api_key_env"])),
            fuser_model=str(brain.get("fuser_model", d_brain["fuser_model"])),
            reducer_model=str(brain.get("reducer_model", d_brain["reducer_model"])),
            worker_model=str(brain.get("worker_model", d_brain["worker_model"])),
            timeout=float(brain.get("timeout", d_brain["timeout"])),
            temperature=float(brain.get("temperature", d_brain["temperature"])),
        ),
        voice=VoiceConfig(
            enabled=bool(voice.get("enabled", d_voice["enabled"])),
            provider=str(voice.get("provider", d_voice["provider"])),
            voice_id=str(voice.get("voice_id", d_voice["voice_id"])),
            model=str(voice.get("model", d_voice["model"])),
            api_key_env=str(voice.get("api_key_env", d_voice["api_key_env"])),
            timeout=float(voice.get("timeout", d_voice["timeout"])),
            repeat_after_minutes=int(
                voice.get("repeat_after_minutes", d_voice["repeat_after_minutes"])
            ),
            function_id=str(voice.get("function_id", d_voice["function_id"])),
            language=str(voice.get("language", d_voice["language"])),
            sample_rate=int(voice.get("sample_rate", d_voice["sample_rate"])),
        ),
        sources=SourcesConfig(
            github=GitHubConfig(
                enabled=bool(github.get("enabled", d_github["enabled"])),
                token_env=str(github.get("token_env", d_github["token_env"])),
                api_base=str(github.get("api_base", d_github["api_base"])).rstrip("/"),
                timeout=float(github.get("timeout", d_github["timeout"])),
            ),
            gmail=GmailConfig(
                enabled=bool(gmail.get("enabled", d_gmail["enabled"])),
                address_env=str(gmail.get("address_env", d_gmail["address_env"])),
                password_env=str(gmail.get("password_env", d_gmail["password_env"])),
                query=str(gmail.get("query", d_gmail["query"])),
                max_messages=int(gmail.get("max_messages", d_gmail["max_messages"])),
                timeout=float(gmail.get("timeout", d_gmail["timeout"])),
                connect_timeout=float(
                    gmail.get("connect_timeout", d_gmail["connect_timeout"])
                ),
            ),
            sessions=SessionsConfig(
                enabled=bool(sessions.get("enabled", d_sessions["enabled"])),
                stale_after_hours=int(
                    sessions.get("stale_after_hours", d_sessions["stale_after_hours"])
                ),
                active_timeout_minutes=int(
                    sessions.get(
                        "active_timeout_minutes", d_sessions["active_timeout_minutes"]
                    )
                ),
            ),
        ),
        router=RouterConfig(
            size_threshold_tokens=int(
                router.get("size_threshold_tokens", d_router["size_threshold_tokens"])
            ),
        ),
    )
