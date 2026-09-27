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
from pathlib import Path
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
        # Per-role models, so each job gets the right shape without touching
        # code. Every one below was chosen by running the *actual prompt for
        # that role* — a model card says nothing useful here. Two lessons paid
        # for on 2026-09-05:
        #
        #   - "high-throughput agentic" can mean a reasoning model that emits
        #     its chain of thought. nemotron-3.5-lightning answered the fuse
        #     prompt with nine paragraphs of deliberation, which would have been
        #     read aloud. It is excellent at tool calling and wrong for prose.
        #   - "free" does not mean reachable. thinkingmachines/inkling:free
        #     returns a hard 403 — it is gated to partner apps.
        #
        # Deliberately two providers. Gemma 429'd across every role at once when
        # they were all Google-served; one busy pool should not be able to do
        # that.

        # Compression, once per source. A trivial job that runs on every wake:
        # speed and availability matter, depth does not.
        "worker_model": "dots-studio/dots-3-note-preview:free",
        # Four sentences of spoken prose. Chosen for instruction-following, not
        # speed: given an empty DECISIONS block and tempting CONTEXT, it has to
        # say "nothing needs you" rather than promote something. Minimax failed
        # that; every model here was re-tested against it on 2026-09-08 and
        # passed. Latency matters least in this role — the briefing is spoken
        # while you are walking back to the desk, not typed at.
        "fuser_model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        # The escalation path: read an oversized payload in chunks and extract
        # what matters. Also unattended, so the same reasoning applies.
        "reducer_model": "nvidia/nemotron-3-super-120b-a12b:free",
        # `mj ask` and `mj chat`. A different job from the briefing roles: those
        # compress a payload once, this has to hold a thread across twenty turns
        # and disagree with you — and you are sitting there while it does, so
        # consistency beats peak quality. Measured over three runs on
        # 2026-09-08: this one 2.0–2.5s, nemotron-ultra 0.7–31.4s. The tail is
        # what you feel.
        "chat_model": "dots-studio/dots-3-note-preview:free",
        # `mj do` and `/agent`. Split from chat_model because the two roles want
        # different things: chat wants a model that talks like a colleague,
        # while the agent wants one that writes code and calls tools reliably —
        # and the agent runs many turns, so its latency compounds where chat's
        # does not. Empty means "use chat_model", which is what it did before
        # this existed.
        "agent_model": "poolside/laguna-s-2.1:free",
        # Tried once, for any role, when its own model fails a retryable way.
        #
        # This exists because free tiers move underneath you. On 2026-09-08
        # every model this file previously named had stopped being free —
        # minimax returned "This model is unavailable for free" and GLM had left
        # the free list entirely. A fallback on a different provider is what
        # keeps one such change from taking out every role at once.
        #
        # Set to "" to disable. A fallback equal to the primary is skipped.
        "fallback_model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        # When a chat's total context passes this, fold the oldest turns into a
        # summary and keep the recent ones verbatim. The working set is expected
        # around 20k; this is the alarm, not the target.
        "chat_compact_threshold_tokens": 32_000,
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

        # ── Speech *in*, as opposed to out ──────────────────────────────────
        # Riva does ASR as well as TTS: same endpoint, same key, same package.
        # What differs is the NVCF function — each model is its own function id,
        # so this cannot be inferred from the TTS one and must be set. Find it
        # on the model's page at build.nvidia.com.
        "asr_function_id": "",
        # Blank means "same as voice.provider". Kept separate so speech in and
        # speech out can come from different vendors without a second config
        # section — you might want a local Whisper reading a cloud voice.
        "asr_provider": "",
        "asr_language": "en-US",
        # 16kHz is the ASR standard and a quarter the bytes of the TTS rate.
        # Speech recognition gains nothing from the extra bandwidth.
        "asr_sample_rate": 16000,
        # A hard cap on one utterance. Recording normally stops when you stop
        # talking; this is the backstop for a noisy room, where silence
        # detection never triggers and a recorder that never stops is worse
        # than one that stops early.
        "listen_max_seconds": 30.0,
        # The key that starts listening from the chat prompt, as a raw control
        # character. Default is Ctrl+N (\x0e).
        #
        # NOT Ctrl+M: that is byte 13, which is exactly what Enter sends — a
        # terminal cannot tell them apart, so binding it would make Enter
        # start recording.
        "listen_key": "\x0e",
        # How long a pause means "I have finished" rather than "I am
        # thinking". 0 uses asr.TRAILING_SILENCE_MS. Raise it if you
        # pause mid-sentence around fillers; lower it if it feels slow.
        "trailing_silence_ms": 0,
        # RMS below which a 30ms chunk counts as quiet. 0 uses
        # asr.SILENCE_RMS. Raise it in a noisy room, where background
        # sound keeps the recorder alive; lower it for a soft voice.
        "silence_rms": 0,
    },
    "sources": {
        "github": {
            "enabled": True,
            # Checked only if `gh auth token` is unavailable.
            "token_env": "MAJORDOMO_GH_TOKEN",
            "api_base": "https://api.github.com",
            "timeout": 30.0,
            # How far back `mj activity` reaches, and how much of the cache is
            # considered current on read. Older entries stay on disk — pruning
            # happens at read time, not by rewriting the log.
            "activity_days": 90,
            # Results per request. 100 is GitHub's maximum.
            "activity_per_page": 100,
            # How many pages to walk per search, so 5 x 100 = 500 each.
            #
            # Set from measurement, not taste: one real 90-day window here held
            # 319 commits, so the first guess of 3 pages still truncated. Five
            # covers a busy quarter with headroom.
            #
            # Bounded on purpose. GitHub caps search at 1000 results however
            # many pages you ask for, and an unbounded loop against a
            # rate-limited endpoint turns a refresh into a stall. When a search
            # still has more than this, `mj activity --refresh` says so rather
            # than silently reporting a ceiling as a count.
            "activity_max_pages": 5,
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
    # What Majordomo has learned about you. One file per fact under
    # ~/.majordomo/memory/, plus an index small enough to send every turn.
    "memory": {
        "enabled": True,
        # How many memory *bodies* get loaded alongside the index. The index is
        # always sent in full; bodies are the expensive part, so only the ones
        # whose description matches the question come along.
        "max_bodies_loaded": 8,
    },
    # `mj start` and the /build command inside a chat.
    "scaffold": {
        # Where new projects are created. `~` is expanded at load time.
        "root": "~/projects",
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


#: The six model roles: label, the ``BrainConfig`` field, and what it is for.
#:
#: One table, because two places need it and they must not disagree. ``mj config``
#: prints it, and ``llm`` reads it backwards — given a model id that a provider
#: rejected permanently, it names which role was pointing at it and therefore
#: which line of config to edit.
#:
#: The purposes are the strings a user reads, so they are phrased for that and
#: are asserted on by the tests for ``mj config``.
MODEL_ROLES: tuple[tuple[str, str, str], ...] = (
    ("worker", "worker_model", "compress each source"),
    ("fuser", "fuser_model", "write the spoken briefing"),
    ("reducer", "reducer_model", "handle an oversized payload"),
    ("chat", "chat_model", "mj ask, mj chat"),
    ("agent", "agent_model", "mj do, /agent"),
    ("fallback", "fallback_model", "when a role's model fails"),
)


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
    # Defaulted so every existing construction of BrainConfig keeps working.
    chat_model: str = "dots-studio/dots-3-note-preview:free"
    agent_model: str = "poolside/laguna-s-2.1:free"
    fallback_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    chat_compact_threshold_tokens: int = 32_000


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
    # Speech in. Defaulted so nothing that builds a VoiceConfig by hand breaks.
    asr_function_id: str = ""
    asr_provider: str = ""
    asr_language: str = "en-US"
    asr_sample_rate: int = 16000
    listen_max_seconds: float = 30.0
    listen_key: str = "\x0e"
    # 0 means "use the module default". Both depend on the speaker and the
    # room, which no default can know: how long a thinking pause runs before
    # it means "I have finished", and how quiet the room actually is.
    trailing_silence_ms: int = 0
    silence_rms: int = 0


@dataclass(frozen=True)
class GitHubConfig:
    enabled: bool
    token_env: str
    api_base: str
    timeout: float
    activity_days: int = 90
    activity_per_page: int = 100
    activity_max_pages: int = 5


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
class MemoryConfig:
    enabled: bool = True
    max_bodies_loaded: int = 8


@dataclass(frozen=True)
class ScaffoldConfig:
    #: Already expanded — ``~`` is resolved in ``build()``, not by the caller.
    root: str = ""


@dataclass(frozen=True)
class Config:
    brain: BrainConfig
    voice: VoiceConfig
    sources: SourcesConfig
    router: RouterConfig
    # Defaulted so a Config built before these existed still constructs.
    memory: MemoryConfig = MemoryConfig()
    scaffold: ScaffoldConfig = ScaffoldConfig()


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
    memory = data.get("memory") or {}
    scaffold = data.get("scaffold") or {}

    d_brain = DEFAULTS["brain"]
    d_voice = DEFAULTS["voice"]
    d_github = DEFAULTS["sources"]["github"]
    d_gmail = DEFAULTS["sources"]["gmail"]
    d_sessions = DEFAULTS["sources"]["sessions"]
    d_router = DEFAULTS["router"]
    d_memory = DEFAULTS["memory"]
    d_scaffold = DEFAULTS["scaffold"]

    return Config(
        brain=BrainConfig(
            provider=str(brain.get("provider", d_brain["provider"])),
            base_url=str(brain.get("base_url", d_brain["base_url"])).rstrip("/"),
            api_key_env=str(brain.get("api_key_env", d_brain["api_key_env"])),
            fuser_model=str(brain.get("fuser_model", d_brain["fuser_model"])),
            reducer_model=str(brain.get("reducer_model", d_brain["reducer_model"])),
            worker_model=str(brain.get("worker_model", d_brain["worker_model"])),
            chat_model=str(brain.get("chat_model", d_brain["chat_model"])),
            agent_model=str(brain.get("agent_model", d_brain["agent_model"])),
            fallback_model=str(brain.get("fallback_model", d_brain["fallback_model"])),
            chat_compact_threshold_tokens=int(
                brain.get(
                    "chat_compact_threshold_tokens",
                    d_brain["chat_compact_threshold_tokens"],
                )
            ),
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
            asr_function_id=str(voice.get("asr_function_id", d_voice["asr_function_id"])),
            asr_provider=str(voice.get("asr_provider", d_voice["asr_provider"])),
            asr_language=str(voice.get("asr_language", d_voice["asr_language"])),
            asr_sample_rate=int(voice.get("asr_sample_rate", d_voice["asr_sample_rate"])),
            listen_max_seconds=float(voice.get("listen_max_seconds", d_voice["listen_max_seconds"])),
            listen_key=str(voice.get("listen_key", d_voice["listen_key"])),
            trailing_silence_ms=int(
                voice.get("trailing_silence_ms", d_voice["trailing_silence_ms"])
            ),
            silence_rms=int(voice.get("silence_rms", d_voice["silence_rms"])),
        ),
        sources=SourcesConfig(
            github=GitHubConfig(
                enabled=bool(github.get("enabled", d_github["enabled"])),
                token_env=str(github.get("token_env", d_github["token_env"])),
                api_base=str(github.get("api_base", d_github["api_base"])).rstrip("/"),
                timeout=float(github.get("timeout", d_github["timeout"])),
                activity_days=int(
                    github.get("activity_days", d_github["activity_days"])
                ),
                activity_per_page=int(
                    github.get("activity_per_page", d_github["activity_per_page"])
                ),
                activity_max_pages=int(
                    github.get("activity_max_pages", d_github["activity_max_pages"])
                ),
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
        memory=MemoryConfig(
            enabled=bool(memory.get("enabled", d_memory["enabled"])),
            max_bodies_loaded=int(
                memory.get("max_bodies_loaded", d_memory["max_bodies_loaded"])
            ),
        ),
        scaffold=ScaffoldConfig(
            # Expanded here so no caller ever has to remember to. A config
            # holding a literal "~/projects" would create a folder called "~".
            root=str(
                Path(str(scaffold.get("root", d_scaffold["root"]))).expanduser()
            ),
        ),
    )
