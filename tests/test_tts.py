"""Tests for the vendored TTS — no real audio, no real network."""
from __future__ import annotations

from unittest import mock

import pytest

from majordomo import tts
from majordomo.config import VoiceConfig
from majordomo.tts import TTSError, chunk_text, speech_text

CFG = VoiceConfig(
    enabled=True,
    provider="elevenlabs",
    voice_id="voice-123",
    model="eleven_multilingual_v2",
    api_key_env="ELEVENLABS_API_KEY",
    timeout=90.0,
)


# ---------------------------------------------------------------------------
# speech_text — the fuser is told to write prose, but models drift into markdown
# ---------------------------------------------------------------------------

def test_strips_heading_markers():
    assert "#" not in speech_text("## Needs you\n### GitHub")


def test_removes_code_blocks_entirely():
    out = speech_text("Run this:\n```\nrm -rf /\n```\nDone.")
    assert "rm -rf" not in out
    assert "Done." in out


def test_turns_bullets_into_plain_speech():
    assert speech_text("- two PRs waiting") == "two PRs waiting"


def test_links_become_their_text():
    out = speech_text("See [the PR](https://github.com/x/y/pull/1) now")
    assert "the PR" in out
    assert "github.com" not in out


def test_inline_markers_are_dropped_but_words_kept():
    out = speech_text("This is `code` and *bold* and _italic_")
    assert "code" in out and "bold" in out and "italic" in out
    assert "*" not in out and "`" not in out and "_" not in out


def test_code_only_input_is_empty():
    assert speech_text("```\njust code\n```").strip() == ""


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

def test_short_text_is_one_chunk():
    assert chunk_text("Two PRs need you.") == ["Two PRs need you."]


def test_long_text_never_cuts_mid_word():
    words = [f"word{i}" for i in range(300)]
    chunks = chunk_text(" ".join(words), max_len=400)

    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)
    assert " ".join(chunks).split() == words


def test_prefers_sentence_boundaries():
    first = "A" * 300 + "."
    chunks = chunk_text(first + " " + "B" * 200 + ".", max_len=400)
    assert chunks[0].rstrip() == first


def test_whitespace_only_is_no_chunks():
    assert chunk_text("   ") == []


# ---------------------------------------------------------------------------
# speak
# ---------------------------------------------------------------------------

def test_missing_key_names_the_configured_env_var(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(TTSError) as exc:
        tts.speak("Hello.", CFG)
    assert "ELEVENLABS_API_KEY" in str(exc.value)


def test_unknown_provider_fails_before_any_network_call(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not reach the network")

    monkeypatch.setattr(tts.httpx, "post", boom)
    bad = VoiceConfig(**{**CFG.__dict__, "provider": "fish"})

    with pytest.raises(TTSError) as exc:
        tts.speak("Hello.", bad)
    assert "fish" in str(exc.value)


def test_happy_path_posts_and_plays(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
    played = []
    monkeypatch.setattr(tts, "_play", lambda p: played.append(p))

    response = mock.MagicMock(is_success=True, content=b"\x01\x02\x03\x04")
    with mock.patch("majordomo.tts.httpx.post", return_value=response) as post:
        tts.speak("Two PRs need your review.", CFG)

    post.assert_called_once()
    args, kwargs = post.call_args
    assert "voice-123" in args[0]
    assert kwargs["headers"]["xi-api-key"] == "el-test"
    assert kwargs["json"]["model_id"] == "eleven_multilingual_v2"
    assert len(played) == 1


def test_missing_voice_id_raises(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
    monkeypatch.setattr(tts, "_play", lambda p: None)
    no_voice = VoiceConfig(**{**CFG.__dict__, "voice_id": ""})

    with pytest.raises(TTSError) as exc:
        tts.speak("Hello.", no_voice)
    assert "voice_id" in str(exc.value)


def test_http_failure_raises_ttserror(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
    monkeypatch.setattr(tts, "_play", lambda p: None)
    response = mock.MagicMock(is_success=False, status_code=401, text="unauthorized")

    with mock.patch("majordomo.tts.httpx.post", return_value=response):
        with pytest.raises(TTSError) as exc:
            tts.speak("Hello.", CFG)
    assert "401" in str(exc.value)


def test_empty_speech_makes_no_call(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
    with mock.patch("majordomo.tts.httpx.post") as post:
        tts.speak("```\njust code\n```", CFG)
    post.assert_not_called()


def test_playback_failure_becomes_ttserror(monkeypatch):
    """So the CLI's 'audio only warns, never blocks' contract holds."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")

    def boom(path):
        raise OSError("audio device busy")

    monkeypatch.setattr(tts, "_play", boom)
    response = mock.MagicMock(is_success=True, content=b"\x01\x02")

    with mock.patch("majordomo.tts.httpx.post", return_value=response):
        with pytest.raises(TTSError):
            tts.speak("Hello.", CFG)


def test_splits_at_the_latest_sentence_boundary():
    """The original compared a raw rfind index against a split_at that already
    had len(sep) added, so a later separator could lose to an earlier one and
    the chunk broke at the wrong sentence."""
    text = "A" * 100 + ". " + "B" * 10 + "! " + "C" * 400
    chunks = chunk_text(text, max_len=200)

    assert chunks[0].endswith("!"), f"broke at the wrong boundary: {chunks[0][-5:]!r}"


def test_chunking_never_loses_or_duplicates_text():
    text = ". ".join(f"sentence number {i} here" for i in range(200))
    chunks = chunk_text(text, max_len=180)
    assert " ".join(chunks).split() == text.split()
