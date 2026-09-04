"""Tests for the vendored TTS — no real audio, no real network."""
from __future__ import annotations

import sys

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


# ---------------------------------------------------------------------------
# NVIDIA Riva — gRPC, not HTTP. All mocked; no real gRPC, no real audio.
# ---------------------------------------------------------------------------

import types  # noqa: E402

NVIDIA_CFG = VoiceConfig(
    enabled=True,
    provider="nvidia",
    voice_id="Magpie-Multilingual.EN-US.Sofia",
    model="",
    api_key_env="NVIDIA_API_KEY",
    timeout=90.0,
    function_id="fid-123",
    language="en-US",
    sample_rate=44100,
)


def _fake_riva():
    """A fake riva.client whose synthesize(future=True) mirrors the gRPC future."""
    fake = types.SimpleNamespace()
    fake.Auth = mock.MagicMock(name="Auth")
    call = mock.MagicMock(name="Call")
    call.result.return_value = types.SimpleNamespace(audio=b"\x01\x02\x03\x04")
    service = mock.MagicMock(name="Service")
    service.synthesize.return_value = call
    fake.SpeechSynthesisService = mock.MagicMock(return_value=service)
    return fake, types.SimpleNamespace(LINEAR_PCM=1), service


def test_nvidia_happy_path(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-test")
    fake, encoding, service = _fake_riva()
    monkeypatch.setattr(tts, "_import_riva", lambda: (fake, encoding))
    played = []
    monkeypatch.setattr(tts, "_play", lambda p: played.append(p))

    tts.speak("Two PRs need your review.", NVIDIA_CFG)

    service.synthesize.assert_called_once()
    _, kwargs = service.synthesize.call_args
    assert kwargs["voice_name"] == "Magpie-Multilingual.EN-US.Sofia"
    assert kwargs["language_code"] == "en-US"
    assert kwargs["sample_rate_hz"] == 44100
    assert kwargs["future"] is True, "the sync path has no timeout and can hang forever"
    assert len(played) == 1


def test_nvidia_sends_function_id_and_key_as_metadata(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-test")
    fake, encoding, _ = _fake_riva()
    monkeypatch.setattr(tts, "_import_riva", lambda: (fake, encoding))
    monkeypatch.setattr(tts, "_play", lambda p: None)

    tts.speak("Hello.", NVIDIA_CFG)

    _, auth_kwargs = fake.Auth.call_args
    meta = dict(auth_kwargs["metadata_args"])
    assert meta["function-id"] == "fid-123"
    assert meta["authorization"] == "Bearer nv-test"


def test_nvidia_missing_function_id_raises(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-test")
    fake, encoding, _ = _fake_riva()
    monkeypatch.setattr(tts, "_import_riva", lambda: (fake, encoding))
    monkeypatch.setattr(tts, "_play", lambda p: None)

    with pytest.raises(TTSError) as exc:
        tts.speak("Hello.", VoiceConfig(**{**NVIDIA_CFG.__dict__, "function_id": ""}))
    assert "function_id" in str(exc.value)


def test_nvidia_missing_package_names_the_extra(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-test")
    monkeypatch.setattr(tts, "_play", lambda p: None)

    def boom():
        raise ImportError("no riva")

    monkeypatch.setattr(tts, "_import_riva", boom)

    with pytest.raises(TTSError) as exc:
        tts.speak("Hello.", NVIDIA_CFG)
    assert "majordomo[nvidia]" in str(exc.value)


def test_nvidia_timeout_cancels_the_call(monkeypatch):
    """A hung synthesis in a scheduled task would sit silently until Windows
    killed it, so the deadline must be enforced and the call cancelled."""
    import grpc

    monkeypatch.setenv("NVIDIA_API_KEY", "nv-test")
    fake, encoding, service = _fake_riva()
    service.synthesize.return_value.result.side_effect = grpc.FutureTimeoutError()
    monkeypatch.setattr(tts, "_import_riva", lambda: (fake, encoding))
    monkeypatch.setattr(tts, "_play", lambda p: None)

    with pytest.raises(TTSError) as exc:
        tts.speak("Hello.", NVIDIA_CFG)
    assert "timed out" in str(exc.value).lower()
    service.synthesize.return_value.cancel.assert_called_once()


def test_nvidia_missing_key_names_the_configured_env(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(TTSError) as exc:
        tts.speak("Hello.", NVIDIA_CFG)
    assert "NVIDIA_API_KEY" in str(exc.value)


def test_riva_is_an_alias_for_nvidia():
    """Voicelog calls this engine 'riva'; accept both names."""
    assert tts.ADAPTERS["riva"] is tts.ADAPTERS["nvidia"]


def test_elevenlabs_still_works_alongside(monkeypatch):
    """Adding a provider must not disturb the one already in use."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test")
    monkeypatch.setattr(tts, "_play", lambda p: None)
    response = mock.MagicMock(is_success=True, content=b"\x01\x02")

    with mock.patch("majordomo.tts.httpx.post", return_value=response) as post:
        tts.speak("Hello.", CFG)
    post.assert_called_once()


# ---------------------------------------------------------------------------
# Playback must be interruptible
# ---------------------------------------------------------------------------

def _fake_wave(seconds: float):
    handle = mock.MagicMock()
    handle.__enter__.return_value.getframerate.return_value = 100
    handle.__enter__.return_value.getnframes.return_value = int(seconds * 100)
    return handle


def test_ctrl_c_stops_playback_instead_of_waiting_it_out(monkeypatch):
    """`winsound.PlaySound` without SND_ASYNC blocks inside C.

    Python's signal handler cannot run there, so Ctrl+C is queued and only
    raises once the clip has finished — while the CLI is printing
    "Ctrl+C to skip". The async-plus-poll form is what makes that true.
    """
    import _thread
    import threading
    import time

    winsound = mock.MagicMock()
    winsound.SND_FILENAME, winsound.SND_ASYNC, winsound.SND_PURGE = 1, 2, 4

    monkeypatch.setitem(sys.modules, "winsound", winsound)
    monkeypatch.setattr(tts.platform, "system", lambda: "Windows")
    monkeypatch.setattr(tts.wave, "open", lambda *a, **k: _fake_wave(10.0))

    threading.Timer(0.2, _thread.interrupt_main).start()
    started = time.monotonic()

    with pytest.raises(KeyboardInterrupt):
        tts._play("clip.wav")

    elapsed = time.monotonic() - started
    assert elapsed < 3.0, "a 10s clip should be cut short, not waited out"
    # and the sound is actually stopped, not left playing over the next output
    assert any(call.args == (None, 4) for call in winsound.PlaySound.call_args_list)


def test_playback_is_started_asynchronously(monkeypatch):
    winsound = mock.MagicMock()
    winsound.SND_FILENAME, winsound.SND_ASYNC, winsound.SND_PURGE = 1, 2, 4

    monkeypatch.setitem(sys.modules, "winsound", winsound)
    monkeypatch.setattr(tts.platform, "system", lambda: "Windows")
    monkeypatch.setattr(tts.wave, "open", lambda *a, **k: _fake_wave(0.0))

    tts._play("clip.wav")

    flags = winsound.PlaySound.call_args_list[0].args[1]
    assert flags & winsound.SND_ASYNC


def test_an_unreadable_wav_does_not_hang_playback(monkeypatch):
    """A bad header must mean 'play and move on', not an unbounded wait."""
    winsound = mock.MagicMock()
    winsound.SND_FILENAME, winsound.SND_ASYNC, winsound.SND_PURGE = 1, 2, 4

    def boom(*_a, **_k):
        raise tts.wave.Error("bad header")

    monkeypatch.setitem(sys.modules, "winsound", winsound)
    monkeypatch.setattr(tts.platform, "system", lambda: "Windows")
    monkeypatch.setattr(tts.wave, "open", boom)

    tts._play("clip.wav")          # must return promptly
    assert winsound.PlaySound.called
