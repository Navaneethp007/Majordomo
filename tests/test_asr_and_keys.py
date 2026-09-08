"""Tests for speech input and the chat trigger key. No microphone, no network."""
from __future__ import annotations

import struct
from unittest import mock

import pytest

from majordomo import asr, chat, keys
from majordomo import config as config_module

CFG = config_module.build(config_module.DEFAULTS)
VOICE = CFG.voice


def tone(chunks: int, amplitude: int, samples: int = 480) -> list[bytes]:
    """`chunks` blocks of 16-bit mono at a fixed level."""
    block = struct.pack(f"<{samples}h", *([amplitude] * samples))
    return [block] * chunks


#: Blocks consumed learning the noise floor before recording begins. Tests that
#: drive `record` must supply room tone first, as the room does in life.
CALIBRATION = asr.CALIBRATE_MS // asr.CHUNK_MS


def room(chunks: int = CALIBRATION) -> list[bytes]:
    """Quiet room tone, for calibration to measure."""
    return tone(chunks, 20)


def fake_stream(blocks):
    """A sounddevice RawInputStream that yields canned audio then silence."""
    supply = list(blocks)

    stream = mock.MagicMock()
    stream.__enter__.return_value = stream
    stream.__exit__.return_value = False

    def read(_n):
        return (supply.pop(0) if supply else tone(1, 0)[0]), False

    stream.read.side_effect = read
    module = mock.MagicMock()
    module.RawInputStream.return_value = stream
    return module


# ---------------------------------------------------------------------------
# Failing before spending anything
# ---------------------------------------------------------------------------


def test_a_missing_capture_library_names_the_extra():
    with mock.patch.object(asr, "_import_sounddevice", side_effect=ImportError):
        with pytest.raises(asr.MicrophoneUnavailable, match=r"majordomo\[voice\]"):
            asr.check_microphone()


def test_no_input_device_is_reported_as_such():
    module = mock.MagicMock()
    module.query_devices.return_value = [{"max_input_channels": 0}]
    with mock.patch.object(asr, "_import_sounddevice", return_value=module):
        with pytest.raises(asr.MicrophoneUnavailable, match="no microphone"):
            asr.check_microphone()


def test_a_present_microphone_passes():
    module = mock.MagicMock()
    module.query_devices.return_value = [{"max_input_channels": 2}]
    with mock.patch.object(asr, "_import_sounddevice", return_value=module):
        asr.check_microphone()


def test_listen_checks_hardware_before_the_api_key(monkeypatch):
    """Mirrors tts._check_playback: fail free rather than after uploading."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with mock.patch.object(asr, "_import_sounddevice", side_effect=ImportError):
        with pytest.raises(asr.MicrophoneUnavailable):
            asr.listen(VOICE)


def test_a_missing_key_is_reported_before_recording(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with mock.patch.object(asr, "check_microphone"), mock.patch.object(
        asr, "record"
    ) as record:
        with pytest.raises(asr.ASRError, match="NVIDIA_API_KEY"):
            asr.listen(VOICE)
    assert not record.called


def test_an_unknown_provider_is_named(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    cfg = config_module.build(
        config_module._deep_merge(
            config_module.DEFAULTS, {"voice": {"asr_provider": "nonsense"}}
        )
    ).voice
    with pytest.raises(asr.ASRError, match="unknown speech provider"):
        asr.listen(cfg)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_recording_stops_after_you_stop_talking():
    speech = tone(10, 8000)          # loud
    module = fake_stream(room() + speech)   # then silence forever

    with mock.patch.object(asr, "_import_sounddevice", return_value=module):
        pcm, rate = asr.record(VOICE)

    assert rate == VOICE.asr_sample_rate
    # speech plus roughly the trailing-silence window, nowhere near the cap
    expected_max = (10 + asr.TRAILING_SILENCE_MS // asr.CHUNK_MS + 2) * 960
    assert 0 < len(pcm) <= expected_max


def test_silence_throughout_gives_up_rather_than_recording_forever():
    module = fake_stream([])         # nothing but silence
    with mock.patch.object(asr, "_import_sounddevice", return_value=module):
        with pytest.raises(asr.ASRError, match="heard nothing"):
            asr.record(VOICE)


def test_a_noisy_room_still_hits_the_hard_cap():
    """Silence detection can be defeated; a recorder that never stops is worse
    than one that stops early."""
    cfg = config_module.build(
        config_module._deep_merge(
            config_module.DEFAULTS, {"voice": {"listen_max_seconds": 0.3}}
        )
    ).voice
    module = fake_stream(room() + tone(1000, 8000))    # never quiet

    with mock.patch.object(asr, "_import_sounddevice", return_value=module):
        pcm, _ = asr.record(cfg)

    assert len(pcm) <= int(0.3 * 1000 / asr.CHUNK_MS) * 960


def test_on_start_fires_once_the_stream_is_open():
    """The prompt should appear when listening becomes true, not before."""
    called = []
    module = fake_stream(room() + tone(6, 8000))
    with mock.patch.object(asr, "_import_sounddevice", return_value=module):
        asr.record(VOICE, on_start=lambda: called.append(True))
    assert called == [True]


def test_an_unopenable_microphone_is_reported():
    module = mock.MagicMock()
    module.RawInputStream.side_effect = OSError("device in use")
    with mock.patch.object(asr, "_import_sounddevice", return_value=module):
        with pytest.raises(asr.MicrophoneUnavailable, match="device in use"):
            asr.record(VOICE)


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


def test_a_missing_asr_function_id_says_it_is_a_different_one():
    """The TTS function id will not work here, and the error should say so."""
    with pytest.raises(asr.ASRError, match="different function"):
        asr._transcribe_nvidia(b"\x00\x00", 16000, "key", VOICE)


# ---------------------------------------------------------------------------
# The line editor
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_history():
    keys.reset_history()
    yield
    keys.reset_history()


def _raw(chars):
    """Patch msvcrt.getwch to hand back these characters in order."""
    supply = list(chars)
    module = mock.MagicMock()
    module.getwch.side_effect = lambda: supply.pop(0)
    return mock.patch.dict("sys.modules", {"msvcrt": module})


TRIGGER = "\x0e"
ENTER = "\r"
BS = "\b"
UP = "\xe0H"
DOWN = "\xe0P"
LEFT = "\xe0K"
RIGHT = "\xe0M"
HOME = "\xe0G"
END = "\xe0O"
DELETE = "\xe0S"


def typed(chars, trigger=TRIGGER):
    """Run read_line over a scripted key sequence."""
    with mock.patch.object(keys, "raw_reads_available", return_value=True), _raw(chars):
        return keys.read_line("> ", trigger)


def test_a_pipe_falls_back_to_plain_input():
    """Scripted input has no use for a hotkey or history."""
    with mock.patch.object(keys, "raw_reads_available", return_value=False):
        with mock.patch("builtins.input", return_value="typed"):
            assert keys.read_line("> ", TRIGGER) == "typed"


def test_ordinary_typing_returns_the_line():
    assert typed("hello" + ENTER) == "hello"


def test_the_trigger_key_returns_the_sentinel():
    assert typed(TRIGGER) is keys.TRIGGERED


def test_ctrl_c_is_re_raised_not_treated_as_a_message():
    """msvcrt hands it back as a character rather than raising, so the REPL
    would take an interrupt as input unless we re-raise it deliberately."""
    with pytest.raises(KeyboardInterrupt):
        typed("\x03")


@pytest.mark.parametrize("char", ["\x1a", "\x04"])
def test_eof_on_an_empty_line_raises_eoferror(char):
    with pytest.raises(EOFError):
        typed(char)


@pytest.mark.parametrize("char", ["\x1a", "\x04"])
def test_eof_mid_message_is_ignored(char):
    """As in a shell: only an empty line ends the session, so a stray Ctrl+D
    cannot close a conversation you were halfway through writing."""
    assert typed("hi" + char + ENTER) == "hi"


def test_enter_on_an_empty_prompt_returns_empty():
    assert typed(ENTER) == ""


# --- the bug that started this ---------------------------------------------


def test_an_arrow_key_does_not_leak_into_the_line():
    """getwch returns a prefix and leaves the scan code queued. Reading only
    the prefix echoed it and handed input() a stray 'H'."""
    assert typed(UP + "hi" + ENTER) == "hi"


def test_up_recalls_the_previous_line():
    typed("first" + ENTER)
    assert typed(UP + ENTER) == "first"


def test_up_walks_further_back_than_one_entry():
    typed("one" + ENTER)
    typed("two" + ENTER)
    assert typed(UP + UP + ENTER) == "one"


def test_down_walks_forward_again():
    typed("one" + ENTER)
    typed("two" + ENTER)
    assert typed(UP + UP + DOWN + ENTER) == "two"


def test_a_recalled_line_becomes_the_most_recent():
    """Submitting something you recalled makes it the newest entry, as a shell
    does — otherwise Up twice in a row would walk past what you just sent."""
    typed("one" + ENTER)
    typed("two" + ENTER)
    typed(UP + UP + ENTER)          # recalls and submits "one"

    assert keys.history() == ["one", "two", "one"]
    assert typed(UP + ENTER) == "one"


def test_down_from_history_restores_what_you_were_typing():
    """Browsing away from a half-written line and back must not lose it."""
    typed("old" + ENTER)
    assert typed("draft" + UP + DOWN + ENTER) == "draft"


def test_up_at_the_start_of_history_stays_put():
    typed("only" + ENTER)
    assert typed(UP + UP + UP + ENTER) == "only"


def test_history_skips_blanks_and_repeats():
    typed("same" + ENTER)
    typed(ENTER)
    typed("same" + ENTER)
    assert keys.history() == ["same"]


def test_history_is_capped():
    for n in range(keys.MAX_HISTORY + 20):
        keys.remember(f"line {n}")
    assert len(keys.history()) == keys.MAX_HISTORY
    assert keys.history()[-1] == f"line {keys.MAX_HISTORY + 19}"


# --- editing ---------------------------------------------------------------


def test_backspace_deletes_before_the_cursor():
    assert typed("hello" + BS + ENTER) == "hell"


def test_backspace_on_an_empty_line_does_nothing():
    assert typed(BS + BS + "hi" + ENTER) == "hi"


def test_left_then_typing_inserts_mid_line():
    assert typed("helo" + LEFT + "l" + ENTER) == "hello"


def test_delete_removes_at_the_cursor_where_backspace_removes_before():
    # cursor sits between 'a' and 'x': Delete takes 'x', Backspace takes 'a'
    assert typed("axb" + LEFT + LEFT + DELETE + ENTER) == "ab"
    assert typed("axb" + LEFT + LEFT + BS + ENTER) == "xb"


def test_backspace_at_the_start_of_a_line_does_nothing():
    assert typed("axb" + HOME + BS + ENTER) == "axb"


def test_delete_at_the_end_of_a_line_does_nothing():
    assert typed("axb" + END + DELETE + ENTER) == "axb"


def test_home_and_end_move_to_the_edges():
    assert typed("world" + HOME + "hello " + ENTER) == "hello world"
    assert typed("hello" + HOME + END + "!" + ENTER) == "hello!"


def test_right_does_not_run_past_the_end():
    assert typed("hi" + RIGHT + RIGHT + "!" + ENTER) == "hi!"


def test_ctrl_u_clears_the_line():
    assert typed("throw away\x15keep" + ENTER) == "keep"


def test_unhandled_control_characters_are_not_inserted():
    """A stray Ctrl+key should not put an invisible byte into a message."""
    assert typed("a\x07b" + ENTER) == "ab"


def test_an_unhandled_scan_code_is_consumed_whole():
    """F1 is prefix + ';'. Consuming both is the point; leaking either half
    is not."""
    assert typed("\xe0;ok" + ENTER) == "ok"


def test_ctrl_m_cannot_be_distinguished_from_enter():
    """Documented rather than supported: Ctrl+M *is* byte 13. Binding it would
    make Enter start recording, which is why the default is Ctrl+N."""
    assert "\r" == "\x0d"
    assert CFG.voice.listen_key != "\r"


# ---------------------------------------------------------------------------
# How the chat loop degrades
# ---------------------------------------------------------------------------


def test_a_missing_microphone_returns_you_to_typing():
    said = []
    with mock.patch.object(asr, "listen", side_effect=asr.MicrophoneUnavailable("no mic")):
        assert chat._listen(CFG, said.append) is None
    assert "no microphone" in said[0]


def test_a_failed_transcription_returns_you_to_typing():
    said = []
    with mock.patch.object(asr, "listen", side_effect=asr.ASRError("heard nothing")):
        assert chat._listen(CFG, said.append) is None
    assert "could not hear you" in said[0]


def test_a_successful_transcription_is_echoed_and_returned():
    said = []
    with mock.patch.object(asr, "listen", return_value="what did I ship"):
        assert chat._listen(CFG, said.append) == "what did I ship"
    assert "what did I ship" in said[0]


# ---------------------------------------------------------------------------
# Silence detection
# ---------------------------------------------------------------------------


import struct


def pcm(*samples):
    return struct.pack(f"<{len(samples)}h", *samples)


def test_rms_matches_the_stdlib_it_replaces():
    """audioop was removed in Python 3.13 and requires-python permits that
    install, so a module-level import of it was a ModuleNotFoundError waiting
    to kill the chat REPL. The arithmetic has to agree with what it replaced."""
    for block in (pcm(0, 0, 0), pcm(1000, -1000), pcm(32767, -32768), pcm(3, -4, 5)):
        expected = (sum(s * s for s in struct.unpack(f"<{len(block)//2}h", block))
                    / (len(block) // 2)) ** 0.5
        assert abs(asr._rms(block) - expected) < 1e-9


def test_rms_of_nothing_is_zero():
    assert asr._rms(b"") == 0.0


def test_silence_reads_below_the_threshold_and_speech_above():
    assert asr._rms(pcm(0, 1, -1)) < asr.SILENCE_RMS
    assert asr._rms(pcm(9000, -9000)) > asr.SILENCE_RMS


def test_the_default_pause_is_long_enough_for_a_thinking_pause():
    """900ms was fine for dictation and wrong for speech: people pause
    mid-sentence around fillers, and being cut off mid-thought is a worse
    failure than waiting a beat too long."""
    assert asr.TRAILING_SILENCE_MS >= 1_500


def test_the_pause_and_threshold_come_from_config():
    from majordomo import config as config_module

    tuned = config_module.build(
        {**config_module.DEFAULTS,
         "voice": {"trailing_silence_ms": 2_500, "silence_rms": 700}}
    ).voice

    assert tuned.trailing_silence_ms == 2_500
    assert tuned.silence_rms == 700


def test_zero_means_use_the_module_default():
    assert CFG.voice.trailing_silence_ms == 0
    assert CFG.voice.silence_rms == 0


def test_recording_stops_after_the_configured_pause():
    """Drives the loop directly: speech, then quiet, and the number of chunks
    captured says which threshold was used."""
    from majordomo import config as config_module

    voice = config_module.build(
        {**config_module.DEFAULTS, "voice": {"trailing_silence_ms": 300}}
    ).voice

    size = int(voice.asr_sample_rate * asr.CHUNK_MS / 1000)
    loud = pcm(*([9000] * size))
    quiet = pcm(*([20] * size))
    blocks = [quiet] * CALIBRATION + [loud] * 3 + [quiet] * 100

    module = mock.MagicMock()
    module.RawInputStream.return_value = canned_stream(blocks)

    with mock.patch.object(asr, "_import_sounddevice", return_value=module):
        audio, rate = asr.record(voice)

    # calibration is consumed, then 3 loud chunks + 10 quiet (300ms / 30ms)
    assert len(audio) == len(loud) * 3 + len(quiet) * 10
    assert rate == voice.asr_sample_rate


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def canned_stream(blocks):
    """A bare stream (not a module) that yields exactly these blocks."""
    stream = mock.MagicMock()
    stream.__enter__ = lambda self: self
    stream.__exit__ = lambda self, *a: False
    stream.read.side_effect = lambda n: (blocks.pop(0), False)
    return stream


def chunk_of(level, cfg=None):
    size = int((cfg or CFG.voice).asr_sample_rate * asr.CHUNK_MS / 1000)
    return pcm(*([level] * size))


def test_the_threshold_is_derived_from_the_room():
    """A fixed threshold is a claim about a microphone's gain that nobody
    measured. Get it wrong and there is no partial failure: either every chunk
    of speech reads as silence, or the room never does."""
    quiet_room = [chunk_of(40) for _ in range(CALIBRATION + 4)]
    stream = canned_stream(quiet_room)

    threshold = asr._calibrate(stream, 480, 16000)

    assert threshold == 40 * asr.NOISE_MULTIPLIER


def test_a_silent_room_does_not_produce_a_threshold_of_nothing():
    """Otherwise the recorder listens until the hard cap."""
    stream = canned_stream([chunk_of(0) for _ in range(20)])

    assert asr._calibrate(stream, 480, 16000) >= asr.MIN_SILENCE_RMS


def test_one_loud_chunk_does_not_raise_the_floor():
    """Someone may already be talking when calibration runs, so the quietest
    moments are the better estimate."""
    blocks = ([chunk_of(40)] * (CALIBRATION - 3)) + ([chunk_of(9000)] * 3)
    assert asr._calibrate(canned_stream(blocks), 480, 16000) < 500


def test_calibration_survives_a_stream_that_fails():
    stream = mock.MagicMock()
    stream.read.side_effect = RuntimeError("device went away")

    assert asr._calibrate(stream, 480, 16000) == float(asr.SILENCE_RMS)


def test_an_explicit_setting_skips_calibration():
    """If you have measured your own microphone, that beats anything derived."""
    from majordomo import config as config_module

    voice = config_module.build(
        {**config_module.DEFAULTS, "voice": {"silence_rms": 90, "trailing_silence_ms": 300}}
    ).voice

    loud = chunk_of(400, voice)
    quiet = chunk_of(10, voice)
    stream = canned_stream([loud] * 3 + [quiet] * 100)   # no calibration: skipped
    module = mock.MagicMock()
    module.RawInputStream.return_value = stream

    with mock.patch.object(asr, "_import_sounddevice", return_value=module):
        audio, _rate = asr.record(voice)

    # 400 clears a threshold of 90, so all three loud chunks counted as speech —
    # with the old fixed 500 they would all have read as silence.
    assert len(audio) == len(loud) * 3 + len(quiet) * 10


def test_speech_quieter_than_the_old_fixed_threshold_is_still_heard():
    """The reported symptom: recording stopped after exactly the trailing pause,
    no matter what you did, because nothing ever cleared 500."""
    from majordomo import config as config_module

    voice = config_module.build(
        {**config_module.DEFAULTS, "voice": {"trailing_silence_ms": 300}}
    ).voice

    ambient = chunk_of(30, voice)
    speech = chunk_of(300, voice)          # real speech, well under the old 500
    blocks = [ambient] * CALIBRATION + [speech] * 20 + [ambient] * 100

    module = mock.MagicMock()
    module.RawInputStream.return_value = canned_stream(blocks)

    with mock.patch.object(asr, "_import_sounddevice", return_value=module):
        audio, _rate = asr.record(voice)

    # Calibration consumes the ambient preamble, then all 20 speech chunks are
    # captured plus the 300ms of quiet that ends the recording. Under the old
    # fixed 500, none of the speech registered and this recorded nothing.
    assert len(audio) == len(speech) * 20 + len(ambient) * 10


def test_measure_reports_what_you_need_to_tune_it():
    from majordomo import config as config_module

    voice = config_module.build(config_module.DEFAULTS).voice
    module = mock.MagicMock()
    module.RawInputStream.return_value = canned_stream(
        [chunk_of(30, voice)] * 10 + [chunk_of(900, voice)] * 400
    )

    with mock.patch.object(asr, "_import_sounddevice", return_value=module):
        result = asr.measure(voice, seconds=6.0, write=lambda *_: None)

    assert result["p50"] > 800
    assert result["threshold"] < 200
    assert result["quiet_fraction"] == 0
    assert result["stops_at_ms"] == asr.TRAILING_SILENCE_MS


def test_a_quiet_room_reads_as_quiet_not_as_speech():
    """Measured: one room's ambient runs 15–103 with a median near 35. A
    threshold below that makes an empty room read as continuous speech, and the
    recorder then never stops. The lower quartile did exactly that."""
    from majordomo import config as config_module

    voice = config_module.build(config_module.DEFAULTS).voice
    ambient = [chunk_of(35, voice) for _ in range(40)]

    threshold = asr._calibrate(canned_stream(ambient), 480, voice.asr_sample_rate)

    assert threshold > 103          # above that room's loudest ambient chunk


def test_the_threshold_still_sits_well_under_speech():
    from majordomo import config as config_module

    voice = config_module.build(config_module.DEFAULTS).voice
    ambient = [chunk_of(35, voice) for _ in range(40)]

    threshold = asr._calibrate(canned_stream(ambient), 480, voice.asr_sample_rate)

    assert threshold < 1_000        # ordinary speech clears this comfortably


def test_mj_mic_fails_cleanly_without_the_capture_extra():
    """It exists to diagnose voice, so it must not traceback on the machine
    most likely to need it: one without the extra installed."""
    with mock.patch.object(asr, "_import_sounddevice", side_effect=ImportError("no module")):
        with pytest.raises(asr.MicrophoneUnavailable, match=r"majordomo\[voice\]"):
            asr.measure(CFG.voice, seconds=1.0, write=lambda *_: None)


def test_record_and_measure_open_the_microphone_the_same_way():
    """`mj mic` exists to report the numbers `record` uses. If the two opened
    the stream differently — a different rate, a different chunk size — the
    diagnosis would describe a stream nobody records with."""
    seen = []

    class Spy:
        def RawInputStream(self, **kwargs):
            seen.append(kwargs)
            # Quiet room for calibration, then endless speech. This test is
            # about how the stream is opened, not about when recording stops,
            # so it must never run out of audio.
            counter = {"n": 0}

            def read(_n):
                counter["n"] += 1
                level = 20 if counter["n"] <= CALIBRATION else 8000
                return tone(1, level)[0], False

            stream = mock.MagicMock()
            stream.__enter__ = lambda self: self
            stream.__exit__ = lambda self, *a: False
            stream.read.side_effect = read
            return stream

    with mock.patch.object(asr, "_import_sounddevice", return_value=Spy()):
        asr.record(VOICE)
        asr.measure(VOICE, seconds=1.0, write=lambda *_: None)

    assert len(seen) == 2
    assert seen[0] == seen[1]


def test_the_stream_helper_reports_a_device_that_will_not_open():
    module = mock.MagicMock()
    module.RawInputStream.side_effect = OSError("device in use")

    with mock.patch.object(asr, "_import_sounddevice", return_value=module):
        with pytest.raises(asr.MicrophoneUnavailable, match="device in use"):
            asr._open_stream(VOICE)


def test_the_chunk_size_matches_the_sample_rate():
    module = mock.MagicMock()
    with mock.patch.object(asr, "_import_sounddevice", return_value=module):
        _stream, chunk = asr._open_stream(VOICE)

    assert chunk == int(VOICE.asr_sample_rate * asr.CHUNK_MS / 1000)
