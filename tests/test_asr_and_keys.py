"""Tests for speech input and the chat trigger key. No microphone, no network."""
from __future__ import annotations

import re
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
        with pytest.raises(asr.MicrophoneUnavailable, match=r"\[voice\]"):
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
    """The TTS function id will not work here, and the error should say so.

    Riva is faked rather than required. `_transcribe_nvidia` checks the import
    before the function id — correctly, since there is no point validating
    configuration for a library that is absent — so without the stub this test
    passed only on a machine where the `nvidia` extra happened to be installed,
    and reported the import error instead of the thing it is about. It passed
    locally for exactly that reason and would have failed on any clean runner,
    Windows included.
    """
    riva = mock.MagicMock()
    with mock.patch.dict(
        "sys.modules",
        {
            "riva": riva,
            "riva.client": riva.client,
            "riva.client.proto": mock.MagicMock(),
            "riva.client.proto.riva_audio_pb2": mock.MagicMock(),
        },
    ):
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
        assert chat._listen(CFG, chat.Terminal(write=said.append)) is None
    assert "no microphone" in said[0]


def test_a_failed_transcription_returns_you_to_typing():
    said = []
    with mock.patch.object(asr, "listen", side_effect=asr.ASRError("heard nothing")):
        assert chat._listen(CFG, chat.Terminal(write=said.append)) is None
    assert "could not hear you" in said[0]


def test_a_successful_transcription_is_echoed_and_returned():
    said = []
    with mock.patch.object(asr, "listen", return_value="what did I ship"):
        assert chat._listen(CFG, chat.Terminal(write=said.append)) == "what did I ship"
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
        with pytest.raises(asr.MicrophoneUnavailable, match=r"\[voice\]"):
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


# ---------------------------------------------------------------------------
# Lines that wrap
# ---------------------------------------------------------------------------


class Screen:
    """A terminal small enough to wrap, that understands what `_redraw` emits.

    Asserting on the raw escape sequence would test the implementation. What
    matters is what ends up on screen, so this applies the codes and reports
    the rows — which is how the repeating-prompt bug is visible at all.
    """

    #: Cursor movement and erasure, plus SGR (`m`) — colour and reverse video,
    #: which move nothing and so are consumed and discarded. Without `m` here,
    #: a highlighted row's escape bytes land in the text and the row no longer
    #: reads as what a person would see.
    ESCAPE = re.compile(r"\x1b\[(\d*)([ABCJm])")

    def __init__(self, width=24):
        self.width = width
        self.rows = [""]
        self.row = self.column = 0

    def _put(self, char):
        while len(self.rows) <= self.row:
            self.rows.append("")
        line = self.rows[self.row].ljust(self.column)
        self.rows[self.row] = line[: self.column] + char + line[self.column + 1 :]
        self.column += 1
        if self.column >= self.width:
            self.column = 0
            self.row += 1

    def write(self, text):
        i = 0
        while i < len(text):
            match = self.ESCAPE.match(text[i:])
            if match:
                count, kind = int(match.group(1) or 0), match.group(2)
                if kind == "A":
                    self.row = max(0, self.row - count)
                elif kind == "B":
                    self.row += count
                elif kind == "C":
                    self.column += count
                elif kind == "J":
                    while len(self.rows) <= self.row:
                        self.rows.append("")
                    self.rows[self.row] = self.rows[self.row][: self.column]
                    del self.rows[self.row + 1 :]
                i += match.end()
                continue
            char = text[i]
            if char == "\r":
                self.column = 0
            elif char == "\n":
                self.row += 1
                self.column = 0
            else:
                self._put(char)
            i += 1

    def flush(self):
        pass

    @property
    def lines(self):
        return [row.rstrip() for row in self.rows]


PROMPT = "you > "


def painted(text, cursor=None, width=24, before=None):
    """Type `text` into a `width`-column terminal; return the screen."""
    screen = Screen(width)
    row = 0
    with mock.patch.object(keys, "_terminal_width", return_value=width),          mock.patch.object(keys, "_ansi_available", return_value=True):
        if before is not None:
            row = keys._redraw(PROMPT, before, len(before), row, stream=screen)
        row = keys._redraw(
            PROMPT, text, len(text) if cursor is None else cursor, row, stream=screen
        )
    return screen


def test_a_wrapped_line_does_not_repeat_itself():
    """The reported bug: past the terminal width, `\r` returns to the start of
    the *visual* row rather than the line, so every keystroke repainted the
    prompt and the first row's worth of text underneath — once per character."""
    screen = painted("tell me about the coffee")

    assert screen.lines == ["you > tell me about the", "coffee"]
    assert screen.lines.count("you > tell me about the") == 1


def test_a_line_that_shrinks_gives_its_second_row_back():
    """The old space-padding only ever cleared the tail of one row, so deleting
    back past a boundary left the rest of the text below the cursor."""
    screen = painted("tell me", before="tell me about the coffee")

    assert screen.lines == ["you > tell me"]


@pytest.mark.parametrize("cursor,row,column", [(0, 0, 6), (3, 0, 9), (24, 1, 6)])
def test_the_cursor_lands_where_it_belongs_on_a_wrapped_line(cursor, row, column):
    screen = painted("tell me about the coffee", cursor=cursor)

    assert (screen.row, screen.column) == (row, column)


def test_text_ending_exactly_at_the_margin():
    """Terminals differ on whether the cursor has wrapped yet, so the
    arithmetic has to hold either way."""
    screen = painted("x" * 18, width=24)

    assert screen.lines[0] == "you > " + "x" * 18


def test_a_line_that_fits_is_unchanged():
    screen = painted("short")
    assert screen.lines == ["you > short"]


def test_finishing_a_wrapped_line_lands_below_all_of_it():
    """A bare newline from mid-wrap leaves the tail above it and the next
    prompt written over the middle."""
    screen = Screen(24)
    with mock.patch.object(keys, "_terminal_width", return_value=24),          mock.patch.object(keys, "_ansi_available", return_value=True):
        row = keys._redraw(PROMPT, "tell me about the coffee", 3, 0, stream=screen)
        keys._finish_line(PROMPT, "tell me about the coffee", row, stream=screen)

    assert screen.row == 2          # past both rows of the line
    assert screen.column == 0


def test_the_width_is_never_zero():
    """A division by it happens on every keystroke."""
    with mock.patch.object(keys.shutil, "get_terminal_size", side_effect=OSError):
        assert keys._terminal_width() > 0


# ---------------------------------------------------------------------------
# The arrow-key picker
#
# Same machinery as the line editor, so the same test approach: drive it from
# scripted keystrokes and assert on what the Screen emulator says is visible.
# Asserting on the escape codes would test the implementation.
# ---------------------------------------------------------------------------

OPTIONS = [f"202609{i:02d}-1200   {i} turns  topic {i}" for i in range(1, 4)]


def pick(chars, options=OPTIONS, width=60):
    """Run `choose` over a scripted key sequence against a fake screen."""
    screen = Screen(width=width)
    with mock.patch.object(keys, "raw_reads_available", return_value=True), _raw(chars):
        with mock.patch.object(keys, "_ansi_available", return_value=True):
            with mock.patch.object(keys, "_terminal_width", return_value=width):
                chosen = keys.choose("Which conversation?", options, stream=screen)
    return chosen, screen


def test_enter_takes_the_first_option_by_default():
    chosen, screen = pick([ENTER])
    assert chosen == 0
    assert "Which conversation?" in screen.rows[0]


def test_down_then_enter_takes_the_second():
    chosen, _ = pick([*DOWN, ENTER])
    assert chosen == 1


def test_up_from_the_top_wraps_to_the_bottom():
    """Wrapping, because a list you cannot get to the end of in one keypress is
    a list you scroll past."""
    chosen, _ = pick([*UP, ENTER])
    assert chosen == len(OPTIONS) - 1


def test_home_and_end_jump():
    assert pick([*END, ENTER])[0] == len(OPTIONS) - 1
    assert pick([*END, *HOME, ENTER])[0] == 0


def test_the_selected_row_is_marked_as_well_as_highlighted():
    """A highlight alone is invisible in a terminal whose theme ignores reverse
    video, and in a screenshot."""
    _, screen = pick([*DOWN, ENTER])
    marked = [row for row in screen.rows if row.strip().startswith(">")]
    assert len(marked) == 1
    assert "topic 2" in marked[0]


def test_escape_cancels_without_choosing():
    chosen, _ = pick(["\x1b"])
    assert chosen is None


def test_ctrl_c_is_an_interrupt_not_a_refusal():
    """`cli.ask` was fixed to keep this distinction: "I changed my mind" is not
    the same as "I picked nothing"."""
    with pytest.raises(KeyboardInterrupt):
        pick(["\x03"])


def test_the_list_does_not_repeat_itself_as_you_move():
    """The bug the line editor had: without tracking how many rows went down,
    every keystroke repaints below the last paint instead of over it."""
    _, screen = pick([*DOWN, *DOWN, *UP, ENTER])
    prompts = [row for row in screen.rows if "Which conversation?" in row]
    assert len(prompts) == 1


def test_a_long_list_scrolls_rather_than_painting_everything():
    """A picker taller than the terminal cannot be repainted — the rows it would
    move back up to have already scrolled away."""
    many = [f"session {i}" for i in range(40)]
    _, screen = pick([*DOWN] * 15 + [ENTER], options=many)

    visible = [row for row in screen.rows if row.strip().startswith(("session", "> session"))]
    assert len(visible) <= keys.VISIBLE_OPTIONS
    assert any("16/40" in row for row in screen.rows)   # says where you are


def test_an_empty_list_chooses_nothing_without_reading_a_key():
    """No keys are scripted, so this would hang if it tried to read one."""
    assert keys.choose("Which?", [], stream=Screen()) is None


def test_without_raw_reads_it_falls_back_to_a_number():
    """A pipe has no arrow keys. Same degradation as `read_line`."""
    screen = Screen(width=60)
    with mock.patch.object(keys, "raw_reads_available", return_value=False):
        with mock.patch("builtins.input", return_value="2"):
            assert keys.choose("Which?", OPTIONS, stream=screen) == 1

    printed = "\n".join(screen.rows)
    assert "1." in printed and "2." in printed      # numbered, so a number means something


@pytest.mark.parametrize("answer", ["", "nonsense", "0", "99"])
def test_a_bad_number_cancels_rather_than_guessing(answer):
    with mock.patch.object(keys, "raw_reads_available", return_value=False):
        with mock.patch("builtins.input", return_value=answer):
            assert keys.choose("Which?", OPTIONS, stream=Screen()) is None


def test_the_picker_prints_nothing_a_windows_console_cannot_encode():
    """It writes straight to the stream, not through `cli.safe_print`, so there
    is nothing to catch a UnicodeEncodeError — and a default Windows console is
    cp1252, which has no arrow glyphs. An "up/down" hint written with ↑↓ would
    crash the picker on the one platform this is written for.
    """
    screen = Screen(width=60)
    pick([*DOWN, ENTER], options=[f"session {i}" for i in range(40)])

    _, screen = pick([ENTER], options=[f"session {i}" for i in range(40)])
    everything = "\n".join(screen.rows)
    everything.encode("cp1252")        # raises if anything is unencodable


# ---------------------------------------------------------------------------
# pick_conversation — the callers, which nothing reached
#
# `mj chat --resume` with no id crashed on its first real use: the signature
# changed and the call site did not follow. Zero tests touched either caller,
# and the reason is worth recording — the branch is guarded by
# `stdin_is_interactive()`, which is the right guard for scripts and also what
# makes it invisible to a test harness. The condition that makes it safe is the
# condition that hides it. The `stream=` seam existed precisely so a test could
# aim it somewhere; it simply had no test using it.
# ---------------------------------------------------------------------------


def saved(home, *stems):
    chats = home / "chats"
    chats.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (chats / f"{stem}.jsonl").write_text(
            '{"role": "user", "content": "hello there", "at": ""}\n', encoding="utf-8"
        )


def test_the_resume_flag_with_no_id_opens_the_picker(isolated_home, monkeypatch):
    """The crash: cmd_chat passed the chat *module* as the output stream."""
    from majordomo import cli

    saved(isolated_home, "20260921-1102", "20260926-1431")
    monkeypatch.setattr(cli, "stdin_is_interactive", lambda: True)

    chosen = {}
    monkeypatch.setattr(
        "majordomo.chat.run",
        lambda config, **kwargs: chosen.update(kwargs),
    )
    # Enter takes the first, which is the newest.
    with mock.patch.object(keys, "choose", return_value=0) as picker:
        cli.main(["chat", "--resume"])

    assert picker.called
    assert chosen["transcript"].name == "20260926-1431.jsonl"
    assert chosen["resume"] is True


def test_cancelling_the_picker_resumes_nothing(isolated_home, monkeypatch, capsys):
    from majordomo import cli

    saved(isolated_home, "20260926-1431")
    monkeypatch.setattr(cli, "stdin_is_interactive", lambda: True)
    monkeypatch.setattr("majordomo.chat.run", lambda *a, **k: pytest.fail("should not open"))

    with mock.patch.object(keys, "choose", return_value=None):
        cli.main(["chat", "--resume"])

    assert "nothing resumed" in capsys.readouterr().out


def test_a_script_passing_resume_still_takes_the_newest(isolated_home, monkeypatch):
    """The guard earns its place: a pipe must not block on a picker."""
    from majordomo import cli

    saved(isolated_home, "20260926-1431")
    monkeypatch.setattr(cli, "stdin_is_interactive", lambda: False)

    seen = {}
    monkeypatch.setattr("majordomo.chat.run", lambda config, **kwargs: seen.update(kwargs))
    with mock.patch.object(keys, "choose", side_effect=AssertionError("no picker")):
        cli.main(["chat", "--resume"])

    assert seen["transcript"] is None      # `run` resolves the newest itself
    assert seen["resume"] is True


def test_the_picker_reports_having_nothing_through_the_caller(isolated_home):
    """`write` is injected, which was the only real reason this lived in cli."""
    from majordomo import session as session_mod

    said = []
    assert session_mod.pick_conversation(said.append) is None
    assert "No saved conversations yet." in said


def test_the_repl_and_the_cli_use_the_same_picker():
    """Two of them would drift, and the second is the one nobody tests."""
    from majordomo import chat, session as session_mod

    assert chat.pick_conversation is session_mod.pick_conversation


def test_the_repl_does_not_import_the_cli():
    """`cli` is the outermost layer — the one module that catches the typed
    exceptions raised below it — and nothing imported it until the picker did,
    which made a chat -> cli -> chat cycle that worked only because the imports
    sat inside functions."""
    import pathlib

    source = pathlib.Path(chat.__file__).read_text(encoding="utf-8")
    assert "import cli" not in source


# --- clamping to the terminal ---------------------------------------------


@pytest.mark.parametrize(
    "lines,expected",
    [(24, 10), (12, 10), (11, 9), (8, 6), (3, 1), (2, 1), (1, 1)],
)
def test_the_window_fits_the_terminal(monkeypatch, lines, expected):
    """A list taller than the terminal cannot be repainted — the rows it would
    move back up to have already scrolled away. The docstring named that hazard
    and the first version measured only the width."""
    monkeypatch.setattr(keys, "_terminal_height", lambda stream=None: lines)
    assert keys._visible_count() == expected


def test_a_short_terminal_paints_within_its_height(monkeypatch):
    """Prompt + options + hint must fit, or the move-up arithmetic lands wrong."""
    monkeypatch.setattr(keys, "_terminal_height", lambda stream=None: 8)
    _, screen = pick([ENTER], options=[f"session {i}" for i in range(40)])
    assert len(screen.rows) <= 8
