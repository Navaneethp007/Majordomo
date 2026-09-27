"""Speech to text — the other half of the voice loop.

``tts.py`` turns text into speech; this turns speech into text. Together with
``chat.send`` in the middle they make a conversation you can have out loud:

    you speak → asr.listen() → chat.send() → tts.speak() → it answers

Deliberately shaped like ``tts.py``, because the same three things matter and
were already solved there:

- **An adapter per provider**, returning plain text, with a dict at the bottom.
  Adding Whisper later is one function and one entry.
- **Check the hardware before spending an API call.** ``tts._check_playback``
  fails free on a machine with no speaker; this fails free on one with no
  microphone, rather than after recording and uploading.
- **Riva speaks gRPC**, which is why ``nvidia-riva-client`` is an optional
  extra. The ``Auth`` block here is identical to the TTS one bar the function
  id — same endpoint, same key, different NVCF function.

── ON RECORDING ─────────────────────────────────────────────────────────────
Capture stops on silence rather than on a keypress. A press-to-start,
press-to-stop design needs the terminal in raw mode for the whole recording,
which fights with the REPL that called us; listening for a pause is both
simpler and closer to how speaking actually feels.

There is always a hard cap as well. Silence detection can be defeated by a noisy
room, and a recorder that never stops is worse than one that stops too early.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

from majordomo import install_hint

import os

from majordomo.config import VoiceConfig


class ASRError(Exception):
    """Capture or transcription failed. Always best-effort — never fatal.

    The caller falls back to typing, which is why this never propagates past
    the REPL: losing a spoken message should cost you the message, not the
    conversation.
    """


class MicrophoneUnavailable(ASRError):
    """No input device, or the capture library is not installed."""


#: Frames per read. 30ms at 16kHz — short enough that silence is noticed
#: promptly, long enough that RMS over the window is meaningful.
CHUNK_MS = 30

#: Fallback threshold when calibration cannot run. A guess, and the reason
#: calibration exists — a fixed number is a claim about a microphone's gain that
#: nobody measured. On one Realtek array mic the room floor sits near 35 while
#: normal speech can sit below 500, so this cut every sentence off mid-word.
SILENCE_RMS = 500

#: How far above the measured noise floor a chunk must be to count as speech.
#: Low enough that a soft filler still registers, high enough that room tone
#: does not hold the recorder open forever.
NOISE_MULTIPLIER = 4.0

#: The threshold never goes below this. Measured: one quiet room's ambient runs
#: 18–87 with a median of 37, so a threshold much under 100 makes an empty room
#: read as continuous speech and the recorder never stops.
MIN_SILENCE_RMS = 100

#: How long to listen to the room before recording. Long enough not to land in
#: one unusually quiet moment and derive a threshold from it.
CALIBRATE_MS = 500

#: Stop after this much continuous quiet once speech has been heard.
#:
#: This was 900ms, which is fine for dictation and wrong for speech. People
#: pause mid-sentence to think — around fillers, between clauses — and 900ms of
#: quiet is a normal part of a sentence, not the end of one. Cutting someone off
#: mid-thought is a worse failure than waiting a beat too long, so the default
#: is generous and ``voice.trailing_silence_ms`` tunes it.
TRAILING_SILENCE_MS = 1_800

#: Give up waiting if nothing is ever said.
LEADING_SILENCE_MS = 6_000


def _rms(block: bytes) -> float:
    """Root-mean-square level of a 16-bit mono buffer.

    Replaces ``audioop.rms``. ``audioop`` was removed from the standard library
    in Python 3.13 and ``requires-python`` permits that install, so a module-level
    import of it was a future ``ModuleNotFoundError` that would have killed the
    chat REPL rather than degrading to typing. One arithmetic loop is a smaller
    thing to own than a dependency that no longer exists.
    """
    if not block:
        return 0.0
    samples = memoryview(block).cast("h")
    total = 0
    for sample in samples:
        total += sample * sample
    return (total / len(samples)) ** 0.5


def _calibrate(stream, chunk: int, rate: int) -> float:
    """Learn this room's noise floor, and derive a threshold from it.

    A fixed threshold assumes a microphone gain. Get that wrong and there is no
    partial failure: either every chunk of your speech reads as silence and the
    recording stops mid-sentence, or the room never reads as silence and it runs
    to the hard cap. Both were observed with a single hard-coded 500.

    Uses the median rather than the mean, so one door slam does not set the
    threshold for the whole recording. Not the lower quartile, which was tried
    first: it sits *below* typical room tone, and a threshold under the room's
    own noise means nothing ever reads as silence and recording runs to the
    hard cap every time.
    """
    samples: list[float] = []
    for _ in range(max(1, int(CALIBRATE_MS / CHUNK_MS))):
        try:
            block, _overflowed = stream.read(chunk)
        except Exception:
            break
        samples.append(_rms(bytes(block)))

    if not samples:
        return float(SILENCE_RMS)

    samples.sort()
    floor = samples[len(samples) // 2]
    return max(float(MIN_SILENCE_RMS), floor * NOISE_MULTIPLIER)


def _open_stream(cfg: VoiceConfig):
    """Open the capture stream, and say what to do when it will not.

    Shared by ``record`` and ``measure``. That sharing is not tidiness: ``mj
    mic`` exists to report the numbers ``record`` actually uses, so if the two
    opened the microphone differently — a different rate, a different chunk
    size — the diagnosis would describe a stream nobody records with.

    Returns ``(stream, chunk_frames)``.
    """
    try:
        sounddevice = _import_sounddevice()
    except ImportError as exc:
        raise MicrophoneUnavailable(
            f"voice input needs an extra package — run: {install_hint('voice')}"
        ) from exc

    rate = cfg.asr_sample_rate
    chunk = int(rate * CHUNK_MS / 1000)
    try:
        stream = sounddevice.RawInputStream(
            samplerate=rate, blocksize=chunk, dtype="int16", channels=1
        )
    except Exception as exc:
        raise MicrophoneUnavailable(f"could not open the microphone: {exc}") from exc
    return stream, chunk


def measure(cfg: VoiceConfig, seconds: float = 10.0, write=print) -> dict:
    """Record without transcribing and report the levels. For tuning.

    Voice failing is almost always one number being wrong for one microphone,
    and that is invisible from the outside: the recording simply stops early,
    or never stops. This makes the number visible.
    """
    stream, chunk = _open_stream(cfg)
    rate = cfg.asr_sample_rate
    levels: list[float] = []
    with stream:
        threshold = float(cfg.silence_rms) if cfg.silence_rms else _calibrate(
            stream, chunk, rate
        )
        write(f"Noise floor measured. Speech must exceed {threshold:.0f}.")
        write(f"Talk normally for {seconds:.0f}s — pauses and fillers included.")
        for _ in range(int(seconds * 1000 / CHUNK_MS)):
            try:
                block, _overflowed = stream.read(chunk)
            except Exception as exc:
                raise ASRError(f"recording failed: {exc}") from exc
            levels.append(_rms(bytes(block)))

    if not levels:
        raise ASRError("captured nothing")

    ordered = sorted(levels)

    def at(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]

    longest = run = 0
    for value in levels:
        run = run + 1 if value < threshold else 0
        longest = max(longest, run)

    return {
        "threshold": threshold,
        "min": ordered[0],
        "p10": at(0.10),
        "p50": at(0.50),
        "p90": at(0.90),
        "max": ordered[-1],
        "quiet_fraction": sum(1 for v in levels if v < threshold) / len(levels),
        "longest_quiet_ms": longest * CHUNK_MS,
        "stops_at_ms": cfg.trailing_silence_ms or TRAILING_SILENCE_MS,
    }


def _import_sounddevice():
    """Import the capture library. Kept thin so tests can patch it."""
    import sounddevice

    return sounddevice


def check_microphone() -> None:
    """Fail before recording or spending an API call. Mirrors _check_playback.

    Raises:
        MicrophoneUnavailable: no capture library, or no input device.
    """
    try:
        sounddevice = _import_sounddevice()
    except ImportError as exc:
        raise MicrophoneUnavailable(
            f"voice input needs an extra package — run: {install_hint('voice')}"
        ) from exc

    try:
        devices = sounddevice.query_devices()
        if not any(d.get("max_input_channels", 0) > 0 for d in devices):
            raise MicrophoneUnavailable("no microphone found on this machine")
    except MicrophoneUnavailable:
        raise
    except Exception as exc:
        raise MicrophoneUnavailable(f"could not query audio devices: {exc}") from exc


def record(cfg: VoiceConfig, on_start=None) -> tuple[bytes, int]:
    """Capture until you stop talking. Returns (pcm_bytes, sample_rate).

    ``on_start`` is called once the stream is actually open, so the caller can
    print "listening…" at the moment it becomes true rather than a moment
    before — the gap is small but it is the difference between a prompt you
    trust and one you learn to ignore.
    """
    rate = cfg.asr_sample_rate
    trailing_ms = cfg.trailing_silence_ms or TRAILING_SILENCE_MS
    trailing_needed = max(1, trailing_ms // CHUNK_MS)
    leading_allowed = LEADING_SILENCE_MS // CHUNK_MS
    max_chunks = int(cfg.listen_max_seconds * 1000 / CHUNK_MS)

    frames: list[bytes] = []
    quiet_run = 0
    heard_speech = False

    stream, chunk = _open_stream(cfg)

    with stream:
        # Before "listening…" appears, so the room is sampled while nobody is
        # deliberately talking. An explicit setting skips it entirely — if you
        # have measured your own microphone, that beats anything derived here.
        quiet_below = float(cfg.silence_rms) if cfg.silence_rms else _calibrate(
            stream, chunk, rate
        )

        if on_start is not None:
            on_start()
        for _ in range(max_chunks):
            try:
                block, _overflowed = stream.read(chunk)
            except Exception as exc:
                raise ASRError(f"recording failed: {exc}") from exc

            data = bytes(block)
            frames.append(data)

            if _rms(data) >= quiet_below:
                heard_speech = True
                quiet_run = 0
                continue

            quiet_run += 1
            if heard_speech and quiet_run >= trailing_needed:
                break
            if not heard_speech and quiet_run >= leading_allowed:
                break

    if not heard_speech:
        # Naming the threshold turns "it didn't work" into something you can act
        # on: if this number is far above your speaking level, voice.silence_rms
        # is the setting to change.
        raise ASRError(f"heard nothing above a level of {quiet_below:.0f}")

    return b"".join(frames), rate


# ---------------------------------------------------------------------------
# Provider adapters — each takes PCM and returns text
# ---------------------------------------------------------------------------


def _transcribe_nvidia(pcm: bytes, rate: int, api_key: str, cfg: VoiceConfig) -> str:
    """NVIDIA Riva ASR over NVCF gRPC. Same auth shape as the TTS adapter."""
    try:
        import riva.client
        from riva.client.proto.riva_audio_pb2 import AudioEncoding
    except ImportError as exc:
        raise ASRError(
            f"NVIDIA Riva ASR not installed — run: {install_hint('nvidia')}"
        ) from exc

    if not cfg.asr_function_id:
        raise ASRError(
            "NVIDIA Riva ASR needs voice.asr_function_id set to an NVCF function "
            "id — find it on the model's page at build.nvidia.com. It is a "
            "different function from the TTS one."
        )

    try:
        auth = riva.client.Auth(
            uri="grpc.nvcf.nvidia.com:443",
            use_ssl=True,
            metadata_args=[
                ["function-id", cfg.asr_function_id],
                ["authorization", f"Bearer {api_key}"],
            ],
        )
        service = riva.client.ASRService(auth)
        config = riva.client.RecognitionConfig(
            encoding=AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=rate,
            language_code=cfg.asr_language,
            max_alternatives=1,
            enable_automatic_punctuation=True,
        )
        response = service.offline_recognize(pcm, config)
    except ASRError:
        raise
    except Exception as exc:
        raise ASRError(str(exc)) from exc

    for result in getattr(response, "results", []):
        alternatives = getattr(result, "alternatives", [])
        if alternatives:
            text = (alternatives[0].transcript or "").strip()
            if text:
                return text

    raise ASRError("nothing recognised in the recording")


#: Adding Whisper later is one function and one entry here.
ADAPTERS = {
    "nvidia": _transcribe_nvidia,
    "riva": _transcribe_nvidia,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def listen(cfg: VoiceConfig, on_start=None) -> str:
    """Record one utterance and return what was said.

    Raises:
        ASRError: no microphone, nothing said, or transcription failed. The
            caller falls back to typing.
    """
    provider = (cfg.asr_provider or cfg.provider).lower()
    adapter = ADAPTERS.get(provider)
    if adapter is None:
        raise ASRError(
            f"unknown speech provider '{provider}' — use one of: "
            f"{', '.join(sorted(ADAPTERS))}"
        )

    # Before anything is recorded or uploaded.
    check_microphone()

    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise ASRError(
            f"Set the {cfg.api_key_env} environment variable with your "
            f"{provider} key."
        )

    pcm, rate = record(cfg, on_start=on_start)
    return adapter(pcm, rate, api_key, cfg)
