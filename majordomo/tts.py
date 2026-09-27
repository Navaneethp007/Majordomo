"""Text-to-speech — speak the briefing aloud.

Vendored from Voicelog's ``tts.py`` with the Riva and OpenAI adapters removed.
The structure is kept exactly as-is, because the two things it gets right are
things worth keeping: the adapter returns raw PCM and shared code does the WAV
stitching and cross-platform playback, so adding Fish Audio or a local engine
later is one function plus one dict entry; and ``_check_playback`` runs *before*
any network call, so a machine with no audio device fails free rather than after
spending an API call.

It was vendored rather than imported because Voicelog's ``speak()`` duck-types a
flat config with ``tts_*`` attributes. Depending on it would have forced
Majordomo's sectioned config to grow a Voicelog-shaped shim.
"""
from __future__ import annotations

from majordomo import install_hint

import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
import wave

import httpx

from majordomo.config import VoiceConfig


class TTSError(Exception):
    """Speech synthesis or playback failed. Always best-effort — never fatal."""


# ---------------------------------------------------------------------------
# Cross-platform playback
# ---------------------------------------------------------------------------

def _linux_player() -> list[str] | None:
    for cmd in (["paplay"], ["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(cmd[0]):
            return cmd
    return None


def _check_playback() -> None:
    """Verify an audio player is available before spending an API call."""
    system = platform.system()
    if system == "Windows":
        try:
            import winsound  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise TTSError("winsound is unavailable on this Windows build") from exc
    elif system == "Darwin":  # pragma: no cover - macOS only
        if not shutil.which("afplay"):
            raise TTSError("no audio player found ('afplay' missing)")
    else:  # pragma: no cover - linux only
        if _linux_player() is None:
            raise TTSError(
                "no audio player found — install one of: pulseaudio (paplay), "
                "alsa-utils (aplay), or ffmpeg (ffplay)"
            )


def _play(path: str) -> None:
    """Play a WAV file, interruptibly. Patchable in tests.

    On Windows this is deliberately asynchronous-plus-poll rather than a plain
    blocking play. ``winsound.PlaySound`` without ``SND_ASYNC`` blocks inside C,
    where Python's signal handler cannot run — so Ctrl+C is queued and only
    raises *after* playback has finished. The CLI prints "Ctrl+C to skip", which
    made that a promise the code could not keep.

    Polling needs an end time, and winsound exposes no "still playing?" query,
    so the duration comes from the WAV header we wrote a moment earlier.
    """
    system = platform.system()
    if system == "Windows":
        import winsound

        try:
            with wave.open(path, "rb") as handle:
                rate = handle.getframerate() or 1
                seconds = handle.getnframes() / float(rate)
        except (OSError, wave.Error):
            # No duration means no deadline to poll against, and a deadline of
            # "now" would return instantly — the caller's `finally` would then
            # delete the file mid-playback and the briefing would stop after a
            # fraction of a second, silently. Fall back to a blocking play: the
            # briefing is heard in full, and the cost is that Ctrl+C cannot
            # interrupt this one. Losing the skip is a far smaller failure than
            # losing the message.
            winsound.PlaySound(path, winsound.SND_FILENAME)
            return

        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
        try:
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                time.sleep(0.05)
        except KeyboardInterrupt:
            # Stop the sound before the caller unwinds, or it keeps playing
            # over whatever is printed next.
            winsound.PlaySound(None, winsound.SND_PURGE)
            raise
        return
    if system == "Darwin":  # pragma: no cover
        subprocess.run(["afplay", path], check=True)
        return
    player = _linux_player()  # pragma: no cover
    if player is None:
        raise TTSError("no audio player found")
    subprocess.run([*player, path], check=True)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def speech_text(markdown: str) -> str:
    """Convert markdown to clean spoken text.

    The fuser is told to return plain prose, but free-tier models drift back
    into bullet points constantly, and hearing a literal "asterisk" read aloud
    is exactly the kind of small failure that makes a voice feature feel cheap.
    """
    text = markdown

    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)

    lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^\s*#{1,6}\s*", "", line)
        line = re.sub(r"^\s*[-*+]\s+", "", line)
        line = line.replace("`", "")
        line = re.sub(r"[*_]", "", line)
        lines.append(line.rstrip())

    text = "\n".join(lines)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_len: int = 400) -> list[str]:
    """Split into chunks <= max_len, breaking at sentence then word boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_len:
        window = remaining[:max_len]
        # Track the boundary position and the cut point separately. Comparing a
        # raw rfind index against a split_at that already had len(sep) added
        # (as the original did) means a later separator can lose to an earlier
        # one, and the chunk breaks at the wrong sentence.
        best_at = -1
        split_at = -1
        for sep in (". ", "! ", "? ", "\n"):
            idx = window.rfind(sep)
            if idx > best_at:
                best_at = idx
                split_at = idx + len(sep)
        if split_at <= 0:
            idx = window.rfind(" ")
            split_at = idx + 1 if idx > 0 else max_len
        piece = remaining[:split_at].strip()
        if piece:
            chunks.append(piece)
        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Provider adapters — each returns (pcm_bytes, sample_rate_hz)
# ---------------------------------------------------------------------------

def _synth_elevenlabs(chunks: list[str], api_key: str, cfg: VoiceConfig) -> tuple[bytes, int]:
    if not cfg.voice_id:
        raise TTSError("ElevenLabs needs voice.voice_id set to a voice id")

    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{cfg.voice_id}"
        "?output_format=pcm_24000"
    )
    headers = {"xi-api-key": api_key}

    pcm = bytearray()
    for piece in chunks:
        try:
            response = httpx.post(
                url,
                headers=headers,
                json={"text": piece, "model_id": cfg.model},
                timeout=cfg.timeout,
            )
            if not response.is_success:
                raise TTSError(f"ElevenLabs HTTP {response.status_code}: {response.text[:200]}")
            pcm.extend(response.content)
        except TTSError:
            raise
        except httpx.HTTPError as exc:
            raise TTSError(f"ElevenLabs request failed: {exc}") from exc

    return bytes(pcm), 24000


def _import_riva():
    """Import riva.client and AudioEncoding. Kept thin so tests can patch it."""
    import riva.client
    from riva.client.proto.riva_audio_pb2 import AudioEncoding

    return riva.client, AudioEncoding


def _synth_nvidia(chunks: list[str], api_key: str, cfg: VoiceConfig) -> tuple[bytes, int]:
    """NVIDIA Riva TTS over NVCF gRPC.

    Ported from Voicelog, where this has been the default engine for a while.
    Unlike the HTTP providers it speaks gRPC, which is why ``nvidia-riva-client``
    is an optional extra rather than a hard dependency — a text-only install
    should not pull in grpc.

    The async future path is deliberate: Riva's synchronous ``synthesize()`` has
    no timeout and can block forever, which in a wake-time briefing would mean a
    scheduled task hanging silently until Windows kills it.
    """
    try:
        riva_client, AudioEncoding = _import_riva()
    except ImportError as exc:
        raise TTSError(
            f"NVIDIA Riva TTS not installed — run: {install_hint('nvidia')}"
        ) from exc

    import grpc  # available whenever riva.client imported

    if not cfg.function_id:
        raise TTSError("NVIDIA Riva needs voice.function_id set to an NVCF function id")

    rate = cfg.sample_rate
    try:
        auth = riva_client.Auth(
            uri="grpc.nvcf.nvidia.com:443",
            use_ssl=True,
            metadata_args=[
                ["function-id", cfg.function_id],
                ["authorization", f"Bearer {api_key}"],
            ],
        )
        service = riva_client.SpeechSynthesisService(auth)

        pcm = bytearray()
        for piece in chunks:
            call = service.synthesize(
                piece,
                voice_name=cfg.voice_id,
                language_code=cfg.language,
                sample_rate_hz=rate,
                encoding=AudioEncoding.LINEAR_PCM,
                future=True,
            )
            try:
                response = call.result(timeout=cfg.timeout)
            except grpc.FutureTimeoutError as exc:
                call.cancel()
                raise TTSError(
                    f"speech synthesis timed out after {cfg.timeout:.0f}s — the TTS "
                    f"service was too slow. Try again, raise voice.timeout, or use "
                    f"--no-speak."
                ) from exc
            pcm.extend(response.audio)
    except TTSError:
        raise
    except Exception as exc:
        raise TTSError(str(exc)) from exc

    return bytes(pcm), rate


#: Adding a local engine later is one function and one entry here.
ADAPTERS = {
    "nvidia": _synth_nvidia,
    "riva": _synth_nvidia,  # alias — Voicelog calls this engine "riva"
    "elevenlabs": _synth_elevenlabs,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def speak(text: str, cfg: VoiceConfig) -> None:
    """Speak ``text`` aloud.

    Raises:
        TTSError: unknown provider, missing key, no audio player, or synthesis
            failure. Callers treat this as best-effort — audio never blocks the
            text output, which has already been printed by the time we get here.
    """
    speech = speech_text(text)
    if not speech:
        return
    chunks = chunk_text(speech)
    if not chunks:
        return

    adapter = ADAPTERS.get(cfg.provider.lower())
    if adapter is None:
        raise TTSError(
            f"unknown voice provider '{cfg.provider}' — use one of: "
            f"{', '.join(sorted(ADAPTERS))}"
        )

    # Fail before spending an API call if we can't play audio here.
    _check_playback()

    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise TTSError(
            f"Set the {cfg.api_key_env} environment variable with your "
            f"{cfg.provider} key."
        )

    pcm, rate = adapter(chunks, api_key, cfg)
    if not pcm:
        return

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    path = tmp.name
    tmp.close()
    try:
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit PCM
            wav.setframerate(rate)
            wav.writeframes(pcm)
        _play(path)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TTSError(f"audio playback failed: {exc}") from exc
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
