"""Server-side room tone for the Pipecat paths.

The classic browser page synthesizes a local ambience bed, but Pipecat browser
sessions and Twilio calls output digitally dead silence between turns — which
reads as IVR/voicemail, not a person on a line. This module synthesizes a
loopable telephone-band noise WAV (deterministic, no shipped assets) and wires
it into pipecat's output mixer, which mixes it continuously — under speech and
through silence — so the line always has faint life on it.

The tone is generated per output sample rate (browser vs. Twilio 8 kHz)
because ``SoundfileMixer`` requires the file to match the transport rate.
Reuses the ``VOICE_AGENT_AMBIENCE_*`` settings; the scene knob is currently
cosmetic (every scene renders the same phone-line tone).
"""
from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger("voice_agent.ambience")

AMBIENCE_DIR = Path("data") / "ambience"
_LOOP_SECONDS = 8
_PEAK = 0.6  # headroom below full scale; the mixer volume scales from here
_SEED = 1309


def ensure_room_tone(sample_rate: int, directory: Path | None = None) -> Path:
    """Synthesize (once) and return the room-tone WAV for a sample rate.

    Shaped noise via FFT over the whole buffer: circular spectral shaping makes
    the loop seam-free, the band-pass (~150 Hz–3 kHz with a soft pink tilt)
    reads as phone-line hiss rather than white noise. Deterministic for a given
    rate, so the file is cache-stable across restarts.
    """
    base = directory or AMBIENCE_DIR
    path = base / f"room_tone_{sample_rate}.wav"
    if path.exists():
        return path

    n = sample_rate * _LOOP_SECONDS
    rng = np.random.default_rng(_SEED)
    noise = rng.standard_normal(n)

    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    gain = np.ones_like(freqs)
    gain[freqs < 150] = 0.0  # no rumble
    ramp = (freqs >= 150) & (freqs < 300)
    gain[ramp] = (freqs[ramp] - 150.0) / 150.0
    high = freqs > 3000
    gain[high] = np.exp(-(freqs[high] - 3000.0) / 800.0)  # soft top rolloff
    audible = freqs > 0
    gain[audible] *= (300.0 / np.maximum(freqs[audible], 300.0)) ** 0.3  # pink tilt

    shaped = np.fft.irfft(spectrum * gain, n)
    peak = float(np.max(np.abs(shaped))) or 1.0
    samples = (shaped / peak * _PEAK * 32767.0).astype(np.int16)

    base.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".wav.tmp")
    with wave.open(str(tmp), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    tmp.replace(path)
    logger.info("synthesized room tone: %s", path)
    return path


def build_room_tone_mixer(settings, sample_rate: int, directory: Path | None = None):
    """Build the output-transport mixer, or None when ambience is off.

    Best-effort: a missing ``soundfile`` dependency or a synthesis failure
    disables ambience with a warning instead of breaking calls.
    """
    if not getattr(settings, "ambience_enabled", False):
        return None
    volume = float(getattr(settings, "ambience_volume", 0.0) or 0.0)
    if volume <= 0:
        return None
    try:
        from pipecat.audio.mixers.soundfile_mixer import SoundfileMixer

        path = ensure_room_tone(sample_rate, directory)
        return SoundfileMixer(
            sound_files={"room": str(path)},
            default_sound="room",
            volume=volume,
            loop=True,
        )
    except Exception:
        logger.exception("room tone unavailable; continuing without ambience")
        return None
