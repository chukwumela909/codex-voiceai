"""Tests for the server-side room tone (app/ambience.py).

The Pipecat/Twilio paths otherwise output digitally dead silence between
turns; the mixer keeps faint phone-line tone on the wire. The WAV must match
the transport sample rate exactly (SoundfileMixer does not resample) and be
deterministic so it is cache-stable across restarts.
"""

import wave
from types import SimpleNamespace

from app.ambience import build_room_tone_mixer, ensure_room_tone


def _settings(**overrides):
    base = dict(ambience_enabled=True, ambience_volume=0.035)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_room_tone_wav_matches_requested_rate(tmp_path):
    for rate in (8000, 16000, 24000):
        path = ensure_room_tone(rate, tmp_path)
        with wave.open(str(path), "rb") as wav:
            assert wav.getframerate() == rate
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            assert wav.getnframes() == rate * 8


def test_room_tone_is_deterministic_and_cached(tmp_path):
    first = ensure_room_tone(8000, tmp_path)
    original = first.read_bytes()
    again = ensure_room_tone(8000, tmp_path)
    assert again == first
    assert again.read_bytes() == original


def test_room_tone_has_signal_but_stays_quiet(tmp_path):
    path = ensure_room_tone(8000, tmp_path)
    with wave.open(str(path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    samples = memoryview(frames).cast("h")
    peak = max(abs(s) for s in samples)
    assert 0 < peak <= int(32767 * 0.65)  # audible, with generous headroom


def test_mixer_disabled_by_flag_or_zero_volume(tmp_path):
    assert build_room_tone_mixer(_settings(ambience_enabled=False), 8000, tmp_path) is None
    assert build_room_tone_mixer(_settings(ambience_volume=0), 8000, tmp_path) is None


def test_mixer_built_with_configured_volume(tmp_path):
    mixer = build_room_tone_mixer(_settings(), 8000, tmp_path)
    assert mixer is not None
    assert mixer._volume == 0.035
