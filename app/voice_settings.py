"""Runtime-selectable active ElevenLabs voice.

The voice id can be chosen from the UI instead of the env. The choice is
persisted server-side (a single small file, atomic write) so it survives
restarts and applies to every path — browser calls, Twilio phone calls, and the
Studio preview. The env ``ELEVENLABS_VOICE_ID`` is the fallback default when no
voice has been chosen. Mirrors the default-character pointer pattern in
``app/characters.py``.
"""
from __future__ import annotations

from pathlib import Path

VOICE_STATE_DIR = Path("data")
ACTIVE_VOICE_FILE = "active_voice_id"


def _voice_path(directory: Path | None = None) -> Path:
    return (directory or VOICE_STATE_DIR) / ACTIVE_VOICE_FILE


def get_active_voice_id(directory: Path | None = None) -> str | None:
    """Read the persisted active voice id, or None if unset/unreadable."""
    try:
        text = _voice_path(directory).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def set_active_voice_id(voice_id: str | None, directory: Path | None = None) -> str | None:
    """Persist (or clear) the active voice id atomically. Returns the stored value.

    An empty/None ``voice_id`` clears the selection so resolution falls back to
    the env default.
    """
    base = directory or VOICE_STATE_DIR
    path = _voice_path(base)
    normalized = (voice_id or "").strip()
    if not normalized:
        try:
            path.unlink()
        except OSError:
            pass
        return None
    base.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{ACTIVE_VOICE_FILE}.tmp")
    tmp.write_text(normalized, encoding="utf-8")
    tmp.replace(path)
    return normalized


def resolve_active_voice_id(
    settings, *, override: str | None = None, directory: Path | None = None
) -> str | None:
    """Resolve the voice to use, in precedence order.

    per-session override (e.g. ``?voice=`` on connect) -> persisted UI choice ->
    env ``ELEVENLABS_VOICE_ID``.
    """
    if override and override.strip():
        return override.strip()
    persisted = get_active_voice_id(directory)
    if persisted:
        return persisted
    return getattr(settings, "elevenlabs_voice_id", None)
