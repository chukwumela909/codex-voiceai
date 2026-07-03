from types import SimpleNamespace

from app.voice_settings import (
    get_active_voice_id,
    resolve_active_voice_id,
    set_active_voice_id,
)


def test_get_returns_none_when_unset(tmp_path):
    assert get_active_voice_id(tmp_path) is None


def test_set_then_get_round_trips_and_trims(tmp_path):
    stored = set_active_voice_id("  voice_abc  ", directory=tmp_path)

    assert stored == "voice_abc"
    assert get_active_voice_id(tmp_path) == "voice_abc"


def test_empty_value_clears_selection(tmp_path):
    set_active_voice_id("voice_abc", directory=tmp_path)

    assert set_active_voice_id("", directory=tmp_path) is None
    assert get_active_voice_id(tmp_path) is None


def test_clearing_when_unset_is_a_noop(tmp_path):
    assert set_active_voice_id(None, directory=tmp_path) is None
    assert get_active_voice_id(tmp_path) is None


def test_resolve_prefers_override_then_persisted_then_env(tmp_path):
    settings = SimpleNamespace(elevenlabs_voice_id="env_voice")

    # env default when nothing persisted and no override
    assert resolve_active_voice_id(settings, directory=tmp_path) == "env_voice"

    # persisted UI choice beats env default
    set_active_voice_id("ui_voice", directory=tmp_path)
    assert resolve_active_voice_id(settings, directory=tmp_path) == "ui_voice"

    # per-session override beats everything
    assert resolve_active_voice_id(settings, override="session_voice", directory=tmp_path) == "session_voice"


def test_resolve_returns_none_when_nothing_configured(tmp_path):
    settings = SimpleNamespace(elevenlabs_voice_id=None)

    assert resolve_active_voice_id(settings, directory=tmp_path) is None
