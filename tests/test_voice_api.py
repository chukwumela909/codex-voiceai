import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("VOICE_AGENT_MODE", "mock")

import app.elevenlabs_tts as elevenlabs_tts
import app.main as main_mod
import app.voice_settings as voice_settings
from app.config import Settings
from app.main import app

client = TestClient(app)


@pytest.fixture()
def isolated_voice_dir(tmp_path, monkeypatch):
    # Keep the persisted-voice file out of the real repo `data/` dir.
    monkeypatch.setattr(voice_settings, "VOICE_STATE_DIR", tmp_path)
    return tmp_path


def test_get_voices_without_key_warns_and_returns_empty(monkeypatch, isolated_voice_dir):
    monkeypatch.setattr(main_mod, "settings", Settings(_env_file=None))  # mock mode, no key

    body = client.get("/voices").json()

    assert body["voices"] == []
    assert body["active"] is None
    assert "ELEVENLABS_API_KEY" in body["warning"]


def test_get_voices_lists_account_voices(monkeypatch, isolated_voice_dir):
    monkeypatch.setattr(
        main_mod, "settings", Settings(_env_file=None, ELEVENLABS_API_KEY="k", VOICE_AGENT_MODE="live")
    )

    async def fake_list_voices(api_key, **kwargs):
        assert api_key == "k"
        return [{"voice_id": "v1", "name": "Rachel", "category": "premade"}]

    monkeypatch.setattr(elevenlabs_tts, "list_voices", fake_list_voices)

    body = client.get("/voices").json()

    assert body["voices"] == [{"voice_id": "v1", "name": "Rachel", "category": "premade"}]
    assert body["warning"] is None


def test_get_voices_survives_fetch_failure(monkeypatch, isolated_voice_dir):
    monkeypatch.setattr(
        main_mod, "settings", Settings(_env_file=None, ELEVENLABS_API_KEY="k", VOICE_AGENT_MODE="live")
    )

    async def boom(api_key, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(elevenlabs_tts, "list_voices", boom)

    body = client.get("/voices").json()

    assert body["voices"] == []
    assert "network down" in body["warning"]


def test_put_voice_persists_and_get_reflects_it(monkeypatch, isolated_voice_dir):
    monkeypatch.setattr(main_mod, "settings", Settings(_env_file=None))

    put = client.put("/voice", json={"voice_id": "  my_voice  "}).json()
    assert put["active"] == "my_voice"
    assert put["persisted"] == "my_voice"

    got = client.get("/voice").json()
    assert got["active"] == "my_voice"
    assert got["persisted"] == "my_voice"


def test_put_empty_voice_clears_selection_and_falls_back_to_env(monkeypatch, isolated_voice_dir):
    monkeypatch.setattr(
        main_mod, "settings", Settings(_env_file=None, ELEVENLABS_VOICE_ID="env_voice")
    )

    client.put("/voice", json={"voice_id": "temp"})
    cleared = client.put("/voice", json={"voice_id": ""}).json()

    assert cleared["persisted"] is None
    assert cleared["active"] == "env_voice"  # falls back to env default


def test_put_voice_rejects_non_string(monkeypatch, isolated_voice_dir):
    monkeypatch.setattr(main_mod, "settings", Settings(_env_file=None))

    resp = client.put("/voice", json={"voice_id": 123})

    assert resp.status_code == 400
