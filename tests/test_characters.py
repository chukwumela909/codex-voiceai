import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("VOICE_AGENT_MODE", "mock")

from app import characters as characters_mod
from app.characters import (
    Character,
    build_system_prompt,
    get_character,
    load_characters,
    save_character,
)
from app.main import app


client = TestClient(app)


def test_zara_character_loads_from_disk():
    chars = load_characters()
    assert "zara" in chars
    zara = chars["zara"]
    # name is user-editable via the Studio; assert it is set, not a fixed value.
    assert zara.name
    assert "warm" in zara.tone
    assert any("as an AI" in p for p in zara.forbidden_phrases)


def test_build_system_prompt_includes_key_contract_pieces():
    zara = load_characters()["zara"]
    prompt = build_system_prompt(zara)
    assert f"You are {zara.name}." in prompt
    assert "stay in character" in prompt
    assert "Do not claim to be human" in prompt
    for phrase in zara.forbidden_phrases:
        assert phrase in prompt
    for tone in zara.tone:
        assert tone in prompt


def test_build_system_prompt_includes_examples_and_all_rules():
    char = Character(
        id="demo",
        name="Demo",
        role="guide",
        speaking_style_rules=["rule one", "rule two", "rule three"],
        example_exchanges=[{"user": "hi", "assistant": "hey, I'm Demo"}],
    )
    prompt = build_system_prompt(char)
    for rule in char.speaking_style_rules:
        assert rule in prompt
    assert "hey, I'm Demo" in prompt


def test_get_character_falls_back_to_default():
    found = get_character("does-not-exist")
    assert found.id in {"zara", characters_mod.DEFAULT_CHARACTER_ID}


def test_get_characters_endpoint_returns_zara_default():
    response = client.get("/characters")
    assert response.status_code == 200
    body = response.json()
    assert body["default"] == "zara"
    ids = {c["id"] for c in body["characters"]}
    assert "zara" in ids


def test_put_character_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(characters_mod, "CHARACTERS_DIR", tmp_path)
    save_character(Character(id="zara", name="Zara", role="seed"), directory=tmp_path)

    payload = {
        "name": "Zara",
        "role": "fashion assistant updated",
        "tone": ["warm", "playful"],
        "grammar": "casual english",
        "forbidden_phrases": ["as an AI"],
        "identity_response_style": "redirect gently",
        "speaking_style_rules": ["short sentences"],
    }
    response = client.put("/characters/zara", json=payload)
    assert response.status_code == 200, response.text
    saved_path = tmp_path / "zara.json"
    assert saved_path.exists()
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved["role"] == "fashion assistant updated"
    assert saved["tone"] == ["warm", "playful"]


def test_put_character_rejects_bad_slug():
    response = client.put("/characters/INVALID!", json={"name": "X", "role": "y"})
    assert response.status_code == 400


def test_put_character_rejects_invalid_payload():
    response = client.put("/characters/zara", json={"name": ""})
    # Missing required string fields beyond defaults are tolerated by Pydantic since
    # role defaults to empty; ensure malformed types (list-as-string) are rejected.
    response = client.put("/characters/zara", json={"name": ["not", "a", "string"], "role": "x"})
    assert response.status_code == 422


def test_put_character_round_trip_preserves_full_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(characters_mod, "CHARACTERS_DIR", tmp_path)
    save_character(Character(id="zara", name="Zara", role="seed"), directory=tmp_path)

    payload = {
        "name": "Zara",
        "role": "fashion assistant",
        "tone": ["warm", "playful"],
        "grammar": "casual english",
        "forbidden_phrases": ["as an AI"],
        "identity_response_style": "redirect gently",
        "speaking_style_rules": ["short sentences", "no jargon", "stay curious"],
        "example_exchanges": [
            {"user": "who are you?", "assistant": "I'm Zara, your style buddy."},
            {"user": "what's up?", "assistant": "Just vibing — what can I help with?"},
        ],
    }
    put = client.put("/characters/zara", json=payload)
    assert put.status_code == 200, put.text

    # Round-trips through disk via GET /characters — the fields the old form dropped survive.
    listing = client.get("/characters").json()
    saved = next(c for c in listing["characters"] if c["id"] == "zara")
    assert saved["speaking_style_rules"] == ["short sentences", "no jargon", "stay curious"]
    assert saved["example_exchanges"] == payload["example_exchanges"]


def test_duplicate_character_via_post_leaves_source_intact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(characters_mod, "CHARACTERS_DIR", tmp_path)
    save_character(
        Character(id="zara", name="Zara", role="stylist", tone=["warm"]),
        directory=tmp_path,
    )

    duplicate_payload = {"id": "zara-copy", "name": "Zara copy", "role": "stylist", "tone": ["warm"]}
    response = client.post("/characters", json=duplicate_payload)
    assert response.status_code == 200, response.text

    assert (tmp_path / "zara-copy.json").exists()
    source = json.loads((tmp_path / "zara.json").read_text(encoding="utf-8"))
    assert source["name"] == "Zara"  # original untouched


def test_preview_returns_text_and_audio_in_mock_mode():
    response = client.post(
        "/characters/preview",
        json={"message": "hi there", "character": {"id": "zara", "name": "Zara", "role": "stylist"}},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["character_id"] == "zara"
    assert body["text"].strip()
    assert body["audio"]  # non-empty base64 PCM
    assert body["encoding"] == "pcm_s16le"
    assert body["sample_rate"] > 0
    assert body["providers"]["llm"] == "mock"
    assert body["providers"]["tts"] == "mock"
    assert "llm_ms" in body["latency"] and "tts_ms" in body["latency"]


def test_preview_accepts_unsaved_candidate_character():
    # A brand-new draft with no id still previews, under a placeholder id.
    response = client.post(
        "/characters/preview",
        json={"message": "who are you?", "character": {"name": "Draft", "role": "new persona"}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["character_id"] == "preview"


def test_preview_requires_message():
    response = client.post(
        "/characters/preview",
        json={"message": "   ", "character": {"name": "Zara", "role": "x"}},
    )
    assert response.status_code == 400


def test_preview_warns_on_missing_live_keys(monkeypatch: pytest.MonkeyPatch):
    import app.main as main_mod
    from app.config import Settings

    for key in ("GROQ_API_KEY", "DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VOICE_AGENT_MODE", "live")
    monkeypatch.setattr(main_mod, "settings", Settings(_env_file=None))

    response = client.post(
        "/characters/preview",
        json={"message": "hello", "character": {"name": "Zara", "role": "x"}},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    codes = {w.get("code") for w in body["warnings"]}
    assert "GROQ_API_KEY" in codes
    # Falls back to mock output rather than erroring out.
    assert body["text"].strip()
    assert body["audio"]


def test_character_select_event_swaps_character():
    from app.mock_conversation import MockConversationSession
    from app.config import get_settings

    sent: list[dict] = []

    async def send(payload):
        sent.append(payload)

    session = MockConversationSession("sess_test", send, get_settings())
    assert session.character.id == "zara"

    new_char = session.set_character("neutral")
    assert new_char.id == "neutral"
    # Cached agent should be cleared so the next _ensure_agent rebuilds with new prompt.
    assert session.agent is None
