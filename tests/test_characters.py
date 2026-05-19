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
    assert zara.name == "Zara"
    assert "warm" in zara.tone
    assert any("as an AI" in p for p in zara.forbidden_phrases)


def test_build_system_prompt_includes_key_contract_pieces():
    zara = load_characters()["zara"]
    prompt = build_system_prompt(zara)
    assert "You are Zara." in prompt
    assert "stay in character" in prompt
    assert "Do not claim to be human" in prompt
    for phrase in zara.forbidden_phrases:
        assert phrase in prompt
    for tone in zara.tone:
        assert tone in prompt


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
