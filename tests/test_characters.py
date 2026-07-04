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


def test_jimmy_character_loads_from_disk():
    chars = load_characters()
    assert "jimmy" in chars
    jimmy = chars["jimmy"]
    # name is user-editable via the Studio; assert it is set, not a fixed value.
    assert jimmy.name
    assert "warm" in jimmy.tone
    assert any("as an AI" in p for p in jimmy.forbidden_phrases)


def test_build_system_prompt_includes_key_contract_pieces():
    jimmy = load_characters()["jimmy"]
    prompt = build_system_prompt(jimmy)
    assert f"You are {jimmy.name}." in prompt
    assert "stay in character" in prompt
    assert "Do not claim to be human" in prompt
    for phrase in jimmy.forbidden_phrases:
        assert phrase in prompt
    for tone in jimmy.tone:
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


def test_get_characters_endpoint_returns_jimmy_default():
    response = client.get("/characters")
    assert response.status_code == 200
    body = response.json()
    assert body["default"] == "jimmy"
    ids = {c["id"] for c in body["characters"]}
    assert "jimmy" in ids


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
    assert session.character.id == "jimmy"

    new_char = session.set_character("neutral")
    assert new_char.id == "neutral"
    # Cached agent should be cleared so the next _ensure_agent rebuilds with new prompt.
    assert session.agent is None


def test_build_system_prompt_includes_life_canon():
    char = Character(
        id="demo",
        name="Demo",
        role="test person",
        backstory="You're Demo, a fictional test person.",
        profile_facts=[{"label": "Age", "value": "40"}, {"label": "City", "value": "Austin"}],
        likes=["tacos"],
        dislikes=["drama"],
        values=["honesty"],
        stories=[{"title": "The move", "content": "Moved to Austin in his twenties."}],
        boundaries=["Never discuss the weather"],
        caller_relationship="An old friend calling to catch up.",
        conversation_setting="A relaxed evening phone call.",
        caller_goals=["Catch up warmly"],
    )
    prompt = build_system_prompt(char)
    assert "You're Demo, a fictional test person." in prompt
    assert "Age: 40" in prompt
    assert "City: Austin" in prompt
    assert "tacos" in prompt
    assert "drama" in prompt
    assert "honesty" in prompt
    assert "The move: Moved to Austin in his twenties." in prompt
    assert "Never discuss the weather" in prompt
    assert "An old friend calling to catch up." in prompt
    assert "A relaxed evening phone call." in prompt
    assert "Catch up warmly" in prompt
    # The anti-drift grounding rule appears whenever there is life canon to stay true to.
    assert "Stay grounded" in prompt


def test_build_system_prompt_omits_canon_block_when_absent():
    # A style-only character (no life canon) must not get the grounding directive
    # or empty canon headers.
    char = Character(id="bare", name="Bare", role="assistant")
    prompt = build_system_prompt(char)
    assert "Stay grounded" not in prompt
    assert "Key facts about your life" not in prompt
    # The base identity contract is still present.
    assert "You are Bare." in prompt
    assert "Do not claim to be human" in prompt


def test_new_canon_fields_round_trip_through_rest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(characters_mod, "CHARACTERS_DIR", tmp_path)
    save_character(Character(id="jimmy", name="Jimmy", role="seed"), directory=tmp_path)

    payload = {
        "name": "Jimmy",
        "role": "a guy on the phone",
        "backstory": "You're Jimmy, 47.",
        "profile_facts": [{"label": "Age", "value": "47"}],
        "likes": ["tacos"],
        "dislikes": ["drama"],
        "values": ["trust"],
        "stories": [{"title": "The band", "content": "Joined as a teen."}],
        "boundaries": ["Don't discuss dad"],
        "caller_relationship": "A friend.",
        "conversation_setting": "A call.",
        "caller_goals": ["Be good company"],
    }
    put = client.put("/characters/jimmy", json=payload)
    assert put.status_code == 200, put.text

    saved = next(c for c in client.get("/characters").json()["characters"] if c["id"] == "jimmy")
    assert saved["backstory"] == "You're Jimmy, 47."
    assert saved["profile_facts"] == [{"label": "Age", "value": "47"}]
    assert saved["stories"] == [{"title": "The band", "content": "Joined as a teen."}]
    assert saved["boundaries"] == ["Don't discuss dad"]
    assert saved["caller_relationship"] == "A friend."
    assert saved["caller_goals"] == ["Be good company"]
