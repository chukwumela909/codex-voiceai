"""Tests for the runtime-switchable default character.

Lets the website pick which character the phone/browser uses without an env
change or redeploy: PUT /characters/default persists a pointer that
resolve_default_character_id() honors, and the phone path reads it per call.
"""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("VOICE_AGENT_MODE", "mock")

from app import characters as characters_mod
from app.characters import (
    Character,
    get_persisted_default_id,
    resolve_default_character_id,
    save_character,
    set_persisted_default_id,
)
from app.main import app


client = TestClient(app)


def _seed(directory: Path) -> None:
    save_character(Character(id="zara", name="Mia", role="neighbor"), directory=directory)
    save_character(Character(id="neutral", name="Alex", role="assistant"), directory=directory)


# ---- resolver precedence (pure, no app) -------------------------------------

def test_pointer_round_trip(tmp_path: Path):
    _seed(tmp_path)
    assert get_persisted_default_id(tmp_path) is None
    set_persisted_default_id("neutral", directory=tmp_path)
    assert get_persisted_default_id(tmp_path) == "neutral"


def test_persisted_pointer_wins_over_env(tmp_path: Path):
    _seed(tmp_path)
    set_persisted_default_id("neutral", directory=tmp_path)
    assert resolve_default_character_id(env_default="zara", directory=tmp_path) == "neutral"


def test_env_default_used_when_no_pointer(tmp_path: Path):
    _seed(tmp_path)
    assert resolve_default_character_id(env_default="neutral", directory=tmp_path) == "neutral"


def test_pointer_to_deleted_character_falls_back(tmp_path: Path):
    _seed(tmp_path)
    set_persisted_default_id("neutral", directory=tmp_path)
    (tmp_path / "neutral.json").unlink()  # the chosen default is removed
    # Must not strand on the dead id — falls back to env, which still exists.
    assert resolve_default_character_id(env_default="zara", directory=tmp_path) == "zara"


def test_empty_directory_returns_env_default(tmp_path: Path):
    assert resolve_default_character_id(env_default="zara", directory=tmp_path) == "zara"


# ---- endpoint ---------------------------------------------------------------

def test_set_default_endpoint_persists_and_lists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(characters_mod, "CHARACTERS_DIR", tmp_path)
    _seed(tmp_path)

    resp = client.put("/characters/default", json={"id": "neutral"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["default"] == "neutral"

    # GET /characters badge now reflects the new default, and it persisted to disk.
    assert client.get("/characters").json()["default"] == "neutral"
    assert (tmp_path / characters_mod.DEFAULT_POINTER_NAME).read_text(encoding="utf-8") == "neutral"


def test_set_default_rejects_unknown_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(characters_mod, "CHARACTERS_DIR", tmp_path)
    _seed(tmp_path)
    resp = client.put("/characters/default", json={"id": "ghost"})
    assert resp.status_code == 404


def test_default_route_not_captured_as_character_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # PUT /characters/default must hit the default handler, not upsert a
    # character literally named "default".
    monkeypatch.setattr(characters_mod, "CHARACTERS_DIR", tmp_path)
    _seed(tmp_path)
    client.put("/characters/default", json={"id": "neutral"})
    assert not (tmp_path / "default.json").exists()
