import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("VOICE_AGENT_MODE", "mock")

import app.llm_models as llm_models
import app.main as main_mod
from app.config import Settings
from app.main import app

client = TestClient(app)


@pytest.fixture()
def isolated_model_dir(tmp_path, monkeypatch):
    # Keep the persisted active-model file out of the real repo `data/` dir.
    monkeypatch.setattr(llm_models, "MODEL_STATE_DIR", tmp_path)
    return tmp_path


def test_list_models_returns_registry_and_active(monkeypatch, isolated_model_dir):
    monkeypatch.setattr(main_mod, "settings", Settings(_env_file=None))

    body = client.get("/models").json()

    keys = [m["key"] for m in body["models"]]
    assert "groq-llama-3.1-8b" in keys
    assert any(m["provider"] == "openrouter" for m in body["models"])
    assert body["active"] == "groq-llama-3.1-8b"  # default when nothing persisted
    assert body["openrouter_configured"] is False  # no key in a bare env


def test_openrouter_configured_reflects_key(monkeypatch, isolated_model_dir):
    monkeypatch.setattr(
        main_mod, "settings", Settings(_env_file=None, OPENROUTER_API_KEY="or-key")
    )
    assert client.get("/models").json()["openrouter_configured"] is True


def test_put_model_persists_and_list_reflects_it(monkeypatch, isolated_model_dir):
    monkeypatch.setattr(main_mod, "settings", Settings(_env_file=None))

    put = client.put("/model", json={"model": "or-gpt-4o-mini"}).json()
    assert put["active"] == "or-gpt-4o-mini"
    assert put["persisted"] == "or-gpt-4o-mini"

    assert client.get("/models").json()["active"] == "or-gpt-4o-mini"


def test_put_empty_model_clears_and_falls_back_to_default(monkeypatch, isolated_model_dir):
    monkeypatch.setattr(main_mod, "settings", Settings(_env_file=None))

    client.put("/model", json={"model": "or-gpt-4o-mini"})
    cleared = client.put("/model", json={"model": ""}).json()

    assert cleared["persisted"] is None
    assert cleared["active"] == "groq-llama-3.1-8b"


def test_put_bare_word_model_is_rejected(monkeypatch, isolated_model_dir):
    monkeypatch.setattr(main_mod, "settings", Settings(_env_file=None))

    # No slash and not a preset key -> not a usable model id.
    resp = client.put("/model", json={"model": "not-a-real-model"})
    assert resp.status_code == 400


def test_put_raw_openrouter_slug_is_accepted(monkeypatch, isolated_model_dir):
    monkeypatch.setattr(main_mod, "settings", Settings(_env_file=None))

    put = client.put("/model", json={"model": "anthropic/claude-sonnet-4.5"}).json()
    assert put["active"] == "anthropic/claude-sonnet-4.5"
    assert put["persisted"] == "anthropic/claude-sonnet-4.5"
    # /models still lists the curated presets; active reflects the pasted slug.
    listing = client.get("/models").json()
    assert listing["active"] == "anthropic/claude-sonnet-4.5"


def test_openrouter_catalog_endpoint(monkeypatch, isolated_model_dir):
    monkeypatch.setattr(
        main_mod, "settings", Settings(_env_file=None, OPENROUTER_API_KEY="k")
    )

    async def fake_catalog(api_key=None, **kwargs):
        return [{"id": "anthropic/claude-sonnet-4.5", "name": "Claude Sonnet 4.5"}]

    monkeypatch.setattr(llm_models, "list_openrouter_models", fake_catalog)

    body = client.get("/openrouter/models").json()
    assert body["models"][0]["id"] == "anthropic/claude-sonnet-4.5"
    assert body["warning"] is None


def test_openrouter_catalog_warns_without_key(monkeypatch, isolated_model_dir):
    monkeypatch.setattr(main_mod, "settings", Settings(_env_file=None))

    async def fake_catalog(api_key=None, **kwargs):
        return [{"id": "openai/gpt-4o-mini", "name": "GPT-4o mini"}]

    monkeypatch.setattr(llm_models, "list_openrouter_models", fake_catalog)

    body = client.get("/openrouter/models").json()
    assert "OPENROUTER_API_KEY" in body["warning"]


def test_put_model_rejects_non_string(monkeypatch, isolated_model_dir):
    monkeypatch.setattr(main_mod, "settings", Settings(_env_file=None))

    assert client.put("/model", json={"model": 5}).status_code == 400
