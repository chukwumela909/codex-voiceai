from types import SimpleNamespace

import pytest

from app.llm_models import (
    MODELS,
    build_llm_service,
    default_model_key,
    get_active_model,
    model_entry_for,
    resolve_active_model_key,
    resolve_model,
    set_active_model,
)


def _settings(**overrides):
    base = dict(
        default_model="groq-llama-3.1-8b",
        groq_api_key="groq-key",
        openrouter_api_key="or-key",
        groq_temperature=0.6,
        groq_max_tokens=200,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_default_model_key_falls_back_when_unknown():
    assert default_model_key(_settings(default_model="nope")) == "groq-llama-3.1-8b"
    assert default_model_key(_settings(default_model="or-gpt-4o-mini")) == "or-gpt-4o-mini"


def test_get_returns_none_when_unset(tmp_path):
    assert get_active_model(tmp_path) is None


def test_set_then_get_round_trips(tmp_path):
    assert set_active_model("or-gpt-4o-mini", directory=tmp_path) == "or-gpt-4o-mini"
    assert get_active_model(tmp_path) == "or-gpt-4o-mini"


def test_set_unknown_model_raises(tmp_path):
    with pytest.raises(ValueError):
        set_active_model("totally-made-up", directory=tmp_path)


def test_empty_clears_selection(tmp_path):
    set_active_model("or-gpt-4o-mini", directory=tmp_path)
    assert set_active_model("", directory=tmp_path) is None
    assert get_active_model(tmp_path) is None


def test_resolve_prefers_override_then_persisted_then_default(tmp_path):
    settings = _settings(default_model="groq-llama-3.1-8b")

    assert resolve_active_model_key(settings, directory=tmp_path) == "groq-llama-3.1-8b"

    set_active_model("or-gemini-2.5-flash", directory=tmp_path)
    assert resolve_active_model_key(settings, directory=tmp_path) == "or-gemini-2.5-flash"

    assert (
        resolve_active_model_key(settings, override="or-gpt-4o-mini", directory=tmp_path)
        == "or-gpt-4o-mini"
    )


def test_resolve_ignores_unknown_override(tmp_path):
    settings = _settings(default_model="groq-llama-3.1-8b")
    assert (
        resolve_active_model_key(settings, override="bogus", directory=tmp_path)
        == "groq-llama-3.1-8b"
    )


def test_build_llm_service_groq_constructs():
    svc = build_llm_service(_settings(), "groq-llama-3.1-8b")
    assert type(svc).__name__ == "OpenAILLMService"


def test_build_llm_service_openrouter_constructs():
    svc = build_llm_service(_settings(), "or-gpt-4o-mini")
    assert type(svc).__name__ == "OpenAILLMService"


def test_build_llm_service_openrouter_requires_key():
    settings = _settings(openrouter_api_key=None)
    with pytest.raises(RuntimeError):
        build_llm_service(settings, "or-gpt-4o-mini")


def test_all_models_have_required_fields():
    for key, entry in MODELS.items():
        assert entry["provider"] in ("groq", "openrouter")
        assert entry["label"] and entry["model"]


# --- raw OpenRouter slugs (pick any / paste a model id) --------------------

def test_model_entry_for_preset_slug_and_unknown():
    s = _settings()
    assert model_entry_for("or-gpt-4o-mini", s)["provider"] == "openrouter"
    assert model_entry_for("groq-llama-3.1-8b", s)["provider"] == "groq"

    slug = model_entry_for("anthropic/claude-sonnet-4.5", s)
    assert slug["provider"] == "openrouter"
    assert slug["model"] == "anthropic/claude-sonnet-4.5"
    assert slug["key"] == "anthropic/claude-sonnet-4.5"

    assert model_entry_for("bogus-no-slash", s) is None
    assert model_entry_for("", s) is None


def test_set_and_resolve_raw_slug(tmp_path):
    assert set_active_model("anthropic/claude-sonnet-4.5", directory=tmp_path) == "anthropic/claude-sonnet-4.5"
    entry = resolve_model(_settings(), directory=tmp_path)
    assert entry["provider"] == "openrouter"
    assert entry["model"] == "anthropic/claude-sonnet-4.5"


def test_set_rejects_bare_word_but_accepts_slug(tmp_path):
    with pytest.raises(ValueError):
        set_active_model("just-a-word", directory=tmp_path)
    assert set_active_model("vendor/model", directory=tmp_path) == "vendor/model"


def test_build_llm_service_raw_slug_uses_openrouter():
    svc = build_llm_service(_settings(), "anthropic/claude-sonnet-4.5")
    assert type(svc).__name__ == "OpenAILLMService"


def test_build_llm_service_raw_slug_requires_key():
    with pytest.raises(RuntimeError):
        build_llm_service(_settings(openrouter_api_key=None), "anthropic/claude-sonnet-4.5")


def test_default_model_can_be_a_raw_slug(tmp_path):
    settings = _settings(default_model="anthropic/claude-sonnet-4.5")
    entry = resolve_model(settings, directory=tmp_path)  # empty dir → falls to default
    assert entry["model"] == "anthropic/claude-sonnet-4.5"
