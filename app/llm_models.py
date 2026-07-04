"""Switchable LLM models for the Pipecat pipeline (performance A/B).

A "model" is chosen from the UI (Pipecat page ?model=, persisted server-side like
the active voice) so you can compare LLMs on a live call. Two providers share one
OpenAI-compatible interface, so switching is just a different base URL + key + id:
  groq       : Groq's OpenAI-compatible endpoint — lowest latency (free tier).
  openrouter : OpenRouter — one key (OPENROUTER_API_KEY), many models
               (Anthropic / OpenAI / Google / Meta …) for side-by-side comparison.

`PipelineParams(enable_metrics=True)` already logs each model's TTFB, so switching
here and reading the bot logs is the measurement.

The keys below are the stable ?model= values; edit the dict to add/remove models.
OpenRouter ids are its model slugs (https://openrouter.ai/models).
"""
from __future__ import annotations

from pathlib import Path

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_STATE_DIR = Path("data")
ACTIVE_MODEL_FILE = "active_model"

MODELS: dict[str, dict] = {
    "groq-llama-3.1-8b": {
        "label": "Groq · Llama 3.1 8B (fast)",
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
    },
    "groq-llama-3.3-70b": {
        "label": "Groq · Llama 3.3 70B",
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
    },
    "or-claude-3.5-sonnet": {
        "label": "OpenRouter · Claude 3.5 Sonnet",
        "provider": "openrouter",
        "model": "anthropic/claude-3.5-sonnet",
    },
    "or-gpt-4o-mini": {
        "label": "OpenRouter · GPT-4o mini",
        "provider": "openrouter",
        "model": "openai/gpt-4o-mini",
    },
    "or-gemini-2.5-flash": {
        "label": "OpenRouter · Gemini 2.5 Flash",
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash",
    },
    "or-llama-3.3-70b": {
        "label": "OpenRouter · Llama 3.3 70B",
        "provider": "openrouter",
        "model": "meta-llama/llama-3.3-70b-instruct",
    },
}

FALLBACK_MODEL_KEY = "groq-llama-3.1-8b"


def default_model_key(settings) -> str:
    """The model used when nothing is persisted/overridden (env DEFAULT_MODEL)."""
    key = getattr(settings, "default_model", None)
    return key if key in MODELS else FALLBACK_MODEL_KEY


def _model_path(directory: Path | None = None) -> Path:
    return (directory or MODEL_STATE_DIR) / ACTIVE_MODEL_FILE


def get_active_model(directory: Path | None = None) -> str | None:
    """Read the persisted active model key, or None if unset/unknown/unreadable."""
    try:
        text = _model_path(directory).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text if text in MODELS else None


def set_active_model(model_key: str | None, directory: Path | None = None) -> str | None:
    """Persist the active model key atomically (must be a known key).

    Empty/None clears the selection so resolution falls back to the default.
    Returns the stored key, or None if cleared.
    """
    base = directory or MODEL_STATE_DIR
    path = _model_path(base)
    normalized = (model_key or "").strip()
    if not normalized:
        try:
            path.unlink()
        except OSError:
            pass
        return None
    if normalized not in MODELS:
        raise ValueError(f"Unknown model key: {normalized!r}")
    base.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{ACTIVE_MODEL_FILE}.tmp")
    tmp.write_text(normalized, encoding="utf-8")
    tmp.replace(path)
    return normalized


def resolve_active_model_key(
    settings, *, override: str | None = None, directory: Path | None = None
) -> str:
    """Resolve which model to use: per-session override -> persisted -> default."""
    if override and override.strip() in MODELS:
        return override.strip()
    persisted = get_active_model(directory)
    if persisted:
        return persisted
    return default_model_key(settings)


def build_llm_service(settings, model_key: str):
    """Construct the Pipecat LLM service for a model key (Groq or via OpenRouter).

    Both providers use Pipecat's OpenAI-compatible ``OpenAILLMService`` with a
    different base URL + key, so the rest of the pipeline is provider-agnostic.
    Temperature / max tokens come from the shared Groq settings so only the model
    varies across an A/B.
    """
    from pipecat.services.openai.llm import OpenAILLMService

    entry = MODELS.get(model_key) or MODELS[default_model_key(settings)]

    llm_settings = OpenAILLMService.Settings(
        model=entry["model"],
        temperature=settings.groq_temperature,
    )
    if settings.groq_max_tokens > 0:
        llm_settings.max_completion_tokens = settings.groq_max_tokens

    if entry["provider"] == "openrouter":
        api_key = settings.openrouter_api_key
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY must be set to use OpenRouter models.")
        return OpenAILLMService(
            api_key=api_key, base_url=OPENROUTER_BASE_URL, settings=llm_settings
        )

    # Default provider: Groq direct (lowest latency).
    return OpenAILLMService(
        api_key=settings.groq_api_key, base_url=GROQ_BASE_URL, settings=llm_settings
    )
