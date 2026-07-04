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
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

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
    "or-claude-haiku-4.5": {
        "label": "OpenRouter · Claude Haiku 4.5 (fast)",
        "provider": "openrouter",
        "model": "anthropic/claude-haiku-4.5",
    },
    "or-claude-sonnet-4.5": {
        "label": "OpenRouter · Claude Sonnet 4.5",
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4.5",
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


def is_openrouter_slug(value: str | None) -> bool:
    # OpenRouter model ids are always "vendor/model"; preset keys never contain "/".
    return bool(value) and "/" in value


def model_entry_for(value: str | None, settings=None) -> dict | None:
    """Map a ?model= value to a full entry, or None if it isn't usable.

    A value is either a curated preset key (in MODELS) or a raw OpenRouter model
    slug like ``anthropic/claude-sonnet-4.5`` (any id from OpenRouter's catalog,
    typed/pasted in the UI). Slugs are treated as OpenRouter models.
    """
    if not value:
        return None
    value = value.strip()
    if value in MODELS:
        return {"key": value, **MODELS[value]}
    if is_openrouter_slug(value):
        return {
            "key": value,
            "label": f"OpenRouter · {value}",
            "provider": "openrouter",
            "model": value,
        }
    return None


def get_active_model(directory: Path | None = None) -> str | None:
    """Read the persisted active model (preset key or OpenRouter slug), or None."""
    try:
        text = _model_path(directory).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def set_active_model(model: str | None, directory: Path | None = None) -> str | None:
    """Persist the active model atomically. Accepts a preset key or an OpenRouter
    slug (``vendor/model``). Empty/None clears the selection. Returns the stored
    value, or None if cleared. Raises ValueError on an unusable value.
    """
    base = directory or MODEL_STATE_DIR
    path = _model_path(base)
    normalized = (model or "").strip()
    if not normalized:
        try:
            path.unlink()
        except OSError:
            pass
        return None
    if normalized not in MODELS and not is_openrouter_slug(normalized):
        raise ValueError(
            f"Unknown model {normalized!r}: use a preset key or an OpenRouter slug "
            "like 'anthropic/claude-sonnet-4.5'."
        )
    base.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{ACTIVE_MODEL_FILE}.tmp")
    tmp.write_text(normalized, encoding="utf-8")
    tmp.replace(path)
    return normalized


def resolve_model(
    settings, *, override: str | None = None, directory: Path | None = None
) -> dict:
    """Resolve the model entry to use: per-session override -> persisted -> default."""
    for candidate in (override, get_active_model(directory), getattr(settings, "default_model", None)):
        entry = model_entry_for(candidate, settings)
        if entry:
            return entry
    key = default_model_key(settings)
    return {"key": key, **MODELS[key]}


def resolve_active_model_key(
    settings, *, override: str | None = None, directory: Path | None = None
) -> str:
    """Convenience: the resolved model's key/slug (see resolve_model)."""
    return resolve_model(settings, override=override, directory=directory)["key"]


def build_llm_service(settings, model: str):
    """Construct the Pipecat LLM service for a model (preset key or OpenRouter slug).

    Both providers use Pipecat's OpenAI-compatible ``OpenAILLMService`` with a
    different base URL + key, so the rest of the pipeline is provider-agnostic.
    Temperature / max tokens come from the shared Groq settings so only the model
    varies across an A/B.
    """
    from pipecat.services.openai.llm import OpenAILLMService

    entry = model_entry_for(model, settings)
    if entry is None:
        key = default_model_key(settings)
        entry = {"key": key, **MODELS[key]}

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


async def list_openrouter_models(api_key: str | None = None, *, timeout: float = 10.0) -> list[dict]:
    """Fetch OpenRouter's model catalog for the UI picker/autocomplete.

    Returns ``[{id, name}]`` sorted by id. The catalog endpoint is public; the key
    is sent when present. Raises on HTTP/transport errors so the caller can warn.
    """
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(OPENROUTER_MODELS_URL, headers=headers)
        response.raise_for_status()
        data = response.json()

    models: list[dict] = []
    for entry in data.get("data", []):
        mid = entry.get("id")
        if not mid:
            continue
        models.append({"id": mid, "name": entry.get("name") or mid})
    models.sort(key=lambda m: m["id"])
    return models
