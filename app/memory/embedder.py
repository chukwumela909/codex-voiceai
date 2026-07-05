"""Embedders behind a tiny async interface so the provider is swappable.

- LocalEmbedder: real semantic embeddings that run on-device with no API key
  and no torch — model2vec static distilled vectors (numpy + tokenizers). The
  model downloads once from HF and is cached; encoding is ~milliseconds, so
  per-turn retrieval stays cheap. This is the recommended way to run memory
  without an OpenAI key.
- OpenAIEmbedder: real embeddings via OpenAI's REST API (httpx, no SDK).
- MockEmbedder: deterministic, offline, keyless — shared tokens map to shared
  dimensions so semantically-overlapping text scores higher. Used in mock mode
  and in tests so retrieval is exercisable without network, keys, or a model.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
from typing import Protocol

import httpx

logger = logging.getLogger("voice_agent.memory.embedder")

_TOKEN_RE = re.compile(r"[a-z0-9']+")
MOCK_DIM = 256
DEFAULT_LOCAL_MODEL = "minishlab/potion-base-8M"


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class Embedder(Protocol):
    name: str

    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class MockEmbedder:
    name = "mock"

    def __init__(self, dim: int = MOCK_DIM) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokenize(text):
            digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = digest % self.dim
            sign = 1.0 if (digest >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm:
            vec = [v / norm for v in vec]
        return vec

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class LocalEmbedder:
    """On-device embeddings via model2vec — real semantics, no key, no torch.

    The model (a static distilled table) loads once and is cached by
    huggingface_hub; loading and encoding both run in a worker thread so the
    event loop is never blocked. If the model can't load (offline first run, a
    missing/renamed model, no writable cache), the embedder degrades to
    :class:`MockEmbedder` and logs once — memory keeps working, just without
    real semantics, instead of silently returning nothing.
    """

    name = "local"

    def __init__(self, model_id: str = DEFAULT_LOCAL_MODEL, *, model=None) -> None:
        self.model_id = model_id or DEFAULT_LOCAL_MODEL
        self._model = model  # allow tests to inject a fake, skipping the download
        self._fallback: MockEmbedder | None = None
        self._lock = asyncio.Lock()

    def _load(self):
        from model2vec import StaticModel

        return StaticModel.from_pretrained(self.model_id)

    async def _ensure_model(self):
        if self._model is not None or self._fallback is not None:
            return self._model
        async with self._lock:
            if self._model is None and self._fallback is None:
                try:
                    self._model = await asyncio.to_thread(self._load)
                    logger.info("local embedder loaded model=%s", self.model_id)
                except Exception:
                    logger.exception(
                        "local embedder failed to load %s; falling back to mock",
                        self.model_id,
                    )
                    self._fallback = MockEmbedder()
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        model = await self._ensure_model()
        if model is None:
            return await self._fallback.embed(texts)
        import numpy as np

        vectors = await asyncio.to_thread(model.encode, list(texts))
        return np.asarray(vectors, dtype=float).tolist()


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, *, api_key: str, model: str = "text-embedding-3-small") -> None:
        self.api_key = api_key
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
        items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        return [item["embedding"] for item in items]


def _resolved_provider(settings) -> str:
    return (getattr(settings, "memory_embedding_provider", "openai") or "").strip().lower()


def _model2vec_available() -> bool:
    try:
        import model2vec  # noqa: F401

        return True
    except Exception:
        return False


def _local_model_id(settings) -> str:
    return getattr(settings, "memory_embedding_local_model", DEFAULT_LOCAL_MODEL) or DEFAULT_LOCAL_MODEL


def embedder_ready(settings) -> bool:
    """Whether a *real* (non-mock) embedder is configured and usable.

    - ``local``: usable whenever model2vec is importable — no key, any mode.
    - ``openai``: usable in live mode with an API key present.
    """
    provider = _resolved_provider(settings)
    if provider == "local":
        return _model2vec_available()
    if provider == "openai" and settings.normalized_mode == "live":
        return bool(getattr(settings, "openai_api_key", None))
    return False


def build_embedder(settings) -> Embedder:
    """Pick the configured real embedder, else the deterministic mock.

    ``local`` is honored in any mode (offline-friendly); ``openai`` still
    requires live mode + a key. Anything unusable falls back to the mock so
    memory never hard-fails on a missing dependency or key.
    """
    provider = _resolved_provider(settings)
    if provider == "local":
        if _model2vec_available():
            return LocalEmbedder(_local_model_id(settings))
        logger.warning("memory provider 'local' set but model2vec is not installed; using mock")
        return MockEmbedder()
    if provider == "openai" and settings.normalized_mode == "live" and getattr(settings, "openai_api_key", None):
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=getattr(settings, "memory_embedding_model", "text-embedding-3-small"),
        )
    return MockEmbedder()


async def warm_embedder(settings) -> None:
    """Best-effort background pre-load so the first caller isn't the one who
    pays the model download/load. No-op unless memory is on and the configured
    embedder is the local model. Never raises.
    """
    if not getattr(settings, "memory_effective_enabled", False):
        return
    if _resolved_provider(settings) != "local" or not _model2vec_available():
        return
    try:
        await LocalEmbedder(_local_model_id(settings)).embed(["warmup"])
        logger.info("local embedder warmed")
    except Exception:
        logger.exception("local embedder warmup failed; will load lazily on first use")
