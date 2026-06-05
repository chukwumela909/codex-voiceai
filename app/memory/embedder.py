"""Embedders behind a tiny async interface so the provider is swappable.

- OpenAIEmbedder: real embeddings via OpenAI's REST API (httpx, no SDK).
- MockEmbedder: deterministic, offline, keyless — shared tokens map to shared
  dimensions so semantically-overlapping text scores higher. Used in mock mode
  and in tests so retrieval is exercisable without network or keys.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

_TOKEN_RE = re.compile(r"[a-z0-9']+")
MOCK_DIM = 256


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


def embedder_ready(settings) -> bool:
    """Whether a *real* embedder is configured (live + provider key present)."""
    provider = getattr(settings, "memory_embedding_provider", "openai").strip().lower()
    if settings.normalized_mode == "live" and provider == "openai":
        return bool(getattr(settings, "openai_api_key", None))
    return False


def build_embedder(settings) -> Embedder:
    """Pick the real embedder when live + keyed, else the deterministic mock."""
    if embedder_ready(settings):
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=getattr(settings, "memory_embedding_model", "text-embedding-3-small"),
        )
    return MockEmbedder()
