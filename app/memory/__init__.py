"""Agent memory: in-session summary + cross-session distilled RAG memory.

A single global person → one store per memory directory, cached for the process
so every session shares the same memory. ``build_manager`` wraps that store with
the configured embedder.
"""
from __future__ import annotations

from pathlib import Path

from app.memory.embedder import build_embedder, embedder_ready
from app.memory.manager import MemoryManager
from app.memory.store import MemoryRecord, VectorMemoryStore

__all__ = [
    "MemoryManager",
    "MemoryRecord",
    "VectorMemoryStore",
    "build_embedder",
    "embedder_ready",
    "get_store",
    "build_manager",
    "reset_store_cache",
]

_STORES: dict[str, VectorMemoryStore] = {}


def _memory_dir(settings) -> str:
    return str(getattr(settings, "memory_dir", "data/memory"))


def get_store(settings) -> VectorMemoryStore:
    directory = _memory_dir(settings)
    store = _STORES.get(directory)
    if store is None:
        store = VectorMemoryStore(
            Path(directory) / "store.json",
            dedupe_threshold=float(getattr(settings, "memory_dedupe_threshold", 0.92)),
        )
        _STORES[directory] = store
    return store


def build_manager(settings) -> MemoryManager:
    return MemoryManager(embedder=build_embedder(settings), store=get_store(settings), settings=settings)


def reset_store_cache() -> None:
    """Drop cached stores — used by tests to isolate temp directories."""
    _STORES.clear()
