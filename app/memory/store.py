"""Local, file-backed vector memory store with brute-force cosine retrieval.

Single-process, single global person — the corpus is small, so plain-Python
cosine over a JSON file is instant and dependency-free. Writes are atomic
(temp file + replace), mirroring the character-file persistence pattern.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class MemoryRecord:
    text: str
    kind: str  # "fact" | "summary"
    embedding: list[float]
    id: str = field(default_factory=lambda: f"mem_{uuid4().hex}")
    created_at: float = field(default_factory=time.time)
    source_session: str | None = None

    def public(self) -> dict:
        """Serialize without the (large) embedding for API responses."""
        return {
            "id": self.id,
            "text": self.text,
            "kind": self.kind,
            "created_at": self.created_at,
            "source_session": self.source_session,
        }


class VectorMemoryStore:
    def __init__(self, path: Path, *, dedupe_threshold: float = 0.92) -> None:
        self.path = Path(path)
        self.dedupe_threshold = dedupe_threshold
        self._records: list[MemoryRecord] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self._records = [MemoryRecord(**row) for row in data]
            except (OSError, json.JSONDecodeError, TypeError):
                self._records = []
        self._loaded = True

    def all(self) -> list[MemoryRecord]:
        self._ensure_loaded()
        return list(self._records)

    def count(self) -> int:
        self._ensure_loaded()
        return len(self._records)

    def add(self, record: MemoryRecord) -> bool:
        """Add a record unless a near-duplicate of the same kind already exists."""
        self._ensure_loaded()
        for existing in self._records:
            if existing.kind == record.kind and cosine(existing.embedding, record.embedding) >= self.dedupe_threshold:
                return False
        self._records.append(record)
        self._save()
        return True

    def query(self, embedding: list[float], k: int) -> list[tuple[MemoryRecord, float]]:
        self._ensure_loaded()
        scored = [(r, cosine(embedding, r.embedding)) for r in self._records]
        # Highest similarity first; break ties toward more recent memories.
        scored.sort(key=lambda pair: (pair[1], pair[0].created_at), reverse=True)
        return scored[: max(0, k)]

    def clear(self) -> int:
        self._ensure_loaded()
        removed = len(self._records)
        self._records = []
        self._save()
        return removed

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps([asdict(r) for r in self._records], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)
