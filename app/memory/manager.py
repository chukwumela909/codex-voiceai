"""Memory orchestration: retrieve (RAG), remember, distill, summarize, format.

Distillation/summarization use Groq in live mode and a lightweight heuristic in
mock mode (or when the LLM is unavailable), so memory works fully offline.
"""
from __future__ import annotations

import json
import re

import httpx

from app.memory.embedder import Embedder
from app.memory.store import MemoryRecord, VectorMemoryStore

MIN_RETRIEVAL_SCORE = 0.12  # below this, a memory is treated as irrelevant

_DISTILL_INSTRUCTION = (
    "You extract durable, long-term memory from a phone conversation. "
    "Return ONLY JSON of the form {\"facts\": [\"...\"], \"summary\": \"...\"}. "
    "facts: short third-person statements about the user worth remembering for future calls "
    "(name, preferences, plans, personal details the user shared). Omit small talk and anything "
    "about the assistant. summary: one sentence on what the call was about. If nothing is worth "
    "remembering, return empty facts and an empty summary."
)

# Heuristic fallback patterns (mock mode / no LLM).
_NAME_RE = re.compile(r"\b(?:my name is|i'm called|call me|this is)\s+([A-Z][a-zA-Z'-]+)", re.IGNORECASE)
_PREF_RE = re.compile(
    r"\bi\s+(?:really\s+)?(like|love|enjoy|prefer|hate|dislike|can't stand|cannot stand)\s+([^.,!?\n]+)",
    re.IGNORECASE,
)
_LIVEWORK_RE = re.compile(r"\bi\s+(live|work)\s+([^.,!?\n]+)", re.IGNORECASE)
_THIRD_PERSON = {
    "like": "likes",
    "love": "loves",
    "enjoy": "enjoys",
    "prefer": "prefers",
    "hate": "hates",
    "dislike": "dislikes",
    "can't stand": "can't stand",
    "cannot stand": "can't stand",
}


class MemoryManager:
    def __init__(self, *, embedder: Embedder, store: VectorMemoryStore, settings) -> None:
        self.embedder = embedder
        self.store = store
        self.settings = settings

    # ---- retrieval --------------------------------------------------------

    async def retrieve(self, query_text: str, k: int | None = None) -> list[MemoryRecord]:
        if not (query_text or "").strip() or self.store.count() == 0:
            return []
        k = k or int(getattr(self.settings, "memory_top_k", 5))
        vectors = await self.embedder.embed([query_text])
        scored = self.store.query(vectors[0], k)
        return [record for record, score in scored if score >= MIN_RETRIEVAL_SCORE]

    # ---- writing ----------------------------------------------------------

    async def remember(self, texts: list[str], *, kind: str = "fact", source_session: str | None = None) -> list[MemoryRecord]:
        clean = [t.strip() for t in texts if t and t.strip()]
        if not clean:
            return []
        vectors = await self.embedder.embed(clean)
        added: list[MemoryRecord] = []
        for text, vector in zip(clean, vectors):
            record = MemoryRecord(text=text, kind=kind, embedding=vector, source_session=source_session)
            if self.store.add(record):
                added.append(record)
        return added

    async def distill_transcript(self, transcript: list[dict[str, str]], *, source_session: str | None = None) -> dict:
        facts, summary = await self._extract(transcript)
        added_facts = await self.remember(facts, kind="fact", source_session=source_session)
        added_summary = await self.remember([summary] if summary else [], kind="summary", source_session=source_session)
        return {"facts": [r.text for r in added_facts], "summary": added_summary[0].text if added_summary else ""}

    # ---- in-session summary ----------------------------------------------

    async def summarize(self, transcript: list[dict[str, str]]) -> str:
        if not transcript:
            return ""
        if self._llm_available():
            try:
                _, summary = await self._llm_extract(transcript)
                if summary:
                    return summary
            except Exception:
                pass
        return _heuristic_summary(transcript)

    # ---- formatting -------------------------------------------------------

    def build_memory_block(self, records: list[MemoryRecord], running_summary: str = "") -> str:
        parts: list[str] = []
        facts = [r.text for r in records if r.kind == "fact"]
        prior = [r.text for r in records if r.kind == "summary"]
        if facts:
            parts.append(
                "What you remember about this person (weave in naturally, do not list, never invent):\n"
                + "\n".join(f"- {f}" for f in facts)
            )
        if prior:
            parts.append("From earlier conversations: " + " ".join(prior))
        if running_summary.strip():
            parts.append("Earlier in this call: " + running_summary.strip())
        return "\n\n".join(parts)

    # ---- extraction internals --------------------------------------------

    async def _extract(self, transcript: list[dict[str, str]]) -> tuple[list[str], str]:
        if self._llm_available():
            try:
                return await self._llm_extract(transcript)
            except Exception:
                pass
        return _heuristic_extract(transcript), _heuristic_summary(transcript)

    def _llm_available(self) -> bool:
        return bool(self.settings.normalized_mode == "live" and getattr(self.settings, "groq_api_key", None))

    async def _llm_extract(self, transcript: list[dict[str, str]]) -> tuple[list[str], str]:
        convo = "\n".join(f"{t.get('role', '')}: {t.get('content', '')}" for t in transcript)
        payload = {
            "model": self.settings.groq_model,
            "messages": [
                {"role": "system", "content": _DISTILL_INSTRUCTION},
                {"role": "user", "content": convo},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.settings.groq_api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        facts = [str(f).strip() for f in parsed.get("facts", []) if str(f).strip()]
        summary = str(parsed.get("summary", "")).strip()
        return facts, summary


# ---- heuristic helpers (mock mode / LLM fallback) -------------------------


def _user_turns(transcript: list[dict[str, str]]) -> list[str]:
    return [t.get("content", "").strip() for t in transcript if t.get("role") == "user" and t.get("content", "").strip()]


def _heuristic_extract(transcript: list[dict[str, str]]) -> list[str]:
    facts: list[str] = []
    seen: set[str] = set()

    def add(fact: str) -> None:
        key = fact.lower()
        if key not in seen:
            seen.add(key)
            facts.append(fact)

    for text in _user_turns(transcript):
        for match in _NAME_RE.finditer(text):
            add(f"The user's name is {match.group(1).strip()}.")
        for verb, obj in _PREF_RE.findall(text):
            third = _THIRD_PERSON.get(verb.lower(), verb.lower() + "s")
            add(f"The user {third} {obj.strip()}.")
        for verb, obj in _LIVEWORK_RE.findall(text):
            add(f"The user {verb.lower()}s {obj.strip()}.")
    return facts


def _heuristic_summary(transcript: list[dict[str, str]]) -> str:
    turns = _user_turns(transcript)
    if not turns:
        return ""
    snippet = " ".join(turns[-3:])
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if len(snippet) > 200:
        snippet = snippet[:197].rstrip() + "…"
    return f"The user talked about: {snippet}"
