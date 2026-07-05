"""Memory wiring for the Pipecat pipeline.

The legacy session (app/mock_conversation.py) retrieves long-term memories per
turn and distills the transcript at session close; this module gives the
Pipecat path the same behavior without putting the embedder's network call on
the frame path. The memory block is refreshed in a background task keyed to the
latest user message and injected synchronously on the next LLM run — so each
turn sees memories retrieved for the previous one (the first turn sees none) —
and the session transcript is distilled into long-term memory when the
WebSocket closes.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger("voice_agent.pipecat.memory")

# Doubles as the marker that identifies the injected system message for
# replacement on later turns, so it must stay a stable prefix.
MEMORY_MARKER = "Long-term memory about this caller:"
# Marker for the rolling in-call summary that replaces turns dropped by the
# context window, so the persona doesn't develop mid-call amnesia.
CALL_SUMMARY_MARKER = "Earlier in this call (older turns, summarized):"
DISTILL_TIMEOUT_SECONDS = 20.0
SUMMARY_TIMEOUT_SECONDS = 10.0
_SUMMARY_MAX_CHARS = 700

_ROLLING_SUMMARY_INSTRUCTION = (
    "You maintain a running summary of an ongoing phone call, written for the "
    "assistant speaking on the call. Merge the previous summary with the new "
    "turns into ONE updated summary of at most 3 short sentences. Keep concrete "
    "facts the caller shared (name, people, plans, preferences) and anything "
    "the assistant promised or claimed about itself. Return ONLY the summary text."
)


def extract_text(content: object) -> str:
    """Flatten message content (plain string or list-of-parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return " ".join(p for p in parts if p)
    return ""


def latest_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            text = extract_text(message.get("content")).strip()
            if text:
                return text
    return ""


def upsert_memory_message(
    messages: list[dict], block: str, marker: str = MEMORY_MARKER
) -> list[dict]:
    """Return messages with the marker-prefixed memory system message updated.

    Any prior memory message is removed; a non-empty ``block`` is inserted
    right after the leading system message(s), where ``window_context_messages``
    preserves it across context trimming.
    """
    result = [
        m
        for m in messages
        if not (m.get("role") == "system" and str(m.get("content", "")).startswith(marker))
    ]
    if not block.strip():
        return result

    prefix_len = 0
    while prefix_len < len(result) and result[prefix_len].get("role") == "system":
        prefix_len += 1
    memory_message = {"role": "system", "content": f"{marker}\n{block.strip()}"}
    return result[:prefix_len] + [memory_message] + result[prefix_len:]


def _heuristic_rolling_summary(prev_summary: str, turns: list[dict]) -> str:
    """Offline/deterministic fallback: keep the tail of a plain-text digest."""
    lines: list[str] = []
    if prev_summary.strip():
        lines.append(prev_summary.strip())
    for turn in turns:
        role = turn.get("role")
        text = extract_text(turn.get("content")).strip()
        if role in ("user", "assistant") and text:
            speaker = "Caller" if role == "user" else "You"
            lines.append(f"{speaker}: {text}")
    digest = " ".join(lines)
    if len(digest) > _SUMMARY_MAX_CHARS:
        digest = "… " + digest[-_SUMMARY_MAX_CHARS:].lstrip()
    return digest


async def summarize_dropped_turns(settings, prev_summary: str, turns: list[dict]) -> str:
    """Fold turns trimmed out of the context window into a rolling summary.

    Uses the Groq LLM in live mode (small, off the frame path); anything else —
    mock mode, missing key, timeout, HTTP failure — degrades to a deterministic
    text digest so the summary never silently disappears.
    """
    if not turns:
        return prev_summary
    if not (settings.normalized_mode == "live" and getattr(settings, "groq_api_key", None)):
        return _heuristic_rolling_summary(prev_summary, turns)

    convo = "\n".join(
        f"{t.get('role', '')}: {extract_text(t.get('content')).strip()}" for t in turns
    )
    user_block = (
        f"Previous summary:\n{prev_summary.strip() or '(none)'}\n\nNew turns:\n{convo}"
    )
    payload = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": _ROLLING_SUMMARY_INSTRUCTION},
            {"role": "user", "content": user_block},
        ],
        "temperature": 0.2,
        "max_tokens": 160,
    }
    try:
        async with httpx.AsyncClient(timeout=SUMMARY_TIMEOUT_SECONDS) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.groq_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            summary = response.json()["choices"][0]["message"]["content"].strip()
        return summary or _heuristic_rolling_summary(prev_summary, turns)
    except Exception:
        logger.exception("rolling summary LLM call failed; using heuristic digest")
        return _heuristic_rolling_summary(prev_summary, turns)


def context_to_transcript(messages: list[dict]) -> list[dict[str, str]]:
    """Reduce context messages to the user/assistant transcript distillation expects."""
    transcript: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        text = extract_text(message.get("content")).strip()
        if text:
            transcript.append({"role": role, "content": text})
    return transcript


class MemoryInjectionProcessor(FrameProcessor):
    """Injects retrieved long-term memories into the shared LLM context.

    Sits between the user aggregator and the context window. On each downstream
    ``LLMContextFrame`` it synchronously upserts the block computed on the
    previous turn (no added latency), then refreshes the block in a background
    task so the next turn sees memories relevant to the latest user message.
    Memory must never break or stall a turn: refresh failures are swallowed.
    """

    def __init__(
        self,
        context: LLMContext,
        *,
        manager_factory: Callable,
        enabled: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._context = context
        self._manager_factory = manager_factory
        self._enabled = enabled
        self._manager = None
        self._block = ""
        self._last_query = ""
        self._refreshing = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if (
            self._enabled
            and isinstance(frame, LLMContextFrame)
            and direction == FrameDirection.DOWNSTREAM
        ):
            messages = self._context.get_messages()
            updated = upsert_memory_message(messages, self._block)
            if updated != messages:
                self._context.set_messages(updated)

            query = latest_user_text(messages)
            if query and query != self._last_query and not self._refreshing:
                self._last_query = query
                self._refreshing = True
                self.create_task(self._refresh(query))
        await self.push_frame(frame, direction)

    async def _refresh(self, query: str) -> None:
        try:
            if self._manager is None:
                self._manager = self._manager_factory()
            records = await self._manager.retrieve(query)
            self._block = self._manager.build_memory_block(records)
        except Exception:
            logger.exception("memory refresh failed; continuing without memories")
        finally:
            self._refreshing = False


async def maybe_distill_context(
    context: LLMContext, settings, archived_messages: list[dict] | None = None
) -> None:
    """Distill the session transcript into long-term memory at session end.

    ``archived_messages`` are turns the context window trimmed during the call
    (the shared context is mutated in place, so without them a long call would
    distill only its final minutes). Bounded and best-effort so a slow or
    failing distillation can never hang session teardown.
    """
    if not getattr(settings, "memory_effective_enabled", False):
        return
    try:
        transcript = context_to_transcript(
            list(archived_messages or []) + context.get_messages()
        )
        if not transcript:
            return
        from app.memory import build_manager

        manager = build_manager(settings)
        result = await asyncio.wait_for(
            manager.distill_transcript(transcript), timeout=DISTILL_TIMEOUT_SECONDS
        )
        if result.get("facts") or result.get("summary"):
            logger.info(
                "distilled session memory: %d facts, summary=%s",
                len(result.get("facts", [])),
                bool(result.get("summary")),
            )
    except Exception:
        logger.exception("memory distillation failed at session end")
