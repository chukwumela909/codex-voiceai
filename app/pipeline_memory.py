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
DISTILL_TIMEOUT_SECONDS = 20.0


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


async def maybe_distill_context(context: LLMContext, settings) -> None:
    """Distill the session transcript into long-term memory at session end.

    Bounded and best-effort so a slow or failing distillation can never hang
    session teardown.
    """
    if not getattr(settings, "memory_effective_enabled", False):
        return
    try:
        transcript = context_to_transcript(context.get_messages())
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
