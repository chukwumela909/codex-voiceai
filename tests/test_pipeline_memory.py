"""Tests for the Pipecat-path memory wiring (app/pipeline_memory.py).

The memory block must be injected without touching the frame path's latency
(refresh happens in the background, one turn behind), must survive context
windowing, and must never break a turn or session teardown when the memory
backend fails.
"""

import asyncio

import pytest

from app.pipeline import window_context_messages
from app.pipeline_memory import (
    MEMORY_MARKER,
    MemoryInjectionProcessor,
    context_to_transcript,
    extract_text,
    latest_user_text,
    maybe_distill_context,
    upsert_memory_message,
)


def _convo() -> list[dict]:
    return [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello!"},
    ]


# ---- pure helpers ----------------------------------------------------------


def test_extract_text_handles_string_and_parts():
    assert extract_text("plain") == "plain"
    assert extract_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a b"
    assert extract_text(["raw", {"type": "image", "url": "x"}]) == "raw"
    assert extract_text(None) == ""


def test_latest_user_text_returns_most_recent_user_message():
    msgs = _convo() + [{"role": "user", "content": "second question"}]
    assert latest_user_text(msgs) == "second question"
    assert latest_user_text([{"role": "system", "content": "persona"}]) == ""


def test_upsert_inserts_after_leading_system_messages():
    out = upsert_memory_message(_convo(), "The user likes jazz.")
    assert out[0] == {"role": "system", "content": "persona"}
    assert out[1]["role"] == "system"
    assert out[1]["content"].startswith(MEMORY_MARKER)
    assert "The user likes jazz." in out[1]["content"]
    assert out[2:] == _convo()[1:]


def test_upsert_replaces_existing_memory_message():
    once = upsert_memory_message(_convo(), "old fact")
    twice = upsert_memory_message(once, "new fact")
    memory_messages = [
        m for m in twice if m["role"] == "system" and m["content"].startswith(MEMORY_MARKER)
    ]
    assert len(memory_messages) == 1
    assert "new fact" in memory_messages[0]["content"]
    assert "old fact" not in memory_messages[0]["content"]


def test_upsert_with_empty_block_removes_memory_message():
    once = upsert_memory_message(_convo(), "old fact")
    cleared = upsert_memory_message(once, "")
    assert cleared == _convo()


def test_memory_message_survives_context_windowing():
    msgs = upsert_memory_message(_convo(), "The user likes jazz.")
    for i in range(50):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    out = window_context_messages(msgs, max_messages=10)
    assert out[0]["content"] == "persona"
    assert out[1]["content"].startswith(MEMORY_MARKER)
    assert len(out) == 12  # 2 leading system + window


def test_context_to_transcript_keeps_only_user_and_assistant_turns():
    msgs = upsert_memory_message(_convo(), "fact")
    msgs.append({"role": "user", "content": [{"type": "text", "text": "parts message"}]})
    transcript = context_to_transcript(msgs)
    assert transcript == [
        {"role": "user", "content": "hi there"},
        {"role": "assistant", "content": "hello!"},
        {"role": "user", "content": "parts message"},
    ]


# ---- refresh / distillation ------------------------------------------------


class StubManager:
    def __init__(self, block: str = "", error: bool = False):
        self.block = block
        self.error = error
        self.queries: list[str] = []
        self.transcripts: list[list[dict]] = []

    async def retrieve(self, query: str, k=None):
        self.queries.append(query)
        if self.error:
            raise RuntimeError("embedder down")
        return ["record"]

    def build_memory_block(self, records, running_summary: str = "") -> str:
        return self.block

    async def distill_transcript(self, transcript, *, source_session=None):
        if self.error:
            raise RuntimeError("distill failed")
        self.transcripts.append(transcript)
        return {"facts": ["f"], "summary": "s"}


class FakeContext:
    def __init__(self, messages: list[dict]):
        self._messages = messages

    def get_messages(self) -> list[dict]:
        return self._messages


class StubSettings:
    def __init__(self, enabled: bool):
        self.memory_effective_enabled = enabled


def _make_processor(manager: StubManager, enabled: bool = True) -> MemoryInjectionProcessor:
    return MemoryInjectionProcessor(
        FakeContext(_convo()), manager_factory=lambda: manager, enabled=enabled
    )


def test_refresh_updates_block_from_manager():
    manager = StubManager(block="What you remember: jazz")
    processor = _make_processor(manager)
    asyncio.run(processor._refresh("hi there"))
    assert processor._block == "What you remember: jazz"
    assert manager.queries == ["hi there"]
    assert processor._refreshing is False


def test_refresh_swallows_manager_errors():
    manager = StubManager(error=True)
    processor = _make_processor(manager)
    asyncio.run(processor._refresh("hi there"))
    assert processor._block == ""
    assert processor._refreshing is False


def test_distill_skipped_when_memory_disabled(monkeypatch: pytest.MonkeyPatch):
    manager = StubManager()
    monkeypatch.setattr("app.memory.build_manager", lambda s: manager)
    asyncio.run(maybe_distill_context(FakeContext(_convo()), StubSettings(enabled=False)))
    assert manager.transcripts == []


def test_distill_receives_user_assistant_transcript(monkeypatch: pytest.MonkeyPatch):
    manager = StubManager()
    monkeypatch.setattr("app.memory.build_manager", lambda s: manager)
    messages = upsert_memory_message(_convo(), "fact")
    asyncio.run(maybe_distill_context(FakeContext(messages), StubSettings(enabled=True)))
    assert manager.transcripts == [
        [
            {"role": "user", "content": "hi there"},
            {"role": "assistant", "content": "hello!"},
        ]
    ]


def test_distill_failures_do_not_raise(monkeypatch: pytest.MonkeyPatch):
    manager = StubManager(error=True)
    monkeypatch.setattr("app.memory.build_manager", lambda s: manager)
    asyncio.run(maybe_distill_context(FakeContext(_convo()), StubSettings(enabled=True)))


def test_distill_skipped_when_transcript_empty(monkeypatch: pytest.MonkeyPatch):
    manager = StubManager()
    monkeypatch.setattr("app.memory.build_manager", lambda s: manager)
    asyncio.run(
        maybe_distill_context(
            FakeContext([{"role": "system", "content": "persona"}]), StubSettings(enabled=True)
        )
    )
    assert manager.transcripts == []
