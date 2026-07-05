"""Regression tests for the LLM context sliding window.

Bug: on long phone calls, the Pipecat LLM context grew unbounded — every turn
re-sent the full transcript, so per-turn token cost climbed and exhausted Groq's
6k TPM budget, stalling replies (429 backoff). The fix windows the context to
the system prompt + the most recent N turns. See app/pipeline.py.
"""

import asyncio

from app.pipeline import (
    ContextWindowProcessor,
    split_context_window,
    window_context_messages,
)


def _convo(num_turns: int) -> list[dict]:
    msgs = [{"role": "system", "content": "persona"}]
    for i in range(num_turns):
        msgs.append({"role": "user", "content": f"u{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


def test_short_conversation_is_untouched():
    msgs = _convo(3)  # 1 system + 6
    assert window_context_messages(msgs, max_messages=20) is msgs


def test_long_conversation_is_bounded_to_window():
    msgs = _convo(50)  # 1 system + 100
    out = window_context_messages(msgs, max_messages=20)
    # system prompt + exactly the window
    assert len(out) == 21
    assert out[0] == {"role": "system", "content": "persona"}


def test_system_prompt_always_preserved():
    msgs = _convo(50)
    out = window_context_messages(msgs, max_messages=20)
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "persona"


def test_window_keeps_the_most_recent_turns():
    msgs = _convo(50)
    out = window_context_messages(msgs, max_messages=20)
    # last message must be the latest assistant reply
    assert out[-1] == {"role": "assistant", "content": "a49"}
    assert out[1] == {"role": "user", "content": "u40"}


def test_zero_disables_windowing():
    msgs = _convo(50)
    assert window_context_messages(msgs, max_messages=0) is msgs


def test_multiple_leading_system_messages_preserved():
    msgs = [
        {"role": "system", "content": "persona"},
        {"role": "system", "content": "extra-directive"},
    ] + _convo(50)[1:]
    out = window_context_messages(msgs, max_messages=10)
    assert out[0]["content"] == "persona"
    assert out[1]["content"] == "extra-directive"
    assert len(out) == 12  # 2 system + 10 window
    assert out[-1] == {"role": "assistant", "content": "a49"}


# ---- split_context_window: trimming must not be amnesia ---------------------


def test_split_returns_no_dropped_when_within_window():
    msgs = _convo(3)
    windowed, dropped = split_context_window(msgs, max_messages=20)
    assert windowed is msgs
    assert dropped == []


def test_split_returns_dropped_oldest_turns():
    msgs = _convo(50)  # 1 system + 100
    windowed, dropped = split_context_window(msgs, max_messages=20)
    assert len(windowed) == 21
    assert len(dropped) == 80
    assert dropped[0] == {"role": "user", "content": "u0"}
    assert dropped[-1] == {"role": "assistant", "content": "a39"}
    # windowed picks up exactly where dropped ends
    assert windowed[1] == {"role": "user", "content": "u40"}


def test_split_excludes_mid_conversation_system_messages_from_dropped():
    msgs = _convo(10)
    msgs.insert(5, {"role": "system", "content": "idle nudge"})
    windowed, dropped = split_context_window(msgs, max_messages=4)
    assert all(m["role"] in ("user", "assistant") for m in dropped)


# ---- rolling summary refresh -------------------------------------------------


class _FakeContext:
    def __init__(self, messages):
        self._messages = messages

    def get_messages(self):
        return self._messages

    def set_messages(self, messages):
        self._messages = messages


def test_refresh_summary_updates_and_clears_flag():
    async def summarizer(prev, turns):
        return f"{prev}+{len(turns)}turns"

    proc = ContextWindowProcessor(_FakeContext(_convo(1)), max_turns=2, summarizer=summarizer)
    proc._summarizing = True
    asyncio.run(proc._refresh_summary([{"role": "user", "content": "x"}]))
    assert proc._summary == "+1turns"
    assert proc._summarizing is False


def test_refresh_summary_failure_requeues_batch():
    async def summarizer(prev, turns):
        raise RuntimeError("groq down")

    proc = ContextWindowProcessor(_FakeContext(_convo(1)), max_turns=2, summarizer=summarizer)
    batch = [{"role": "user", "content": "x"}]
    proc._summarizing = True
    asyncio.run(proc._refresh_summary(batch))
    assert proc._summary == ""
    assert proc._pending == batch
    assert proc._summarizing is False
