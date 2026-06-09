"""Regression tests for the LLM context sliding window.

Bug: on long phone calls, the Pipecat LLM context grew unbounded — every turn
re-sent the full transcript, so per-turn token cost climbed and exhausted Groq's
6k TPM budget, stalling replies (429 backoff). The fix windows the context to
the system prompt + the most recent N turns. See app/pipeline.py.
"""

from app.pipeline import window_context_messages


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
