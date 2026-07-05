"""Tests for the idle check-in cap on the Pipecat path.

Bug: the user-idle timer re-arms every time the bot stops speaking, so a
silent caller was nudged with an identical check-in every idle timeout,
forever — an instant bot tell. The handler now counts consecutive nudges
since the caller's last message and stops at the configured maximum.
"""

from app.pipeline import IDLE_NUDGE_INSTRUCTION, count_trailing_idle_nudges


def _nudge() -> dict:
    return {"role": "system", "content": IDLE_NUDGE_INSTRUCTION}


def test_no_nudges_yet():
    msgs = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hey"},
        {"role": "assistant", "content": "hey man"},
    ]
    assert count_trailing_idle_nudges(msgs) == 0


def test_counts_consecutive_nudges_after_last_user_message():
    msgs = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hey"},
        {"role": "assistant", "content": "hey man"},
        _nudge(),
        {"role": "assistant", "content": "you still there?"},
        _nudge(),
        {"role": "assistant", "content": "alright, take your time"},
    ]
    assert count_trailing_idle_nudges(msgs) == 2


def test_user_reply_resets_the_count():
    msgs = [
        {"role": "system", "content": "persona"},
        _nudge(),
        {"role": "assistant", "content": "you there?"},
        {"role": "user", "content": "yeah sorry"},
        {"role": "assistant", "content": "all good"},
    ]
    assert count_trailing_idle_nudges(msgs) == 0


def test_other_system_messages_are_not_counted():
    msgs = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hey"},
        {"role": "system", "content": "some other instruction"},
        {"role": "assistant", "content": "hey"},
    ]
    assert count_trailing_idle_nudges(msgs) == 0
