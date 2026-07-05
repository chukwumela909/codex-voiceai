"""Tests for the canned call-open greeting on the Pipecat path.

A fixed ``TTSSpeakFrame`` line replaces the old LLM-composed greeting: the
caller hears it immediately (no LLM round trip on top of pipeline warmup) and
a flat "Hello?" is how a real person answers a phone.
"""

from app.characters import Character, get_character
from app.pipeline import DEFAULT_GREETING_LINE, resolve_greeting_line


def test_character_greeting_wins():
    character = Character(id="t", name="T", role="tester", greeting="Yeah, hello?")
    assert resolve_greeting_line(character) == "Yeah, hello?"


def test_blank_greeting_falls_back_to_default():
    assert resolve_greeting_line(Character(id="t", name="T", role="tester")) == (
        DEFAULT_GREETING_LINE
    )
    assert resolve_greeting_line(
        Character(id="t", name="T", role="tester", greeting="   ")
    ) == DEFAULT_GREETING_LINE


def test_jimmy_answers_like_a_phone_pickup():
    jimmy = get_character("jimmy")
    assert resolve_greeting_line(jimmy) == "Yeah, hello?"
