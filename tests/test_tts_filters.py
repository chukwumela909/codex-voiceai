"""Tests for the LLM -> TTS speakability filter used by the Pipecat path."""
import asyncio

from app.tts_filters import SpokenTextFilter, make_speakable


def test_strips_emojis():
    assert make_speakable("Sounds good, man! 😂🔥") == "Sounds good, man!"
    assert make_speakable("Let's go 🚀 to the gym 💪 today") == "Let's go to the gym today"


def test_strips_star_wrapped_stage_directions():
    assert make_speakable("*laughs* Yeah, that got me too.") == "Yeah, that got me too."
    assert make_speakable("Man, that's wild. *chuckles softly*") == "Man, that's wild."


def test_strips_parenthesized_and_bracketed_actions():
    assert make_speakable("(sighs) Long day at work.") == "Long day at work."
    assert make_speakable("[clears throat] So anyway.") == "So anyway."


def test_keeps_plain_parentheticals_and_emphasis():
    # A normal aside is speech, not a stage direction.
    assert make_speakable("My son (he's twelve) loves that game.") == (
        "My son (he's twelve) loves that game."
    )
    # Emphasis is left for the markdown filter to unwrap into plain words.
    assert make_speakable("I *really* mean it.") == "I *really* mean it."


def test_strips_control_tags_but_keeps_inner_text():
    assert make_speakable('Hey <emotion value="warm"/> there') == "Hey there"
    assert make_speakable("<spell>WD40</spell> fixed it") == "WD40 fixed it"


def test_collapses_leftover_whitespace_and_punctuation_gaps():
    assert make_speakable("Yeah *laughs* , that's funny") == "Yeah, that's funny"


def test_plain_speech_passes_through_unchanged():
    text = "Not much, man — just got back from the gym. What's good with you?"
    assert make_speakable(text) == text


def test_empty_and_all_junk_chunks_become_empty():
    assert make_speakable("") == ""
    assert make_speakable("😂") == ""
    assert make_speakable("*laughs*") == ""


def test_filter_wrapper_applies_make_speakable():
    filtered = asyncio.run(SpokenTextFilter().filter("*laughs* Okay 😂 deal."))
    assert filtered == "Okay deal."
