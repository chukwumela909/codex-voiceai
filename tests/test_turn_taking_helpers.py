from app.turn_taking import (
    is_explicit_interrupt,
    is_short_backchannel,
    normalize_spoken_phrase,
)


def test_short_listener_signals_are_backchannels():
    for phrase in ("yeah", "Mm-hm.", "uh huh", "right", "I see", "okay"):
        assert is_short_backchannel(phrase)


def test_takeover_words_are_never_treated_as_backchannels():
    for phrase in ("stop", "wait", "no", "hold on"):
        assert is_explicit_interrupt(phrase)
        assert not is_short_backchannel(phrase)


def test_longer_turn_is_not_mistaken_for_a_backchannel():
    assert not is_short_backchannel("yeah but that's not what I meant")
    assert normalize_spoken_phrase("  Mm-hm... ") == "mm hm"
