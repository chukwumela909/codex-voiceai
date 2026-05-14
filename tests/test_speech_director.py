from app.speech_director import SpeechDirectionConfig, direct_speech_for_cartesia


def test_director_leaves_short_acknowledgement_plain():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True)

    assert direct_speech_for_cartesia("Okay.", config) == "Okay."


def test_director_adds_pause_after_contextual_discourse_marker():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True)

    directed = direct_speech_for_cartesia("Well, I think you were asking about the microphone.", config)

    assert directed == 'Well,<break time="180ms"/> I think you were asking about the microphone.'


def test_director_adds_clarification_pause_when_inferring_intent():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True)

    directed = direct_speech_for_cartesia("I think you meant the Deepgram model is missing words.", config)

    assert directed == 'I think you meant<break time="250ms"/> the Deepgram model is missing words.'


def test_director_spells_code_like_tokens():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True)

    directed = direct_speech_for_cartesia("Set API mode on port 8000.", config)

    assert directed == "Set <spell>API</spell> mode on port <spell>8000</spell>."


def test_director_can_be_disabled():
    config = SpeechDirectionConfig(enabled=False, ssml_enabled=True)

    assert direct_speech_for_cartesia("Well, this should stay plain.", config) == "Well, this should stay plain."


def test_director_can_disable_ssml_only():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=False)

    assert direct_speech_for_cartesia("Set API mode.", config) == "Set API mode."
