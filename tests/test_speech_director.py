from app.speech_director import SpeechDirectionConfig, direct_speech_for_cartesia


def test_director_leaves_plain_text_untouched():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True)

    assert direct_speech_for_cartesia("Okay.", config) == "Okay."


def test_director_passes_through_allowed_break_tag():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True)

    text = 'Well,<break time="180ms"/> let me think.'

    assert direct_speech_for_cartesia(text, config) == text


def test_director_passes_through_allowed_emotion_and_spell():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True)

    text = '<emotion value="excited"/> The <spell>API</spell> is live!'

    assert direct_speech_for_cartesia(text, config) == text


def test_director_strips_unknown_tag_but_keeps_inner_text():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True)

    text = "Hello <prosody rate=\"fast\">world</prosody> there."

    assert direct_speech_for_cartesia(text, config) == "Hello world there."


def test_director_strips_emotion_with_unknown_label():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True)

    text = '<emotion value="grumpy"/> Hi.'

    assert direct_speech_for_cartesia(text, config) == " Hi."


def test_director_strips_break_with_malformed_time():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True)

    text = 'Sure<break time="forever"/> let me think.'

    assert direct_speech_for_cartesia(text, config) == "Sure let me think."


def test_director_strips_unbalanced_paired_tag():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True)

    text = "Read <spell>API and continue."

    assert direct_speech_for_cartesia(text, config) == "Read API and continue."


def test_director_passes_through_stray_xml_sensitive_chars():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True)

    text = "Use R&D API on x < 8000."

    assert direct_speech_for_cartesia(text, config) == text


def test_director_can_disable_ssml_handling():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=False)

    text = '<emotion value="excited"/> Hi.'

    assert direct_speech_for_cartesia(text, config) == text


def test_director_drops_emotion_tag_when_emotion_disabled():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True, emotion_tags_enabled=False)

    text = '<emotion value="excited"/> Hi.'

    assert direct_speech_for_cartesia(text, config) == " Hi."


def test_director_can_be_disabled_entirely():
    config = SpeechDirectionConfig(enabled=False, ssml_enabled=True)

    text = '<emotion value="excited"/> Hi.'

    assert direct_speech_for_cartesia(text, config) == text
