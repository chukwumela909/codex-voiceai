from app.speech_director import strip_markup_for_tts, strip_markup_for_tts_detailed


def test_plain_text_is_untouched():
    assert strip_markup_for_tts("Okay.") == "Okay."


def test_break_tag_is_stripped():
    assert strip_markup_for_tts('Well,<break time="180ms"/> let me think.') == "Well, let me think."


def test_emotion_and_spell_tags_are_stripped_keeping_inner_text():
    text = '<emotion value="excited"/> The <spell>API</spell> is live!'

    assert strip_markup_for_tts(text) == " The API is live!"


def test_unknown_tag_is_stripped_but_inner_text_survives():
    assert strip_markup_for_tts('Hello <prosody rate="fast">world</prosody> there.') == "Hello world there."


def test_unbalanced_paired_tag_is_stripped():
    assert strip_markup_for_tts("Read <spell>API and continue.") == "Read API and continue."


def test_stray_xml_sensitive_chars_pass_through():
    text = "Use R&D API on x < 8000."

    assert strip_markup_for_tts(text) == text


def test_empty_text_is_returned_as_is():
    assert strip_markup_for_tts("") == ""


def test_detailed_reports_stripped_count():
    result = strip_markup_for_tts_detailed('<emotion value="excited"/> Hi.')

    assert result.text == " Hi."
    assert result.stripped == 1
