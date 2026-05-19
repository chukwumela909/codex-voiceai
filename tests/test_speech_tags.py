from app.speech_tags import sanitize


def test_sanitize_empty():
    result = sanitize("")
    assert result.text == ""
    assert result.stripped == 0


def test_sanitize_passes_known_tags():
    text = 'Hi<break time="200ms"/> <emotion value="warm"/> <spell>API</spell>.'
    result = sanitize(text)
    assert result.text == text
    assert result.stripped == 0


def test_sanitize_strips_unknown_tag_keeps_inner_text():
    result = sanitize("Hello <prosody rate=\"fast\">world</prosody>!")
    assert result.text == "Hello world!"
    assert result.stripped == 2


def test_sanitize_strips_self_closing_paired_tag():
    result = sanitize("<spell/>API")
    assert result.text == "API"
    assert result.stripped == 1


def test_sanitize_strips_close_for_void_tag():
    result = sanitize('<break time="100ms"/></break>')
    assert result.text == '<break time="100ms"/>'
    assert result.stripped == 1


def test_sanitize_strips_unknown_attribute():
    result = sanitize('<break time="200ms" volume="loud"/>')
    assert result.text == ""
    assert result.stripped == 1


def test_sanitize_strips_break_with_invalid_time_format():
    result = sanitize('<break time="forever"/>')
    assert result.text == ""
    assert result.stripped == 1


def test_sanitize_accepts_decimal_break_time():
    text = '<break time="1.5s"/>'
    assert sanitize(text).text == text


def test_sanitize_strips_emotion_with_invalid_label():
    result = sanitize('<emotion value="grumpy"/> hi')
    assert result.text == " hi"
    assert result.stripped == 1


def test_sanitize_accepts_emotion_label_case_insensitively():
    text = '<emotion value="EXCITED"/>'
    result = sanitize(text)
    assert result.text == text
    assert result.stripped == 0


def test_sanitize_drops_unclosed_paired_tag_keeping_inner_text():
    result = sanitize("Read <spell>API and keep going.")
    assert result.text == "Read API and keep going."
    assert result.stripped == 1


def test_sanitize_drops_stray_close_with_no_open():
    result = sanitize("hi </spell> there")
    assert result.text == "hi  there"
    assert result.stripped == 1


def test_sanitize_passes_stray_xml_sensitive_chars():
    result = sanitize("Use R&D API on x < 8000.")
    assert result.text == "Use R&D API on x < 8000."
    assert result.stripped == 0
