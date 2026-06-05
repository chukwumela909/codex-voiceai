import json

from app.groq_agent import parse_groq_stream_line, pop_speakable_chunks


def test_parse_groq_stream_line_returns_delta_content():
    line = "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}}]})

    assert parse_groq_stream_line(line) == "Hello"


def test_parse_groq_stream_line_ignores_done_and_empty_lines():
    assert parse_groq_stream_line("") is None
    assert parse_groq_stream_line("data: [DONE]") is None


def test_pop_speakable_chunks_flushes_sentence_and_keeps_remainder():
    chunks, remainder = pop_speakable_chunks("This is a complete sentence. This is still forming")

    assert chunks == ["This is a complete sentence. "]
    assert remainder == " This is still forming"


def test_pop_speakable_chunks_force_flushes_remainder():
    chunks, remainder = pop_speakable_chunks("A short final phrase", force=True)

    assert chunks == ["A short final phrase"]
    assert remainder == ""


def test_pop_speakable_chunks_releases_long_phone_call_phrase_before_full_sentence():
    text = (
        "We can keep the line moving with a short thought before the sentence fully ends "
        "while still sounding natural"
    )
    split_at = text.rfind(" ", 0, 90)

    chunks, remainder = pop_speakable_chunks(text)

    assert chunks == [text[:split_at].strip() + " "]
    assert remainder == text[split_at:]


def test_pop_speakable_chunks_does_not_split_inside_tag_brackets():
    text = 'This is a complete sentence. <break time="200ms'

    chunks, remainder = pop_speakable_chunks(text)

    assert chunks == ["This is a complete sentence. "]
    assert remainder == ' <break time="200ms'


def test_pop_speakable_chunks_does_not_split_between_open_and_close_spell():
    text = "We talked about <spell>API</spell>. Then we moved on completely."

    chunks, remainder = pop_speakable_chunks(text)

    assert chunks == [
        "We talked about <spell>API</spell>. ",
        "Then we moved on completely. ",
    ]
    assert remainder == ""


def test_pop_speakable_chunks_holds_punctuation_inside_open_spell():
    text = "Look at <spell>A.B.C</spell> please now alright friends here we go"

    chunks, remainder = pop_speakable_chunks(text)

    assert chunks == []
    assert remainder == text


def test_pop_speakable_chunks_holds_decimal_attribute_atomically():
    text = 'Hi.<break time="1.05s"/> Sure.'

    chunks, remainder = pop_speakable_chunks(text, force=True)

    assert chunks == ['Hi.<break time="1.05s"/> Sure. ']
    assert remainder == ""


def test_pop_speakable_chunks_force_flushes_unclosed_tag():
    text = "Some words <spell>ABC"

    chunks, remainder = pop_speakable_chunks(text, force=True)

    assert chunks == [text]
    assert remainder == ""
