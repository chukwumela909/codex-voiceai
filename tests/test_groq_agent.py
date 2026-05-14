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
    text = "We can keep the line moving with a short thought before the sentence fully ends"
    split_at = text.rfind(" ", 0, 70)

    chunks, remainder = pop_speakable_chunks(text)

    assert chunks == [text[:split_at].strip() + " "]
    assert remainder == text[split_at:]
