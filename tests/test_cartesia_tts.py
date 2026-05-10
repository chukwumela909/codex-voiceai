import json

from app.cartesia_tts import generate_cartesia_context_id, parse_cartesia_message


def test_parse_cartesia_chunk_message():
    parsed = parse_cartesia_message(
        json.dumps(
            {
                "type": "chunk",
                "data": "YWJj",
                "done": False,
                "context_id": "ctx_123",
            }
        )
    )

    assert parsed == {
        "type": "chunk",
        "audio": "YWJj",
        "context_id": "ctx_123",
        "done": False,
    }


def test_parse_cartesia_done_message():
    parsed = parse_cartesia_message(json.dumps({"type": "done", "done": True, "context_id": "ctx_123"}))

    assert parsed == {
        "type": "done",
        "context_id": "ctx_123",
        "done": True,
    }


def test_parse_cartesia_error_message():
    parsed = parse_cartesia_message(json.dumps({"type": "error", "error": "bad voice", "done": True}))

    assert parsed["type"] == "error"
    assert parsed["message"] == "bad voice"
    assert parsed["done"] is True


def test_parse_cartesia_error_message_field():
    parsed = parse_cartesia_message(json.dumps({"type": "error", "message": "bad api key", "done": True}))

    assert parsed["type"] == "error"
    assert parsed["message"] == "bad api key"
    assert parsed["done"] is True


def test_generate_cartesia_context_id_matches_response_id():
    assert generate_cartesia_context_id("resp_abc") == "ctx_abc"
