import json

from app.deepgram import parse_deepgram_message


def test_parse_deepgram_interim_transcript():
    parsed = parse_deepgram_message(
        json.dumps(
            {
                "is_final": False,
                "speech_final": False,
                "channel": {
                    "alternatives": [
                        {
                            "transcript": "hello there",
                            "confidence": 0.91,
                        }
                    ]
                },
            }
        )
    )

    assert parsed == {
        "text": "hello there",
        "confidence": 0.91,
        "is_final": False,
        "speech_final": False,
        "provider": "deepgram",
    }


def test_parse_deepgram_speech_final_transcript_from_bytes():
    parsed = parse_deepgram_message(
        json.dumps(
            {
                "is_final": True,
                "speech_final": True,
                "channel": {
                    "alternatives": [
                        {
                            "transcript": "testing the browser microphone",
                            "confidence": 0.97,
                        }
                    ]
                },
            }
        ).encode("utf-8")
    )

    assert parsed["text"] == "testing the browser microphone"
    assert parsed["is_final"] is True
    assert parsed["speech_final"] is True


def test_parse_deepgram_empty_transcript_is_ignored():
    parsed = parse_deepgram_message(
        json.dumps(
            {
                "is_final": False,
                "speech_final": False,
                "channel": {"alternatives": [{"transcript": ""}]},
            }
        )
    )

    assert parsed is None
