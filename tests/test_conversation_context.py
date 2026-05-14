from app.conversation_context import INTENT_INFERENCE_INSTRUCTION, agent_transcript_with_intent_inference


def test_intent_inference_preserves_raw_turns_and_inserts_hidden_instruction_before_latest_user():
    transcript = [
        {"role": "user", "content": "I want to test the browser mic."},
        {"role": "assistant", "content": "Sure, say something short."},
        {"role": "user", "content": "You're what's up can you hear me now"},
    ]

    prepared = agent_transcript_with_intent_inference(transcript)

    assert prepared[:-2] == transcript[:-1]
    assert prepared[-2]["role"] == "system"
    assert "speech-to-text errors" in prepared[-2]["content"]
    assert "Infer the user's likely intent" in prepared[-2]["content"]
    assert prepared[-1] == transcript[-1]
    assert transcript[-1]["content"] == "You're what's up can you hear me now"
    assert INTENT_INFERENCE_INSTRUCTION in prepared[-2]["content"]


def test_intent_inference_returns_copy_when_disabled():
    transcript = [{"role": "user", "content": "raw words stay raw"}]

    prepared = agent_transcript_with_intent_inference(transcript, enabled=False)

    assert prepared == transcript
    assert prepared is not transcript


def test_intent_inference_skips_when_latest_turn_is_not_user():
    transcript = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi."},
    ]

    assert agent_transcript_with_intent_inference(transcript) == transcript
