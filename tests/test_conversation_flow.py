from app.conversation_flow import (
    TURN_GUIDANCE_MARKER,
    build_turn_guidance,
    inject_turn_guidance,
)


def test_guidance_matches_short_user_energy_and_blocks_question_streak():
    messages = [
        {"role": "user", "content": "Rough day."},
        {"role": "assistant", "content": "Yeah, man. What happened?"},
        {"role": "user", "content": "Just work."},
    ]

    guidance = build_turn_guidance(messages, character_name="Jimmy")

    assert "tiny turn" in guidance
    assert "previous turn already asked a question" in guidance
    assert "participant with a point of view" in guidance
    assert "helper, host, interviewer" in guidance
    assert "Never invent what you are doing right now" in guidance
    assert "hopeful reassurance" in guidance
    assert "Do not ask another one this turn" in guidance
    assert "No private memory is relevant" in guidance


def test_guidance_reduces_recent_filler_repetition_and_names_openers():
    messages = [
        {"role": "user", "content": "First thing"},
        {"role": "assistant", "content": "Yeah, honestly, I mean, that's a lot."},
        {"role": "user", "content": "Second thing"},
        {"role": "assistant", "content": "Well, you know, I can see that."},
        {"role": "user", "content": "Anyway, I'm fine now."},
    ]

    guidance = build_turn_guidance(messages, character_name="Jimmy")

    assert "several discourse markers" in guidance
    assert "Do not reuse these recent openings verbatim" in guidance


def test_injection_replaces_old_brief_without_mutating_raw_transcript():
    original = [
        {"role": "system", "content": "You are Jimmy."},
        {"role": "system", "content": TURN_GUIDANCE_MARKER + "\nold"},
        {"role": "user", "content": "You're what's up can you hear me"},
    ]

    prepared = inject_turn_guidance(original, character_name="Jimmy")

    guidance_messages = [
        message
        for message in prepared
        if message.get("role") == "system"
        and str(message.get("content", "")).startswith(TURN_GUIDANCE_MARKER)
    ]
    assert len(guidance_messages) == 1
    assert guidance_messages[0]["content"] != TURN_GUIDANCE_MARKER + "\nold"
    assert prepared[-1] == original[-1]
    assert original[1]["content"].endswith("old")


def test_disabled_injection_removes_stale_brief_and_adds_nothing():
    messages = [
        {"role": "system", "content": TURN_GUIDANCE_MARKER + "\nstale"},
        {"role": "user", "content": "Hello"},
    ]

    assert inject_turn_guidance(messages, character_name="Alex", enabled=False) == [
        {"role": "user", "content": "Hello"}
    ]


def test_injection_keeps_relevant_private_context_inside_hidden_brief():
    prepared = inject_turn_guidance(
        [{"role": "user", "content": "How's your son?"}],
        character_name="Jimmy",
        private_context="# Relevant private memory\n- Son: He is twelve.",
    )

    assert prepared[0]["role"] == "system"
    assert "Son: He is twelve" in prepared[0]["content"]
    assert prepared[-1] == {"role": "user", "content": "How's your son?"}
