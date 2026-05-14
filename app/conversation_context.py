INTENT_INFERENCE_INSTRUCTION = (
    "The latest user message may contain speech-to-text errors. "
    "Infer the user's likely intent from recent conversation context. "
    "Do not mention transcription uncertainty unless the ambiguity blocks a useful answer. "
    "If the utterance is genuinely ambiguous, ask one short clarifying question. "
    "Keep the response spoken, brief, and phone-call friendly."
)


def agent_transcript_with_intent_inference(
    transcript: list[dict[str, str]],
    *,
    enabled: bool = True,
) -> list[dict[str, str]]:
    prepared = [_copy_turn(turn) for turn in transcript]
    if not enabled or not prepared:
        return prepared

    latest = prepared[-1]
    if latest.get("role") != "user" or not latest.get("content", "").strip():
        return prepared

    return [
        *prepared[:-1],
        {"role": "system", "content": INTENT_INFERENCE_INSTRUCTION},
        latest,
    ]


def _copy_turn(turn: dict[str, str]) -> dict[str, str]:
    role = turn.get("role", "")
    content = turn.get("content", "")
    return {"role": role, "content": content}
