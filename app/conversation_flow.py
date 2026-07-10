"""Shared social-conversation direction for every text-LLM surface.

The character prompt defines who is speaking.  This module supplies the small,
turn-specific brief that chat-tuned models otherwise tend to invent for
themselves (usually: acknowledge, help, ask another question).  It is entirely
local and deterministic, so it adds no model round trip or perceptible latency.
"""
from __future__ import annotations

import re


TURN_GUIDANCE_MARKER = "[conversation-turn-guidance]"
_WORD_RE = re.compile(r"\b[\w'’]+\b", re.UNICODE)
_FILLER_RE = re.compile(
    r"\b(?:uh+|um+|well|i mean|you know|y['’]?know|honestly)\b",
    re.IGNORECASE,
)


def build_turn_guidance(
    messages: list[dict],
    *,
    character_name: str,
    private_context: str = "",
) -> str:
    """Build a compact direction for the latest user turn.

    The brief controls three failure modes that a static persona prompt cannot
    track well: monotonous reply length, question streaks, and repeated
    acknowledgement/filler openings.
    """
    if not messages or messages[-1].get("role") != "user":
        return ""

    latest = _text(messages[-1].get("content"))
    if not latest.strip():
        return ""

    assistant_turns = [
        _text(message.get("content"))
        for message in messages
        if message.get("role") == "assistant" and _text(message.get("content")).strip()
    ][-3:]
    word_count = len(_WORD_RE.findall(latest))

    if word_count <= 3:
        length_direction = (
            "The caller gave a tiny turn. Usually answer with one natural line; "
            "do not inflate it into a mini-speech."
        )
    elif word_count <= 14:
        length_direction = "Keep this beat compact: roughly one or two spoken sentences."
    elif word_count <= 45:
        length_direction = "Take a normal turn: roughly two or three spoken sentences if needed."
    else:
        length_direction = (
            "The caller took a substantial turn. You may answer with a few connected spoken "
            "sentences, but do not summarize their whole message."
        )

    recent_question_turns = sum("?" in turn for turn in assistant_turns)
    if assistant_turns and "?" in assistant_turns[-1]:
        question_direction = (
            "Your previous turn already asked a question. Do not ask another one this turn; "
            "interpret a terse answer from context and end on a statement."
        )
    elif recent_question_turns >= 2:
        question_direction = (
            "Questions have been doing too much work recently. Do not ask one this turn; react, "
            "contribute, and leave room."
        )
    else:
        question_direction = (
            "A question is optional, never the automatic handoff. Ask at most one and only when "
            "it follows directly from what the caller said."
        )

    recent_fillers = sum(len(_FILLER_RE.findall(turn)) for turn in assistant_turns)
    if recent_fillers >= 3:
        filler_direction = (
            "Recent turns already contain several discourse markers. Keep this one cleaner and "
            "do not reuse the same filler."
        )
    else:
        filler_direction = (
            "A brief filler, hesitation, or self-correction is welcome only where a person is "
            "actually forming a thought—usually zero or one, at most two in a longer story."
        )

    openers = _recent_openers(assistant_turns)
    variety_direction = (
        " Do not reuse these recent openings verbatim: " + "; ".join(openers) + "."
        if openers
        else ""
    )

    memory_direction = (
        "A relevant private-memory block is included below. It is the only specific personal "
        "history available for this turn, and mentioning it is still optional."
        if private_context.strip()
        else (
            "No private memory is relevant to this turn. Do not mention an event from your past "
            "or claim a current activity; contribute an opinion or immediate reaction instead."
        )
    )

    lines = [
        TURN_GUIDANCE_MARKER,
        "# This live conversation turn",
        (
            "Silently read the caller's latest words as a social move—a question, story, "
            "feeling, joke, correction, disagreement, or simple backchannel—and continue that "
            "move. Treat odd wording as possible speech-to-text noise."
        ),
        (
            f"Speak as {character_name}, a participant with a point of view—not as a helper, "
            "host, interviewer, coach, or customer-service agent. React specifically, add one "
            "real contribution of your own, and let unresolved things stay unresolved."
        ),
        (
            "Base your contribution on established character memory or an honest opinion. Never "
            "invent what you are doing right now or a personal anecdote merely to mirror the caller."
        ),
        memory_direction,
        length_direction,
        question_direction,
        filler_direction + variety_direction,
    ]
    if private_context.strip():
        lines.append(private_context.strip())
    lines.append(
        (
            "Do not thank the caller for sharing, paraphrase them as validation, offer a menu "
            "of help, announce what you can do, or tack on a hopeful reassurance just to soften "
            "the ending. Output only words that would be spoken aloud; use contractions, fragments, "
            "and light punctuation for rhythm."
        )
    )
    return "\n".join(lines)


def inject_turn_guidance(
    messages: list[dict],
    *,
    character_name: str,
    enabled: bool = True,
    private_context: str = "",
) -> list[dict]:
    """Replace the prior hidden brief immediately before the latest user turn.

    That position gives the live social policy recency over generic transcript-
    repair instructions. The visible transcript remains byte-for-byte unchanged.
    """
    cleaned = [
        dict(message)
        for message in messages
        if not _is_turn_guidance(message)
    ]
    if not enabled:
        return cleaned

    guidance = build_turn_guidance(
        cleaned,
        character_name=character_name,
        private_context=private_context,
    )
    if not guidance:
        return cleaned

    # Place the live brief immediately before the latest user turn. In the
    # classic path this lets it supersede the generic STT-repair instruction;
    # in Pipecat it also keeps turn-specific policy close to the generation.
    insert_at = len(cleaned) - 1
    return [
        *cleaned[:insert_at],
        {"role": "system", "content": guidance},
        *cleaned[insert_at:],
    ]


def _is_turn_guidance(message: dict) -> bool:
    return message.get("role") == "system" and _text(message.get("content")).startswith(
        TURN_GUIDANCE_MARKER
    )


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return " ".join(parts)
    return ""


def _recent_openers(turns: list[str]) -> list[str]:
    openers: list[str] = []
    for turn in turns[-2:]:
        words = turn.strip().split()
        if not words:
            continue
        opener = " ".join(words[:3]).strip(" \t\r\n,.;:!?—-")
        if opener and opener.casefold() not in {item.casefold() for item in openers}:
            openers.append(opener)
    return openers
