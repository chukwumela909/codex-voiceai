import re
from dataclasses import dataclass


DISCOURSE_MARKERS = ("Well,", "Hmm,", "Okay,", "Right,", "So,")
MIN_WORDS_FOR_DISCOURSE_PAUSE = 6
SSML_TAG_PATTERN = re.compile(r"<[^>]+>")
SPELLABLE_TOKEN_PATTERN = re.compile(r"\b(?:[A-Z]{2,6}|\d{3,}|\w+-\w+)\b")


@dataclass(frozen=True)
class SpeechDirectionConfig:
    enabled: bool = True
    ssml_enabled: bool = True
    emotion_tags_enabled: bool = False


def direct_speech_for_cartesia(text: str, config: SpeechDirectionConfig) -> str:
    if not config.enabled or not config.ssml_enabled or not text.strip():
        return text
    if SSML_TAG_PATTERN.search(text):
        return text

    directed = _add_discourse_pause(text)
    directed = _add_clarification_pause(directed)
    directed = _spell_code_like_tokens(directed)
    return directed


def _add_discourse_pause(text: str) -> str:
    if len(text.split()) < MIN_WORDS_FOR_DISCOURSE_PAUSE:
        return text
    for marker in DISCOURSE_MARKERS:
        if text.startswith(f"{marker} "):
            return f'{marker}<break time="180ms"/> {text[len(marker) + 1:]}'
    return text


def _add_clarification_pause(text: str) -> str:
    phrase = "I think you meant "
    if phrase in text:
        return text.replace(phrase, 'I think you meant<break time="250ms"/> ', 1)
    return text


def _spell_code_like_tokens(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in {"OK"}:
            return token
        return f"<spell>{token}</spell>"

    pieces = re.split(r"(<[^>]+>)", text)
    return "".join(
        piece if piece.startswith("<") and piece.endswith(">") else SPELLABLE_TOKEN_PATTERN.sub(replace, piece)
        for piece in pieces
    )
