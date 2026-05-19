from dataclasses import dataclass

from app.speech_tags import ALLOWED_TAGS, SanitizeResult, sanitize


@dataclass(frozen=True)
class SpeechDirectionConfig:
    enabled: bool = True
    ssml_enabled: bool = True
    emotion_tags_enabled: bool = True


def direct_speech_for_cartesia(text: str, config: SpeechDirectionConfig) -> str:
    return direct_speech_for_cartesia_detailed(text, config).text


def direct_speech_for_cartesia_detailed(text: str, config: SpeechDirectionConfig) -> SanitizeResult:
    if not text:
        return SanitizeResult(text=text, stripped=0)
    if not config.enabled or not config.ssml_enabled:
        return SanitizeResult(text=text, stripped=0)

    allowed = ALLOWED_TAGS
    if not config.emotion_tags_enabled:
        allowed = {name: spec for name, spec in ALLOWED_TAGS.items() if name != "emotion"}

    return sanitize(text, allowed_tags=allowed)
