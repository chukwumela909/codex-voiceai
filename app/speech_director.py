"""Strip inline markup before sending text to ElevenLabs.

ElevenLabs `eleven_flash_v2_5` has no SSML/emotion/break tag support, so any
control tags a character prompt might still emit (`<emotion …>`, `<break …>`,
`<spell>…</spell>`) must be removed or they'd be read aloud. We reuse the
existing sanitizer with an empty allow-list: every tag is stripped while the
inner text survives. Expressiveness now lives in ElevenLabs voice settings.
"""
from app.speech_tags import SanitizeResult, sanitize


def strip_markup_for_tts(text: str) -> str:
    return strip_markup_for_tts_detailed(text).text


def strip_markup_for_tts_detailed(text: str) -> SanitizeResult:
    if not text:
        return SanitizeResult(text=text, stripped=0)
    return sanitize(text, allowed_tags={})
