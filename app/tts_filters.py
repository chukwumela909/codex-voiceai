"""Pipecat text filter that keeps LLM output speakable.

The Pipecat pipeline sends aggregated LLM sentences straight to ElevenLabs,
which reads whatever it receives. Chat-tuned models drift into chat idioms —
emojis, stage directions like ``*laughs*`` or ``(sighs)``, and stray control
tags — all of which are garbled or read aloud by TTS. This filter removes them
while leaving the spoken words intact. Markdown structure (bullets, headers,
emphasis) is handled separately by pipecat's ``MarkdownTextFilter``; this
filter runs first so action asterisks are removed before the markdown pass
would unwrap them into bare words ("laughs").
"""
from __future__ import annotations

import re

from pipecat.utils.text.base_text_filter import BaseTextFilter

from app.speech_director import strip_markup_for_tts

# Emoji and pictograph ranges plus the joiners/selectors that compose them.
_EMOJI_RE = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # regional indicator flags
    "\U0001F300-\U0001F5FF"  # symbols & pictographs (incl. skin tones)
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F900-\U0001FAFF"  # supplemental symbols & pictographs
    "☀-➿"          # misc symbols & dingbats
    "⬀-⯿"          # misc symbols & arrows (incl. star)
    "︎️"           # variation selectors
    "‍"                 # zero-width joiner
    "]+"
)

# Stage directions the model narrates instead of speaks. Only spans wrapped in
# *asterisks*, (parens), or [brackets] that contain one of these verbs are
# removed — a plain parenthetical aside or emphasized word passes through.
_ACTION_WORDS = (
    "laughs?|laughing|chuckles?|chuckling|giggles?|giggling|sighs?|sighing|"
    "smiles?|smiling|grins?|grinning|winks?|nods?|nodding|shrugs?|pauses?|"
    "coughs?|snorts?|whistles?|hums?|scoffs?|smirks?|groans?|gasps?|"
    "clears (?:his |her |their )?throat|takes a (?:deep )?breath|deep breath"
)
_ACTION_RE = re.compile(
    r"[*(\[]\s*(?:\w+[ '-]){0,3}?(?:" + _ACTION_WORDS + r")(?:[ '-]\w+){0,3}\s*[*)\]]",
    re.IGNORECASE,
)

_WHITESPACE_RE = re.compile(r"[ \t]{2,}")
_SPACE_BEFORE_PUNCT_RE = re.compile(r" +([,.!?;:])")


def make_speakable(text: str) -> str:
    """Strip emojis, stage directions, and control tags from one text chunk."""
    if not text:
        return text
    cleaned = strip_markup_for_tts(text)  # <emotion .../> etc., inner text kept
    cleaned = _ACTION_RE.sub(" ", cleaned)
    cleaned = _EMOJI_RE.sub("", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return cleaned.strip()


class SpokenTextFilter(BaseTextFilter):
    """Stateless pipecat text filter wrapping :func:`make_speakable`."""

    async def filter(self, text: str) -> str:
        return make_speakable(text)
