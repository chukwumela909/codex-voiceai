"""Conversation-aware helpers for barge-in and short backchannels."""
from __future__ import annotations

import re

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy


_NON_WORD_RE = re.compile(r"[^a-z0-9']+")

# These are listener signals when uttered alone while the other person is still
# speaking.  Deliberate takeover words ("stop", "wait", "no") are intentionally
# absent and must always be allowed to interrupt.
_SHORT_BACKCHANNELS = frozenset(
    {
        "ah",
        "got it",
        "gotcha",
        "hm",
        "hmm",
        "i see",
        "mhm",
        "mm",
        "mm hm",
        "mm hmm",
        "okay",
        "ok",
        "right",
        "sure",
        "uh",
        "uh huh",
        "um",
        "yeah",
        "yep",
    }
)

_EXPLICIT_INTERRUPTS = frozenset(
    {
        "enough",
        "hang on",
        "hold on",
        "no",
        "pause",
        "stop",
        "wait",
        "wait a second",
    }
)


def normalize_spoken_phrase(text: str) -> str:
    normalized = _NON_WORD_RE.sub(" ", (text or "").casefold())
    return " ".join(normalized.split())


def is_short_backchannel(text: str) -> bool:
    return normalize_spoken_phrase(text) in _SHORT_BACKCHANNELS


def is_explicit_interrupt(text: str) -> bool:
    return normalize_spoken_phrase(text) in _EXPLICIT_INTERRUPTS


class BackchannelAwareMinWordsUserTurnStartStrategy(MinWordsUserTurnStartStrategy):
    """Do not cancel Jimmy's sentence for a lone ``mm-hm`` or ``yeah``.

    Once the bot has stopped, the inherited strategy still accepts every
    one-word turn.  While it is speaking, only the narrow listener-signal set is
    ignored; explicit one-word takeovers such as ``stop`` remain immediate.
    """

    async def _handle_transcription(
        self, frame: TranscriptionFrame | InterimTranscriptionFrame
    ) -> ProcessFrameResult:
        if self._bot_speaking and is_short_backchannel(frame.text):
            await self.trigger_reset_aggregation()
            return ProcessFrameResult.CONTINUE
        return await super()._handle_transcription(frame)
