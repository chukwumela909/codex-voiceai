"""Pipecat 1.x voice pipeline (spike).

Runs side-by-side with the legacy `app.mock_conversation` pipeline. Wires
FastAPI WebSocket transport → Deepgram STT → Groq LLM (via OpenAI-compatible
endpoint) → ElevenLabs TTS. Turn-taking lives on the user aggregator (pipecat
1.2 moved VAD/interruption config off the transport): Silero VAD gates speech
start/stop, a local smart-turn model decides whether a pause is end-of-thought
or mid-sentence, and barge-in requires a minimum word count so noise can't cut
the bot off.
"""

from __future__ import annotations

import asyncio
import logging

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMMessagesAppendFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.text.markdown_text_filter import MarkdownTextFilter

from app.ambience import build_room_tone_mixer
from app.config import Settings
from app.llm_models import build_llm_service, resolve_model
from app.pipeline_memory import (
    CALL_SUMMARY_MARKER,
    MemoryInjectionProcessor,
    maybe_distill_context,
    summarize_dropped_turns,
    upsert_memory_message,
)
from app.tts_filters import SpokenTextFilter
from app.voice_settings import resolve_active_voice_id

logger = logging.getLogger("voice_agent.pipecat")


def window_context_messages(messages: list[dict], max_messages: int) -> list[dict]:
    """Keep leading system message(s) + the most recent ``max_messages`` messages.

    The LLM context grows unbounded otherwise — every turn re-sends the entire
    transcript, so per-turn token cost climbs with the conversation and quickly
    exhausts Groq's tokens-per-minute budget, stalling replies on long calls.
    This caps the history while always preserving the leading system prompt.

    ``max_messages <= 0`` disables windowing (unbounded). Returns the original
    list object unchanged when no trimming is needed.
    """
    if max_messages <= 0 or len(messages) <= max_messages:
        return messages

    prefix_len = 0
    while prefix_len < len(messages) and messages[prefix_len].get("role") == "system":
        prefix_len += 1

    rest = messages[prefix_len:]
    if len(rest) <= max_messages:
        return messages

    return messages[:prefix_len] + rest[-max_messages:]


def split_context_window(
    messages: list[dict], max_messages: int
) -> tuple[list[dict], list[dict]]:
    """Window the context and also return the user/assistant turns dropped.

    The dropped turns feed the rolling in-call summary and the end-of-call
    memory distillation — without them, trimming is pure amnesia.
    """
    windowed = window_context_messages(messages, max_messages)
    if len(windowed) == len(messages):
        return windowed, []

    prefix_len = 0
    while prefix_len < len(messages) and messages[prefix_len].get("role") == "system":
        prefix_len += 1

    dropped_count = len(messages) - len(windowed)
    dropped = messages[prefix_len : prefix_len + dropped_count]
    return windowed, [m for m in dropped if m.get("role") in ("user", "assistant")]


class ContextWindowProcessor(FrameProcessor):
    """Trims the shared LLM context to a sliding window before each LLM run.

    Sits just before the LLM service. Every LLM run funnels a downstream
    ``LLMContextFrame`` through here; we bound the shared context's message list
    in place so the LLM (and the next turn's aggregation) stay within budget.

    Dropped turns are not discarded: they accumulate in ``archived_messages``
    (used to distill the *full* call into long-term memory at session end) and,
    when a ``summarizer`` is provided, are folded in the background into a
    rolling summary that is upserted into the leading system block — so the
    persona keeps the gist of minutes-old conversation instead of blanking on
    it. The summary refresh mirrors the memory injector: computed off the frame
    path, visible from the next turn on.
    """

    # Summarize only once this many dropped messages accumulate, so a long call
    # triggers a small background LLM call every few turns instead of every turn.
    SUMMARY_BATCH_MIN = 6

    def __init__(self, context: LLMContext, max_turns: int, summarizer=None, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._max_messages = max_turns * 2  # one user + one assistant per turn
        self._summarizer = summarizer
        self._summary = ""
        self._summarizing = False
        self._pending: list[dict] = []
        self._archived: list[dict] = []

    @property
    def archived_messages(self) -> list[dict]:
        """All user/assistant messages trimmed out of the context this session."""
        return self._archived

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            messages = self._context.get_messages()
            updated = upsert_memory_message(
                messages, self._summary, marker=CALL_SUMMARY_MARKER
            )
            windowed, dropped = split_context_window(updated, self._max_messages)
            if dropped:
                self._archived.extend(dropped)
                self._pending.extend(dropped)
                logger.debug(
                    "context windowed: %d -> %d messages (%d turns archived)",
                    len(updated),
                    len(windowed),
                    len(dropped),
                )
            if windowed != messages:
                self._context.set_messages(windowed)

            if (
                self._summarizer
                and not self._summarizing
                and len(self._pending) >= self.SUMMARY_BATCH_MIN
            ):
                batch, self._pending = self._pending, []
                self._summarizing = True
                self.create_task(self._refresh_summary(batch))
        await self.push_frame(frame, direction)

    async def _refresh_summary(self, batch: list[dict]) -> None:
        try:
            self._summary = await self._summarizer(self._summary, batch)
        except Exception:
            # Put the batch back so its content is retried with the next one.
            self._pending = batch + self._pending
            logger.exception("rolling summary refresh failed; will retry")
        finally:
            self._summarizing = False

IDLE_NUDGE_INSTRUCTION = (
    "The caller has gone quiet. Offer one brief, warm check-in to let them know "
    "you're still on the line. Keep it under 18 words and do not ask why they went silent."
)
# Spoken when a character defines no greeting of its own.
DEFAULT_GREETING_LINE = "Hello?"


def resolve_greeting_line(character) -> str:
    """The fixed line spoken the moment a call connects.

    A canned ``TTSSpeakFrame`` beats an LLM-composed greeting twice over: the
    caller hears it immediately (no LLM round trip on top of pipeline warmup),
    and a flat "Hello?" is how a real person answers a phone.
    """
    greeting = (getattr(character, "greeting", "") or "").strip()
    return greeting or DEFAULT_GREETING_LINE


def build_vad_params(settings: Settings) -> VADParams:
    return VADParams(
        confidence=settings.vad_confidence,
        start_secs=settings.vad_start_secs,
        stop_secs=settings.vad_stop_secs,
        min_volume=settings.vad_min_volume,
    )


def create_turn_analyzer(settings: Settings) -> LocalSmartTurnAnalyzerV3 | None:
    """Construct the smart-turn analyzer (or None when disabled).

    Loading the ONNX session + Whisper feature extractor takes hundreds of
    milliseconds of pure CPU, so callers on the event loop must run this via
    ``asyncio.to_thread`` — constructing it inline stalls audio for every
    other live session. Also used at app startup to pre-warm imports and the
    OS file cache so the first caller doesn't pay the full cold start.
    """
    if not settings.smart_turn_enabled:
        return None
    return LocalSmartTurnAnalyzerV3(
        params=SmartTurnParams(stop_secs=settings.smart_turn_stop_secs)
    )


def build_user_turn_strategies(
    settings: Settings,
    turn_analyzer: LocalSmartTurnAnalyzerV3 | None = None,
) -> UserTurnStrategies:
    """Map turn-taking settings onto pipecat user turn strategies.

    Start: with ``interruption_min_words > 0``, a turn (and therefore an
    interruption while the bot speaks) only starts after that many transcribed
    words, so noise can't barge in. MinWords must be the sole start strategy —
    a VAD start strategy would open the turn first and the controller ignores
    later start triggers, so MinWords would never gate interruptions.

    Stop: smart turn hands VAD pauses to a local end-of-turn model (holds
    through mid-thought pauses up to ``smart_turn_stop_secs``); with it
    disabled, a flat ``speech_timeout_stop_secs`` pause ends the turn.

    ``turn_analyzer`` accepts a pre-built analyzer (see
    :func:`create_turn_analyzer`); when omitted one is constructed inline.
    """
    start = None
    if settings.interruption_min_words > 0:
        start = [MinWordsUserTurnStartStrategy(min_words=settings.interruption_min_words)]

    if settings.smart_turn_enabled:
        stop = [
            TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=turn_analyzer or create_turn_analyzer(settings)
            )
        ]
    else:
        stop = [
            SpeechTimeoutUserTurnStopStrategy(
                user_speech_timeout=settings.speech_timeout_stop_secs
            )
        ]

    return UserTurnStrategies(start=start, stop=stop)


def resolve_session_character_id(requested: str | None, settings: Settings) -> str:
    """Honor a requested character id when it exists; otherwise the default."""
    from app.characters import load_characters, resolve_default_character_id

    normalized = (requested or "").strip().lower()
    if normalized and normalized in load_characters():
        return normalized
    return resolve_default_character_id(
        env_default=getattr(settings, "default_character_id", None)
    )


def count_trailing_idle_nudges(messages: list[dict]) -> int:
    """Consecutive idle nudges issued since the caller last said anything.

    Scans backwards to the most recent user message; used to cap check-ins so
    a silent caller isn't nudged every idle-timeout forever.
    """
    count = 0
    for message in reversed(messages):
        if message.get("role") == "user":
            break
        if (
            message.get("role") == "system"
            and message.get("content") == IDLE_NUDGE_INSTRUCTION
        ):
            count += 1
    return count


def build_session_task(
    transport: FastAPIWebsocketTransport,
    settings: Settings,
    character_id: str | None = None,
    voice_id: str | None = None,
    model_id: str | None = None,
    turn_analyzer: LocalSmartTurnAnalyzerV3 | None = None,
) -> tuple[PipelineTask, LLMContext, ContextWindowProcessor]:
    """Wire STT/LLM/TTS, aggregators, and idle/greeting handlers onto a transport.

    Transport-agnostic so the browser (`run_pipecat_session`) and telephony
    (`run_twilio_session`) paths share identical pipeline behavior and only
    differ in how the transport is constructed. Returns the task, the shared
    LLM context, and the context-window processor so callers can distill the
    full call (current context + archived trimmed turns) after the session.
    """

    stt = DeepgramSTTService(
        api_key=settings.deepgram_api_key,
        settings=DeepgramSTTService.Settings(
            model=settings.deepgram_model,
            endpointing=settings.deepgram_endpointing_ms,
            utterance_end_ms=settings.deepgram_utterance_end_ms,
        ),
    )

    # The LLM is chosen per session from the UI (?model=), else the persisted/default
    # model — a curated preset (app/llm_models.py MODELS) or any OpenRouter slug.
    model_entry = resolve_model(settings, override=model_id)
    llm = build_llm_service(settings, model_entry["key"])
    logger.info(
        "pipeline llm model=%s (%s:%s)",
        model_entry["key"],
        model_entry["provider"],
        model_entry["model"],
    )

    tts = ElevenLabsTTSService(
        api_key=settings.elevenlabs_api_key,
        sample_rate=settings.elevenlabs_sample_rate,
        # SpokenTextFilter must run first: it removes *stage directions*
        # wholesale before the markdown pass would unwrap them into bare
        # spoken words ("laughs").
        text_filters=[SpokenTextFilter(), MarkdownTextFilter()],
        settings=ElevenLabsTTSService.Settings(
            model=settings.elevenlabs_model,
            voice=resolve_active_voice_id(settings, override=voice_id),
            stability=settings.elevenlabs_stability,
            similarity_boost=settings.elevenlabs_similarity_boost,
            style=settings.elevenlabs_style,
            use_speaker_boost=settings.elevenlabs_use_speaker_boost,
            speed=settings.elevenlabs_speed,
        ),
    )

    from app.characters import build_system_prompt, get_character
    from app.memory import build_manager

    active_character = get_character(resolve_session_character_id(character_id, settings))
    context = LLMContext(
        messages=[{"role": "system", "content": build_system_prompt(active_character)}]
    )

    idle_timeout_seconds = max(1.0, settings.proactive_effective_silence_timeout_ms / 1000)
    user_aggregator = LLMUserAggregator(
        context=context,
        params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=build_vad_params(settings)),
            user_turn_strategies=build_user_turn_strategies(settings, turn_analyzer),
            user_idle_timeout=idle_timeout_seconds,
        ),
    )
    assistant_aggregator = LLMAssistantAggregator(context=context)

    memory_injector = MemoryInjectionProcessor(
        context,
        manager_factory=lambda: build_manager(settings),
        enabled=settings.memory_effective_enabled,
    )

    async def _rolling_summarizer(prev_summary: str, turns: list[dict]) -> str:
        return await summarize_dropped_turns(settings, prev_summary, turns)

    context_window = ContextWindowProcessor(
        context,
        max_turns=getattr(settings, "llm_context_max_turns", 0),
        summarizer=_rolling_summarizer,
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            memory_injector,
            context_window,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @user_aggregator.event_handler("on_user_turn_idle")
    async def _on_user_idle(_aggregator) -> None:
        # The idle timer re-arms every time the bot stops speaking, so without
        # a cap a silent caller gets an identical check-in every timeout
        # forever. Allow at most N consecutive nudges (resets once the caller
        # says something) — a real person checks in once, then stays quiet.
        max_nudges = max(1, settings.proactive_effective_max_consecutive_prompts)
        if count_trailing_idle_nudges(context.get_messages()) >= max_nudges:
            logger.debug("idle nudge suppressed: %d consecutive already sent", max_nudges)
            return
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[{"role": "system", "content": IDLE_NUDGE_INSTRUCTION}],
                    run_llm=True,
                )
            ]
        )

    greeting_line = resolve_greeting_line(active_character)

    @transport.event_handler("on_client_connected")
    async def _on_client_connected(_transport, _client) -> None:
        logger.info("pipecat client connected")
        # Canned greeting straight to TTS: instant, and the assistant
        # aggregator records the spoken text into the context afterwards.
        await task.queue_frames([TTSSpeakFrame(greeting_line)])

    @transport.event_handler("on_client_disconnected")
    async def _on_client_disconnected(_transport, _client) -> None:
        logger.info("pipecat client disconnected")
        await task.cancel()

    return task, context, context_window


async def run_pipecat_session(
    websocket,
    settings: Settings,
    character_id: str | None = None,
    voice_id: str | None = None,
    model_id: str | None = None,
) -> None:
    """Build and run a Pipecat pipeline against an accepted browser WebSocket."""

    # Pin the transport output to the TTS rate: no pointless resample from the
    # 24 kHz pipeline default, and the room-tone mixer needs a known rate.
    out_sample_rate = settings.elevenlabs_sample_rate
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_out_sample_rate=out_sample_rate,
            audio_out_mixer=await asyncio.to_thread(
                build_room_tone_mixer, settings, out_sample_rate
            ),
            serializer=ProtobufFrameSerializer(),
        ),
    )

    turn_analyzer = await asyncio.to_thread(create_turn_analyzer, settings)
    task, context, context_window = build_session_task(
        transport,
        settings,
        character_id=character_id,
        voice_id=voice_id,
        model_id=model_id,
        turn_analyzer=turn_analyzer,
    )
    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    finally:
        await maybe_distill_context(
            context, settings, archived_messages=context_window.archived_messages
        )


async def run_twilio_session(
    websocket,
    stream_sid: str,
    call_sid: str,
    settings: Settings,
) -> None:
    """Build and run a Pipecat pipeline against a Twilio Media Streams WebSocket.

    Twilio media is 8 kHz µ-law; the TwilioFrameSerializer transcodes to/from
    the pipeline's PCM and speaks the Twilio Media Streams protocol. The
    `connected`/`start` handshake (which yields `stream_sid`/`call_sid`) must
    already have been consumed by the caller before this runs.
    """

    from pipecat.serializers.twilio import TwilioFrameSerializer

    has_credentials = bool(settings.twilio_account_sid and settings.twilio_auth_token)
    serializer = TwilioFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
        account_sid=settings.twilio_account_sid,
        auth_token=settings.twilio_auth_token,
        params=TwilioFrameSerializer.InputParams(auto_hang_up=has_credentials),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            audio_out_mixer=await asyncio.to_thread(build_room_tone_mixer, settings, 8000),
            serializer=serializer,
        ),
    )

    turn_analyzer = await asyncio.to_thread(create_turn_analyzer, settings)
    task, context, context_window = build_session_task(
        transport, settings, turn_analyzer=turn_analyzer
    )
    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    finally:
        await maybe_distill_context(
            context, settings, archived_messages=context_window.archived_messages
        )
