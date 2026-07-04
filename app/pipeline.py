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

import logging

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import Frame, LLMContextFrame, LLMMessagesAppendFrame
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

from app.config import Settings
from app.llm_models import MODELS, build_llm_service, resolve_active_model_key
from app.pipeline_memory import MemoryInjectionProcessor, maybe_distill_context
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


class ContextWindowProcessor(FrameProcessor):
    """Trims the shared LLM context to a sliding window before each LLM run.

    Sits just before the LLM service. Every LLM run funnels a downstream
    ``LLMContextFrame`` through here; we bound the shared context's message list
    in place so the LLM (and the next turn's aggregation) stay within budget.
    """

    def __init__(self, context: LLMContext, max_turns: int, **kwargs):
        super().__init__(**kwargs)
        self._context = context
        self._max_messages = max_turns * 2  # one user + one assistant per turn

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
            messages = self._context.get_messages()
            windowed = window_context_messages(messages, self._max_messages)
            if len(windowed) != len(messages):
                self._context.set_messages(windowed)
                logger.debug(
                    "context windowed: %d -> %d messages", len(messages), len(windowed)
                )
        await self.push_frame(frame, direction)

IDLE_NUDGE_INSTRUCTION = (
    "The caller has gone quiet. Offer one brief, warm check-in to let them know "
    "you're still on the line. Keep it under 18 words and do not ask why they went silent."
)
GREETING_INSTRUCTION = (
    "Open the call with one concise, spoken-friendly greeting. Sound natural and present. "
    "Do not mention the time of day. Keep it under 16 words."
)


def build_vad_params(settings: Settings) -> VADParams:
    return VADParams(
        confidence=settings.vad_confidence,
        start_secs=settings.vad_start_secs,
        stop_secs=settings.vad_stop_secs,
        min_volume=settings.vad_min_volume,
    )


def build_user_turn_strategies(settings: Settings) -> UserTurnStrategies:
    """Map turn-taking settings onto pipecat user turn strategies.

    Start: with ``interruption_min_words > 0``, a turn (and therefore an
    interruption while the bot speaks) only starts after that many transcribed
    words, so noise can't barge in. MinWords must be the sole start strategy —
    a VAD start strategy would open the turn first and the controller ignores
    later start triggers, so MinWords would never gate interruptions.

    Stop: smart turn hands VAD pauses to a local end-of-turn model (holds
    through mid-thought pauses up to ``smart_turn_stop_secs``); with it
    disabled, a flat ``speech_timeout_stop_secs`` pause ends the turn.
    """
    start = None
    if settings.interruption_min_words > 0:
        start = [MinWordsUserTurnStartStrategy(min_words=settings.interruption_min_words)]

    if settings.smart_turn_enabled:
        stop = [
            TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=LocalSmartTurnAnalyzerV3(
                    params=SmartTurnParams(stop_secs=settings.smart_turn_stop_secs)
                )
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


def build_session_task(
    transport: FastAPIWebsocketTransport,
    settings: Settings,
    character_id: str | None = None,
    voice_id: str | None = None,
    model_id: str | None = None,
) -> tuple[PipelineTask, LLMContext]:
    """Wire STT/LLM/TTS, aggregators, and idle/greeting handlers onto a transport.

    Transport-agnostic so the browser (`run_pipecat_session`) and telephony
    (`run_twilio_session`) paths share identical pipeline behavior and only
    differ in how the transport is constructed. Returns the task plus the
    shared LLM context so callers can distill memory after the session ends.
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
    # model — see app/llm_models.py MODELS (Groq direct or an OpenRouter model).
    model_key = resolve_active_model_key(settings, override=model_id)
    llm = build_llm_service(settings, model_key)
    logger.info(
        "pipeline llm model=%s (%s:%s)",
        model_key,
        MODELS[model_key]["provider"],
        MODELS[model_key]["model"],
    )

    tts = ElevenLabsTTSService(
        api_key=settings.elevenlabs_api_key,
        sample_rate=settings.elevenlabs_sample_rate,
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
            user_turn_strategies=build_user_turn_strategies(settings),
            user_idle_timeout=idle_timeout_seconds,
        ),
    )
    assistant_aggregator = LLMAssistantAggregator(context=context)

    memory_injector = MemoryInjectionProcessor(
        context,
        manager_factory=lambda: build_manager(settings),
        enabled=settings.memory_effective_enabled,
    )

    context_window = ContextWindowProcessor(
        context, max_turns=getattr(settings, "llm_context_max_turns", 0)
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
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[{"role": "system", "content": IDLE_NUDGE_INSTRUCTION}],
                    run_llm=True,
                )
            ]
        )

    @transport.event_handler("on_client_connected")
    async def _on_client_connected(_transport, _client) -> None:
        logger.info("pipecat client connected")
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[{"role": "system", "content": GREETING_INSTRUCTION}],
                    run_llm=True,
                )
            ]
        )

    @transport.event_handler("on_client_disconnected")
    async def _on_client_disconnected(_transport, _client) -> None:
        logger.info("pipecat client disconnected")
        await task.cancel()

    return task, context


async def run_pipecat_session(
    websocket,
    settings: Settings,
    character_id: str | None = None,
    voice_id: str | None = None,
    model_id: str | None = None,
) -> None:
    """Build and run a Pipecat pipeline against an accepted browser WebSocket."""

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=ProtobufFrameSerializer(),
        ),
    )

    task, context = build_session_task(
        transport, settings, character_id=character_id, voice_id=voice_id, model_id=model_id
    )
    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    finally:
        await maybe_distill_context(context, settings)


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
            serializer=serializer,
        ),
    )

    task, context = build_session_task(transport, settings)
    runner = PipelineRunner(handle_sigint=False)
    try:
        await runner.run(task)
    finally:
        await maybe_distill_context(context, settings)
