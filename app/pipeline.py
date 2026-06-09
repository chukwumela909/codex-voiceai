"""Pipecat 1.x voice pipeline (spike).

Runs side-by-side with the legacy `app.mock_conversation` pipeline. Wires
FastAPI WebSocket transport → Deepgram STT → Groq LLM (via OpenAI-compatible
endpoint) → Cartesia TTS, with Silero VAD for barge-in and a single
UserIdleController for the silence-nudge behavior.
"""

from __future__ import annotations

import logging

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, LLMContextFrame, LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_idle_controller import UserIdleController

from app.config import Settings


class UserIdleObserver(FrameProcessor):
    """Passthrough processor that feeds frames to a UserIdleController.

    UserIdleController is not itself a FrameProcessor in pipecat 1.x — it must
    be driven externally. This wrapper forwards every frame downstream and
    also hands it to the controller so its idle timer can advance.
    """

    def __init__(self, controller: UserIdleController, **kwargs):
        super().__init__(**kwargs)
        self._controller = controller

    async def setup(self, setup) -> None:
        await super().setup(setup)
        await self._controller.setup(setup.task_manager)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self._controller.process_frame(frame)
        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        await super().cleanup()
        await self._controller.cleanup()

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

GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
IDLE_NUDGE_INSTRUCTION = (
    "The caller has gone quiet. Offer one brief, warm check-in to let them know "
    "you're still on the line. Keep it under 18 words and do not ask why they went silent."
)
GREETING_INSTRUCTION = (
    "Open the call with one concise, spoken-friendly greeting. Sound natural and present. "
    "Do not mention the time of day. Keep it under 16 words."
)


def build_session_task(transport: FastAPIWebsocketTransport, settings: Settings) -> PipelineTask:
    """Wire STT/LLM/TTS, aggregators, and idle/greeting handlers onto a transport.

    Transport-agnostic so the browser (`run_pipecat_session`) and telephony
    (`run_twilio_session`) paths share identical pipeline behavior and only
    differ in how the transport is constructed.
    """

    stt = DeepgramSTTService(api_key=settings.deepgram_api_key)

    llm = OpenAILLMService(
        api_key=settings.groq_api_key,
        base_url=GROQ_OPENAI_BASE_URL,
        model=settings.groq_model,
    )

    tts = CartesiaTTSService(
        api_key=settings.cartesia_api_key,
        voice_id=settings.cartesia_voice_id,
        model=settings.cartesia_model,
        sample_rate=settings.cartesia_sample_rate,
    )

    from app.characters import build_system_prompt, get_character

    active_character = get_character(getattr(settings, "default_character_id", None))
    context = LLMContext(
        messages=[{"role": "system", "content": build_system_prompt(active_character)}]
    )
    user_aggregator = LLMUserAggregator(context=context)
    assistant_aggregator = LLMAssistantAggregator(context=context)

    idle_timeout_seconds = max(1.0, settings.proactive_effective_silence_timeout_ms / 1000)
    idle_controller = UserIdleController(user_idle_timeout=idle_timeout_seconds)
    idle_observer = UserIdleObserver(idle_controller)

    context_window = ContextWindowProcessor(
        context, max_turns=getattr(settings, "llm_context_max_turns", 0)
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            idle_observer,
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
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @idle_controller.event_handler("on_user_turn_idle")
    async def _on_user_idle(_controller) -> None:
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

    return task


async def run_pipecat_session(websocket, settings: Settings) -> None:
    """Build and run a Pipecat pipeline against an accepted browser WebSocket."""

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=ProtobufFrameSerializer(),
        ),
    )

    task = build_session_task(transport, settings)
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


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
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    task = build_session_task(transport, settings)
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
