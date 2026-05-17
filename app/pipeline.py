"""Pipecat 1.x voice pipeline (spike).

Runs side-by-side with the legacy `app.mock_conversation` pipeline. Wires
FastAPI WebSocket transport → Deepgram STT → Groq LLM (via OpenAI-compatible
endpoint) → Cartesia TTS, with Silero VAD for barge-in and a single
UserIdleController for the silence-nudge behavior.
"""

from __future__ import annotations

import logging

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, LLMMessagesAppendFrame
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

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self._controller.process_frame(frame)
        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        await super().cleanup()
        await self._controller.cleanup()

logger = logging.getLogger("voice_agent.pipecat")

GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
IDLE_NUDGE_INSTRUCTION = (
    "The caller has gone quiet. Offer one brief, warm check-in to let them know "
    "you're still on the line. Keep it under 18 words and do not ask why they went silent."
)
GREETING_INSTRUCTION = (
    "Open the call with one concise, spoken-friendly greeting. Sound natural and present. "
    "Do not mention the time of day. Keep it under 16 words."
)


async def run_pipecat_session(websocket, settings: Settings) -> None:
    """Build and run a Pipecat pipeline against an accepted FastAPI WebSocket."""

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

    context = LLMContext(messages=[{"role": "system", "content": settings.persona}])
    user_aggregator = LLMUserAggregator(context=context)
    assistant_aggregator = LLMAssistantAggregator(context=context)

    idle_timeout_seconds = max(1.0, settings.proactive_effective_silence_timeout_ms / 1000)
    idle_controller = UserIdleController(user_idle_timeout=idle_timeout_seconds)
    idle_observer = UserIdleObserver(idle_controller)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            idle_observer,
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

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
