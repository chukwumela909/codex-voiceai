"""Pipecat-based voice pipeline (spike).

Runs side-by-side with the legacy `app.mock_conversation` pipeline. Wires
SmallWebRTC transport → Deepgram STT → Groq LLM (via OpenAI-compatible
endpoint) → Cartesia TTS, with Silero VAD for barge-in and a single
UserIdleProcessor for the silence-nudge behavior.
"""

from __future__ import annotations

import logging

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

from app.config import Settings

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


async def run_pipecat_session(webrtc_connection, settings: Settings) -> None:
    """Build and run a Pipecat pipeline against an established WebRTC connection."""

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
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

    messages: list[dict[str, str]] = [{"role": "system", "content": settings.persona}]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    async def on_user_idle(processor: UserIdleProcessor, retry_count: int) -> bool:
        if retry_count > 2:
            return False
        await processor.push_frame(
            LLMMessagesFrame(
                [{"role": "system", "content": IDLE_NUDGE_INSTRUCTION}]
            )
        )
        return True

    idle_timeout_seconds = max(1, settings.proactive_effective_silence_timeout_ms // 1000)
    idle = UserIdleProcessor(callback=on_user_idle, timeout=idle_timeout_seconds)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            idle,
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client) -> None:
        logger.info("pipecat client connected")
        await task.queue_frames(
            [LLMMessagesFrame([{"role": "system", "content": GREETING_INSTRUCTION}])]
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client) -> None:
        logger.info("pipecat client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
