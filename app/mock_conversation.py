import asyncio
import base64
import math
import struct
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from uuid import uuid4

from app.cartesia_tts import CartesiaStreamingTTS, generate_cartesia_context_id
from app.config import is_uuid
from app.conversation_context import agent_transcript_with_intent_inference
from app.deepgram import DeepgramStreamingTranscriber
from app.events import event
from app.exceptions import ClientConnectionClosed
from app.groq_agent import GroqStreamingAgent, pop_speakable_chunks
from app.proactive import (
    TRIGGER_CONTEXTUAL_FOLLOW_UP,
    TRIGGER_SILENCE_NUDGE,
    TRIGGER_STARTUP_GREETING,
    ProactiveContext,
    ProactivePolicy,
    ProactivePolicyConfig,
)
from app.speech_director import SpeechDirectionConfig, direct_speech_for_cartesia

SendEvent = Callable[[dict], Awaitable[None]]
RECENT_CONTEXT_TURN_LIMIT = 8
MOCK_TURN_SPEECH_BYTES_THRESHOLD = 24000
MOCK_SPEECH_RMS_THRESHOLD = 0.01
PROACTIVE_STARTUP_GREETING_INSTRUCTION = (
    "Initiate one concise, spoken-friendly opening greeting for an ambiguous open phone call. "
    "Sound natural, present, and conversational, as if you are still on the line with the caller. "
    "Do not mention the time of day. Do not imply who called whom. "
    "Do not use canned phrases. Keep it under 16 words."
)
PROACTIVE_FOLLOWUP_INSTRUCTION = (
    "Initiate one concise, spoken-friendly proactive turn for an ambiguous open phone call "
    "based on the recent conversation. "
    "Use one brief question or observation. Stay within the configured persona. Do not introduce a new topic. "
    "Do not ask why the user went silent. Do not apologize. Keep it under 25 words."
)
PROACTIVE_SILENCE_NUDGE_INSTRUCTION = (
    "Initiate one concise, spoken-friendly brief check-in for an idle open phone call. "
    "Make it clear you are still on the line without pressuring the caller. "
    "Sound natural and present, not canned. Avoid repeating recent assistant wording. "
    "Do not ask why the user went silent. Do not apologize. Keep it under 18 words."
)


class MockConversationSession:
    def __init__(self, session_id: str, send_event: SendEvent, settings) -> None:
        self.session_id = session_id
        self.send_event = send_event
        self.settings = settings
        self.audio_config: dict | None = None
        self.transcriber: DeepgramStreamingTranscriber | None = None
        self.agent: GroqStreamingAgent | None = None
        self.synthesizer: CartesiaStreamingTTS | None = None
        self.proactive_config = ProactivePolicyConfig.from_settings(settings)
        self.proactive_policy = ProactivePolicy(self.proactive_config)
        self.transcript: list[dict[str, str]] = []
        self.bytes_received = 0
        self.mock_speech_bytes_received = 0
        self.frames_received = 0
        self.turn_started_at: float | None = None
        self.current_task: asyncio.Task | None = None
        self.pending_transcript_task: asyncio.Task | None = None
        self.pending_proactive_task: asyncio.Task | None = None
        self.pending_proactive_trigger_reason: str | None = None
        self.live_transcript_sequence = 0
        self.active_response_id: str | None = None
        self.active_response_started_at: float | None = None
        self.active_user_text: str | None = None
        self.active_assistant_text = ""
        self.active_response_metadata: dict[str, object] = {}
        self.text_latency_reported_response_ids: set[str] = set()
        self.final_transcript_buffer: list[str] = []
        self.latest_partial_text: str | None = None
        self.interrupted_response_ids: set[str] = set()
        self.audio_stream_active = False
        self.received_audio_frame = False
        self.user_has_spoken = False
        self.startup_greeting_sent = False
        self.consecutive_proactive_prompts = 0
        self.proactive_failures = 0
        self.proactive_cooldown_until_ms: int | None = None
        self.failed_proactive_turn_ids: set[str] = set()
        self.closed = False

    async def configure_audio(self, payload: dict) -> None:
        self.audio_config = {
            "encoding": payload.get("encoding", "pcm_s16le"),
            "sample_rate": int(payload.get("sample_rate", 16000)),
            "channels": int(payload.get("channels", 1)),
            "frame_duration_ms": int(payload.get("frame_duration_ms", 20)),
        }
        self.audio_stream_active = True
        if self.settings.normalized_mode == "live":
            await self._start_live_transcriber()
        await self.send_event(event("status.changed", self.session_id, {"state": "listening"}))
        await self.send_event(
            event(
                "pipeline.stage",
                self.session_id,
                {"stage": "stt_listening", "provider": "deepgram" if self.transcriber else "mock"},
            )
        )

    async def receive_audio(self, frame: bytes) -> None:
        if self.closed:
            return

        self.frames_received += 1
        if frame:
            self.received_audio_frame = True
        self.bytes_received += len(frame)
        input_gain = float(getattr(self.settings, "input_gain", 1.0))
        raw_level = calculate_pcm_level(frame)
        processed_frame = apply_pcm_gain(frame, input_gain)
        level = calculate_pcm_level(processed_frame)
        mock_speech_detected = self.transcriber is None and is_mock_speech_frame(level)
        if self.frames_received == 1 or self.frames_received % 100 == 0:
            await self.send_event(
                event(
                    "audio.input",
                    self.session_id,
                    {
                        "frames_received": self.frames_received,
                        "bytes_received": self.bytes_received,
                        "frame_bytes": len(frame),
                        "rms": level["rms"],
                        "peak": level["peak"],
                        "raw_rms": raw_level["rms"],
                        "raw_peak": raw_level["peak"],
                        "input_gain": input_gain,
                        "mock_speech_bytes_received": self.mock_speech_bytes_received,
                        "speech_detected": mock_speech_detected,
                        "sample_rate": self.audio_config["sample_rate"] if self.audio_config else None,
                        "mode": self.settings.normalized_mode,
                        "provider": "deepgram" if self.transcriber else "mock",
                    },
                )
            )
        if self.frames_received == 1:
            await self._maybe_schedule_startup_greeting()

        if self.transcriber:
            try:
                await self.transcriber.send_audio(processed_frame)
                return
            except Exception as exc:
                await self.handle_live_error(f"Deepgram audio send failed: {exc}")
                mock_speech_detected = is_mock_speech_frame(level)

        if not mock_speech_detected:
            return

        self.mock_speech_bytes_received += len(processed_frame)
        if self.turn_started_at is None:
            self.turn_started_at = time.perf_counter()

        if self.current_task is None and self.mock_speech_bytes_received >= MOCK_TURN_SPEECH_BYTES_THRESHOLD:
            await self._mark_user_speech_started("mock_audio_threshold")
            self.current_task = asyncio.create_task(self._run_mock_turn(self.mock_speech_bytes_received))
            self.mock_speech_bytes_received = 0

    async def stop_audio(self) -> None:
        self.audio_stream_active = False
        await self._cancel_pending_proactive("audio_stop")

    async def close(self) -> None:
        self.closed = True
        if self.transcriber:
            await self.transcriber.close()
        if self.pending_proactive_task and not self.pending_proactive_task.done():
            self.pending_proactive_task.cancel()
            try:
                await self.pending_proactive_task
            except asyncio.CancelledError:
                pass
        if self.pending_transcript_task and not self.pending_transcript_task.done():
            self.pending_transcript_task.cancel()
            try:
                await self.pending_transcript_task
            except asyncio.CancelledError:
                pass
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass
        self.current_task = None
        self.active_response_id = None
        self.active_response_started_at = None
        self.active_user_text = None
        self.active_assistant_text = ""
        self.active_response_metadata = {}
        self.final_transcript_buffer = []
        self.latest_partial_text = None
        self.pending_proactive_task = None
        self.pending_proactive_trigger_reason = None
        self.proactive_cooldown_until_ms = None
        self.active_response_started_at = None

    async def _start_live_transcriber(self) -> None:
        if self.transcriber:
            return
        if not self.settings.deepgram_api_key:
            await self.send_event(
                event(
                    "error",
                    self.session_id,
                    {
                        "message": "VOICE_AGENT_MODE=live requires DEEPGRAM_API_KEY for Phase 3 transcription.",
                        "provider": "deepgram",
                    },
                )
            )
            return

        assert self.audio_config is not None
        self.transcriber = DeepgramStreamingTranscriber(
            api_key=self.settings.deepgram_api_key,
            model=self.settings.deepgram_model,
            encoding=self.audio_config["encoding"].replace("pcm_s16le", "linear16"),
            sample_rate=self.audio_config["sample_rate"],
            channels=self.audio_config["channels"],
            endpointing_ms=self.settings.deepgram_endpointing_ms,
            utterance_end_ms=self.settings.deepgram_utterance_end_ms,
            on_transcript=self.handle_live_transcript,
            on_error=self.handle_live_error,
        )
        try:
            await self.transcriber.start()
        except Exception as exc:
            self.transcriber = None
            await self.send_provider_error("deepgram", f"Deepgram connection failed: {exc}")
            await self.send_event(
                event(
                    "pipeline.stage",
                    self.session_id,
                    {"stage": "stt_fallback", "provider": "mock", "fallback_from": "deepgram"},
                )
            )
            return

        await self.send_event(
            event(
                "status.changed",
                self.session_id,
                {
                    "state": "transcriber_connected",
                    "provider": "deepgram",
                    "model": self.settings.deepgram_model,
                },
            )
        )

    async def handle_live_transcript(self, transcript: dict) -> None:
        if transcript.get("type") == "speech_started":
            await self._handle_speech_started(transcript)
            return

        if transcript.get("type") == "utterance_end":
            await self._handle_utterance_end(transcript)
            return

        if is_meaningful_user_speech(transcript.get("text", "")):
            await self._mark_user_speech_started("live_transcript")

        if self.turn_started_at is None:
            self.turn_started_at = time.perf_counter()

        await self.send_event(
            event(
                "transcript.final" if transcript["is_final"] else "transcript.partial",
                self.session_id,
                {
                    "text": transcript["text"],
                    "confidence": transcript["confidence"],
                    "is_final": transcript["is_final"],
                    "speech_final": transcript["speech_final"],
                    "provider": "deepgram",
                },
            )
        )
        if transcript["is_final"]:
            self._buffer_final_transcript(transcript["text"])
        else:
            self.latest_partial_text = transcript["text"]

        if self.current_task is not None:
            if not should_interrupt_active_response(
                transcript,
                active_user_text=self.active_user_text,
                active_assistant_text=self.active_assistant_text,
            ):
                return
            await self._interrupt_active_response(transcript["text"])

        if self.current_task is not None:
            return

        self.live_transcript_sequence += 1
        if transcript["speech_final"]:
            await self._start_buffered_agent_response(transcript["text"], reason="speech_final")
            return

        if self.pending_transcript_task and not self.pending_transcript_task.done():
            self.pending_transcript_task.cancel()
        self.pending_transcript_task = asyncio.create_task(
            self._finalize_partial_transcript_after_idle(transcript["text"], self.live_transcript_sequence)
        )

    async def _handle_speech_started(self, transcript: dict) -> None:
        await self._mark_user_speech_started("live_speech_started")
        if self.turn_started_at is None:
            self.turn_started_at = time.perf_counter()
        await self.send_event(event("status.changed", self.session_id, {"state": "hearing"}))
        await self.send_event(
            event(
                "pipeline.stage",
                self.session_id,
                {
                    "stage": "stt_speech_started",
                    "provider": "deepgram",
                    "timestamp": transcript.get("timestamp"),
                },
            )
        )

    async def handle_live_error(self, message: str) -> None:
        failed_transcriber = self.transcriber
        self.transcriber = None
        if failed_transcriber:
            await failed_transcriber.close()
        await self.send_provider_error("deepgram", message)
        await self.send_event(
            event(
                "pipeline.stage",
                self.session_id,
                {"stage": "stt_fallback", "provider": "mock", "fallback_from": "deepgram"},
            )
        )

    async def _handle_utterance_end(self, transcript: dict) -> None:
        if self.current_task is not None:
            return
        await self.send_event(
            event(
                "pipeline.stage",
                self.session_id,
                {
                    "stage": "stt_utterance_end",
                    "provider": "deepgram",
                    "last_word_end": transcript.get("last_word_end"),
                },
            )
        )
        await self._start_buffered_agent_response(self.latest_partial_text, reason="utterance_end")

    async def _finalize_partial_transcript_after_idle(self, text: str, sequence: int) -> None:
        idle_ms = max(1, int(getattr(self.settings, "partial_idle_finalize_ms", 650)))
        idle_seconds = idle_ms / 1000
        try:
            await asyncio.sleep(idle_seconds)
        except asyncio.CancelledError:
            raise

        if self.closed or self.current_task is not None or sequence != self.live_transcript_sequence:
            return

        await self.send_event(
            event(
                "transcript.final",
                self.session_id,
                {
                    "text": text,
                    "confidence": None,
                    "is_final": False,
                    "speech_final": False,
                    "provider": "deepgram",
                    "finalization_reason": "partial_idle_timeout",
                },
            )
        )
        await self.send_event(
            event(
                "latency.metric",
                self.session_id,
                {
                    "name": "partial_idle_finalize_wait_ms",
                    "value_ms": idle_ms,
                    "finalization_reason": "partial_idle_timeout",
                },
            )
        )
        await self._start_buffered_agent_response(text, reason="partial_idle_timeout")

    async def _start_buffered_agent_response(self, fallback_text: str | None, *, reason: str) -> None:
        text = self._consume_buffered_transcript(fallback_text)
        if not text:
            return
        await self._start_live_agent_response(text, reason=reason)

    async def _start_live_agent_response(self, text: str, *, reason: str) -> None:
        active_task = asyncio.current_task()
        if (
            self.pending_transcript_task
            and self.pending_transcript_task is not active_task
            and not self.pending_transcript_task.done()
        ):
            self.pending_transcript_task.cancel()
        await self.send_event(
            event(
                "pipeline.stage",
                self.session_id,
                {"stage": "stt_final", "provider": "deepgram", "text": text, "reason": reason},
            )
        )
        self.current_task = asyncio.create_task(self._run_agent_response(text))

    async def _maybe_schedule_startup_greeting(self) -> None:
        if self.closed or self.startup_greeting_sent or self.user_has_spoken:
            return
        if self.pending_proactive_task and not self.pending_proactive_task.done():
            return
        if self.current_task is not None and not self.current_task.done():
            return
        context = self._proactive_context()
        decision = self.proactive_policy.evaluate(TRIGGER_STARTUP_GREETING, context, now_ms=monotonic_ms())
        if not decision.allowed and decision.skip_reason != "waiting_for_audio_frame":
            await self.send_event(event(decision.event_type, self.session_id, decision.payload))
            return
        await self._schedule_proactive_timer(
            TRIGGER_STARTUP_GREETING,
            self.proactive_config.startup_greeting_delay_ms,
            cooldown=False,
        )

    async def _schedule_proactive_timer(self, trigger_reason: str, delay_ms: int, *, cooldown: bool) -> None:
        if self.closed:
            return
        if self.pending_proactive_task and not self.pending_proactive_task.done():
            return
        delay_ms = max(0, delay_ms)
        self.pending_proactive_trigger_reason = trigger_reason
        if cooldown:
            self.proactive_cooldown_until_ms = monotonic_ms() + delay_ms
            await self.send_event(
                event(
                    "proactive.cooldown",
                    self.session_id,
                    {
                        "trigger_reason": trigger_reason,
                        "cooldown_ms": delay_ms,
                        "next_eligible_at_ms": self.proactive_cooldown_until_ms,
                    },
                )
            )
        self.pending_proactive_task = asyncio.create_task(self._run_proactive_timer(trigger_reason, delay_ms=delay_ms))

    async def _run_proactive_timer(self, trigger_reason: str, *, delay_ms: int | None = None) -> None:
        try:
            if delay_ms is None:
                delay_ms = self.proactive_config.startup_greeting_delay_ms
            await asyncio.sleep(max(0, delay_ms) / 1000)
            if self.closed:
                return

            if self.proactive_cooldown_until_ms is not None:
                self.proactive_cooldown_until_ms = None
            context = self._proactive_context()
            decision = self.proactive_policy.evaluate(trigger_reason, context, now_ms=monotonic_ms())
            proactive_turn_id = f"pro_{uuid4().hex}"
            payload = {**decision.payload, "proactive_turn_id": proactive_turn_id}
            await self.send_event(event(decision.event_type, self.session_id, payload))
            if not decision.allowed:
                if decision.skip_reason == "max_consecutive_prompts_reached":
                    await self.send_event(
                        event(
                            "proactive.state",
                            self.session_id,
                            {
                                "state": "backed_off",
                                "trigger_reason": trigger_reason,
                                "consecutive_prompts": self.consecutive_proactive_prompts,
                                "max_consecutive_prompts": self.proactive_config.max_consecutive_prompts,
                            },
                        )
                    )
                return

            metadata = proactive_metadata(trigger_reason, proactive_turn_id)
            if trigger_reason == TRIGGER_STARTUP_GREETING:
                self.startup_greeting_sent = True
            prompt_index = self.consecutive_proactive_prompts
            self.consecutive_proactive_prompts += 1
            if self._should_use_live_proactive_agent(trigger_reason):
                self.current_task = asyncio.create_task(self._run_proactive_agent_response(metadata=metadata))
            else:
                message = scripted_proactive_message(
                    trigger_reason,
                    prompt_index,
                    transcript=self.transcript,
                )
                self.current_task = asyncio.create_task(
                    self._run_proactive_scripted_response(message, metadata=metadata)
                )
        except asyncio.CancelledError:
            raise
        finally:
            self.pending_proactive_task = None
            self.pending_proactive_trigger_reason = None

    def _should_use_live_proactive_agent(self, trigger_reason: str) -> bool:
        return bool(
            self.settings.normalized_mode == "live"
            and self.settings.groq_api_key
            and trigger_reason in {TRIGGER_STARTUP_GREETING, TRIGGER_CONTEXTUAL_FOLLOW_UP, TRIGGER_SILENCE_NUDGE}
        )

    def _proactive_context(self) -> ProactiveContext:
        active_response = self.current_task is not None and not self.current_task.done()
        return ProactiveContext(
            session_state="listening" if self.audio_stream_active else "connected",
            audio_stream_active=self.audio_stream_active,
            received_audio_frame=self.received_audio_frame,
            closing=self.closed,
            active_response=active_response,
            user_has_spoken=self.user_has_spoken,
            startup_greeting_sent=self.startup_greeting_sent,
            consecutive_prompts=self.consecutive_proactive_prompts,
            proactive_failures=self.proactive_failures,
            cooldown_until_ms=self.proactive_cooldown_until_ms,
            has_recent_user_context=has_recent_user_context(self.transcript),
            last_assistant_asked_question=last_assistant_asked_question(self.transcript),
            last_assistant_has_next_step=last_assistant_has_next_step(self.transcript),
        )

    async def _mark_user_speech_started(self, reason: str) -> None:
        self.user_has_spoken = True
        self.consecutive_proactive_prompts = 0
        self.proactive_cooldown_until_ms = None
        await self._cancel_pending_proactive(reason)

    async def _cancel_pending_proactive(self, reason: str) -> None:
        if not self.pending_proactive_task or self.pending_proactive_task.done():
            return
        trigger_reason = self.pending_proactive_trigger_reason or TRIGGER_STARTUP_GREETING
        self.pending_proactive_task.cancel()
        try:
            await self.pending_proactive_task
        except asyncio.CancelledError:
            pass
        self.pending_proactive_task = None
        self.pending_proactive_trigger_reason = None
        self.proactive_cooldown_until_ms = None
        await self.send_event(
            event(
                "proactive.cancelled",
                self.session_id,
                {
                    "trigger_reason": trigger_reason,
                    "reason": reason,
                },
            )
        )

    def _buffer_final_transcript(self, text: str) -> None:
        normalized_text = normalize_turn_text(text)
        if not normalized_text:
            return
        if any(normalize_turn_text(existing) == normalized_text for existing in self.final_transcript_buffer):
            return
        self.final_transcript_buffer.append(text)
        self.latest_partial_text = text

    def _consume_buffered_transcript(self, fallback_text: str | None) -> str:
        if fallback_text:
            self._buffer_final_transcript(fallback_text)
        text = " ".join(part.strip() for part in self.final_transcript_buffer if part.strip()).strip()
        if not text and fallback_text:
            text = fallback_text.strip()
        self.final_transcript_buffer = []
        self.latest_partial_text = None
        return text

    async def _interrupt_active_response(
        self,
        next_user_text: str,
        *,
        reason: str = "user_speech_during_response",
    ) -> None:
        interrupted_response_id = self.active_response_id
        metadata = dict(self.active_response_metadata)
        await self.send_event(
            event(
                "interruption.started",
                self.session_id,
                {
                    "interrupted_response_id": interrupted_response_id,
                    "next_user_text": next_user_text,
                    "reason": reason,
                    **metadata,
                },
            )
        )
        await self.send_event(
            event(
                "pipeline.stage",
                self.session_id,
                {
                    "stage": "interrupted",
                    "provider": "conversation",
                    "interrupted_response_id": interrupted_response_id,
                    **metadata,
                },
            )
        )
        if interrupted_response_id:
            self.interrupted_response_ids.add(interrupted_response_id)
        if self.current_task and not self.current_task.done():
            self.current_task.cancel()
            try:
                await self.current_task
            except asyncio.CancelledError:
                pass
        self.current_task = None
        self.active_response_id = None
        self.active_user_text = None
        self.active_response_metadata = {}
        await self.send_event(event("status.changed", self.session_id, {"state": "listening"}))

    async def _run_mock_turn(self, bytes_seen: int) -> None:
        await self.send_event(event("status.changed", self.session_id, {"state": "hearing"}))
        await self.send_event(
            event(
                "transcript.partial",
                self.session_id,
                {
                    "text": "I can hear the microphone stream...",
                    "bytes_seen": bytes_seen,
                    "is_final": False,
                    "provider": "mock",
                },
            )
        )
        await asyncio.sleep(0.08)
        await self.send_event(
            event(
                "transcript.final",
                self.session_id,
                {
                    "text": "Testing the browser microphone.",
                    "bytes_seen": bytes_seen,
                    "is_final": True,
                    "speech_final": True,
                    "provider": "mock",
                },
            )
        )
        await self._run_agent_response("Testing the browser microphone.")

    async def _run_agent_response(self, user_text: str) -> None:
        response_id = f"resp_{uuid4().hex}"
        self.active_response_id = response_id
        self.active_response_started_at = time.perf_counter()
        self.active_user_text = user_text
        self.active_assistant_text = ""
        self.active_response_metadata = {}
        turn_started_at = self.turn_started_at or time.perf_counter()
        final_detected_at = time.perf_counter()

        await self.send_event(
            event(
                "latency.metric",
                self.session_id,
                {
                    "name": "mock_turn_detection_ms",
                    "value_ms": round((final_detected_at - turn_started_at) * 1000, 2),
                },
            )
        )

        await self.send_event(event("status.changed", self.session_id, {"state": "thinking"}))
        await self.send_event(
            event(
                "pipeline.stage",
                self.session_id,
                {"stage": "llm_streaming", "provider": "groq" if self.settings.normalized_mode == "live" else "mock"},
            )
        )
        self.transcript.append({"role": "user", "content": user_text})

        speech_already_streamed = False
        if (
            self.settings.normalized_mode == "live"
            and self.settings.groq_api_key
            and self._has_usable_cartesia_config()
            and self._uses_default_live_agent_response()
        ):
            message = await self._stream_live_agent_response_with_streaming_speech(response_id, final_detected_at)
            speech_already_streamed = True
        elif self.settings.normalized_mode == "live":
            message = await self._stream_live_agent_response(response_id)
        else:
            message = await self._stream_mock_agent_response(response_id, user_text)

        if response_id in self.interrupted_response_ids:
            return

        await self.send_event(event("agent.text.final", self.session_id, {"response_id": response_id, "text": message}))
        await self.send_event(
            event(
                "pipeline.stage",
                self.session_id,
                {"stage": "llm_done", "provider": "groq" if self.settings.normalized_mode == "live" else "mock"},
            )
        )
        if not speech_already_streamed:
            await self._stream_speech(response_id, message, final_detected_at)
        if response_id in self.interrupted_response_ids:
            return

        self.transcript.append({"role": "assistant", "content": message})
        await self.send_event(event("status.changed", self.session_id, {"state": "listening"}))
        self.turn_started_at = None
        self.current_task = None
        self.active_response_id = None
        self.active_response_started_at = None
        self.active_user_text = None
        self.active_assistant_text = ""
        self.active_response_metadata = {}
        await self._schedule_next_idle_trigger(after_trigger_reason=None)

    async def _run_proactive_scripted_response(self, message: str, *, metadata: dict) -> None:
        response_id = f"resp_{uuid4().hex}"
        self.active_response_id = response_id
        self.active_response_started_at = time.perf_counter()
        self.active_user_text = None
        self.active_assistant_text = ""
        self.active_response_metadata = dict(metadata)
        final_detected_at = time.perf_counter()

        try:
            await self.send_event(event("status.changed", self.session_id, {"state": "thinking"}))
            await self.send_event(
                event(
                    "pipeline.stage",
                    self.session_id,
                    {
                        "stage": "llm_streaming",
                        "provider": "scripted",
                        **metadata,
                    },
                )
            )
            await self._emit_agent_text_chunks(response_id, split_mock_response(message), metadata=metadata)
            if response_id in self.interrupted_response_ids:
                return

            await self.send_event(
                event("agent.text.final", self.session_id, {"response_id": response_id, "text": message, **metadata})
            )
            await self.send_event(
                event(
                    "pipeline.stage",
                    self.session_id,
                    {
                        "stage": "llm_done",
                        "provider": "scripted",
                        **metadata,
                    },
                )
            )
            await self._stream_speech(response_id, message, final_detected_at, metadata=metadata)
            if response_id in self.interrupted_response_ids:
                return

            self.transcript.append({"role": "assistant", "content": message})
            await self.send_event(event("status.changed", self.session_id, {"state": "listening"}))
        finally:
            was_interrupted = response_id in self.interrupted_response_ids
            trigger_reason = str(metadata.get("trigger_reason", ""))
            if self.active_response_id == response_id:
                self.active_response_id = None
                self.active_response_started_at = None
            self.current_task = None
            self.active_user_text = None
            self.active_assistant_text = ""
            self.active_response_metadata = {}
            if not was_interrupted and not self._proactive_turn_failed(metadata):
                self.proactive_failures = 0
            if not was_interrupted:
                await self._schedule_next_idle_trigger(after_trigger_reason=trigger_reason)

    async def _run_proactive_contextual_agent_response(self, *, metadata: dict) -> None:
        await self._run_proactive_agent_response(metadata=metadata)

    async def _run_proactive_agent_response(self, *, metadata: dict) -> None:
        response_id = f"resp_{uuid4().hex}"
        self.active_response_id = response_id
        self.active_response_started_at = time.perf_counter()
        self.active_user_text = None
        self.active_assistant_text = ""
        self.active_response_metadata = dict(metadata)
        final_detected_at = time.perf_counter()

        try:
            await self.send_event(event("status.changed", self.session_id, {"state": "thinking"}))
            await self.send_event(
                event(
                    "pipeline.stage",
                    self.session_id,
                    {
                        "stage": "llm_streaming",
                        "provider": "groq" if self.settings.groq_api_key else "scripted",
                        **metadata,
                    },
                )
            )
            message = await self._stream_live_proactive_response(response_id, metadata=metadata)
            if not message:
                return
            if response_id in self.interrupted_response_ids:
                return

            await self.send_event(
                event("agent.text.final", self.session_id, {"response_id": response_id, "text": message, **metadata})
            )
            await self.send_event(
                event(
                    "pipeline.stage",
                    self.session_id,
                    {
                        "stage": "llm_done",
                        "provider": "groq" if self.settings.groq_api_key else "scripted",
                        **metadata,
                    },
                )
            )
            await self._stream_speech(response_id, message, final_detected_at, metadata=metadata)
            if response_id in self.interrupted_response_ids:
                return

            self.transcript.append({"role": "assistant", "content": message})
            await self.send_event(event("status.changed", self.session_id, {"state": "listening"}))
        finally:
            was_interrupted = response_id in self.interrupted_response_ids
            trigger_reason = str(metadata.get("trigger_reason", ""))
            if self.active_response_id == response_id:
                self.active_response_id = None
                self.active_response_started_at = None
            self.current_task = None
            self.active_user_text = None
            self.active_assistant_text = ""
            self.active_response_metadata = {}
            if not was_interrupted and not self._proactive_turn_failed(metadata):
                self.proactive_failures = 0
            if not was_interrupted:
                await self._schedule_next_idle_trigger(after_trigger_reason=trigger_reason)

    async def _schedule_next_idle_trigger(self, *, after_trigger_reason: str | None) -> None:
        if not self.proactive_config.enabled:
            return
        if self.closed or not self.audio_stream_active or not self.received_audio_frame:
            return
        if self.pending_proactive_task and not self.pending_proactive_task.done():
            return
        if self.current_task is not None and not self.current_task.done():
            return

        context = self._proactive_context()
        trigger_reason = self.proactive_policy.select_idle_trigger(context)
        decision = self.proactive_policy.evaluate(trigger_reason, context, now_ms=monotonic_ms())
        if not decision.allowed:
            await self.send_event(event(decision.event_type, self.session_id, decision.payload))
            if decision.skip_reason in {"failure_backoff", "max_consecutive_prompts_reached"}:
                await self.send_event(
                    event(
                        "proactive.state",
                        self.session_id,
                        {
                            "state": "backed_off",
                            "trigger_reason": trigger_reason,
                            "consecutive_prompts": self.consecutive_proactive_prompts,
                            "max_consecutive_prompts": self.proactive_config.max_consecutive_prompts,
                        },
                    )
                )
            return

        use_repeat_cooldown = after_trigger_reason in {TRIGGER_CONTEXTUAL_FOLLOW_UP, TRIGGER_SILENCE_NUDGE}
        delay_ms = (
            self.proactive_config.repeat_cooldown_ms
            if use_repeat_cooldown
            else self.proactive_config.silence_timeout_ms
        )
        await self.send_event(
            event(
                "proactive.state",
                self.session_id,
                {
                    "state": "idle_monitoring",
                    "trigger_reason": trigger_reason,
                    "delay_ms": delay_ms,
                    "consecutive_prompts": self.consecutive_proactive_prompts,
                },
            )
        )
        await self._schedule_proactive_timer(trigger_reason, delay_ms, cooldown=use_repeat_cooldown)

    async def _stream_speech(
        self,
        response_id: str,
        message: str,
        final_detected_at: float,
        *,
        metadata: dict | None = None,
    ) -> None:
        metadata = metadata or {}
        await self.send_event(event("status.changed", self.session_id, {"state": "speaking"}))
        await self.send_event(
            event(
                "pipeline.stage",
                self.session_id,
                {
                    "stage": "tts_streaming",
                    "provider": "cartesia" if self.settings.normalized_mode == "live" else "mock",
                    **metadata,
                },
            )
        )
        if self._has_usable_cartesia_config():
            cartesia_succeeded = await self._stream_cartesia_speech(
                response_id,
                message,
                final_detected_at,
                metadata=metadata,
            )
            if cartesia_succeeded:
                return

            await self._stream_tts_fallback(
                response_id,
                final_detected_at,
                fallback_from="cartesia",
                metadata=metadata,
            )
            return

        if (
            self.settings.normalized_mode == "live"
            and self.settings.cartesia_api_key
            and self.settings.cartesia_voice_id
            and not is_uuid(self.settings.cartesia_voice_id)
        ):
            await self.send_provider_error(
                "cartesia",
                "CARTESIA_VOICE_ID must be a UUID. Falling back to mock audio.",
                metadata=metadata,
            )
            await self._stream_tts_fallback(
                response_id,
                final_detected_at,
                fallback_from="cartesia",
                metadata=metadata,
            )
            return

        await self._stream_mock_speech(response_id, final_detected_at, metadata=metadata)
        await self.send_event(
            event("pipeline.stage", self.session_id, {"stage": "tts_done", "provider": "mock", **metadata})
        )

    def _has_usable_cartesia_config(self) -> bool:
        return bool(
            self.settings.normalized_mode == "live"
            and self.settings.cartesia_api_key
            and self.settings.cartesia_voice_id
            and is_uuid(self.settings.cartesia_voice_id)
        )

    def _ensure_synthesizer(self) -> None:
        if self.synthesizer is not None:
            return
        self.synthesizer = CartesiaStreamingTTS(
            api_key=self.settings.cartesia_api_key,
            model_id=self.settings.cartesia_model,
            voice_id=self.settings.cartesia_voice_id,
            sample_rate=self.settings.cartesia_sample_rate,
            cartesia_version=self.settings.cartesia_version,
            speed=getattr(self.settings, "cartesia_speed", None),
            open_timeout_seconds=getattr(self.settings, "cartesia_open_timeout_seconds", 8.0),
            connect_retries=getattr(self.settings, "cartesia_connect_retries", 1),
        )

    async def _direct_cartesia_text(self, text: str, *, metadata: dict | None = None) -> str:
        metadata = metadata or {}
        config = SpeechDirectionConfig(
            enabled=bool(getattr(self.settings, "cartesia_speech_director_enabled", True)),
            ssml_enabled=bool(getattr(self.settings, "cartesia_ssml_enabled", True)),
            emotion_tags_enabled=bool(getattr(self.settings, "cartesia_emotion_tags_enabled", False)),
        )
        try:
            directed = direct_speech_for_cartesia(text, config)
        except Exception as exc:
            await self.send_provider_error("cartesia", f"Speech direction failed: {exc}", metadata=metadata)
            return text

        if directed != text:
            await self.send_event(
                event(
                    "pipeline.stage",
                    self.session_id,
                    {
                        "stage": "tts_speech_direction",
                        "provider": "cartesia",
                        "directed": True,
                        **metadata,
                    },
                )
            )
        return directed

    async def _stream_tts_fallback(
        self,
        response_id: str,
        final_detected_at: float,
        *,
        fallback_from: str,
        metadata: dict | None = None,
    ) -> None:
        metadata = metadata or {}
        await self.send_event(
            event(
                "pipeline.stage",
                self.session_id,
                {"stage": "tts_fallback", "provider": "mock", "fallback_from": fallback_from, **metadata},
            )
        )
        await self._stream_mock_speech(response_id, final_detected_at, metadata=metadata)
        await self.send_event(
            event("pipeline.stage", self.session_id, {"stage": "tts_done", "provider": "mock", **metadata})
        )

    async def _stream_cartesia_speech(
        self,
        response_id: str,
        message: str,
        final_detected_at: float,
        *,
        metadata: dict | None = None,
    ) -> bool:
        metadata = metadata or {}
        self._ensure_synthesizer()

        first_audio_sent = False
        audio_chunks = 0
        audio_bytes = 0
        try:
            directed_message = await self._direct_cartesia_text(message, metadata=metadata)
            async for chunk in self.synthesizer.stream_speech(
                directed_message,
                context_id=generate_cartesia_context_id(response_id),
            ):
                if chunk["type"] == "error":
                    await self.send_provider_error("cartesia", chunk["message"], metadata=metadata)
                    return False
                if chunk["type"] != "chunk" or not chunk["audio"]:
                    continue
                if response_id in self.interrupted_response_ids:
                    return True

                try:
                    decoded_audio = base64.b64decode(chunk["audio"])
                except Exception:
                    decoded_audio = b""
                audio_chunks += 1
                audio_bytes += len(decoded_audio)

                if not first_audio_sent:
                    first_audio_sent = True
                    await self.send_event(
                        event(
                            "latency.metric",
                            self.session_id,
                            {
                                "name": "cartesia_time_to_first_audio_ms",
                                "response_id": response_id,
                                "value_ms": round((time.perf_counter() - final_detected_at) * 1000, 2),
                            },
                        )
                    )

                await self.send_event(
                    event(
                        "audio.chunk",
                        self.session_id,
                        {
                            "response_id": response_id,
                            "encoding": "pcm_s16le",
                            "sample_rate": self.settings.cartesia_sample_rate,
                            "channels": 1,
                            "is_final": False,
                            "audio": chunk["audio"],
                            "provider": "cartesia",
                            "chunk_index": audio_chunks,
                            "chunk_bytes": len(decoded_audio),
                            "rms": calculate_pcm_level(decoded_audio)["rms"],
                            "peak": calculate_pcm_level(decoded_audio)["peak"],
                            **metadata,
                        },
                    )
                )
            await self.send_event(
                event(
                    "pipeline.stage",
                    self.session_id,
                    {
                        "stage": "tts_done",
                        "provider": "cartesia",
                        "audio_chunks": audio_chunks,
                        "audio_bytes": audio_bytes,
                        **metadata,
                    },
                )
            )
            if audio_chunks == 0:
                await self.send_provider_error(
                    "cartesia",
                    "Cartesia completed without sending audio chunks. Check CARTESIA_API_KEY, CARTESIA_VOICE_ID, model access, and account permissions.",
                    metadata=metadata,
                )
                return False
            return True
        except ClientConnectionClosed:
            self.closed = True
            raise
        except Exception as exc:
            if self.closed:
                return True
            await self.send_provider_error("cartesia", str(exc), metadata=metadata)
            return False

    async def _stream_cartesia_speech_chunks(
        self,
        response_id: str,
        chunks: AsyncIterator[str],
        final_detected_at: float,
        *,
        metadata: dict | None = None,
    ) -> bool:
        metadata = metadata or {}
        self._ensure_synthesizer()
        await self.send_event(event("status.changed", self.session_id, {"state": "speaking"}))
        await self.send_event(
            event(
                "pipeline.stage",
                self.session_id,
                {"stage": "tts_streaming", "provider": "cartesia", "streaming_input": True, **metadata},
            )
        )

        first_audio_sent = False
        audio_chunks = 0
        audio_bytes = 0
        try:
            async def directed_chunks() -> AsyncIterator[str]:
                async for text in chunks:
                    yield await self._direct_cartesia_text(text, metadata=metadata)

            async for chunk in self.synthesizer.stream_speech_chunks(
                directed_chunks(),
                context_id=generate_cartesia_context_id(response_id),
            ):
                if chunk["type"] == "error":
                    await self.send_provider_error("cartesia", chunk["message"], metadata=metadata)
                    return False
                if chunk["type"] != "chunk" or not chunk["audio"]:
                    continue
                if response_id in self.interrupted_response_ids:
                    return True

                try:
                    decoded_audio = base64.b64decode(chunk["audio"])
                except Exception:
                    decoded_audio = b""
                audio_chunks += 1
                audio_bytes += len(decoded_audio)

                if not first_audio_sent:
                    first_audio_sent = True
                    await self.send_event(
                        event(
                            "latency.metric",
                            self.session_id,
                            {
                                "name": "cartesia_time_to_first_audio_ms",
                                "response_id": response_id,
                                "value_ms": round((time.perf_counter() - final_detected_at) * 1000, 2),
                            },
                        )
                    )

                level = calculate_pcm_level(decoded_audio)
                await self.send_event(
                    event(
                        "audio.chunk",
                        self.session_id,
                        {
                            "response_id": response_id,
                            "encoding": "pcm_s16le",
                            "sample_rate": self.settings.cartesia_sample_rate,
                            "channels": 1,
                            "is_final": False,
                            "audio": chunk["audio"],
                            "provider": "cartesia",
                            "chunk_index": audio_chunks,
                            "chunk_bytes": len(decoded_audio),
                            "rms": level["rms"],
                            "peak": level["peak"],
                            **metadata,
                        },
                    )
                )
            await self.send_event(
                event(
                    "pipeline.stage",
                    self.session_id,
                    {
                        "stage": "tts_done",
                        "provider": "cartesia",
                        "streaming_input": True,
                        "audio_chunks": audio_chunks,
                        "audio_bytes": audio_bytes,
                        **metadata,
                    },
                )
            )
            if audio_chunks == 0:
                await self.send_provider_error(
                    "cartesia",
                    "Cartesia completed without sending audio chunks. Check CARTESIA_API_KEY, CARTESIA_VOICE_ID, model access, and account permissions.",
                    metadata=metadata,
                )
                return False
            return True
        except ClientConnectionClosed:
            self.closed = True
            raise
        except Exception as exc:
            if self.closed:
                return True
            await self.send_provider_error("cartesia", str(exc), metadata=metadata)
            return False

    async def _stream_mock_speech(
        self,
        response_id: str,
        final_detected_at: float,
        *,
        metadata: dict | None = None,
    ) -> None:
        metadata = metadata or {}
        if response_id in self.interrupted_response_ids:
            return
        first_audio_at = time.perf_counter()
        audio = generate_mock_pcm(sample_rate=16000, duration_seconds=0.38)
        await self.send_event(
            event(
                "audio.chunk",
                self.session_id,
                {
                    "response_id": response_id,
                    "encoding": "pcm_s16le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "is_final": True,
                    "audio": base64.b64encode(audio).decode("ascii"),
                    "provider": "mock",
                    **metadata,
                },
            )
        )
        await self.send_event(
            event(
                "latency.metric",
                self.session_id,
                {
                    "name": "mock_time_to_first_audio_ms",
                    "response_id": response_id,
                    "value_ms": round((first_audio_at - final_detected_at) * 1000, 2),
                },
            )
        )

    def _ensure_agent(self) -> None:
        if self.agent is not None:
            return
        self.agent = GroqStreamingAgent(
            api_key=self.settings.groq_api_key,
            model=self.settings.groq_model,
            persona=self.settings.persona,
            temperature=self.settings.groq_temperature,
        )

    async def _live_user_agent_transcript(self, *, metadata: dict | None = None) -> list[dict[str, str]]:
        metadata = metadata or {}
        enabled = bool(getattr(self.settings, "intent_inference_enabled", True))
        transcript = agent_transcript_with_intent_inference(self.transcript, enabled=enabled)
        if enabled and transcript != self.transcript:
            await self.send_event(
                event(
                    "pipeline.stage",
                    self.session_id,
                    {
                        "stage": "llm_context",
                        "provider": "groq",
                        "intent_inference": True,
                        "turns_sent": len(transcript),
                        **metadata,
                    },
                )
            )
        return transcript

    def _uses_default_live_agent_response(self) -> bool:
        return getattr(self._stream_live_agent_response, "__func__", None) is MockConversationSession._stream_live_agent_response

    async def _stream_live_agent_response_with_streaming_speech(
        self,
        response_id: str,
        final_detected_at: float,
        *,
        metadata: dict | None = None,
    ) -> str:
        metadata = metadata or {}
        self._ensure_agent()
        agent_transcript = await self._live_user_agent_transcript(metadata=metadata)
        full_response = ""
        pending = ""
        emitted_text = ""
        stream_complete = False
        sentinel = object()
        chunk_queue: asyncio.Queue[str | object] = asyncio.Queue()

        async def drain_agent_response() -> None:
            nonlocal emitted_text, full_response, pending, stream_complete
            try:
                async for delta in self.agent.stream_response(agent_transcript):
                    full_response += delta
                    pending += delta
                    chunks, pending = pop_speakable_chunks(pending)
                    emitted_text = await self._emit_agent_text_chunks(
                        response_id,
                        chunks,
                        text_so_far=emitted_text,
                        metadata=metadata,
                    )
                    for chunk in chunks:
                        await chunk_queue.put(chunk)

                chunks, pending = pop_speakable_chunks(pending, force=True)
                emitted_text = await self._emit_agent_text_chunks(
                    response_id,
                    chunks,
                    text_so_far=emitted_text,
                    metadata=metadata,
                )
                for chunk in chunks:
                    await chunk_queue.put(chunk)
            except ClientConnectionClosed:
                self.closed = True
                raise
            except Exception as exc:
                if self.closed:
                    raise
                await self.send_provider_error("groq", str(exc), metadata=metadata)
                fallback = "I heard you, but Groq did not return a response. Check the server logs and API key."
                full_response = fallback
                chunks = split_mock_response(fallback)
                emitted_text = await self._emit_agent_text_chunks(
                    response_id,
                    chunks,
                    text_so_far=emitted_text,
                    metadata=metadata,
                )
                for chunk in chunks:
                    await chunk_queue.put(chunk)
            finally:
                stream_complete = True
                await chunk_queue.put(sentinel)

        async def speakable_chunks() -> AsyncIterator[str]:
            while True:
                chunk = await chunk_queue.get()
                if chunk is sentinel:
                    return
                yield str(chunk)

        drain_task = asyncio.create_task(drain_agent_response())
        try:
            cartesia_succeeded = await self._stream_cartesia_speech_chunks(
                response_id,
                speakable_chunks(),
                final_detected_at,
                metadata=metadata,
            )
            if not cartesia_succeeded:
                if not stream_complete:
                    await drain_task
                if not full_response.strip():
                    full_response = await self._stream_live_agent_response(response_id)
                await self._stream_tts_fallback(
                    response_id,
                    final_detected_at,
                    fallback_from="cartesia",
                    metadata=metadata,
                )
            await drain_task
        finally:
            if not drain_task.done():
                drain_task.cancel()
                try:
                    await drain_task
                except asyncio.CancelledError:
                    pass

        return full_response.strip()

    async def _stream_live_agent_response(self, response_id: str) -> str:
        if not self.settings.groq_api_key:
            message = "I can hear you now, but GROQ_API_KEY is missing so my real response engine is not connected yet."
            await self._emit_agent_text_chunks(response_id, split_mock_response(message))
            return message

        self._ensure_agent()
        agent_transcript = await self._live_user_agent_transcript()

        full_response = ""
        pending = ""
        try:
            emitted_text = ""
            async for delta in self.agent.stream_response(agent_transcript):
                full_response += delta
                pending += delta
                chunks, pending = pop_speakable_chunks(pending)
                emitted_text = await self._emit_agent_text_chunks(response_id, chunks, text_so_far=emitted_text)

            chunks, pending = pop_speakable_chunks(pending, force=True)
            await self._emit_agent_text_chunks(response_id, chunks, text_so_far=emitted_text)
            return full_response.strip()
        except ClientConnectionClosed:
            self.closed = True
            raise
        except Exception as exc:
            if self.closed:
                raise
            await self.send_provider_error("groq", str(exc))
            message = "I heard you, but Groq did not return a response. Check the server logs and API key."
            await self._emit_agent_text_chunks(response_id, split_mock_response(message))
            return message

    async def _stream_live_contextual_followup(self, response_id: str, *, metadata: dict) -> str:
        return await self._stream_live_proactive_response(response_id, metadata=metadata)

    async def _stream_live_proactive_response(self, response_id: str, *, metadata: dict) -> str:
        trigger_reason = str(metadata.get("trigger_reason") or "")
        if not self.settings.groq_api_key:
            await self.send_provider_error("groq", "GROQ_API_KEY is missing for live proactive speech.", metadata=metadata)
            return ""

        self._ensure_agent()

        full_response = ""
        pending = ""
        proactive_transcript = proactive_agent_transcript(trigger_reason, self.transcript)
        try:
            emitted_text = ""
            async for delta in self.agent.stream_response(proactive_transcript):
                full_response += delta
                pending += delta
                chunks, pending = pop_speakable_chunks(pending)
                emitted_text = await self._emit_agent_text_chunks(
                    response_id,
                    chunks,
                    text_so_far=emitted_text,
                    metadata=metadata,
                )

            chunks, pending = pop_speakable_chunks(pending, force=True)
            await self._emit_agent_text_chunks(response_id, chunks, text_so_far=emitted_text, metadata=metadata)
            message = full_response.strip()
            if message:
                return message
        except ClientConnectionClosed:
            self.closed = True
            raise
        except Exception as exc:
            if self.closed:
                raise
            await self.send_provider_error("groq", str(exc), metadata=metadata)
            return ""

        return ""

    async def _stream_mock_agent_response(self, response_id: str, user_text: str) -> str:
        message = f"I heard: {user_text}. The downstream voice loop is still mocked."
        await self._emit_agent_text_chunks(response_id, split_mock_response(message), delay_seconds=0.08)
        return message

    async def _emit_agent_text_chunks(
        self,
        response_id: str,
        chunks: list[str],
        *,
        delay_seconds: float = 0,
        text_so_far: str = "",
        metadata: dict | None = None,
    ) -> str:
        if not chunks:
            return text_so_far
        metadata = metadata or {}

        for chunk in chunks:
            if delay_seconds:
                await asyncio.sleep(delay_seconds)
            if (
                not text_so_far
                and self.active_response_started_at is not None
                and response_id not in self.text_latency_reported_response_ids
            ):
                self.text_latency_reported_response_ids.add(response_id)
                await self.send_event(
                    event(
                        "latency.metric",
                        self.session_id,
                        {
                            "name": "time_to_first_text_ms",
                            "response_id": response_id,
                            "value_ms": round((time.perf_counter() - self.active_response_started_at) * 1000, 2),
                            **metadata,
                        },
                    )
                )
            text_so_far += chunk
            self.active_assistant_text = text_so_far
            await self.send_event(
                event(
                    "agent.text.delta",
                    self.session_id,
                    {"response_id": response_id, "text": chunk, "text_so_far": text_so_far, **metadata},
                )
            )
        return text_so_far

    async def send_provider_error(self, provider: str, message: str, *, metadata: dict | None = None) -> None:
        if self.closed:
            return
        metadata = metadata or {}
        if metadata.get("proactive"):
            await self._record_proactive_failure(metadata)
        await self.send_event(
            event(
                "pipeline.stage",
                self.session_id,
                {"stage": f"{provider}_failed", "provider": provider, "message": message, **metadata},
            )
        )
        await self.send_event(event("error", self.session_id, {"message": message, "provider": provider, **metadata}))

    async def _record_proactive_failure(self, metadata: dict) -> None:
        proactive_turn_id = str(metadata.get("proactive_turn_id") or "")
        if proactive_turn_id and proactive_turn_id in self.failed_proactive_turn_ids:
            return
        if proactive_turn_id:
            self.failed_proactive_turn_ids.add(proactive_turn_id)

        self.proactive_failures += 1
        if self.proactive_failures < self.proactive_config.failure_backoff_threshold:
            return

        self.proactive_cooldown_until_ms = monotonic_ms() + self.proactive_config.failure_backoff_ms
        await self.send_event(
            event(
                "proactive.cooldown",
                self.session_id,
                {
                    "trigger_reason": metadata.get("trigger_reason"),
                    "cooldown_ms": self.proactive_config.failure_backoff_ms,
                    "next_eligible_at_ms": self.proactive_cooldown_until_ms,
                    "reason": "failure_backoff",
                    "proactive_failures": self.proactive_failures,
                    "failure_backoff_threshold": self.proactive_config.failure_backoff_threshold,
                },
            )
        )

    def _proactive_turn_failed(self, metadata: dict) -> bool:
        proactive_turn_id = str(metadata.get("proactive_turn_id") or "")
        return bool(proactive_turn_id and proactive_turn_id in self.failed_proactive_turn_ids)


def generate_mock_pcm(sample_rate: int, duration_seconds: float) -> bytes:
    total_samples = int(sample_rate * duration_seconds)
    frames = bytearray()
    for index in range(total_samples):
        fade = min(index / 800, (total_samples - index) / 800, 1)
        value = int(12000 * fade * math.sin(2 * math.pi * 440 * index / sample_rate))
        frames.extend(struct.pack("<h", value))
    return bytes(frames)


def calculate_pcm_level(frame: bytes) -> dict[str, float]:
    if len(frame) < 2:
        return {"rms": 0.0, "peak": 0.0}

    sample_count = len(frame) // 2
    samples = struct.unpack(f"<{sample_count}h", frame[: sample_count * 2])
    if not samples:
        return {"rms": 0.0, "peak": 0.0}

    peak = max(abs(sample) for sample in samples) / 32768
    mean_square = sum(sample * sample for sample in samples) / sample_count
    rms = math.sqrt(mean_square) / 32768
    return {"rms": round(rms, 4), "peak": round(peak, 4)}


def apply_pcm_gain(frame: bytes, gain: float) -> bytes:
    if len(frame) < 2 or gain == 1.0:
        return frame

    sample_count = len(frame) // 2
    samples = struct.unpack(f"<{sample_count}h", frame[: sample_count * 2])
    boosted = bytearray()
    for sample in samples:
        boosted_sample = round(sample * gain)
        boosted_sample = max(-32768, min(32767, boosted_sample))
        boosted.extend(struct.pack("<h", boosted_sample))
    boosted.extend(frame[sample_count * 2 :])
    return bytes(boosted)


def is_mock_speech_frame(level: dict[str, float]) -> bool:
    return level["rms"] >= MOCK_SPEECH_RMS_THRESHOLD


def split_mock_response(message: str) -> list[str]:
    words = message.split(" ")
    chunks: list[str] = []
    for index in range(0, len(words), 4):
        chunks.append(" ".join(words[index : index + 4]) + " ")
    return chunks


def scripted_proactive_message(
    trigger_reason: str,
    consecutive_prompts: int,
    *,
    transcript: list[dict[str, str]] | None = None,
) -> str:
    if trigger_reason == TRIGGER_STARTUP_GREETING:
        return "I'm on the line with you."
    if trigger_reason == TRIGGER_SILENCE_NUDGE:
        silence_messages = [
            "I'm here with you on the line.",
            "Still with you when you're ready.",
            "I'll stay quiet on the line for a moment.",
        ]
        return silence_messages[max(0, consecutive_prompts - 1) % len(silence_messages)]
    if trigger_reason == TRIGGER_CONTEXTUAL_FOLLOW_UP:
        return "A quick thought while we're on the line?"
    return "I'm still on the line."


def contextual_followup_transcript(transcript: list[dict[str, str]]) -> list[dict[str, str]]:
    recent_turns = recent_conversation_context(transcript)
    return [
        *recent_turns,
        {"role": "user", "content": PROACTIVE_FOLLOWUP_INSTRUCTION},
    ]


def proactive_agent_transcript(trigger_reason: str, transcript: list[dict[str, str]]) -> list[dict[str, str]]:
    recent_turns = recent_conversation_context(transcript)
    if trigger_reason == TRIGGER_STARTUP_GREETING:
        instruction = PROACTIVE_STARTUP_GREETING_INSTRUCTION
    elif trigger_reason == TRIGGER_CONTEXTUAL_FOLLOW_UP:
        instruction = PROACTIVE_FOLLOWUP_INSTRUCTION
    else:
        instruction = PROACTIVE_SILENCE_NUDGE_INSTRUCTION
    return [
        *recent_turns,
        {"role": "user", "content": instruction},
    ]


def recent_conversation_context(transcript: list[dict[str, str]]) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for turn in transcript[-RECENT_CONTEXT_TURN_LIMIT:]:
        role = turn.get("role")
        content = turn.get("content", "").strip()
        if role in {"user", "assistant"} and content:
            turns.append({"role": role, "content": content})
    return turns


def has_recent_user_context(transcript: list[dict[str, str]]) -> bool:
    return any(
        turn["role"] == "user" and is_meaningful_user_speech(turn["content"])
        for turn in recent_conversation_context(transcript)
    )


def latest_turn_text(transcript: list[dict[str, str]], role: str) -> str | None:
    for turn in reversed(transcript):
        if turn.get("role") == role:
            content = turn.get("content", "").strip()
            if content:
                return content
    return None


def last_assistant_asked_question(transcript: list[dict[str, str]]) -> bool:
    text = latest_turn_text(transcript, "assistant")
    return bool(text and "?" in text)


def last_assistant_has_next_step(transcript: list[dict[str, str]]) -> bool:
    text = latest_turn_text(transcript, "assistant")
    if not text:
        return False
    normalized = normalize_turn_text(text)
    next_step_phrases = (
        "next step",
        "start by",
        "try ",
        "you should",
        "go ahead",
        "say the word",
        "whenever you're ready",
        "want me to",
    )
    return any(phrase in normalized for phrase in next_step_phrases)


def proactive_metadata(trigger_reason: str, proactive_turn_id: str) -> dict[str, str | bool]:
    return {
        "proactive": True,
        "trigger_reason": trigger_reason,
        "proactive_turn_id": proactive_turn_id,
    }


def monotonic_ms() -> int:
    return round(time.monotonic() * 1000)


def is_duplicate_active_turn(incoming_text: str, active_user_text: str | None) -> bool:
    if not active_user_text:
        return False

    normalized_incoming = normalize_turn_text(incoming_text)
    normalized_active = normalize_turn_text(active_user_text)
    if not normalized_incoming or not normalized_active:
        return False

    return normalized_incoming == normalized_active


def should_interrupt_active_response(
    transcript: dict,
    *,
    active_user_text: str | None,
    active_assistant_text: str,
) -> bool:
    text = transcript.get("text", "")
    if not text or not is_meaningful_user_speech(text):
        return False
    if is_duplicate_active_turn(text, active_user_text):
        return False
    if is_likely_assistant_echo(text, active_assistant_text):
        return False
    if not (transcript.get("is_final") or transcript.get("speech_final")):
        return False
    confidence = transcript.get("confidence")
    if confidence is not None and confidence < 0.75 and not transcript.get("speech_final"):
        return False
    return True


def is_meaningful_user_speech(text: str) -> bool:
    normalized = normalize_turn_text(text)
    if not normalized:
        return False
    words = normalized.split()
    return len(words) >= 2 or len(normalized) >= 12


def is_likely_assistant_echo(incoming_text: str, active_assistant_text: str) -> bool:
    incoming_tokens = set(normalize_turn_text(incoming_text).split())
    assistant_tokens = set(normalize_turn_text(active_assistant_text).split())
    if not incoming_tokens or not assistant_tokens:
        return False
    overlap = len(incoming_tokens & assistant_tokens) / len(incoming_tokens)
    return overlap >= 0.6


def normalize_turn_text(text: str) -> str:
    return " ".join(text.strip().casefold().split())
