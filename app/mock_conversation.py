import asyncio
import base64
import math
import struct
import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

from app.cartesia_tts import CartesiaStreamingTTS, generate_cartesia_context_id
from app.deepgram import DeepgramStreamingTranscriber
from app.events import event
from app.groq_agent import GroqStreamingAgent, pop_speakable_chunks

SendEvent = Callable[[dict], Awaitable[None]]


class MockConversationSession:
    def __init__(self, session_id: str, send_event: SendEvent, settings) -> None:
        self.session_id = session_id
        self.send_event = send_event
        self.settings = settings
        self.audio_config: dict | None = None
        self.transcriber: DeepgramStreamingTranscriber | None = None
        self.agent: GroqStreamingAgent | None = None
        self.synthesizer: CartesiaStreamingTTS | None = None
        self.transcript: list[dict[str, str]] = []
        self.bytes_received = 0
        self.frames_received = 0
        self.turn_started_at: float | None = None
        self.current_task: asyncio.Task | None = None
        self.pending_transcript_task: asyncio.Task | None = None
        self.live_transcript_sequence = 0
        self.active_response_id: str | None = None
        self.active_user_text: str | None = None
        self.interrupted_response_ids: set[str] = set()
        self.closed = False

    async def configure_audio(self, payload: dict) -> None:
        self.audio_config = {
            "encoding": payload.get("encoding", "pcm_s16le"),
            "sample_rate": int(payload.get("sample_rate", 16000)),
            "channels": int(payload.get("channels", 1)),
            "frame_duration_ms": int(payload.get("frame_duration_ms", 20)),
        }
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
        self.bytes_received += len(frame)
        if self.frames_received == 1 or self.frames_received % 100 == 0:
            await self.send_event(
                event(
                    "audio.input",
                    self.session_id,
                    {
                        "frames_received": self.frames_received,
                        "bytes_received": self.bytes_received,
                        "frame_bytes": len(frame),
                        "rms": calculate_pcm_level(frame)["rms"],
                        "peak": calculate_pcm_level(frame)["peak"],
                        "sample_rate": self.audio_config["sample_rate"] if self.audio_config else None,
                        "mode": self.settings.normalized_mode,
                        "provider": "deepgram" if self.transcriber else "mock",
                    },
                )
            )

        if self.transcriber:
            await self.transcriber.send_audio(frame)
            return

        if self.turn_started_at is None:
            self.turn_started_at = time.perf_counter()

        if self.current_task is None and self.bytes_received >= 24000:
            self.current_task = asyncio.create_task(self._run_mock_turn(self.bytes_received))
            self.bytes_received = 0

    async def close(self) -> None:
        self.closed = True
        if self.transcriber:
            await self.transcriber.close()
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
        self.active_user_text = None

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
            on_transcript=self.handle_live_transcript,
            on_error=self.handle_live_error,
        )
        await self.transcriber.start()
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
        if self.current_task is not None:
            if is_duplicate_active_turn(transcript["text"], self.active_user_text):
                return
            await self._interrupt_active_response(transcript["text"])

        if self.current_task is not None:
            return

        self.live_transcript_sequence += 1
        if transcript["speech_final"] or transcript["is_final"]:
            await self._start_live_agent_response(transcript["text"], reason="provider_final")
            return

        if self.pending_transcript_task and not self.pending_transcript_task.done():
            self.pending_transcript_task.cancel()
        self.pending_transcript_task = asyncio.create_task(
            self._finalize_partial_transcript_after_idle(transcript["text"], self.live_transcript_sequence)
        )

    async def handle_live_error(self, message: str) -> None:
        await self.send_provider_error("deepgram", message)

    async def _finalize_partial_transcript_after_idle(self, text: str, sequence: int) -> None:
        idle_seconds = max(0.8, self.settings.deepgram_endpointing_ms / 1000 * 2)
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
        await self._start_live_agent_response(text, reason="partial_idle_timeout")

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

    async def _interrupt_active_response(self, next_user_text: str) -> None:
        interrupted_response_id = self.active_response_id
        await self.send_event(
            event(
                "interruption.started",
                self.session_id,
                {
                    "interrupted_response_id": interrupted_response_id,
                    "next_user_text": next_user_text,
                    "reason": "user_speech_during_response",
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
        self.active_user_text = user_text
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

        if self.settings.normalized_mode == "live":
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
        await self._stream_speech(response_id, message, final_detected_at)
        if response_id in self.interrupted_response_ids:
            return

        self.transcript.append({"role": "assistant", "content": message})
        await self.send_event(event("status.changed", self.session_id, {"state": "listening"}))
        self.turn_started_at = None
        self.current_task = None
        self.active_response_id = None
        self.active_user_text = None

    async def _stream_speech(self, response_id: str, message: str, final_detected_at: float) -> None:
        await self.send_event(event("status.changed", self.session_id, {"state": "speaking"}))
        await self.send_event(
            event(
                "pipeline.stage",
                self.session_id,
                {"stage": "tts_streaming", "provider": "cartesia" if self.settings.normalized_mode == "live" else "mock"},
            )
        )
        if self.settings.normalized_mode == "live" and self.settings.cartesia_api_key and self.settings.cartesia_voice_id:
            await self._stream_cartesia_speech(response_id, message, final_detected_at)
            return

        await self._stream_mock_speech(response_id, final_detected_at)
        await self.send_event(event("pipeline.stage", self.session_id, {"stage": "tts_done", "provider": "mock"}))

    async def _stream_cartesia_speech(self, response_id: str, message: str, final_detected_at: float) -> None:
        if self.synthesizer is None:
            self.synthesizer = CartesiaStreamingTTS(
                api_key=self.settings.cartesia_api_key,
                model_id=self.settings.cartesia_model,
                voice_id=self.settings.cartesia_voice_id,
                sample_rate=self.settings.cartesia_sample_rate,
                cartesia_version=self.settings.cartesia_version,
            )

        first_audio_sent = False
        audio_chunks = 0
        audio_bytes = 0
        try:
            async for chunk in self.synthesizer.stream_speech(
                message,
                context_id=generate_cartesia_context_id(response_id),
            ):
                if chunk["type"] == "error":
                    await self.send_provider_error("cartesia", chunk["message"])
                    return
                if chunk["type"] != "chunk" or not chunk["audio"]:
                    continue

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
                    },
                )
            )
            if audio_chunks == 0:
                await self.send_provider_error(
                    "cartesia",
                    "Cartesia completed without sending audio chunks. Check CARTESIA_API_KEY, CARTESIA_VOICE_ID, model access, and account permissions.",
                )
        except Exception as exc:
            await self.send_provider_error("cartesia", str(exc))

    async def _stream_mock_speech(self, response_id: str, final_detected_at: float) -> None:
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

    async def _stream_live_agent_response(self, response_id: str) -> str:
        if not self.settings.groq_api_key:
            message = "I can hear you now, but GROQ_API_KEY is missing so my real response engine is not connected yet."
            await self._emit_agent_text_chunks(response_id, split_mock_response(message))
            return message

        if self.agent is None:
            self.agent = GroqStreamingAgent(
                api_key=self.settings.groq_api_key,
                model=self.settings.groq_model,
                persona=self.settings.persona,
                temperature=self.settings.groq_temperature,
            )

        full_response = ""
        pending = ""
        try:
            emitted_text = ""
            async for delta in self.agent.stream_response(self.transcript):
                full_response += delta
                pending += delta
                chunks, pending = pop_speakable_chunks(pending)
                emitted_text = await self._emit_agent_text_chunks(response_id, chunks, text_so_far=emitted_text)

            chunks, pending = pop_speakable_chunks(pending, force=True)
            await self._emit_agent_text_chunks(response_id, chunks, text_so_far=emitted_text)
            return full_response.strip()
        except Exception as exc:
            await self.send_provider_error("groq", str(exc))
            message = "I heard you, but Groq did not return a response. Check the server logs and API key."
            await self._emit_agent_text_chunks(response_id, split_mock_response(message))
            return message

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
    ) -> str:
        if not chunks:
            return text_so_far

        for chunk in chunks:
            if delay_seconds:
                await asyncio.sleep(delay_seconds)
            text_so_far += chunk
            await self.send_event(
                event(
                    "agent.text.delta",
                    self.session_id,
                    {"response_id": response_id, "text": chunk, "text_so_far": text_so_far},
                )
            )
        return text_so_far

    async def send_provider_error(self, provider: str, message: str) -> None:
        await self.send_event(
            event(
                "pipeline.stage",
                self.session_id,
                {"stage": f"{provider}_failed", "provider": provider, "message": message},
            )
        )
        await self.send_event(event("error", self.session_id, {"message": message, "provider": provider}))


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


def split_mock_response(message: str) -> list[str]:
    words = message.split(" ")
    chunks: list[str] = []
    for index in range(0, len(words), 4):
        chunks.append(" ".join(words[index : index + 4]) + " ")
    return chunks


def is_duplicate_active_turn(incoming_text: str, active_user_text: str | None) -> bool:
    if not active_user_text:
        return False

    normalized_incoming = normalize_turn_text(incoming_text)
    normalized_active = normalize_turn_text(active_user_text)
    if not normalized_incoming or not normalized_active:
        return False

    return normalized_incoming == normalized_active


def normalize_turn_text(text: str) -> str:
    return " ".join(text.strip().casefold().split())
