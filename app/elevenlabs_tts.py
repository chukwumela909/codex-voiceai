import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import quote
from uuid import uuid4

import websockets

# Buffer thresholds (in characters) at which ElevenLabs flushes generation. The
# low first value keeps time-to-first-audio short on short replies; later values
# grow so longer text is generated in fewer, more prosodically coherent passes.
DEFAULT_CHUNK_LENGTH_SCHEDULE = [50, 160, 250, 290]


class ElevenLabsConnectionError(RuntimeError):
    pass


class ElevenLabsStreamingTTS:
    """Streaming TTS over the ElevenLabs realtime WebSocket (`stream-input`).

    Mirrors the public surface of the prior streaming TTS client so call sites
    only rename: ``stream_speech`` / ``stream_speech_chunks`` yield the dict shape
    ``{"type": "chunk"|"done"|"error", "audio": <base64 pcm_s16le>, "context_id",
    "done"}`` and the connection-pooling helpers behave identically.

    ElevenLabs single-context streaming has no server-side context id, so
    ``context_id`` is a local tag for event correlation only.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model_id: str,
        voice_id: str,
        sample_rate: int,
        stability: float = 0.5,
        similarity_boost: float = 0.8,
        style: float = 0.0,
        use_speaker_boost: bool = False,
        speed: float | None = None,
        open_timeout_seconds: float = 8.0,
        connect_retries: int = 1,
        retry_backoff_seconds: float = 0.2,
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.voice_id = voice_id
        self.sample_rate = sample_rate
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.style = style
        self.use_speaker_boost = use_speaker_boost
        self.speed = speed
        self.open_timeout_seconds = open_timeout_seconds
        self.connect_retries = max(0, connect_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self._idle_ws = None

    async def stream_speech(self, transcript: str, *, context_id: str | None = None) -> AsyncIterator[dict]:
        context_id = context_id or f"ctx_{uuid4().hex}"
        async with self._connect_websocket() as websocket:
            await websocket.send(json.dumps(self._init_message()))
            await websocket.send(json.dumps({"text": _frame_text(transcript)}))
            await websocket.send(json.dumps({"text": ""}))
            async for raw_message in websocket:
                parsed = parse_elevenlabs_message(raw_message, context_id=context_id)
                if parsed is None:
                    continue
                yield parsed
                if parsed["type"] in ("done", "error"):
                    break

    async def stream_speech_chunks(
        self,
        transcripts: AsyncIterator[str],
        *,
        context_id: str | None = None,
    ) -> AsyncIterator[dict]:
        context_id = context_id or f"ctx_{uuid4().hex}"
        done_sentinel = object()

        async with self._connect_websocket() as websocket:
            queue: asyncio.Queue[dict | object] = asyncio.Queue()

            async def read_messages() -> None:
                try:
                    async for raw_message in websocket:
                        parsed = parse_elevenlabs_message(raw_message, context_id=context_id)
                        if parsed is None:
                            continue
                        await queue.put(parsed)
                        if parsed["type"] in ("done", "error"):
                            break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await queue.put(
                        {
                            "type": "error",
                            "context_id": context_id,
                            "message": str(exc),
                            "done": True,
                        }
                    )
                finally:
                    await queue.put(done_sentinel)

            async def send_messages() -> None:
                try:
                    await websocket.send(json.dumps(self._init_message()))
                    async for transcript in transcripts:
                        if not transcript:
                            continue
                        await websocket.send(json.dumps({"text": _frame_text(transcript)}))
                    await websocket.send(json.dumps({"text": ""}))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await queue.put(
                        {
                            "type": "error",
                            "context_id": context_id,
                            "message": str(exc),
                            "done": True,
                        }
                    )
                    await queue.put(done_sentinel)

            reader = asyncio.create_task(read_messages())
            sender = asyncio.create_task(send_messages())
            finished_normally = False
            try:
                while True:
                    parsed = await queue.get()
                    if parsed is done_sentinel:
                        finished_normally = True
                        break
                    yield parsed
                    if isinstance(parsed, dict) and parsed["type"] == "error":
                        break

                if finished_normally:
                    await sender
            finally:
                for task in (reader, sender):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(reader, sender, return_exceptions=True)

    def _voice_settings(self) -> dict:
        settings: dict = {
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "use_speaker_boost": self.use_speaker_boost,
        }
        if self.speed is not None:
            settings["speed"] = self.speed
        return settings

    def _init_message(self) -> dict:
        # BOS frame: a lone space opens the stream and carries voice settings;
        # ElevenLabs only honors voice_settings on the first message of a stream.
        return {
            "text": " ",
            "voice_settings": self._voice_settings(),
            "generation_config": {"chunk_length_schedule": DEFAULT_CHUNK_LENGTH_SCHEDULE},
        }

    def _ws_url(self) -> str:
        return (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{quote(self.voice_id)}/stream-input"
            f"?model_id={quote(self.model_id)}&output_format=pcm_{self.sample_rate}"
        )

    async def pre_warm(self) -> None:
        if self._idle_ws is not None and not self._idle_ws.closed:
            return
        try:
            manager = websockets.connect(
                self._ws_url(),
                additional_headers={"xi-api-key": self.api_key},
                open_timeout=self.open_timeout_seconds,
            )
            ws = await manager.__aenter__()
        except Exception:
            return
        if self._idle_ws is not None and not self._idle_ws.closed:
            try:
                await ws.close()
            except Exception:
                pass
            return
        self._idle_ws = ws

    async def close_idle_connection(self) -> None:
        ws, self._idle_ws = self._idle_ws, None
        if ws and not ws.closed:
            try:
                await ws.close()
            except Exception:
                pass

    @asynccontextmanager
    async def _connect_websocket(self) -> AsyncIterator[object]:
        ws = self._idle_ws
        if ws is not None and not ws.closed:
            self._idle_ws = None
            try:
                yield ws
            except BaseException:
                try:
                    await ws.close()
                except Exception:
                    pass
                asyncio.create_task(self.pre_warm())
                raise
            else:
                self._idle_ws = ws
            return

        attempts = self.connect_retries + 1
        for attempt in range(1, attempts + 1):
            manager = websockets.connect(
                self._ws_url(),
                additional_headers={"xi-api-key": self.api_key},
                open_timeout=self.open_timeout_seconds,
            )
            try:
                websocket = await manager.__aenter__()
            except TimeoutError as exc:
                if attempt >= attempts:
                    raise ElevenLabsConnectionError(
                        "ElevenLabs WebSocket opening handshake timed out "
                        f"after {attempts} attempts ({self.open_timeout_seconds:g}s each)."
                    ) from exc
                if self.retry_backoff_seconds:
                    await asyncio.sleep(self.retry_backoff_seconds)
                continue

            try:
                yield websocket
            except BaseException as exc:
                suppress = await manager.__aexit__(type(exc), exc, exc.__traceback__)
                asyncio.create_task(self.pre_warm())
                if not suppress:
                    raise
            else:
                self._idle_ws = websocket
            return


def _frame_text(transcript: str) -> str:
    # ElevenLabs concatenates consecutive frames verbatim; a trailing space keeps
    # words from running together across frames.
    return transcript if transcript.endswith(" ") else transcript + " "


def parse_elevenlabs_message(raw_message: str | bytes, *, context_id: str | None = None) -> dict | None:
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")

    data = json.loads(raw_message)

    error = data.get("error")
    detail = data.get("detail")
    if error or (detail and not data.get("audio")):
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("status")
        message = error or detail or data.get("message") or "ElevenLabs returned an unknown error."
        return {
            "type": "error",
            "context_id": context_id,
            "message": str(message),
            "done": True,
        }

    is_final = bool(data.get("isFinal"))
    audio = data.get("audio")

    if audio:
        return {
            "type": "chunk",
            "audio": audio,
            "context_id": context_id,
            "done": is_final,
        }

    if is_final:
        return {
            "type": "done",
            "context_id": context_id,
            "done": True,
        }

    return None


def generate_context_id(response_id: str) -> str:
    return response_id.replace("resp_", "ctx_", 1)
