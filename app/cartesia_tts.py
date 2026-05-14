import asyncio
import json
from collections.abc import AsyncIterator
from uuid import uuid4

import websockets


class CartesiaStreamingTTS:
    def __init__(
        self,
        *,
        api_key: str,
        model_id: str,
        voice_id: str,
        sample_rate: int,
        cartesia_version: str,
        speed: float | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.voice_id = voice_id
        self.sample_rate = sample_rate
        self.cartesia_version = cartesia_version
        self.speed = speed

    async def stream_speech(self, transcript: str, *, context_id: str | None = None) -> AsyncIterator[dict]:
        context_id = context_id or f"ctx_{uuid4().hex}"
        async with websockets.connect(
            "wss://api.cartesia.ai/tts/websocket",
            additional_headers={
                "X-API-Key": self.api_key,
                "Cartesia-Version": self.cartesia_version,
            },
        ) as websocket:
            await websocket.send(json.dumps(self._generation_request(transcript, context_id)))
            async for raw_message in websocket:
                parsed = parse_cartesia_message(raw_message)
                if parsed is None:
                    continue
                yield parsed
                if parsed["type"] == "done":
                    break

    async def stream_speech_chunks(
        self,
        transcripts: AsyncIterator[str],
        *,
        context_id: str | None = None,
    ) -> AsyncIterator[dict]:
        context_id = context_id or f"ctx_{uuid4().hex}"
        done_sentinel = object()

        async with websockets.connect(
            "wss://api.cartesia.ai/tts/websocket",
            additional_headers={
                "X-API-Key": self.api_key,
                "Cartesia-Version": self.cartesia_version,
            },
        ) as websocket:
            queue: asyncio.Queue[dict | object] = asyncio.Queue()

            async def read_messages() -> None:
                try:
                    async for raw_message in websocket:
                        parsed = parse_cartesia_message(raw_message)
                        if parsed is None:
                            continue
                        await queue.put(parsed)
                        if parsed["type"] == "done":
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
                    async for transcript in transcripts:
                        if not transcript:
                            continue
                        request = self._generation_request(transcript, context_id, continue_=True)
                        await websocket.send(json.dumps(request))
                    await websocket.send(json.dumps(self._generation_request("", context_id, continue_=False)))
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

    def _generation_request(self, transcript: str, context_id: str, *, continue_: bool = False) -> dict:
        request = {
            "model_id": self.model_id,
            "transcript": transcript,
            "voice": {
                "mode": "id",
                "id": self.voice_id,
            },
            "language": "en",
            "context_id": context_id,
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
            "add_timestamps": False,
            "continue": continue_,
        }
        if self.speed is not None:
            request["generation_config"] = {"speed": self.speed}
        return request


def parse_cartesia_message(raw_message: str | bytes) -> dict | None:
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")

    data = json.loads(raw_message)
    message_type = data.get("type")

    if message_type == "chunk":
        return {
            "type": "chunk",
            "audio": data.get("data", ""),
            "context_id": data.get("context_id"),
            "done": bool(data.get("done", False)),
        }

    if message_type == "done":
        return {
            "type": "done",
            "context_id": data.get("context_id"),
            "done": True,
        }

    if message_type == "error" or data.get("error"):
        message = data.get("error") or data.get("message") or data.get("detail") or "Cartesia returned an unknown error."
        return {
            "type": "error",
            "context_id": data.get("context_id"),
            "message": message,
            "done": bool(data.get("done", True)),
        }

    return None


def generate_cartesia_context_id(response_id: str) -> str:
    return response_id.replace("resp_", "ctx_", 1)
