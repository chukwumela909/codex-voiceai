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
    ) -> None:
        self.api_key = api_key
        self.model_id = model_id
        self.voice_id = voice_id
        self.sample_rate = sample_rate
        self.cartesia_version = cartesia_version

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

    def _generation_request(self, transcript: str, context_id: str) -> dict:
        return {
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
            "continue": False,
        }


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
