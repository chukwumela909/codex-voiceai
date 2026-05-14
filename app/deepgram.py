import asyncio
import json
from collections.abc import Awaitable, Callable
from urllib.parse import urlencode

import websockets

from app.exceptions import ClientConnectionClosed


DeepgramTranscriptHandler = Callable[[dict], Awaitable[None]]
DeepgramErrorHandler = Callable[[str], Awaitable[None]]


class DeepgramStreamingTranscriber:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        encoding: str,
        sample_rate: int,
        channels: int,
        endpointing_ms: int,
        utterance_end_ms: int,
        on_transcript: DeepgramTranscriptHandler,
        on_error: DeepgramErrorHandler,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.channels = channels
        self.endpointing_ms = endpointing_ms
        self.utterance_end_ms = utterance_end_ms
        self.on_transcript = on_transcript
        self.on_error = on_error
        self.websocket = None
        self.receive_task: asyncio.Task | None = None
        self.closed = False

    async def start(self) -> None:
        query = urlencode(
            {
                "model": self.model,
                "encoding": self.encoding,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "interim_results": "true",
                "endpointing": self.endpointing_ms,
                "utterance_end_ms": self.utterance_end_ms,
                "vad_events": "true",
                "smart_format": "true",
                "punctuate": "true",
            }
        )
        self.websocket = await websockets.connect(
            f"wss://api.deepgram.com/v1/listen?{query}",
            additional_headers={"Authorization": f"Token {self.api_key}"},
        )
        self.receive_task = asyncio.create_task(self._receive_loop())

    async def send_audio(self, frame: bytes) -> None:
        if self.websocket and not self.closed:
            await self.websocket.send(frame)

    async def close(self) -> None:
        self.closed = True
        if self.websocket:
            await self.websocket.close()
        if self.receive_task and not self.receive_task.done() and self.receive_task is not asyncio.current_task():
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass

    async def _receive_loop(self) -> None:
        try:
            assert self.websocket is not None
            async for raw_message in self.websocket:
                parsed = parse_deepgram_message(raw_message)
                if parsed:
                    await self.on_transcript(parsed)
        except asyncio.CancelledError:
            raise
        except ClientConnectionClosed:
            self.closed = True
        except Exception as exc:
            if not self.closed:
                await self.on_error(str(exc))


def parse_deepgram_message(raw_message: str | bytes) -> dict | None:
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")

    data = json.loads(raw_message)
    if data.get("type") == "UtteranceEnd":
        return {
            "type": "utterance_end",
            "last_word_end": data.get("last_word_end"),
            "provider": "deepgram",
        }

    channel = data.get("channel", {})
    if not isinstance(channel, dict):
        return None

    alternative = channel.get("alternatives", [{}])[0]
    transcript = alternative.get("transcript", "").strip()
    if not transcript:
        return None

    return {
        "text": transcript,
        "confidence": alternative.get("confidence"),
        "is_final": bool(data.get("is_final", False)),
        "speech_final": bool(data.get("speech_final", False)),
        "provider": "deepgram",
    }
