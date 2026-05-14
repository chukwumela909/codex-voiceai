import asyncio
import base64
import json

import app.cartesia_tts as cartesia_tts
from app.cartesia_tts import CartesiaStreamingTTS


def test_generation_request_includes_speed_and_continuation_flag():
    synthesizer = CartesiaStreamingTTS(
        api_key="cartesia-key",
        model_id="sonic-3",
        voice_id="voice-id",
        sample_rate=16000,
        cartesia_version="2026-03-01",
        speed=1.2,
    )

    request = synthesizer._generation_request("Hello.", "ctx_test", continue_=True)

    assert request["generation_config"] == {"speed": 1.2}
    assert request["continue"] is True


def test_stream_speech_chunks_sends_continuations_before_final(monkeypatch):
    audio = base64.b64encode(b"\x00\x00" * 120).decode("ascii")

    class FakeWebSocket:
        def __init__(self):
            self.sent = []
            self.inbound = asyncio.Queue()

        async def send(self, message):
            request = json.loads(message)
            self.sent.append(request)
            if request["transcript"]:
                await self.inbound.put(
                    json.dumps(
                        {
                            "type": "chunk",
                            "data": audio,
                            "context_id": request["context_id"],
                        }
                    )
                )
            if request["continue"] is False:
                await self.inbound.put(json.dumps({"type": "done", "context_id": request["context_id"]}))

        def __aiter__(self):
            return self

        async def __anext__(self):
            return await self.inbound.get()

    class FakeConnect:
        def __init__(self, websocket):
            self.websocket = websocket

        async def __aenter__(self):
            return self.websocket

        async def __aexit__(self, exc_type, exc, tb):
            return False

    fake_websocket = FakeWebSocket()
    monkeypatch.setattr(cartesia_tts.websockets, "connect", lambda *args, **kwargs: FakeConnect(fake_websocket))

    async def input_chunks():
        yield "First sentence. "
        await asyncio.sleep(0)
        yield "Second sentence."

    async def run_stream():
        synthesizer = CartesiaStreamingTTS(
            api_key="cartesia-key",
            model_id="sonic-3",
            voice_id="voice-id",
            sample_rate=16000,
            cartesia_version="2026-03-01",
            speed=1.2,
        )
        messages = []
        async for message in synthesizer.stream_speech_chunks(input_chunks(), context_id="ctx_test"):
            messages.append(message)
        return messages

    messages = asyncio.run(run_stream())

    assert [request["transcript"] for request in fake_websocket.sent] == [
        "First sentence. ",
        "Second sentence.",
        "",
    ]
    assert [request["continue"] for request in fake_websocket.sent] == [True, True, False]
    assert all(request["generation_config"] == {"speed": 1.2} for request in fake_websocket.sent)
    assert [message["type"] for message in messages] == ["chunk", "chunk", "done"]
