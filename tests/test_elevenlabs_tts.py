import asyncio
import base64
import json

import app.elevenlabs_tts as elevenlabs_tts
from app.elevenlabs_tts import ElevenLabsStreamingTTS


def _make_synth(**overrides) -> ElevenLabsStreamingTTS:
    params = dict(
        api_key="eleven-key",
        model_id="eleven_flash_v2_5",
        voice_id="voice-id",
        sample_rate=16000,
        speed=1.0,
    )
    params.update(overrides)
    return ElevenLabsStreamingTTS(**params)


def test_ws_url_carries_model_and_pcm_output_format():
    synthesizer = _make_synth()

    url = synthesizer._ws_url()

    assert url.startswith("wss://api.elevenlabs.io/v1/text-to-speech/voice-id/stream-input")
    assert "model_id=eleven_flash_v2_5" in url
    assert "output_format=pcm_16000" in url


def test_init_message_carries_voice_settings():
    synthesizer = _make_synth(stability=0.3, similarity_boost=0.7, style=0.1, use_speaker_boost=True, speed=1.1)

    init = synthesizer._init_message()

    assert init["text"] == " "
    assert init["voice_settings"] == {
        "stability": 0.3,
        "similarity_boost": 0.7,
        "style": 0.1,
        "use_speaker_boost": True,
        "speed": 1.1,
    }
    assert "chunk_length_schedule" in init["generation_config"]


def test_init_message_omits_speed_when_unset():
    synthesizer = _make_synth(speed=None)

    assert "speed" not in synthesizer._init_message()["voice_settings"]


class _FakeWebSocket:
    """Models the ElevenLabs stream-input protocol: BOS (with voice_settings)
    produces no audio, each text frame yields one audio message, and the empty
    EOS frame yields a final ``isFinal`` message.
    """

    def __init__(self, audio: str):
        self.sent: list[dict] = []
        self.inbound: asyncio.Queue = asyncio.Queue()
        self._audio = audio
        self.closed = False

    async def send(self, message):
        request = json.loads(message)
        self.sent.append(request)
        if "voice_settings" in request:
            return  # BOS init frame — no audio
        if request.get("text", "") == "":
            await self.inbound.put(json.dumps({"audio": None, "isFinal": True}))
        else:
            await self.inbound.put(json.dumps({"audio": self._audio}))

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.inbound.get()

    async def close(self):
        self.closed = True

    @property
    def text_frames(self) -> list[str]:
        return [r["text"] for r in self.sent if "voice_settings" not in r]


class _FakeConnect:
    def __init__(self, websocket):
        self.websocket = websocket

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_stream_speech_chunks_sends_init_frames_then_eos(monkeypatch):
    audio = base64.b64encode(b"\x00\x00" * 120).decode("ascii")
    fake_websocket = _FakeWebSocket(audio)
    monkeypatch.setattr(
        elevenlabs_tts.websockets, "connect", lambda *args, **kwargs: _FakeConnect(fake_websocket)
    )

    async def input_chunks():
        yield "First sentence. "
        await asyncio.sleep(0)
        yield "Second sentence."

    async def run_stream():
        synthesizer = _make_synth()
        messages = []
        async for message in synthesizer.stream_speech_chunks(input_chunks(), context_id="ctx_test"):
            messages.append(message)
        return messages

    messages = asyncio.run(run_stream())

    # First frame is the BOS init carrying voice settings.
    assert "voice_settings" in fake_websocket.sent[0]
    assert fake_websocket.sent[0]["text"] == " "
    # Each fragment is sent verbatim with a trailing space, then an empty EOS frame.
    assert fake_websocket.text_frames == ["First sentence. ", "Second sentence. ", ""]
    assert [m["type"] for m in messages] == ["chunk", "chunk", "done"]
    assert all(m["context_id"] == "ctx_test" for m in messages)


def test_stream_speech_retries_opening_handshake_timeout(monkeypatch):
    audio = base64.b64encode(b"\x00\x00" * 120).decode("ascii")
    fake_websocket = _FakeWebSocket(audio)

    class _TimeoutConnect:
        async def __aenter__(self):
            raise TimeoutError("timed out during opening handshake")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    connect_kwargs = []

    def fake_connect(*args, **kwargs):
        connect_kwargs.append(kwargs)
        if len(connect_kwargs) == 1:
            return _TimeoutConnect()
        return _FakeConnect(fake_websocket)

    monkeypatch.setattr(elevenlabs_tts.websockets, "connect", fake_connect)

    async def run_stream():
        synthesizer = _make_synth(open_timeout_seconds=3, connect_retries=1, retry_backoff_seconds=0)
        messages = []
        async for message in synthesizer.stream_speech("Hello.", context_id="ctx_test"):
            messages.append(message)
        return messages

    messages = asyncio.run(run_stream())

    assert len(connect_kwargs) == 2
    assert all(kwargs["open_timeout"] == 3 for kwargs in connect_kwargs)
    assert fake_websocket.text_frames == ["Hello. ", ""]
    assert [m["type"] for m in messages] == ["chunk", "done"]


def test_parse_message_surfaces_error():
    parsed = elevenlabs_tts.parse_elevenlabs_message(
        json.dumps({"error": "voice not found", "code": 404}), context_id="ctx_x"
    )

    assert parsed == {"type": "error", "context_id": "ctx_x", "message": "voice not found", "done": True}


def test_generate_context_id_rewrites_prefix():
    assert elevenlabs_tts.generate_context_id("resp_abc123") == "ctx_abc123"
