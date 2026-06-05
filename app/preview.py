"""One-shot character preview: run a candidate persona through the same
LLM + TTS building blocks a live session uses, without a WebSocket or any
persistence. Honors VOICE_AGENT_MODE — in mock mode it returns a canned reply
and synthesized tone so the Personality Studio works without provider keys.
"""
from __future__ import annotations

import base64
import time

from app.cartesia_tts import CartesiaStreamingTTS
from app.characters import Character, build_system_prompt
from app.config import Settings, is_uuid
from app.groq_agent import GroqStreamingAgent
from app.mock_conversation import generate_mock_pcm

MOCK_SAMPLE_RATE = 16000
MOCK_MIN_SECONDS = 0.4
MOCK_MAX_SECONDS = 2.5


def _usable_cartesia(settings: Settings) -> bool:
    return bool(
        settings.normalized_mode == "live"
        and settings.cartesia_api_key
        and settings.cartesia_voice_id
        and is_uuid(settings.cartesia_voice_id)
    )


def _mock_reply(character: Character, message: str) -> str:
    return (
        f"Hi, I'm {character.name}. I heard: \"{message}\". "
        "This is a mock preview — connect live provider keys to hear a real in-character reply."
    )


async def preview_character(character: Character, message: str, settings: Settings) -> dict:
    """Generate a single text + audio preview for ``character`` replying to ``message``.

    Never raises for provider problems: any missing key, invalid config, or
    provider error is recorded in ``warnings`` and the corresponding stage falls
    back to mock output, so the caller always gets a playable result.
    """
    mode = settings.normalized_mode
    warnings: list[dict] = []

    if mode == "live":
        for key in settings.missing_live_keys():
            warnings.append(
                {"provider": "config", "code": key, "message": f"{key} is not configured; using mock fallback."}
            )
        for key in settings.invalid_live_keys():
            warnings.append(
                {"provider": "config", "code": key, "message": f"{key} is invalid; using mock fallback."}
            )

    # --- LLM: reuse the same agent + system-prompt builder as a live turn ---
    persona = build_system_prompt(character)
    transcript = [{"role": "user", "content": message}]
    llm_provider = "mock"
    text = ""
    llm_start = time.perf_counter()
    if mode == "live" and settings.groq_api_key:
        agent = GroqStreamingAgent(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            persona=persona,
            temperature=settings.groq_temperature,
        )
        try:
            async for delta in agent.stream_response(transcript):
                text += delta
            llm_provider = "groq"
        except Exception as exc:  # noqa: BLE001 — surface as warning, fall back to mock
            warnings.append({"provider": "groq", "message": str(exc)})
            text = ""
    text = text.strip()
    if not text:
        text = _mock_reply(character, message)
        llm_provider = "mock"
    llm_ms = round((time.perf_counter() - llm_start) * 1000, 2)

    # --- TTS: reuse the same Cartesia client as a live turn ---
    tts_provider = "mock"
    sample_rate = MOCK_SAMPLE_RATE
    audio_bytes = b""
    tts_start = time.perf_counter()
    if _usable_cartesia(settings):
        synthesizer = CartesiaStreamingTTS(
            api_key=settings.cartesia_api_key,
            model_id=settings.cartesia_model,
            voice_id=settings.cartesia_voice_id,
            sample_rate=settings.cartesia_sample_rate,
            cartesia_version=settings.cartesia_version,
            speed=getattr(settings, "cartesia_speed", None),
            open_timeout_seconds=getattr(settings, "cartesia_open_timeout_seconds", 8.0),
            connect_retries=getattr(settings, "cartesia_connect_retries", 1),
        )
        try:
            async for chunk in synthesizer.stream_speech(text):
                if chunk["type"] == "error":
                    warnings.append({"provider": "cartesia", "message": chunk["message"]})
                    audio_bytes = b""
                    break
                if chunk["type"] == "chunk" and chunk["audio"]:
                    audio_bytes += base64.b64decode(chunk["audio"])
            if audio_bytes:
                tts_provider = "cartesia"
                sample_rate = settings.cartesia_sample_rate
        except Exception as exc:  # noqa: BLE001 — surface as warning, fall back to mock
            warnings.append({"provider": "cartesia", "message": str(exc)})
            audio_bytes = b""
    if not audio_bytes:
        duration = max(MOCK_MIN_SECONDS, min(MOCK_MAX_SECONDS, len(text) / 25))
        audio_bytes = generate_mock_pcm(sample_rate=MOCK_SAMPLE_RATE, duration_seconds=duration)
        sample_rate = MOCK_SAMPLE_RATE
        tts_provider = "mock"
    tts_ms = round((time.perf_counter() - tts_start) * 1000, 2)

    return {
        "character_id": character.id,
        "text": text,
        "audio": base64.b64encode(audio_bytes).decode("ascii"),
        "encoding": "pcm_s16le",
        "sample_rate": sample_rate,
        "channels": 1,
        "latency": {"llm_ms": llm_ms, "tts_ms": tts_ms},
        "providers": {"llm": llm_provider, "tts": tts_provider},
        "warnings": warnings,
    }
