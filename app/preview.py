"""One-shot character preview: run a candidate persona through the same
LLM + TTS building blocks a live session uses, without a WebSocket or any
persistence. Honors VOICE_AGENT_MODE — in mock mode it returns a canned reply
and synthesized tone so the Personality Studio works without provider keys.
"""
from __future__ import annotations

import base64
import time

from app.characters import Character, build_relevant_canon, build_system_prompt
from app.config import Settings
from app.conversation_context import agent_transcript_with_context
from app.elevenlabs_tts import ElevenLabsStreamingTTS
from app.groq_agent import GroqStreamingAgent
from app.llm_models import resolve_chat_target
from app.mock_conversation import generate_mock_pcm
from app.voice_settings import resolve_active_voice_id

MOCK_SAMPLE_RATE = 16000
MOCK_MIN_SECONDS = 0.4
MOCK_MAX_SECONDS = 2.5


def _usable_elevenlabs(settings: Settings) -> bool:
    return bool(
        settings.normalized_mode == "live"
        and settings.elevenlabs_api_key
        and resolve_active_voice_id(settings)
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
    transcript = agent_transcript_with_context(
        [{"role": "user", "content": message}],
        intent_enabled=settings.intent_inference_enabled,
        conversation_flow_enabled=(
            settings.conversation_flow_enabled and character.conversation_mode == "social"
        ),
        character_name=character.name,
        private_context=build_relevant_canon(
            character,
            [{"role": "user", "content": message}],
        ),
    )
    llm_provider = "mock"
    text = ""
    llm_start = time.perf_counter()
    model_entry = resolve_chat_target(settings)
    if model_entry["fallback_from"]:
        warnings.append(
            {
                "provider": "openrouter",
                "message": "OPENROUTER_API_KEY is not configured; preview used the Groq fallback.",
            }
        )
    llm_key = model_entry["api_key"]
    if mode == "live" and llm_key:
        agent = GroqStreamingAgent(
            api_key=llm_key,
            model=model_entry["model"],
            persona=persona,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
            reasoning_effort=settings.groq_reasoning_effort,
            endpoint_url=model_entry["endpoint_url"],
        )
        try:
            async for delta in agent.stream_response(transcript):
                text += delta
            llm_provider = model_entry["provider"]
        except Exception as exc:  # noqa: BLE001 — surface as warning, fall back to mock
            warnings.append({"provider": model_entry["provider"], "message": str(exc)})
            text = ""
    text = text.strip()
    if not text:
        text = _mock_reply(character, message)
        llm_provider = "mock"
    llm_ms = round((time.perf_counter() - llm_start) * 1000, 2)

    # --- TTS: reuse the same ElevenLabs client as a live turn ---
    tts_provider = "mock"
    sample_rate = MOCK_SAMPLE_RATE
    audio_bytes = b""
    tts_start = time.perf_counter()
    if _usable_elevenlabs(settings):
        synthesizer = ElevenLabsStreamingTTS(
            api_key=settings.elevenlabs_api_key,
            model_id=settings.elevenlabs_model,
            voice_id=resolve_active_voice_id(settings),
            sample_rate=settings.elevenlabs_sample_rate,
            stability=settings.elevenlabs_stability,
            similarity_boost=settings.elevenlabs_similarity_boost,
            style=settings.elevenlabs_style,
            use_speaker_boost=settings.elevenlabs_use_speaker_boost,
            speed=getattr(settings, "elevenlabs_speed", None),
            open_timeout_seconds=getattr(settings, "elevenlabs_open_timeout_seconds", 8.0),
            connect_retries=getattr(settings, "elevenlabs_connect_retries", 1),
        )
        try:
            async for chunk in synthesizer.stream_speech(text):
                if chunk["type"] == "error":
                    warnings.append({"provider": "elevenlabs", "message": chunk["message"]})
                    audio_bytes = b""
                    break
                if chunk["type"] == "chunk" and chunk["audio"]:
                    audio_bytes += base64.b64decode(chunk["audio"])
            if audio_bytes:
                tts_provider = "elevenlabs"
                sample_rate = settings.elevenlabs_sample_rate
        except Exception as exc:  # noqa: BLE001 — surface as warning, fall back to mock
            warnings.append({"provider": "elevenlabs", "message": str(exc)})
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
