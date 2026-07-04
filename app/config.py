from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


MOCK_PROACTIVE_SILENCE_TIMEOUT_MS = 5000
LIVE_PROACTIVE_SILENCE_TIMEOUT_MS = 30000
MOCK_PROACTIVE_REPEAT_COOLDOWN_MS = 8000
LIVE_PROACTIVE_REPEAT_COOLDOWN_MS = 60000
MOCK_PROACTIVE_MAX_CONSECUTIVE_PROMPTS = 3
LIVE_PROACTIVE_MAX_CONSECUTIVE_PROMPTS = 1
DEFAULT_DEEPGRAM_ENDPOINTING_MS = 200
MIN_DEEPGRAM_UTTERANCE_END_MS = 1000
DEFAULT_DEEPGRAM_UTTERANCE_END_MS = MIN_DEEPGRAM_UTTERANCE_END_MS
DEFAULT_PARTIAL_IDLE_FINALIZE_MS = 500
DEFAULT_AMBIENCE_VOLUME = 0.035
DEFAULT_INPUT_GAIN = 2.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mode: str = Field(default="mock", alias="VOICE_AGENT_MODE")
    host: str = Field(default="0.0.0.0", alias="VOICE_AGENT_HOST")
    port: int = Field(default=8000, validation_alias=AliasChoices("VOICE_AGENT_PORT", "PORT"))
    ws_ping_interval: float = Field(default=30.0, alias="VOICE_AGENT_WS_PING_INTERVAL")
    ws_ping_timeout: float = Field(default=120.0, alias="VOICE_AGENT_WS_PING_TIMEOUT")
    cors_origins: str = Field(
        default="http://localhost:8000,http://127.0.0.1:8000",
        alias="VOICE_AGENT_CORS_ORIGINS",
    )

    deepgram_api_key: str | None = Field(default=None, alias="DEEPGRAM_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    elevenlabs_api_key: str | None = Field(default=None, alias="ELEVENLABS_API_KEY")

    deepgram_model: str = Field(default="nova-3", alias="DEEPGRAM_MODEL")
    deepgram_endpointing_ms: int = Field(default=DEFAULT_DEEPGRAM_ENDPOINTING_MS, alias="DEEPGRAM_ENDPOINTING_MS")
    deepgram_utterance_end_ms: int = Field(default=DEFAULT_DEEPGRAM_UTTERANCE_END_MS, alias="DEEPGRAM_UTTERANCE_END_MS")
    partial_idle_finalize_ms: int = Field(
        default=DEFAULT_PARTIAL_IDLE_FINALIZE_MS,
        alias="VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS",
    )
    input_gain: float = Field(default=DEFAULT_INPUT_GAIN, alias="VOICE_AGENT_INPUT_GAIN")
    # Pipecat turn-taking. VAD gates when a user turn can start/stop; smart turn
    # is a local model that distinguishes a finished thought from a mid-sentence
    # pause, so the bot neither jumps in early nor pads every turn with silence.
    vad_confidence: float = Field(default=0.7, ge=0.0, le=1.0, alias="VOICE_AGENT_VAD_CONFIDENCE")
    vad_start_secs: float = Field(default=0.2, gt=0, alias="VOICE_AGENT_VAD_START_SECS")
    vad_stop_secs: float = Field(default=0.2, gt=0, alias="VOICE_AGENT_VAD_STOP_SECS")
    vad_min_volume: float = Field(default=0.6, ge=0.0, le=1.0, alias="VOICE_AGENT_VAD_MIN_VOLUME")
    smart_turn_enabled: bool = Field(default=True, alias="VOICE_AGENT_SMART_TURN_ENABLED")
    smart_turn_stop_secs: float = Field(default=3.0, gt=0, alias="VOICE_AGENT_SMART_TURN_STOP_SECS")
    speech_timeout_stop_secs: float = Field(
        default=0.8, gt=0, alias="VOICE_AGENT_SPEECH_TIMEOUT_STOP_SECS"
    )
    # Words required in a transcript before barge-in interrupts the bot, so a
    # cough or background noise can't cut it off. 0 = interrupt on any speech.
    interruption_min_words: int = Field(default=2, ge=0, alias="VOICE_AGENT_INTERRUPT_MIN_WORDS")
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")
    groq_temperature: float = Field(default=0.7, alias="GROQ_TEMPERATURE")
    # Caps reply length to voice-appropriate size. 0 = no cap.
    groq_max_tokens: int = Field(default=200, ge=0, alias="GROQ_MAX_TOKENS")
    # Sliding-window cap on conversation turns sent to the LLM each turn. Bounds
    # per-turn token cost so a long call doesn't keep re-sending the full
    # transcript (which exhausts Groq's TPM budget and stalls replies). 0 = unbounded.
    llm_context_max_turns: int = Field(default=12, ge=0, alias="VOICE_AGENT_LLM_CONTEXT_MAX_TURNS")
    # Which LLM answers by default (a key in app/llm_models.py MODELS). The UI model
    # picker overrides this per session via ?model=. Groq or an OpenRouter model.
    default_model: str = Field(default="groq-llama-3.1-8b", alias="DEFAULT_MODEL")
    elevenlabs_model: str = Field(default="eleven_flash_v2_5", alias="ELEVENLABS_MODEL")
    elevenlabs_speed: float = Field(default=1.0, alias="ELEVENLABS_SPEED")
    elevenlabs_voice_id: str | None = Field(default=None, alias="ELEVENLABS_VOICE_ID")
    elevenlabs_sample_rate: int = Field(default=16000, alias="ELEVENLABS_SAMPLE_RATE")
    elevenlabs_stability: float = Field(default=0.5, ge=0.0, le=1.0, alias="ELEVENLABS_STABILITY")
    elevenlabs_similarity_boost: float = Field(default=0.8, ge=0.0, le=1.0, alias="ELEVENLABS_SIMILARITY_BOOST")
    elevenlabs_style: float = Field(default=0.0, ge=0.0, le=1.0, alias="ELEVENLABS_STYLE")
    elevenlabs_use_speaker_boost: bool = Field(default=False, alias="ELEVENLABS_USE_SPEAKER_BOOST")
    elevenlabs_open_timeout_seconds: float = Field(default=8.0, gt=0, alias="ELEVENLABS_OPEN_TIMEOUT_SECONDS")
    elevenlabs_connect_retries: int = Field(default=1, ge=0, alias="ELEVENLABS_CONNECT_RETRIES")
    twilio_account_sid: str | None = Field(default=None, alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str | None = Field(default=None, alias="TWILIO_AUTH_TOKEN")
    public_host: str | None = Field(default=None, alias="PUBLIC_HOST")
    persona: str = Field(
        default=(
            "You are a concise, warm voice on an ambiguous open phone call. "
            "Sound present and spoken, as if you are still on the line with the caller. "
            "Keep replies brief, natural, and phone-call appropriate. Never sound like a web chat assistant."
        ),
        alias="VOICE_AGENT_PERSONA",
    )
    default_character_id: str = Field(default="zara", alias="VOICE_AGENT_DEFAULT_CHARACTER")
    intent_inference_enabled: bool = Field(default=True, alias="VOICE_AGENT_INTENT_INFERENCE_ENABLED")
    ambience_enabled: bool = Field(default=True, alias="VOICE_AGENT_AMBIENCE_ENABLED")
    ambience_scene: str = Field(default="room_line", alias="VOICE_AGENT_AMBIENCE_SCENE")
    ambience_volume: float = Field(default=DEFAULT_AMBIENCE_VOLUME, alias="VOICE_AGENT_AMBIENCE_VOLUME")
    proactive_enabled: str = Field(default="auto", alias="VOICE_AGENT_PROACTIVE_ENABLED")
    proactive_startup_greeting_delay_ms: int = Field(default=500, alias="VOICE_AGENT_PROACTIVE_GREETING_DELAY_MS")
    proactive_silence_timeout_ms: int | None = Field(default=None, alias="VOICE_AGENT_PROACTIVE_SILENCE_TIMEOUT_MS")
    proactive_repeat_cooldown_ms: int | None = Field(default=None, alias="VOICE_AGENT_PROACTIVE_REPEAT_COOLDOWN_MS")
    proactive_max_consecutive_prompts: int | None = Field(
        default=None,
        alias="VOICE_AGENT_PROACTIVE_MAX_CONSECUTIVE_PROMPTS",
    )
    proactive_failure_backoff_threshold: int = Field(default=2, alias="VOICE_AGENT_PROACTIVE_FAILURE_BACKOFF_THRESHOLD")
    proactive_failure_backoff_ms: int = Field(default=30000, alias="VOICE_AGENT_PROACTIVE_FAILURE_BACKOFF_MS")
    proactive_contextual_followups_enabled: bool = Field(
        default=True,
        alias="VOICE_AGENT_PROACTIVE_CONTEXTUAL_FOLLOWUPS_ENABLED",
    )
    memory_enabled: str = Field(default="auto", alias="VOICE_AGENT_MEMORY_ENABLED")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    memory_embedding_provider: str = Field(default="openai", alias="VOICE_AGENT_MEMORY_EMBEDDING_PROVIDER")
    memory_embedding_model: str = Field(default="text-embedding-3-small", alias="VOICE_AGENT_MEMORY_EMBEDDING_MODEL")
    memory_top_k: int = Field(default=5, ge=1, le=50, alias="VOICE_AGENT_MEMORY_TOP_K")
    memory_dedupe_threshold: float = Field(default=0.92, ge=0.0, le=1.0, alias="VOICE_AGENT_MEMORY_DEDUPE_THRESHOLD")
    memory_dir: str = Field(default="data/memory", alias="VOICE_AGENT_MEMORY_DIR")
    memory_session_summary_enabled: bool = Field(
        default=True,
        alias="VOICE_AGENT_MEMORY_SESSION_SUMMARY_ENABLED",
    )

    @field_validator("elevenlabs_voice_id", mode="before")
    @classmethod
    def normalize_elevenlabs_voice_id(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip().strip("'\"")
        return text or None

    @field_validator("elevenlabs_speed")
    @classmethod
    def validate_elevenlabs_speed(cls, value: float) -> float:
        if not 0.7 <= value <= 1.2:
            raise ValueError("ELEVENLABS_SPEED must be between 0.7 and 1.2")
        return value

    @field_validator("partial_idle_finalize_ms")
    @classmethod
    def validate_partial_idle_finalize_ms(cls, value: int) -> int:
        if value < 1:
            raise ValueError("VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS must be at least 1")
        return value

    @field_validator("input_gain")
    @classmethod
    def validate_input_gain(cls, value: float) -> float:
        if not 0.1 <= value <= 8.0:
            raise ValueError("VOICE_AGENT_INPUT_GAIN must be between 0.1 and 8.0")
        return round(value, 2)

    @field_validator("deepgram_utterance_end_ms")
    @classmethod
    def normalize_deepgram_utterance_end_ms(cls, value: int) -> int:
        return max(value, MIN_DEEPGRAM_UTTERANCE_END_MS)

    @field_validator("ambience_scene")
    @classmethod
    def normalize_ambience_scene(cls, value: str) -> str:
        normalized = value.strip().lower().replace("-", "_")
        if not normalized:
            return "room_line"
        return normalized

    @field_validator("ambience_volume")
    @classmethod
    def validate_ambience_volume(cls, value: float) -> float:
        if not 0 <= value <= 0.2:
            raise ValueError("VOICE_AGENT_AMBIENCE_VOLUME must be between 0 and 0.2")
        return round(value, 3)

    @property
    def normalized_mode(self) -> str:
        return self.mode.strip().lower()

    @property
    def parsed_cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def allow_all_cors_origins(self) -> bool:
        return not self.parsed_cors_origins or "*" in self.parsed_cors_origins

    @property
    def cors_allow_credentials(self) -> bool:
        return not self.allow_all_cors_origins

    @property
    def normalized_proactive_enabled(self) -> str:
        return self.proactive_enabled.strip().lower()

    @property
    def proactive_effective_enabled(self) -> bool:
        configured = self.normalized_proactive_enabled
        if configured in {"", "auto", "default"}:
            return self.normalized_mode != "live"
        if configured in {"1", "true", "yes", "on", "enabled"}:
            return True
        if configured in {"0", "false", "no", "off", "disabled"}:
            return False
        return False

    @property
    def proactive_effective_silence_timeout_ms(self) -> int:
        if self.proactive_silence_timeout_ms is not None:
            return self.proactive_silence_timeout_ms
        if self.normalized_mode == "live":
            return LIVE_PROACTIVE_SILENCE_TIMEOUT_MS
        return MOCK_PROACTIVE_SILENCE_TIMEOUT_MS

    @property
    def proactive_effective_repeat_cooldown_ms(self) -> int:
        if self.proactive_repeat_cooldown_ms is not None:
            return self.proactive_repeat_cooldown_ms
        if self.normalized_mode == "live":
            return LIVE_PROACTIVE_REPEAT_COOLDOWN_MS
        return MOCK_PROACTIVE_REPEAT_COOLDOWN_MS

    @property
    def proactive_effective_max_consecutive_prompts(self) -> int:
        if self.proactive_max_consecutive_prompts is not None:
            return self.proactive_max_consecutive_prompts
        if self.normalized_mode == "live":
            return LIVE_PROACTIVE_MAX_CONSECUTIVE_PROMPTS
        return MOCK_PROACTIVE_MAX_CONSECUTIVE_PROMPTS

    @property
    def normalized_memory_enabled(self) -> str:
        return self.memory_enabled.strip().lower()

    @property
    def memory_effective_enabled(self) -> bool:
        configured = self.normalized_memory_enabled
        if configured in {"", "auto", "default"}:
            # On by default in mock; explicit opt-in for live (like proactive).
            return self.normalized_mode != "live"
        if configured in {"1", "true", "yes", "on", "enabled"}:
            return True
        return False

    def missing_live_keys(self) -> list[str]:
        missing: list[str] = []
        if not self.deepgram_api_key:
            missing.append("DEEPGRAM_API_KEY")
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.elevenlabs_api_key:
            missing.append("ELEVENLABS_API_KEY")
        if not self.elevenlabs_voice_id:
            missing.append("ELEVENLABS_VOICE_ID")
        return missing

    def invalid_live_keys(self) -> list[str]:
        # ElevenLabs voice ids are opaque strings (not UUIDs), so there is no
        # format to validate here. Kept for the health-report contract.
        return []

    def _llm_status(self) -> dict:
        from app.llm_models import MODELS, resolve_model

        entry = resolve_model(self)
        return {
            "active_model": entry["key"],
            "provider": entry["provider"],
            "model": entry["model"],
            "openrouter_configured": bool(self.openrouter_api_key),
            "available": list(MODELS.keys()),
        }

    def public_config_status(self) -> dict:
        missing = self.missing_live_keys() if self.normalized_mode == "live" else []
        invalid = self.invalid_live_keys() if self.normalized_mode == "live" else []
        return {
            "mode": self.normalized_mode,
            "live_ready": not missing and not invalid,
            "missing_live_keys": missing,
            "invalid_live_keys": invalid,
            "server": {
                "host": self.host,
                "port": self.port,
            },
            "cors": {
                "allow_all_origins": self.allow_all_cors_origins,
                "origin_count": 0 if self.allow_all_cors_origins else len(self.parsed_cors_origins),
                "allow_credentials": self.cors_allow_credentials,
            },
            "providers": {
                "stt": "deepgram",
                "llm": "groq",
                "tts": "elevenlabs",
            },
            "audio": {
                "input_gain": self.input_gain,
            },
            "conversation": {
                "intent_inference_enabled": self.intent_inference_enabled,
            },
            "llm": self._llm_status(),
            "memory": {
                "configured": self.normalized_memory_enabled,
                "enabled": self.memory_effective_enabled,
                "embedding_provider": self.memory_embedding_provider,
                "embedder_ready": (
                    self.normalized_mode == "live"
                    and self.memory_embedding_provider.strip().lower() == "openai"
                    and bool(self.openai_api_key)
                ),
                "top_k": self.memory_top_k,
                "session_summary_enabled": self.memory_session_summary_enabled,
            },
            "elevenlabs": {
                "model": self.elevenlabs_model,
                "sample_rate": self.elevenlabs_sample_rate,
                "connection": {
                    "open_timeout_seconds": self.elevenlabs_open_timeout_seconds,
                    "connect_retries": self.elevenlabs_connect_retries,
                },
                "voice_settings": {
                    "stability": self.elevenlabs_stability,
                    "similarity_boost": self.elevenlabs_similarity_boost,
                    "style": self.elevenlabs_style,
                    "use_speaker_boost": self.elevenlabs_use_speaker_boost,
                    "speed": self.elevenlabs_speed,
                },
            },
            "ambience": {
                "enabled": self.ambience_enabled,
                "scene": self.ambience_scene,
                "volume": self.ambience_volume,
            },
            "turn_timing": {
                "deepgram_endpointing_ms": self.deepgram_endpointing_ms,
                "deepgram_utterance_end_ms": self.deepgram_utterance_end_ms,
                "partial_idle_finalize_ms": self.partial_idle_finalize_ms,
                "vad": {
                    "confidence": self.vad_confidence,
                    "start_secs": self.vad_start_secs,
                    "stop_secs": self.vad_stop_secs,
                    "min_volume": self.vad_min_volume,
                },
                "smart_turn": {
                    "enabled": self.smart_turn_enabled,
                    "stop_secs": self.smart_turn_stop_secs,
                },
                "speech_timeout_stop_secs": self.speech_timeout_stop_secs,
                "interruption_min_words": self.interruption_min_words,
            },
            "proactive": {
                "configured": self.normalized_proactive_enabled,
                "enabled": self.proactive_effective_enabled,
                "startup_greeting_delay_ms": self.proactive_startup_greeting_delay_ms,
                "silence_timeout_ms": self.proactive_effective_silence_timeout_ms,
                "repeat_cooldown_ms": self.proactive_effective_repeat_cooldown_ms,
                "max_consecutive_prompts": self.proactive_effective_max_consecutive_prompts,
                "failure_backoff_threshold": self.proactive_failure_backoff_threshold,
                "failure_backoff_ms": self.proactive_failure_backoff_ms,
                "contextual_followups_enabled": self.proactive_contextual_followups_enabled,
            },
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
