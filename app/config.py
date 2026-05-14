from functools import lru_cache
import re
from uuid import UUID

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


UUID_PREFIX_PATTERN = re.compile(
    r"^\s*['\"]?([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
MOCK_PROACTIVE_SILENCE_TIMEOUT_MS = 5000
LIVE_PROACTIVE_SILENCE_TIMEOUT_MS = 30000
MOCK_PROACTIVE_REPEAT_COOLDOWN_MS = 8000
LIVE_PROACTIVE_REPEAT_COOLDOWN_MS = 60000
MOCK_PROACTIVE_MAX_CONSECUTIVE_PROMPTS = 3
LIVE_PROACTIVE_MAX_CONSECUTIVE_PROMPTS = 1
DEFAULT_DEEPGRAM_ENDPOINTING_MS = 220
MIN_DEEPGRAM_UTTERANCE_END_MS = 1000
DEFAULT_DEEPGRAM_UTTERANCE_END_MS = MIN_DEEPGRAM_UTTERANCE_END_MS
DEFAULT_PARTIAL_IDLE_FINALIZE_MS = 1000
DEFAULT_AMBIENCE_VOLUME = 0.035
DEFAULT_INPUT_GAIN = 2.0


def normalize_leading_uuid(value: object) -> str | None:
    if value is None:
        return None

    text = str(value).strip().strip("'\"")
    if not text:
        return None

    match = UUID_PREFIX_PATTERN.match(text)
    if not match:
        return text

    return str(UUID(match.group(1)))


def is_uuid(value: str | None) -> bool:
    if not value:
        return False
    try:
        UUID(value)
    except ValueError:
        return False
    return True


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
    cartesia_api_key: str | None = Field(default=None, alias="CARTESIA_API_KEY")

    deepgram_model: str = Field(default="nova-3", alias="DEEPGRAM_MODEL")
    deepgram_endpointing_ms: int = Field(default=DEFAULT_DEEPGRAM_ENDPOINTING_MS, alias="DEEPGRAM_ENDPOINTING_MS")
    deepgram_utterance_end_ms: int = Field(default=DEFAULT_DEEPGRAM_UTTERANCE_END_MS, alias="DEEPGRAM_UTTERANCE_END_MS")
    partial_idle_finalize_ms: int = Field(
        default=DEFAULT_PARTIAL_IDLE_FINALIZE_MS,
        alias="VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS",
    )
    input_gain: float = Field(default=DEFAULT_INPUT_GAIN, alias="VOICE_AGENT_INPUT_GAIN")
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")
    groq_temperature: float = Field(default=0.7, alias="GROQ_TEMPERATURE")
    cartesia_model: str = Field(default="sonic-3", alias="CARTESIA_MODEL")
    cartesia_speed: float = Field(default=1.2, alias="CARTESIA_SPEED")
    cartesia_voice_id: str | None = Field(default=None, alias="CARTESIA_VOICE_ID")
    cartesia_sample_rate: int = Field(default=16000, alias="CARTESIA_SAMPLE_RATE")
    cartesia_version: str = Field(default="2026-03-01", alias="CARTESIA_VERSION")
    cartesia_open_timeout_seconds: float = Field(default=8.0, gt=0, alias="CARTESIA_OPEN_TIMEOUT_SECONDS")
    cartesia_connect_retries: int = Field(default=1, ge=0, alias="CARTESIA_CONNECT_RETRIES")
    persona: str = Field(
        default=(
            "You are a concise, warm voice on an ambiguous open phone call. "
            "Sound present and spoken, as if you are still on the line with the caller. "
            "Keep replies brief, natural, and phone-call appropriate. Never sound like a web chat assistant."
        ),
        alias="VOICE_AGENT_PERSONA",
    )
    intent_inference_enabled: bool = Field(default=True, alias="VOICE_AGENT_INTENT_INFERENCE_ENABLED")
    cartesia_speech_director_enabled: bool = Field(
        default=True,
        alias="VOICE_AGENT_CARTESIA_SPEECH_DIRECTOR_ENABLED",
    )
    cartesia_ssml_enabled: bool = Field(default=True, alias="VOICE_AGENT_CARTESIA_SSML_ENABLED")
    cartesia_emotion_tags_enabled: bool = Field(
        default=False,
        alias="VOICE_AGENT_CARTESIA_EMOTION_TAGS_ENABLED",
    )
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

    @field_validator("cartesia_voice_id", mode="before")
    @classmethod
    def normalize_cartesia_voice_id(cls, value: object) -> str | None:
        return normalize_leading_uuid(value)

    @field_validator("cartesia_speed")
    @classmethod
    def validate_cartesia_speed(cls, value: float) -> float:
        if not 0.6 <= value <= 1.5:
            raise ValueError("CARTESIA_SPEED must be between 0.6 and 1.5")
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

    def missing_live_keys(self) -> list[str]:
        missing: list[str] = []
        if not self.deepgram_api_key:
            missing.append("DEEPGRAM_API_KEY")
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if not self.cartesia_api_key:
            missing.append("CARTESIA_API_KEY")
        if not self.cartesia_voice_id:
            missing.append("CARTESIA_VOICE_ID")
        return missing

    def invalid_live_keys(self) -> list[str]:
        invalid: list[str] = []
        if self.cartesia_voice_id and not is_uuid(self.cartesia_voice_id):
            invalid.append("CARTESIA_VOICE_ID")
        return invalid

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
                "tts": "cartesia",
            },
            "audio": {
                "input_gain": self.input_gain,
            },
            "conversation": {
                "intent_inference_enabled": self.intent_inference_enabled,
            },
            "cartesia": {
                "connection": {
                    "open_timeout_seconds": self.cartesia_open_timeout_seconds,
                    "connect_retries": self.cartesia_connect_retries,
                },
                "speech_direction": {
                    "enabled": self.cartesia_speech_director_enabled,
                    "ssml_enabled": self.cartesia_ssml_enabled,
                    "emotion_tags_enabled": self.cartesia_emotion_tags_enabled,
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
