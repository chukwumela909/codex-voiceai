from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    mode: str = Field(default="mock", alias="VOICE_AGENT_MODE")
    host: str = Field(default="0.0.0.0", alias="VOICE_AGENT_HOST")
    port: int = Field(default=8000, validation_alias=AliasChoices("VOICE_AGENT_PORT", "PORT"))
    cors_origins: str = Field(
        default="http://localhost:8000,http://127.0.0.1:8000",
        alias="VOICE_AGENT_CORS_ORIGINS",
    )

    deepgram_api_key: str | None = Field(default=None, alias="DEEPGRAM_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    cartesia_api_key: str | None = Field(default=None, alias="CARTESIA_API_KEY")

    deepgram_model: str = Field(default="nova-3", alias="DEEPGRAM_MODEL")
    deepgram_endpointing_ms: int = Field(default=300, alias="DEEPGRAM_ENDPOINTING_MS")
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")
    groq_temperature: float = Field(default=0.7, alias="GROQ_TEMPERATURE")
    cartesia_model: str = Field(default="sonic-3", alias="CARTESIA_MODEL")
    cartesia_voice_id: str | None = Field(default=None, alias="CARTESIA_VOICE_ID")
    cartesia_sample_rate: int = Field(default=16000, alias="CARTESIA_SAMPLE_RATE")
    cartesia_version: str = Field(default="2026-03-01", alias="CARTESIA_VERSION")
    persona: str = Field(
        default="You are a playful, vivid conversational voice agent. Keep replies concise, warm, and naturally spoken.",
        alias="VOICE_AGENT_PERSONA",
    )

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

    def public_config_status(self) -> dict:
        missing = self.missing_live_keys() if self.normalized_mode == "live" else []
        return {
            "mode": self.normalized_mode,
            "live_ready": not missing,
            "missing_live_keys": missing,
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
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
