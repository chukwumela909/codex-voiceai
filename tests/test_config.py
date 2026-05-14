from app.config import Settings


def test_port_can_come_from_platform_port_env(monkeypatch):
    monkeypatch.delenv("VOICE_AGENT_PORT", raising=False)
    monkeypatch.setenv("PORT", "9001")

    settings = Settings(_env_file=None)

    assert settings.port == 9001
    assert settings.public_config_status()["server"]["port"] == 9001


def test_voice_agent_port_wins_over_platform_port(monkeypatch):
    monkeypatch.setenv("VOICE_AGENT_PORT", "8123")
    monkeypatch.setenv("PORT", "9001")

    settings = Settings(_env_file=None)

    assert settings.port == 8123


def test_wildcard_cors_disables_credentials(monkeypatch):
    monkeypatch.setenv("VOICE_AGENT_CORS_ORIGINS", "*")

    settings = Settings(_env_file=None)
    status = settings.public_config_status()

    assert settings.parsed_cors_origins == ["*"]
    assert settings.allow_all_cors_origins is True
    assert settings.cors_allow_credentials is False
    assert status["cors"] == {
        "allow_all_origins": True,
        "origin_count": 0,
        "allow_credentials": False,
    }


def test_public_config_status_never_exposes_secret_values(monkeypatch):
    voice_secret = "6bf6d6c3-9d45-48fb-94a9-4840f83eb385"
    monkeypatch.setenv("VOICE_AGENT_MODE", "live")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-secret")
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setenv("CARTESIA_API_KEY", "cartesia-secret")
    monkeypatch.setenv("CARTESIA_VOICE_ID", voice_secret)

    status = Settings(_env_file=None).public_config_status()

    rendered = str(status)
    assert status["live_ready"] is True
    assert "dg-secret" not in rendered
    assert "groq-secret" not in rendered
    assert "cartesia-secret" not in rendered
    assert voice_secret not in rendered


def test_cartesia_voice_id_normalizes_leading_uuid_from_accidental_trailing_text(monkeypatch):
    voice_id = "6bf6d6c3-9d45-48fb-94a9-4840f83eb385"
    monkeypatch.setenv("VOICE_AGENT_MODE", "live")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-secret")
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setenv("CARTESIA_API_KEY", "cartesia-secret")
    monkeypatch.setenv("CARTESIA_VOICE_ID", f"{voice_id} pasted words by mistake")

    settings = Settings(_env_file=None)
    status = settings.public_config_status()

    assert settings.cartesia_voice_id == voice_id
    assert status["live_ready"] is True
    assert status["invalid_live_keys"] == []


def test_live_ready_rejects_unrecoverable_cartesia_voice_id(monkeypatch):
    bad_voice_id = "not-a-real-voice-id"
    monkeypatch.setenv("VOICE_AGENT_MODE", "live")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-secret")
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setenv("CARTESIA_API_KEY", "cartesia-secret")
    monkeypatch.setenv("CARTESIA_VOICE_ID", bad_voice_id)

    status = Settings(_env_file=None).public_config_status()

    assert status["live_ready"] is False
    assert status["missing_live_keys"] == []
    assert status["invalid_live_keys"] == ["CARTESIA_VOICE_ID"]
    assert bad_voice_id not in str(status)


def test_websocket_keepalive_defaults_and_overrides(monkeypatch):
    monkeypatch.setenv("VOICE_AGENT_WS_PING_INTERVAL", "45")
    monkeypatch.setenv("VOICE_AGENT_WS_PING_TIMEOUT", "180")
    monkeypatch.setenv("DEEPGRAM_UTTERANCE_END_MS", "1200")

    settings = Settings(_env_file=None)

    assert settings.ws_ping_interval == 45
    assert settings.ws_ping_timeout == 180
    assert settings.deepgram_utterance_end_ms == 1200


def test_balanced_fast_voice_timing_defaults(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_ENDPOINTING_MS", raising=False)
    monkeypatch.delenv("DEEPGRAM_UTTERANCE_END_MS", raising=False)
    monkeypatch.delenv("VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS", raising=False)

    settings = Settings(_env_file=None)
    status = settings.public_config_status()

    assert settings.deepgram_endpointing_ms == 220
    assert settings.deepgram_utterance_end_ms == 700
    assert settings.partial_idle_finalize_ms == 650
    assert status["turn_timing"] == {
        "deepgram_endpointing_ms": 220,
        "deepgram_utterance_end_ms": 700,
        "partial_idle_finalize_ms": 650,
    }


def test_ambience_public_config_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("VOICE_AGENT_AMBIENCE_ENABLED", raising=False)
    monkeypatch.delenv("VOICE_AGENT_AMBIENCE_SCENE", raising=False)
    monkeypatch.delenv("VOICE_AGENT_AMBIENCE_VOLUME", raising=False)

    defaults = Settings(_env_file=None).public_config_status()

    monkeypatch.setenv("VOICE_AGENT_AMBIENCE_ENABLED", "false")
    monkeypatch.setenv("VOICE_AGENT_AMBIENCE_SCENE", "quiet_office")
    monkeypatch.setenv("VOICE_AGENT_AMBIENCE_VOLUME", "0.025")
    overridden = Settings(_env_file=None).public_config_status()

    assert defaults["ambience"] == {
        "enabled": True,
        "scene": "room_line",
        "volume": 0.035,
    }
    assert overridden["ambience"] == {
        "enabled": False,
        "scene": "quiet_office",
        "volume": 0.025,
    }


def test_cartesia_speed_defaults_to_faster_spoken_agent_and_can_be_overridden(monkeypatch):
    monkeypatch.delenv("CARTESIA_SPEED", raising=False)

    default_settings = Settings(_env_file=None)

    monkeypatch.setenv("CARTESIA_SPEED", "1.35")
    overridden_settings = Settings(_env_file=None)

    assert default_settings.cartesia_speed == 1.2
    assert overridden_settings.cartesia_speed == 1.35


def test_proactive_auto_enabled_in_mock_and_live_opt_in(monkeypatch):
    monkeypatch.setenv("VOICE_AGENT_PROACTIVE_ENABLED", "auto")
    monkeypatch.setenv("VOICE_AGENT_MODE", "mock")

    mock_settings = Settings(_env_file=None)

    monkeypatch.setenv("VOICE_AGENT_MODE", "live")
    live_settings = Settings(_env_file=None)

    assert mock_settings.proactive_effective_enabled is True
    assert live_settings.proactive_effective_enabled is False


def test_live_proactive_opt_in_uses_patient_defaults_when_timing_is_unset(monkeypatch):
    monkeypatch.setenv("VOICE_AGENT_MODE", "live")
    monkeypatch.setenv("VOICE_AGENT_PROACTIVE_ENABLED", "true")
    monkeypatch.delenv("VOICE_AGENT_PROACTIVE_SILENCE_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("VOICE_AGENT_PROACTIVE_REPEAT_COOLDOWN_MS", raising=False)
    monkeypatch.delenv("VOICE_AGENT_PROACTIVE_MAX_CONSECUTIVE_PROMPTS", raising=False)

    status = Settings(_env_file=None).public_config_status()

    assert status["proactive"]["enabled"] is True
    assert status["proactive"]["silence_timeout_ms"] == 30000
    assert status["proactive"]["repeat_cooldown_ms"] == 60000
    assert status["proactive"]["max_consecutive_prompts"] == 1


def test_proactive_public_config_reports_safe_tuning(monkeypatch):
    monkeypatch.setenv("VOICE_AGENT_MODE", "live")
    monkeypatch.setenv("VOICE_AGENT_PROACTIVE_ENABLED", "true")
    monkeypatch.setenv("VOICE_AGENT_PROACTIVE_GREETING_DELAY_MS", "250")
    monkeypatch.setenv("VOICE_AGENT_PROACTIVE_SILENCE_TIMEOUT_MS", "1500")
    monkeypatch.setenv("VOICE_AGENT_PROACTIVE_REPEAT_COOLDOWN_MS", "2500")
    monkeypatch.setenv("VOICE_AGENT_PROACTIVE_MAX_CONSECUTIVE_PROMPTS", "2")
    monkeypatch.setenv("VOICE_AGENT_PROACTIVE_FAILURE_BACKOFF_THRESHOLD", "4")
    monkeypatch.setenv("VOICE_AGENT_PROACTIVE_FAILURE_BACKOFF_MS", "45000")
    monkeypatch.setenv("VOICE_AGENT_PROACTIVE_CONTEXTUAL_FOLLOWUPS_ENABLED", "false")

    status = Settings(_env_file=None).public_config_status()

    assert status["proactive"] == {
        "configured": "true",
        "enabled": True,
        "startup_greeting_delay_ms": 250,
        "silence_timeout_ms": 1500,
        "repeat_cooldown_ms": 2500,
        "max_consecutive_prompts": 2,
        "failure_backoff_threshold": 4,
        "failure_backoff_ms": 45000,
        "contextual_followups_enabled": False,
    }
