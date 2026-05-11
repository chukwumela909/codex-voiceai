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
    monkeypatch.setenv("VOICE_AGENT_MODE", "live")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-secret")
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setenv("CARTESIA_API_KEY", "cartesia-secret")
    monkeypatch.setenv("CARTESIA_VOICE_ID", "voice-secret")

    status = Settings(_env_file=None).public_config_status()

    rendered = str(status)
    assert status["live_ready"] is True
    assert "dg-secret" not in rendered
    assert "groq-secret" not in rendered
    assert "cartesia-secret" not in rendered
    assert "voice-secret" not in rendered


def test_websocket_keepalive_defaults_and_overrides(monkeypatch):
    monkeypatch.setenv("VOICE_AGENT_WS_PING_INTERVAL", "45")
    monkeypatch.setenv("VOICE_AGENT_WS_PING_TIMEOUT", "180")
    monkeypatch.setenv("DEEPGRAM_UTTERANCE_END_MS", "1200")

    settings = Settings(_env_file=None)

    assert settings.ws_ping_interval == 45
    assert settings.ws_ping_timeout == 180
    assert settings.deepgram_utterance_end_ms == 1200
