import os
from pathlib import Path

from fastapi.testclient import TestClient

os.environ["VOICE_AGENT_MODE"] = "mock"
os.environ["VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS"] = "1000"

from app.main import app, log_server_event


client = TestClient(app)


def mock_speech_frame(sample_count=12000, amplitude=12000):
    return int(amplitude).to_bytes(2, "little", signed=True) * sample_count


def test_health_reports_service_and_config_status():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "codex-voiceai"
    assert body["config"]["mode"] in {"mock", "live"}
    assert "missing_live_keys" in body["config"]
    assert "server" in body["config"]
    assert "cors" in body["config"]
    assert "proactive" in body["config"]
    assert body["config"]["ambience"] == {
        "enabled": True,
        "scene": "room_line",
        "volume": 0.035,
    }
    assert body["config"]["turn_timing"]["partial_idle_finalize_ms"] == 1000


def test_events_contract_documents_initial_and_planned_events():
    response = client.get("/events")

    assert response.status_code == 200
    body = response.json()
    assert "session.started" in body["server"]
    assert "transcript.partial" in body["server"]
    assert "audio.input" in body["server"]
    assert "pipeline.stage" in body["server"]
    assert "interruption.started" in body["server"]
    assert "proactive.triggered" in body["server"]
    assert "proactive.skipped" in body["server"]
    assert "proactive.cancelled" in body["server"]
    assert "proactive.cooldown" in body["server"]
    assert "proactive.state" in body["server"]
    assert "client.hello" in body["client"]
    assert "audio.start" in body["client"]


def test_readme_documents_proactive_tuning():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "## Proactive Conversation" in readme
    assert "VOICE_AGENT_PROACTIVE_ENABLED" in readme
    assert "proactive.triggered" in readme
    assert "failure_backoff" in readme
    assert "generated through Groq" in readme


def test_event_logging_includes_operational_stage_and_provider(caplog):
    caplog.set_level("INFO", logger="voice_agent")

    log_server_event(
        {
            "type": "pipeline.stage",
            "session_id": "sess_test",
            "payload": {
                "stage": "tts_streaming",
                "provider": "cartesia",
                "response_id": "resp_test",
            },
        }
    )

    assert "event=pipeline.stage" in caplog.text
    assert "stage=tts_streaming" in caplog.text
    assert "provider=cartesia" in caplog.text
    assert "response_id=resp_test" in caplog.text


def test_event_logging_includes_proactive_decisions(caplog):
    caplog.set_level("INFO", logger="voice_agent")

    log_server_event(
        {
            "type": "proactive.skipped",
            "session_id": "sess_test",
            "payload": {
                "trigger_reason": "silence_nudge",
                "skip_reason": "cooldown",
                "source_state": "listening",
            },
        }
    )

    assert "event=proactive.skipped" in caplog.text
    assert "trigger=silence_nudge" in caplog.text
    assert "skip_reason=cooldown" in caplog.text


def test_event_logging_includes_proactive_tuning_fields(caplog):
    caplog.set_level("INFO", logger="voice_agent")

    log_server_event(
        {
            "type": "proactive.cooldown",
            "session_id": "sess_test",
            "payload": {
                "trigger_reason": "contextual_follow_up",
                "cooldown_ms": 30000,
                "next_eligible_at_ms": 123456,
                "reason": "failure_backoff",
                "proactive_failures": 2,
                "failure_backoff_threshold": 2,
                "consecutive_prompts": 3,
            },
        }
    )

    assert "event=proactive.cooldown" in caplog.text
    assert "trigger=contextual_follow_up" in caplog.text
    assert "cooldown_ms=30000" in caplog.text
    assert "next_eligible_at_ms=123456" in caplog.text
    assert "proactive_failures=2" in caplog.text
    assert "failure_backoff_threshold=2" in caplog.text
    assert "consecutive_prompts=3" in caplog.text


def test_event_logging_includes_audio_input_levels(caplog):
    caplog.set_level("INFO", logger="voice_agent")

    log_server_event(
        {
            "type": "audio.input",
            "session_id": "sess_test",
            "payload": {
                "rms": 0.045,
                "peak": 0.12,
                "raw_rms": 0.022,
                "raw_peak": 0.06,
                "input_gain": 2.0,
            },
        }
    )

    assert "event=audio.input" in caplog.text
    assert "rms=0.045" in caplog.text
    assert "raw_rms=0.022" in caplog.text
    assert "input_gain=2.0" in caplog.text


def test_browser_websocket_starts_session_and_accepts_hello():
    with client.websocket_connect("/ws/browser") as websocket:
        started = websocket.receive_json()
        connected = websocket.receive_json()

        assert started["type"] == "session.started"
        assert started["session_id"].startswith("sess_")
        assert started["payload"]["ambience"] == {
            "enabled": True,
            "scene": "room_line",
            "volume": 0.035,
        }
        assert started["payload"]["turn_timing"]["partial_idle_finalize_ms"] == 1000
        assert connected["type"] == "status.changed"
        assert connected["payload"]["state"] == "connected"

        websocket.send_json({"type": "client.hello", "payload": {"client": "test"}})
        ready = websocket.receive_json()

        assert ready["type"] == "status.changed"
        assert ready["payload"]["state"] == "client_ready"
        assert ready["payload"]["client"] == "test"


def test_browser_websocket_mock_audio_loop_emits_transcript_agent_audio_and_latency():
    with client.websocket_connect("/ws/browser") as websocket:
        websocket.receive_json()
        websocket.receive_json()

        websocket.send_json(
            {
                "type": "audio.start",
                "payload": {
                    "encoding": "pcm_s16le",
                    "sample_rate": 16000,
                    "channels": 1,
                    "frame_duration_ms": 20,
                },
            }
        )
        listening = websocket.receive_json()
        assert listening["type"] == "status.changed"
        assert listening["payload"]["state"] == "listening"

        websocket.send_bytes(mock_speech_frame())

        seen_types = []
        response_id = None
        for _ in range(22):
            received = websocket.receive_json()
            seen_types.append(received["type"])
            if received["type"] == "agent.text.delta":
                response_id = received["payload"]["response_id"]
            if received["type"] == "audio.chunk":
                assert received["payload"]["response_id"] == response_id
                assert received["payload"]["encoding"] == "pcm_s16le"
                assert received["payload"]["audio"]
                assert received["payload"]["provider"] == "mock"
                break
            if received["type"] == "audio.input":
                assert "rms" in received["payload"]
                assert "peak" in received["payload"]
                assert received["payload"]["frame_bytes"] > 0

        assert "transcript.partial" in seen_types
        assert "transcript.final" in seen_types
        assert "audio.input" in seen_types
        assert "agent.text.delta" in seen_types
        assert "agent.text.final" in seen_types
        assert "latency.metric" in seen_types
        assert "pipeline.stage" in seen_types
        assert "audio.chunk" in seen_types


def test_frontend_wires_local_barge_in_audio_behavior():
    index = Path("frontend/index.html").read_text(encoding="utf-8")
    app_js = Path("frontend/app.js").read_text(encoding="utf-8")

    assert "/static/audio-behavior.js" in index
    assert "activeTextResponseId" in app_js
    assert "activeAudioResponseId" in app_js
    assert "locallyInterruptedAudioResponseIds" in app_js
    assert "evaluateLocalBargeIn" in app_js
    assert "VoiceAudioBehavior.pcmLevelFromArrayBuffer" in app_js
    assert "VoiceAudioBehavior.shouldTriggerLocalBargeIn" in app_js
