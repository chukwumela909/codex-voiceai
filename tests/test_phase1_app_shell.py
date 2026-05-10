import os

from fastapi.testclient import TestClient

os.environ["VOICE_AGENT_MODE"] = "mock"

from app.main import app, log_server_event


client = TestClient(app)


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


def test_events_contract_documents_initial_and_planned_events():
    response = client.get("/events")

    assert response.status_code == 200
    body = response.json()
    assert "session.started" in body["server"]
    assert "transcript.partial" in body["server"]
    assert "audio.input" in body["server"]
    assert "pipeline.stage" in body["server"]
    assert "interruption.started" in body["server"]
    assert "client.hello" in body["client"]
    assert "audio.start" in body["client"]


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


def test_browser_websocket_starts_session_and_accepts_hello():
    with client.websocket_connect("/ws/browser") as websocket:
        started = websocket.receive_json()
        connected = websocket.receive_json()

        assert started["type"] == "session.started"
        assert started["session_id"].startswith("sess_")
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

        websocket.send_bytes(b"\x00\x00" * 12000)

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
