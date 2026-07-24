import json
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("VOICE_AGENT_MODE", "mock")

import app.main as main_mod
import app.twilio_outbound as twilio_outbound
from app.config import Settings
from app.main import app
from app.outbound_calls import outbound_call_store


client = TestClient(app)
AUTH = {"Authorization": "Bearer test-outbound-token"}


@pytest.fixture(autouse=True)
def isolated_calls():
    outbound_call_store.reset()
    yield
    outbound_call_store.reset()


@pytest.fixture()
def outbound_settings(monkeypatch):
    configured = Settings(
        _env_file=None,
        TWILIO_ACCOUNT_SID="ACtest",
        TWILIO_AUTH_TOKEN="auth-token",
        TWILIO_FROM_NUMBER="+15551230000",
        VOICE_AGENT_OUTBOUND_ACCESS_TOKEN="test-outbound-token",
        PUBLIC_HOST="voice.example.com",
    )
    monkeypatch.setattr(main_mod, "settings", configured)
    return configured


def test_outbound_call_requires_access_token(outbound_settings):
    response = client.post("/api/outbound-calls", json={"to": "+15551234567"})

    assert response.status_code == 401


def test_outbound_call_validates_e164_number(outbound_settings):
    response = client.post(
        "/api/outbound-calls",
        headers=AUTH,
        json={"to": "0801 234 5678"},
    )

    assert response.status_code == 422
    assert "E.164" in response.text


def test_create_and_read_outbound_call(outbound_settings, monkeypatch):
    captured = {}

    def fake_place(call, settings):
        captured["call"] = call
        captured["settings"] = settings
        return SimpleNamespace(sid="CA123")

    monkeypatch.setattr(twilio_outbound, "place_outbound_call", fake_place)

    response = client.post(
        "/api/outbound-calls",
        headers=AUTH,
        json={
            "to": "+1 (555) 123-4567",
            "character_id": "jimmy",
            "voice_id": "voice-1",
            "model_id": "groq-gpt-oss-20b",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["to"] == "••••4567"
    assert body["status"] == "queued"
    assert body["character_id"] == "jimmy"
    assert captured["call"].to_number == "+15551234567"
    assert captured["settings"] is outbound_settings

    read = client.get(f"/api/outbound-calls/{body['id']}", headers=AUTH)
    assert read.status_code == 200
    assert read.json()["status"] == "queued"


def test_outbound_twiml_carries_only_opaque_call_credentials(outbound_settings):
    call = outbound_call_store.create(
        to_number="+15551234567",
        character_id="jimmy",
        voice_id="voice-1",
        model_id="groq-gpt-oss-20b",
    )

    xml = twilio_outbound.build_outbound_twiml(call, outbound_settings)

    assert 'url="wss://voice.example.com/api/twilio-ws"' in xml
    assert f'value="{call.id}"' in xml
    assert f'value="{call.stream_token}"' in xml
    assert "+15551234567" not in xml
    assert "voice-1" not in xml


def test_place_outbound_call_uses_twilio_calls_api(outbound_settings, monkeypatch):
    call = outbound_call_store.create(
        to_number="+15551234567",
        character_id=None,
        voice_id=None,
        model_id=None,
    )
    captured = {}

    class FakeCalls:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(sid="CA123")

    import twilio.rest

    monkeypatch.setattr(
        twilio.rest,
        "Client",
        lambda account_sid, auth_token: SimpleNamespace(calls=FakeCalls()),
    )

    result = twilio_outbound.place_outbound_call(call, outbound_settings)

    assert result.sid == "CA123"
    assert captured["to"] == "+15551234567"
    assert captured["from_"] == "+15551230000"
    assert captured["status_callback"] == (
        f"https://voice.example.com/twilio/call-status/{call.id}"
    )
    assert captured["status_callback_event"] == [
        "initiated",
        "ringing",
        "answered",
        "completed",
    ]


def test_signed_status_callback_updates_call(outbound_settings):
    call = outbound_call_store.create(
        to_number="+15551234567",
        character_id=None,
        voice_id=None,
        model_id=None,
    )
    outbound_call_store.attach_twilio_call(call.id, "CA123")
    callback_url = f"https://voice.example.com/twilio/call-status/{call.id}"
    params = {"CallSid": "CA123", "CallStatus": "ringing"}
    from twilio.request_validator import RequestValidator

    signature = RequestValidator("auth-token").compute_signature(callback_url, params)

    response = client.post(
        f"/twilio/call-status/{call.id}",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Twilio-Signature": signature,
        },
        data=params,
    )

    assert response.status_code == 204
    assert outbound_call_store.get(call.id).status == "ringing"


def test_outbound_stream_uses_snapshotted_pipeline_selections(outbound_settings, monkeypatch):
    call = outbound_call_store.create(
        to_number="+15551234567",
        character_id="jimmy",
        voice_id="voice-1",
        model_id="groq-gpt-oss-20b",
    )
    outbound_call_store.attach_twilio_call(call.id, "CA123")
    captured = {}

    async def fake_run(websocket, stream_sid, call_sid, settings, **kwargs):
        captured.update(
            stream_sid=stream_sid,
            call_sid=call_sid,
            settings=settings,
            **kwargs,
        )

    import app.pipeline as pipeline

    monkeypatch.setattr(pipeline, "run_twilio_session", fake_run)

    with client.websocket_connect("/api/twilio-ws") as websocket:
        websocket.send_text(json.dumps({"event": "connected", "protocol": "Call"}))
        websocket.send_text(
            json.dumps(
                {
                    "event": "start",
                    "start": {
                        "streamSid": "MZ123",
                        "callSid": "CA123",
                        "customParameters": {
                            "call_id": call.id,
                            "stream_token": call.stream_token,
                        },
                    },
                }
            )
        )

    assert captured["stream_sid"] == "MZ123"
    assert captured["call_sid"] == "CA123"
    assert captured["character_id"] == "jimmy"
    assert captured["voice_id"] == "voice-1"
    assert captured["model_id"] == "groq-gpt-oss-20b"
