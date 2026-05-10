from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


SERVER_EVENT_TYPES = {
    "session.started": "A browser WebSocket session has been accepted.",
    "status.changed": "The session changed state, such as connected or disconnected.",
    "config.warning": "The backend is runnable, but optional or live-mode configuration is incomplete.",
    "transcript.partial": "Mock or provider STT partial transcript update.",
    "transcript.final": "Mock or provider STT final user utterance.",
    "agent.text.delta": "Streaming assistant text delta for a response.",
    "agent.text.final": "Final assistant text for a response.",
    "audio.input": "Inbound microphone audio frame diagnostic.",
    "transcriber.event": "Non-transcript provider event or diagnostic.",
    "audio.chunk": "Assistant audio chunk with response metadata.",
    "interruption.started": "User barge-in cancelled an active assistant response.",
    "latency.metric": "Latency span measurement.",
    "pipeline.stage": "Current provider pipeline stage for observability.",
    "error": "A recoverable or terminal error occurred.",
    "session.ended": "The WebSocket session is closing or has closed.",
}

CLIENT_EVENT_TYPES = {
    "client.hello": "Client handshake or debug hello event.",
    "audio.start": "Client audio stream configuration.",
    "audio.stop": "Client audio stream stopped.",
    "session.stop": "Client requested session shutdown.",
}

PLANNED_EVENT_TYPES = {}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_session_id() -> str:
    return f"sess_{uuid4().hex}"


def event(event_type: str, session_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": event_type,
        "session_id": session_id,
        "timestamp": utc_now_iso(),
        "payload": payload or {},
    }
