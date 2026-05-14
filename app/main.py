import asyncio
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.events import CLIENT_EVENT_TYPES, PLANNED_EVENT_TYPES, SERVER_EVENT_TYPES, event, new_session_id
from app.exceptions import ClientConnectionClosed
from app.mock_conversation import MockConversationSession


settings = get_settings()


class SessionIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "session_id"):
            record.session_id = "-"
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s session_id=%(session_id)s %(message)s",
)
for handler in logging.getLogger().handlers:
    handler.addFilter(SessionIdFilter())
logger = logging.getLogger("voice_agent")

app = FastAPI(title="Browser-First Conversational Voice Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_cors_origins or ["*"],
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


def log_info(message: str, session_id: str = "-", **extra: object) -> None:
    logger.info(message, extra={"session_id": session_id, **extra})


def log_server_event(payload: dict) -> None:
    event_type = payload.get("type", "-")
    event_payload = payload.get("payload", {})
    if event_type not in {
        "status.changed",
        "pipeline.stage",
        "latency.metric",
        "error",
        "interruption.started",
        "proactive.triggered",
        "proactive.skipped",
        "proactive.cancelled",
        "proactive.cooldown",
        "proactive.state",
        "session.ended",
        "audio.input",
    }:
        return

    logger.info(
        (
            "event=%s state=%s stage=%s provider=%s response_id=%s latency_ms=%s reason=%s "
            "trigger=%s skip_reason=%s cooldown_ms=%s next_eligible_at_ms=%s proactive_failures=%s "
            "failure_backoff_threshold=%s consecutive_prompts=%s rms=%s peak=%s raw_rms=%s raw_peak=%s "
            "input_gain=%s message=%s"
        ),
        event_type,
        event_payload.get("state", "-"),
        event_payload.get("stage", "-"),
        event_payload.get("provider", "-"),
        event_payload.get("response_id") or event_payload.get("interrupted_response_id", "-"),
        event_payload.get("value_ms", "-"),
        event_payload.get("reason", "-"),
        event_payload.get("trigger_reason", "-"),
        event_payload.get("skip_reason", "-"),
        event_payload.get("cooldown_ms", "-"),
        event_payload.get("next_eligible_at_ms", "-"),
        event_payload.get("proactive_failures", "-"),
        event_payload.get("failure_backoff_threshold", "-"),
        event_payload.get("consecutive_prompts", "-"),
        event_payload.get("rms", "-"),
        event_payload.get("peak", "-"),
        event_payload.get("raw_rms", "-"),
        event_payload.get("raw_peak", "-"),
        event_payload.get("input_gain", "-"),
        event_payload.get("message", "-"),
        extra={"session_id": payload.get("session_id", "-")},
    )


async def send_server_event(websocket: WebSocket, payload: dict) -> None:
    log_server_event(payload)
    try:
        await websocket.send_json(payload)
    except WebSocketDisconnect as exc:
        raise ClientConnectionClosed from exc
    except RuntimeError as exc:
        raise ClientConnectionClosed from exc


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("frontend/index.html")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "codex-voiceai",
        "config": settings.public_config_status(),
    }


@app.get("/events")
async def events_contract() -> dict:
    return {
        "server": SERVER_EVENT_TYPES,
        "client": CLIENT_EVENT_TYPES,
        "planned": PLANNED_EVENT_TYPES,
    }


@app.websocket("/ws/browser")
async def browser_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = new_session_id()
    conversation: MockConversationSession | None = None

    async def send_event_to_client(payload: dict) -> None:
        try:
            await send_server_event(websocket, payload)
        except ClientConnectionClosed:
            if conversation and not conversation.closed:
                asyncio.create_task(conversation.close())
            raise

    conversation = MockConversationSession(session_id, send_event_to_client, settings)
    config_status = settings.public_config_status()
    log_info("browser websocket accepted", session_id=session_id)

    await send_server_event(
        websocket,
        event(
            "session.started",
            session_id,
            {
                "mode": settings.normalized_mode,
                "event_contract_url": "/events",
                "ambience": config_status["ambience"],
                "turn_timing": config_status["turn_timing"],
                "audio_contract": {
                    "inbound_preferred_encoding": "pcm_s16le",
                    "channels": 1,
                    "sample_rate": "metadata_required",
                },
            },
        )
    )
    await send_server_event(websocket, event("status.changed", session_id, {"state": "connected"}))

    missing = config_status["missing_live_keys"]
    invalid = config_status.get("invalid_live_keys", [])
    if missing or invalid:
        await send_server_event(
            websocket,
            event(
                "config.warning",
                session_id,
                {
                    "message": "Live mode is selected, but required provider configuration is missing or invalid.",
                    "missing": missing,
                    "invalid": invalid,
                },
            )
        )

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "text" in message and message["text"] is not None:
                should_continue = await handle_text_message(websocket, conversation, message["text"])
                if not should_continue:
                    break
            elif "bytes" in message and message["bytes"] is not None:
                await conversation.receive_audio(message["bytes"])
    except (WebSocketDisconnect, ClientConnectionClosed):
        log_info("browser websocket disconnected", session_id=session_id)
    finally:
        if conversation:
            await conversation.close()
        log_info("browser websocket session ended", session_id=session_id)


async def handle_text_message(websocket: WebSocket, conversation: MockConversationSession, text: str) -> bool:
    session_id = conversation.session_id
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        await send_server_event(websocket, event("error", session_id, {"message": "Invalid JSON message."}))
        return True

    message_type = data.get("type")
    if message_type == "client.hello":
        await send_server_event(
            websocket,
            event(
                "status.changed",
                session_id,
                {
                    "state": "client_ready",
                    "client": data.get("payload", {}).get("client", "browser"),
                },
            )
        )
        return True

    if message_type == "audio.start":
        await conversation.configure_audio(data.get("payload", {}))
        return True

    if message_type == "audio.stop":
        await conversation.stop_audio()
        await send_server_event(websocket, event("status.changed", session_id, {"state": "connected"}))
        return True

    if message_type == "session.stop":
        await send_server_event(websocket, event("session.ended", session_id, {"reason": "client_requested"}))
        await websocket.close(code=1000)
        return False

    await send_server_event(
        websocket,
        event(
            "error",
            session_id,
            {
                "message": "Unsupported client event.",
                "received_type": message_type,
                "supported_types": sorted(CLIENT_EVENT_TYPES),
            },
        )
    )
    return True
