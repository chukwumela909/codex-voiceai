import asyncio
import json
import logging

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.characters import (
    Character,
    DEFAULT_CHARACTER_ID,
    SLUG_PATTERN,
    load_characters,
    resolve_default_character_id,
    save_character,
    set_persisted_default_id,
)
from app.config import get_settings
from app.events import CLIENT_EVENT_TYPES, PLANNED_EVENT_TYPES, SERVER_EVENT_TYPES, event, new_session_id
from app.exceptions import ClientConnectionClosed
from app.mock_conversation import MockConversationSession
from app.preview import preview_character
from app.voice_settings import get_active_voice_id, resolve_active_voice_id, set_active_voice_id
from app.llm_models import (
    MODELS,
    get_active_model,
    resolve_model,
    set_active_model,
)
from fastapi import HTTPException
from pydantic import ValidationError


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


@app.on_event("startup")
async def warm_pipecat_pipeline() -> None:
    """Pre-warm the Pipecat stack so the first caller doesn't pay the cold start.

    The pipecat/transformers/onnxruntime imports plus the smart-turn ONNX model
    load take multiple seconds; without this the first `/api/ws` or Twilio
    session sits in dead air while they happen (and stalls any concurrent
    session's audio). Best-effort: a legacy-only deploy without the pipecat
    extras still boots.
    """

    def _warm() -> None:
        from app.pipeline import create_turn_analyzer

        create_turn_analyzer(settings)

    try:
        await asyncio.to_thread(_warm)
        log_info("pipecat pipeline warmed (imports + smart-turn model)")
    except Exception as exc:  # noqa: BLE001 — warmup must never block startup
        logger.warning("pipecat warmup skipped: %s", exc)

    # Pre-load the local embedding model in the background (best-effort) so the
    # first caller doesn't pay the model download/load on the memory path. Runs
    # detached: it must not delay the server accepting connections.
    async def _warm_embedder() -> None:
        from app.memory.embedder import warm_embedder

        await warm_embedder(settings)

    asyncio.create_task(_warm_embedder())


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
async def index() -> RedirectResponse:
    # Pipecat page is the default landing page; the classic UI lives at /classic.
    return RedirectResponse(url="/pipecat")


@app.get("/classic")
async def classic_index() -> FileResponse:
    return FileResponse("frontend/index.html")


@app.get("/pipecat")
async def pipecat_spike_page() -> FileResponse:
    return FileResponse("frontend/pipecat.html")


@app.get("/studio")
async def personality_studio_page() -> FileResponse:
    return FileResponse("frontend/studio.html")


@app.websocket("/api/ws")
async def pipecat_ws(websocket: WebSocket) -> None:
    """WebSocket transport for the Pipecat spike pipeline.

    Lazy-imports pipecat so the legacy /ws/browser path keeps working even if
    pipecat-ai is not yet installed.
    """
    await websocket.accept()
    session_id = new_session_id()
    requested_character = websocket.query_params.get("character")
    requested_voice = websocket.query_params.get("voice")
    requested_model = websocket.query_params.get("model")
    log_info("pipecat ws client connected", session_id=session_id)

    from app.pipeline import run_pipecat_session

    try:
        await run_pipecat_session(
            websocket,
            settings,
            character_id=requested_character,
            voice_id=requested_voice,
            model_id=requested_model,
        )
    except WebSocketDisconnect:
        log_info("pipecat ws client disconnected", session_id=session_id)
    except Exception:
        logger.exception("pipecat session failed", extra={"session_id": session_id})
        raise


@app.api_route("/twiml", methods=["GET", "POST"])
async def twiml(request: Request) -> Response:
    """TwiML that bridges an inbound Twilio call to our Media Streams WebSocket.

    `<Connect><Stream>` is bidirectional so the caller hears the agent's TTS;
    `<Start><Stream>` would be listen-only. `PUBLIC_HOST` is preferred over the
    request hostname, which behind a proxy can resolve to the internal bind
    address and yield a wss:// URL Twilio cannot reach.
    """
    host = settings.public_host or request.url.hostname
    ws_url = f"wss://{host}/api/twilio-ws"
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response><Connect><Stream url="{ws_url}"/></Connect></Response>'
    )
    return Response(content=xml, media_type="text/xml")


@app.websocket("/api/twilio-ws")
async def twilio_ws(websocket: WebSocket) -> None:
    """Twilio Media Streams transport for the Pipecat pipeline.

    Twilio sends a `connected` text frame, then a `start` frame carrying the
    streamSid/callSid. Both must be consumed here before the serializer is
    built, so we read them before handing off to the pipeline.
    """
    await websocket.accept()
    session_id = new_session_id()

    start_iter = websocket.iter_text()
    try:
        await start_iter.__anext__()  # {"event": "connected", ...}
        call_data = json.loads(await start_iter.__anext__())  # {"event": "start", ...}
    except (StopAsyncIteration, WebSocketDisconnect):
        log_info("twilio ws closed before start", session_id=session_id)
        return
    stream_sid = call_data["start"]["streamSid"]
    call_sid = call_data["start"]["callSid"]
    log_info("twilio media stream started", session_id=session_id)

    from app.pipeline import run_twilio_session

    try:
        await run_twilio_session(websocket, stream_sid, call_sid, settings)
    except WebSocketDisconnect:
        log_info("twilio ws disconnected", session_id=session_id)
    except Exception:
        logger.exception("twilio session failed", extra={"session_id": session_id})
        raise


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "codex-voiceai",
        "config": settings.public_config_status(),
    }


def _default_character_id() -> str:
    return resolve_default_character_id(
        env_default=getattr(settings, "default_character_id", DEFAULT_CHARACTER_ID)
    )


@app.get("/characters")
async def list_characters() -> dict:
    characters = load_characters()
    default_id = _default_character_id()
    if default_id not in characters and characters:
        default_id = next(iter(characters))
    return {
        "default": default_id,
        "characters": [c.model_dump(exclude_none=True) for c in characters.values()],
    }


def _validate_character_payload(character_id: str, payload: dict) -> Character:
    if not SLUG_PATTERN.match(character_id):
        raise HTTPException(status_code=400, detail="Invalid character id.")
    data = {**payload, "id": character_id}
    try:
        return Character.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


@app.put("/characters/default")
async def set_default_character_endpoint(payload: dict) -> dict:
    """Persist which character is active. Takes effect on the next call/session.

    Declared before `/characters/{character_id}` so the literal `default` path is
    not captured as a character id by the dynamic route.
    """
    character_id = str(payload.get("id", "") or "").strip().lower()
    if character_id not in load_characters():
        raise HTTPException(status_code=404, detail="Unknown character id.")
    set_persisted_default_id(character_id)
    return {"default": character_id}


@app.put("/characters/{character_id}")
async def upsert_character(character_id: str, payload: dict) -> dict:
    character = _validate_character_payload(character_id, payload)
    save_character(character)
    return character.model_dump(exclude_none=True)


@app.post("/characters")
async def create_character(payload: dict) -> dict:
    character_id = str(payload.get("id", "")).strip().lower()
    character = _validate_character_payload(character_id, payload)
    save_character(character)
    return character.model_dump(exclude_none=True)


def _coerce_preview_character(data: dict) -> Character:
    """Build a Character from an in-editor draft that may be unsaved/incomplete.

    Lenient on purpose: a brand-new draft can lack a valid id and have empty
    name/role, so we fill safe placeholders rather than reject the preview.
    """
    raw_id = str(data.get("id", "") or "").strip().lower()
    char_id = raw_id if SLUG_PATTERN.match(raw_id) else "preview"
    payload = {
        **data,
        "id": char_id,
        "name": (str(data.get("name") or "").strip() or "Preview"),
        "role": (str(data.get("role") or "").strip() or "voice assistant"),
    }
    try:
        return Character.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


@app.post("/characters/preview")
async def preview_character_endpoint(payload: dict) -> dict:
    message = str(payload.get("message", "") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="preview requires a non-empty message.")
    char_data = payload.get("character")
    if not isinstance(char_data, dict):
        raise HTTPException(status_code=400, detail="preview requires a character object.")
    character = _coerce_preview_character(char_data)
    return await preview_character(character, message, settings)


@app.get("/voices")
async def list_voices_endpoint() -> dict:
    """List the account's ElevenLabs voices for the UI picker + report the active one.

    Never raises for provider problems: a missing key or a fetch failure returns
    an empty list plus a warning, so the paste-a-voice-id path still works.
    """
    from app.elevenlabs_tts import list_voices

    voices: list[dict] = []
    warning: str | None = None
    if settings.elevenlabs_api_key:
        try:
            voices = await list_voices(settings.elevenlabs_api_key)
        except Exception as exc:  # noqa: BLE001 — surface as a warning, not a 500
            warning = f"Could not load ElevenLabs voices: {exc}"
    else:
        warning = "ELEVENLABS_API_KEY is not configured; paste a voice id manually."
    return {
        "voices": voices,
        "active": resolve_active_voice_id(settings),
        "persisted": get_active_voice_id(),
        "env_default": settings.elevenlabs_voice_id,
        "warning": warning,
    }


@app.get("/voice")
async def get_voice_endpoint() -> dict:
    return {
        "active": resolve_active_voice_id(settings),
        "persisted": get_active_voice_id(),
        "env_default": settings.elevenlabs_voice_id,
    }


@app.put("/voice")
async def set_voice_endpoint(payload: dict) -> dict:
    """Persist the active ElevenLabs voice id chosen in the UI (empty string clears it)."""
    raw = payload.get("voice_id", None)
    if raw is not None and not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="voice_id must be a string.")
    stored = set_active_voice_id(raw)
    return {
        "active": resolve_active_voice_id(settings),
        "persisted": stored,
        "env_default": settings.elevenlabs_voice_id,
    }


def _model_status_payload() -> dict:
    return {
        "models": [
            {"key": k, "label": v["label"], "provider": v["provider"], "model": v["model"]}
            for k, v in MODELS.items()
        ],
        "active": resolve_model(settings)["key"],
        "persisted": get_active_model(),
        "default": settings.default_model,
        "openrouter_configured": bool(settings.openrouter_api_key),
    }


@app.get("/models")
async def list_models_endpoint() -> dict:
    """List the curated LLM presets (Groq + OpenRouter) plus the active model.

    Feeds the UI model picker; the `key` is what the browser sends as ?model= on
    connect (a preset key or a raw OpenRouter slug). `openrouter_configured` is
    false when OPENROUTER_API_KEY is unset, so the UI can warn that OpenRouter
    models won't work yet.
    """
    return _model_status_payload()


@app.get("/openrouter/models")
async def list_openrouter_catalog_endpoint() -> dict:
    """OpenRouter's full model catalog for the UI autocomplete/paste field.

    Lets you pick or paste ANY OpenRouter model id, not just the curated presets.
    Never raises: a fetch failure returns an empty list plus a warning.
    """
    from app.llm_models import list_openrouter_models

    warning: str | None = None
    if not settings.openrouter_api_key:
        warning = "OPENROUTER_API_KEY is not set; models will error until you add it."
    try:
        catalog = await list_openrouter_models(settings.openrouter_api_key)
    except Exception as exc:  # noqa: BLE001 — surface as a warning, not a 500
        return {"models": [], "warning": f"Could not load OpenRouter catalog: {exc}"}
    return {"models": catalog, "warning": warning}


@app.put("/model")
async def set_model_endpoint(payload: dict) -> dict:
    """Persist the active LLM model chosen in the UI (empty string clears it)."""
    raw = payload.get("model", None)
    if raw is not None and not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="model must be a string.")
    try:
        set_active_model(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _model_status_payload()


@app.get("/memory")
async def list_memory() -> dict:
    from app.memory import get_store

    store = get_store(settings)
    return {
        "enabled": settings.memory_effective_enabled,
        "count": store.count(),
        "memories": [record.public() for record in store.all()],
    }


@app.delete("/memory")
async def clear_memory() -> dict:
    from app.memory import get_store

    removed = get_store(settings).clear()
    return {"cleared": removed}


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
        payload = data.get("payload", {}) or {}
        character_id = payload.get("character_id")
        if character_id:
            character = conversation.set_character(str(character_id))
            await send_server_event(
                websocket,
                event(
                    "character.changed",
                    session_id,
                    {"character": character.model_dump(exclude_none=True)},
                ),
            )
        await send_server_event(
            websocket,
            event(
                "status.changed",
                session_id,
                {
                    "state": "client_ready",
                    "client": payload.get("client", "browser"),
                    "character_id": conversation.character.id,
                },
            )
        )
        return True

    if message_type == "character.select":
        payload = data.get("payload", {}) or {}
        character_id = str(payload.get("character_id", "")).strip()
        if not character_id:
            await send_server_event(
                websocket,
                event("error", session_id, {"message": "character.select requires character_id."}),
            )
            return True
        character = conversation.set_character(character_id)
        await send_server_event(
            websocket,
            event(
                "character.changed",
                session_id,
                {"character": character.model_dump(exclude_none=True)},
            ),
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
