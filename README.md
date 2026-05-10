# Browser-First Conversational Voice Agent

Local browser voice agent with Deepgram streaming transcription, Groq streaming responses, Cartesia speech, browser playback, and Phase 6 barge-in interruption.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
python -m app.server
```

Open `http://localhost:8000/` for the browser client.

For fast local UI/backend iteration, you can still run Uvicorn directly:

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Runtime Modes

`VOICE_AGENT_MODE=mock` is the default for local development and does not require provider keys.

`VOICE_AGENT_MODE=live` enables:

- Deepgram streaming STT
- Groq streaming LLM responses
- Cartesia streaming TTS
- Browser playback and barge-in cancellation

Live mode expects these values:

- `DEEPGRAM_API_KEY`
- `GROQ_API_KEY`
- `CARTESIA_API_KEY`
- `CARTESIA_VOICE_ID`

The app still boots if live keys are missing. `/health` and WebSocket `config.warning` events report missing variable names without exposing secret values.

## Configuration

Server:

- `VOICE_AGENT_HOST`: bind host, default `0.0.0.0`
- `VOICE_AGENT_PORT`: bind port, default `8000`
- `PORT`: deployment-platform fallback if `VOICE_AGENT_PORT` is unset
- `VOICE_AGENT_CORS_ORIGINS`: comma-separated allowed origins, or `*`

Provider tuning:

- `DEEPGRAM_MODEL`, default `nova-3`
- `DEEPGRAM_ENDPOINTING_MS`, default `300`
- `GROQ_MODEL`, default `llama-3.1-8b-instant`
- `GROQ_TEMPERATURE`, default `0.7`
- `CARTESIA_MODEL`, default `sonic-3`
- `CARTESIA_SAMPLE_RATE`, default `16000`
- `CARTESIA_VERSION`, default `2026-03-01`
- `VOICE_AGENT_PERSONA`

## Routes

- `GET /`: local browser client
- `GET /health`: deployment health and safe config status
- `GET /events`: JSON event contract
- `WS /ws/browser`: one browser voice session per WebSocket

## Health Check

Use `/health` for Coolify or other deployment probes. A healthy response looks like:

```json
{
  "status": "ok",
  "service": "codex-voiceai",
  "config": {
    "mode": "live",
    "live_ready": true,
    "missing_live_keys": [],
    "server": { "host": "0.0.0.0", "port": 8000 },
    "cors": {
      "allow_all_origins": false,
      "origin_count": 2,
      "allow_credentials": true
    },
    "providers": {
      "stt": "deepgram",
      "llm": "groq",
      "tts": "cartesia"
    }
  }
}
```

Secret values are never returned by `/health`.

## Event Contract

Server events include:

- `session.started`
- `status.changed`
- `config.warning`
- `transcript.partial`
- `transcript.final`
- `agent.text.delta`
- `agent.text.final`
- `audio.input`
- `transcriber.event`
- `audio.chunk`
- `interruption.started`
- `latency.metric`
- `pipeline.stage`
- `error`
- `session.ended`

Client events include:

- `client.hello`
- `audio.start`
- `audio.stop`
- `session.stop`

Every assistant text/audio event includes a `response_id` so stale output can be ignored after interruption.

## Logs

The server writes structured-ish operational log lines for:

- session lifecycle
- status transitions
- provider pipeline stages
- latency metrics
- interruption events
- client-visible errors

Each conversation log line includes a `session_id`.

## Docker And Coolify

Build and run locally:

```powershell
docker build -t codex-voiceai .
docker run --env-file .env -p 8000:8000 codex-voiceai
```

Coolify notes:

- Use the included `Dockerfile`.
- Set provider keys as environment variables in Coolify.
- Route traffic to container port `8000`, or set `VOICE_AGENT_PORT`/`PORT` to match your platform.
- Use `/health` as the health check path.
- Set `VOICE_AGENT_CORS_ORIGINS` to your deployed origin. Use `*` only for quick tests.

## Tests

```powershell
pytest
```

The suite covers config loading, safe health reporting, WebSocket startup/shutdown, provider adapters, turn detection, interruption behavior, and event logging.
