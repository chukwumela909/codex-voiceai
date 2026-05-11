# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env

# Dev server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Tests
pytest
pytest tests/test_specific.py  # single file
pytest -k "test_name"          # single test

# Docker
docker build -t codex-voiceai .
docker run -p 8000:8000 --env-file .env codex-voiceai
```

## Runtime Modes

Set `VOICE_AGENT_MODE` in `.env`:
- **`mock`** (default) — no API keys needed; uses simulated transcript, sine-wave audio, canned agent responses
- **`live`** — requires `DEEPGRAM_API_KEY`, `GROQ_API_KEY`, `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID`

The health endpoint `GET /health` reports which providers are configured.

## Architecture

### Request Flow

```
Browser (frontend/app.js)
  └─ WebSocket /ws/browser
       └─ MockConversationSession (app/mock_conversation.py)
            ├─ DeepgramStreamingTranscriber (app/deepgram.py)   ← STT
            ├─ GroqStreamingAgent (app/groq_agent.py)           ← LLM
            └─ CartesiaStreamingTTS (app/cartesia_tts.py)       ← TTS
```

One WebSocket connection = one conversation session. The server streams JSON events back to the browser at every pipeline stage.

### Key Server Modules

- **`app/main.py`** — FastAPI app; mounts `frontend/` as static files; owns the `/ws/browser` WebSocket endpoint and routes `/health`, `/events`
- **`app/config.py`** — Pydantic Settings; all env vars prefixed `VOICE_AGENT_*`; exposes `public_status()` for health reporting
- **`app/events.py`** — Canonical event schema; all server↔client messages are typed JSON objects with `type`, `session_id`, `timestamp`, `payload`; use `event_factory()` to construct them
- **`app/mock_conversation.py`** — Core session logic (~530 lines); handles audio ingestion (PCM S16LE framing, RMS/peak analysis), turn detection, provider orchestration, and graceful fallback to mock when providers fail
- **`app/groq_agent.py`** — Groq LLM adapter; `pop_speakable_chunks()` splits streaming delta text into TTS-ready sentence fragments (≥24 chars on `.!?\n`, or every 90 chars at a word boundary)
- **`app/deepgram.py`** — Deepgram WebSocket STT; emits on `is_final` and `speech_final` signals
- **`app/cartesia_tts.py`** — Cartesia WebSocket TTS; returns base64-encoded PCM audio chunks with context IDs

### Frontend

- **`frontend/app.js`** — WebSocket lifecycle, microphone capture (AudioContext at 16 kHz), PCM conversion, audio playback scheduling, HUD metrics
- **`frontend/pcm-worklet.js`** — AudioWorklet that converts float32 samples → 16-bit signed PCM in the audio thread
- **`frontend/index.html`** / **`frontend/styles.css`** — Static shell; no build step required

### Event Contract

All WebSocket messages are JSON. Server events include: `session.started`, `session.ended`, `transcript.partial`, `transcript.final`, `agent.response_start`, `agent.response_chunk`, `agent.response_end`, `audio.chunk`, `pipeline.stage`, `latency.report`, `error`. Client events: `client.hello`, `audio.start`, `audio.stop`, `session.stop`. See `GET /events` for the live contract or `app/events.py` for definitions.

## Deployment

Uses the included `Dockerfile` (python:3.12-slim). Set `VOICE_AGENT_PORT` and point the health check at `/health`. Compatible with Coolify.
