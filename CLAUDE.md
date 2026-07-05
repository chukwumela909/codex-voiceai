# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
- **`live`** — requires `DEEPGRAM_API_KEY`, `GROQ_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`

The health endpoint `GET /health` reports which providers are configured.

## Architecture

### Request Flow

```
Browser (frontend/pipecat.html, Pipecat JS SDK)
  └─ WebSocket /api/ws ────────────┐
Twilio Media Streams               │
  └─ WebSocket /api/twilio-ws ─────┤
                                   └─ app/pipeline.py (Pipecat)
                                        ├─ Deepgram STT
                                        ├─ Groq/OpenRouter LLM (app/llm_models.py)
                                        ├─ ElevenLabs TTS
                                        └─ Silero VAD + local smart-turn model (turn-taking)
```

One WebSocket connection = one conversation session. `/` redirects to `/pipecat`; the browser speaks the Pipecat client protocol over `/api/ws`, Twilio Media Streams connect to `/api/twilio-ws` (8 kHz µ-law, transcoded by `TwilioFrameSerializer`), and both run the same `app/pipeline.py`. The legacy path (`/classic` UI → `/ws/browser` → `app/mock_conversation.py`) is still mounted.

### Key Server Modules

- **`app/main.py`** — FastAPI app; mounts `frontend/` as static files; owns the `/api/ws`, `/api/twilio-ws`, and legacy `/ws/browser` WebSocket endpoints plus REST routes (`/health`, `/characters`, `/voices`, `/models`, `/memory`, `/events`)
- **`app/config.py`** — Pydantic Settings; all env vars prefixed `VOICE_AGENT_*`; exposes `public_status()` for health reporting
- **`app/events.py`** — Canonical event schema; all server↔client messages are typed JSON objects with `type`, `session_id`, `timestamp`, `payload`; use `event_factory()` to construct them
- **`app/pipeline.py`** — Pipecat pipeline shared by browser and Twilio sessions; context windowing + rolling call summary, canned `TTSSpeakFrame` greeting, idle-nudge cap
- **`app/pipeline_memory.py`** — Memory for the Pipecat path: background memory injection, rolling summary of dropped turns, full-call distillation on session close
- **`app/llm_models.py`** — Switchable LLM models (Groq / OpenRouter behind one OpenAI-compatible interface); selected from the UI, persisted server-side
- **`app/tts_filters.py`** — LLM→TTS speakability filter; strips emojis, `*stage directions*`, and control tags before ElevenLabs
- **`app/ambience.py`** — Server-side room-tone mixer; loops a synthesized noise bed through Pipecat's output mixer, per transport sample rate
- **`app/voice_settings.py`** — Persisted active ElevenLabs voice, UI-selected and applied to all paths; env `ELEVENLABS_VOICE_ID` is the fallback
- **`app/characters.py`** — Persona definitions loaded from JSON (`app/characters/`), including the `greeting` used as the canned opener
- **`app/mock_conversation.py`** — Legacy `/ws/browser` session logic (~530 lines); handles audio ingestion (PCM S16LE framing, RMS/peak analysis), turn detection, provider orchestration, and graceful fallback to mock when providers fail
- **`app/groq_agent.py`** — Groq LLM adapter; `pop_speakable_chunks()` splits streaming delta text into TTS-ready sentence fragments (≥24 chars on `.!?\n`, or every 90 chars at a word boundary)
- **`app/deepgram.py`** — Deepgram WebSocket STT; emits on `is_final` and `speech_final` signals
- **`app/elevenlabs_tts.py`** — ElevenLabs WebSocket (`stream-input`) TTS; returns base64-encoded PCM audio chunks
- **`app/preview.py`** — Stateless one-shot character preview (LLM→TTS) powering the Personality Studio's `POST /characters/preview`; composes the same adapters a live turn uses, mock-fallbacks without keys
- **`app/memory/`** — Agent memory: `embedder.py` (swappable: OpenAI via httpx, deterministic mock), `store.py` (local file-backed numpy-free cosine vector store, atomic writes), `manager.py` (RAG retrieve / distill / summarize). Injects at the `conversation_context.py` seam alongside intent inference; distills on session close. Inspect/reset via `GET`/`DELETE /memory`. Single global person (no identity keying yet)

### Frontend

- **`frontend/pipecat.html`** — Default UI (served at `/pipecat`); self-contained page using the Pipecat JS SDK (`@pipecat-ai/client-js` + `@pipecat-ai/websocket-transport` ESM from jsdelivr) over `/api/ws`
- **`frontend/app.js`** — Classic UI (`/classic`): WebSocket lifecycle, microphone capture (AudioContext at 16 kHz), PCM conversion, audio playback scheduling, HUD metrics
- **`frontend/pcm-worklet.js`** — AudioWorklet that converts float32 samples → 16-bit signed PCM in the audio thread
- **`frontend/index.html`** / **`frontend/styles.css`** — Classic static shell; no build step required

### Event Contract

Legacy `/ws/browser` contract — the Pipecat endpoints speak the Pipecat client protocol instead. All messages are JSON. Server events include: `session.started`, `session.ended`, `transcript.partial`, `transcript.final`, `agent.response_start`, `agent.response_chunk`, `agent.response_end`, `audio.chunk`, `pipeline.stage`, `latency.report`, `error`. Client events: `client.hello`, `audio.start`, `audio.stop`, `session.stop`. See `GET /events` for the live contract or `app/events.py` for definitions.

## Deployment

Uses the included `Dockerfile` (python:3.12-slim). Set `VOICE_AGENT_PORT` and point the health check at `/health`. Compatible with Coolify.
