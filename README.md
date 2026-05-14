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
- `VOICE_AGENT_WS_PING_INTERVAL`: browser WebSocket ping interval, default `30`
- `VOICE_AGENT_WS_PING_TIMEOUT`: browser WebSocket ping timeout, default `120`
- `VOICE_AGENT_CORS_ORIGINS`: comma-separated allowed origins, or `*`

Provider tuning:

- `DEEPGRAM_MODEL`, default `nova-3`
- `DEEPGRAM_ENDPOINTING_MS`, default `220`
- `DEEPGRAM_UTTERANCE_END_MS`, default `1000`
- `VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS`, default `1000`; fallback debounce for useful partial transcripts when Deepgram has not emitted `speech_final`
- `VOICE_AGENT_INPUT_GAIN`, default `2.0`; server-side PCM gain applied before STT for quiet microphones
- `VOICE_AGENT_INTENT_INFERENCE_ENABLED`, default `true`; keeps raw transcripts visible while adding hidden Groq guidance to infer likely intent from recent context
- `GROQ_MODEL`, default `llama-3.1-8b-instant`
- `GROQ_TEMPERATURE`, default `0.7`
- `CARTESIA_MODEL`, default `sonic-3`
- `CARTESIA_SPEED`, default `1.2` (`0.6` to `1.5`; higher is faster)
- `CARTESIA_SAMPLE_RATE`, default `16000`
- `CARTESIA_VERSION`, default `2026-03-01`
- `CARTESIA_OPEN_TIMEOUT_SECONDS`, default `8`; WebSocket opening-handshake timeout per attempt
- `CARTESIA_CONNECT_RETRIES`, default `1`; retry count for transient Cartesia opening-handshake timeouts
- `VOICE_AGENT_CARTESIA_SPEECH_DIRECTOR_ENABLED`, default `true`
- `VOICE_AGENT_CARTESIA_SSML_ENABLED`, default `true`
- `VOICE_AGENT_CARTESIA_EMOTION_TAGS_ENABLED`, default `false`
- `VOICE_AGENT_PERSONA`

Phone-call ambience:

- `VOICE_AGENT_AMBIENCE_ENABLED`, default `true`
- `VOICE_AGENT_AMBIENCE_SCENE`, default `room_line`
- `VOICE_AGENT_AMBIENCE_VOLUME`, default `0.035`

The ambience bed is generated in the browser with Web Audio after microphone permission is granted. It is connected only to local playback, never sent to Deepgram, never mixed into assistant `audio.chunk` events, and ramps down when the mic or session stops.

## Contextual Speech

Live mode preserves raw Deepgram transcripts in `transcript.partial`, `transcript.final`, logs, and stored conversation turns. When `VOICE_AGENT_INTENT_INFERENCE_ENABLED=true`, the Groq request also receives hidden guidance that the latest user turn may include speech-to-text errors, so it should infer likely intent from recent context and ask a short clarifying question only when ambiguity blocks a useful answer.

`VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS` controls the app fallback used when Deepgram has not emitted `speech_final`. The default is `1000ms` to leave more room for natural thinking pauses. Lower it for snappier demos; raise it when the assistant interrupts too early.

When Cartesia is configured, `VOICE_AGENT_CARTESIA_SPEECH_DIRECTOR_ENABLED=true` applies conservative SSML-style speech direction to TTS input only. Frontend assistant text stays plain. The director adds short context-relevant pauses after discourse markers, pauses before inferred clarifications, and spells code-like tokens such as API names or numeric IDs. Emotion tags are disabled by default with `VOICE_AGENT_CARTESIA_EMOTION_TAGS_ENABLED=false`.

Proactive conversation tuning:

- `VOICE_AGENT_PROACTIVE_ENABLED`: `auto`, `true`, or `false`. `auto` enables proactive behavior in mock mode and keeps live mode opt-in.
- `VOICE_AGENT_PROACTIVE_GREETING_DELAY_MS`: startup greeting delay after the first inbound audio frame, default `500`.
- `VOICE_AGENT_PROACTIVE_SILENCE_TIMEOUT_MS`: idle delay before a proactive silence nudge or contextual follow-up. If unset, mock mode defaults to `5000`; live mode defaults to `30000`.
- `VOICE_AGENT_PROACTIVE_REPEAT_COOLDOWN_MS`: delay between repeated proactive prompts. If unset, mock mode defaults to `8000`; live mode defaults to `60000`.
- `VOICE_AGENT_PROACTIVE_MAX_CONSECUTIVE_PROMPTS`: maximum proactive prompts before backing off until the user speaks. If unset, mock mode defaults to `3`; live mode defaults to `1`.
- `VOICE_AGENT_PROACTIVE_FAILURE_BACKOFF_THRESHOLD`: provider failures before proactive failure backoff, default `2`.
- `VOICE_AGENT_PROACTIVE_FAILURE_BACKOFF_MS`: observable cooldown duration after failure_backoff, default `30000`.
- `VOICE_AGENT_PROACTIVE_CONTEXTUAL_FOLLOWUPS_ENABLED`: lets the idle policy choose contextual follow-ups when recent user context exists, default `true`.

## Proactive Conversation

Proactive behavior makes the agent feel present without waiting forever for a user event. It uses the same assistant response slot, transcript, TTS path, interruption logic, and `response_id` stale-output protection as normal user-driven turns. The default persona and proactive instructions frame the experience as an ambiguous open phone call: warm, brief, and still on the line without implying who called whom.

In mock mode the defaults are demo-ready: after audio starts and the backend receives the first audio frame, the agent sends deterministic mock proactive text. If the user stays quiet, the same idle scheduler waits 5 seconds, then chooses either a mock silence nudge or a mock contextual follow-up. Repeated nudges use the repeat cooldown so they do not stack rapidly.

In live mode, `VOICE_AGENT_PROACTIVE_ENABLED=auto` keeps proactive behavior disabled. Set `VOICE_AGENT_PROACTIVE_ENABLED=true` to opt in. When Groq is configured, proactive startup greetings, silence nudges, and contextual follow-ups are generated through Groq with short internal instructions, while Groq still applies the configured `VOICE_AGENT_PERSONA`. Scripted proactive copy is used only for mock mode or provider fallback.

Proactive diagnostics:

- `proactive.triggered`: a proactive turn was allowed. Payload includes `trigger_reason`, `source_state`, prompt counts, and failure counts.
- `proactive.skipped`: a candidate turn was blocked. `skip_reason` explains why, such as `cooldown`, `active_response`, `question_already_pending`, or `failure_backoff`.
- `proactive.cancelled`: pending proactive work was cancelled by user speech, audio stop, shutdown, or test/setup state.
- `proactive.cooldown`: reports cooldown and backoff windows, including `cooldown_ms`, `next_eligible_at_ms`, and failure counts when relevant.
- `proactive.state`: reports idle monitoring and backed-off states.

Tuning guidance:

- For local demos, keep the 5 second silence timeout and 8 second repeat cooldown.
- For live demos, opt in deliberately and watch `proactive.skipped`, `proactive.cooldown`, and `proactive.state` logs.
- Lower `VOICE_AGENT_PROACTIVE_MAX_CONSECUTIVE_PROMPTS` if the agent feels too eager.
- Raise `VOICE_AGENT_PROACTIVE_SILENCE_TIMEOUT_MS` when real callers need more thinking time.
- Keep failure backoff enabled so provider problems do not create repeated proactive retries.

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
    "invalid_live_keys": [],
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
    },
    "audio": {
      "input_gain": 2.0
    },
    "conversation": {
      "intent_inference_enabled": true
    },
    "cartesia": {
      "connection": {
        "open_timeout_seconds": 8.0,
        "connect_retries": 1
      },
      "speech_direction": {
        "enabled": true,
        "ssml_enabled": true,
        "emotion_tags_enabled": false
      }
    },
    "ambience": {
      "enabled": true,
      "scene": "room_line",
      "volume": 0.035
    },
    "turn_timing": {
      "deepgram_endpointing_ms": 220,
      "deepgram_utterance_end_ms": 1000,
      "partial_idle_finalize_ms": 1000
    },
    "proactive": {
      "configured": "auto",
      "enabled": false,
      "startup_greeting_delay_ms": 500,
      "silence_timeout_ms": 30000,
      "repeat_cooldown_ms": 60000,
      "max_consecutive_prompts": 1,
      "failure_backoff_threshold": 2,
      "failure_backoff_ms": 30000,
      "contextual_followups_enabled": true
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
- `proactive.triggered`
- `proactive.skipped`
- `proactive.cancelled`
- `proactive.cooldown`
- `proactive.state`
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
- proactive trigger, skip, cancellation, cooldown, and backoff events
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

The suite covers config loading, safe health reporting, WebSocket startup/shutdown, provider adapters, turn detection, proactive startup greetings, silence nudges, contextual follow-ups, interruption behavior, provider failure backoff, and event logging.
