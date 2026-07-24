# Browser-First Conversational Voice Agent

Local browser voice agent with Deepgram streaming transcription, Groq streaming responses, ElevenLabs speech, browser playback, and Phase 6 barge-in interruption.

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
- ElevenLabs streaming TTS
- Browser playback and barge-in cancellation

Live mode expects these values:

- `DEEPGRAM_API_KEY`
- `GROQ_API_KEY`
- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID`

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
- `DEEPGRAM_ENDPOINTING_MS`, default `200`
- `DEEPGRAM_UTTERANCE_END_MS`, default `1000`
- `VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS`, default `500`; legacy-path fallback debounce for useful partial transcripts when Deepgram has not emitted `speech_final`
- `VOICE_AGENT_INPUT_GAIN`, default `2.0`; server-side PCM gain applied before STT for quiet microphones
- `VOICE_AGENT_INTENT_INFERENCE_ENABLED`, default `true`; keeps raw transcripts visible while adding hidden Groq guidance to infer likely intent from recent context
- `VOICE_AGENT_CONVERSATION_FLOW_ENABLED`, default `true`; adds a local per-turn social brief for characters whose `conversation_mode` is `social` (reply length, question streaks, opener/filler variety, and non-assistant stance)
- `GROQ_MODEL`, default `openai/gpt-oss-120b` for the classic/preview path
- `DEFAULT_MODEL`, default `groq-gpt-oss-120b` for the Pipecat path
- `GROQ_TEMPERATURE`, default `0.8`
- `GROQ_REASONING_EFFORT`, default `low`; GPT-OSS only
- `GROQ_MAX_TOKENS`, default `320`; includes the reasoning budget, while the conversation director keeps spoken replies compact
- `ELEVENLABS_MODEL`, default `eleven_flash_v2_5`. Note: `eleven_v3` is **not** supported on the realtime WebSocket the pipeline uses — keep a streaming model (`eleven_flash_v2_5` or `eleven_turbo_v2_5`)
- `ELEVENLABS_VOICE_ID`; an ElevenLabs voice id (opaque string, e.g. `21m00Tcm4TlvDq8ikWAM`). This is only the fallback default — the active voice can be picked from the UI (see below), which overrides it for all calls
- `ELEVENLABS_SAMPLE_RATE`, default `24000`; the Pipecat browser reads this value from `/health` and constructs playback at the same rate to avoid pitch/speed distortion
- `ELEVENLABS_SPEED`, default `1.0` (`0.7` to `1.2`; higher is faster)
- `ELEVENLABS_STABILITY`, default `0.5` (`0.0` to `1.0`)
- `ELEVENLABS_SIMILARITY_BOOST`, default `0.8` (`0.0` to `1.0`)
- `ELEVENLABS_STYLE`, default `0.0` (`0.0` to `1.0`)
- `ELEVENLABS_USE_SPEAKER_BOOST`, default `false`
- `ELEVENLABS_OPEN_TIMEOUT_SECONDS`, default `8`; WebSocket opening-handshake timeout per attempt
- `ELEVENLABS_CONNECT_RETRIES`, default `1`; retry count for transient opening-handshake timeouts
- `VOICE_AGENT_PERSONA`

Phone-call ambience:

- `VOICE_AGENT_AMBIENCE_ENABLED`, default `true`
- `VOICE_AGENT_AMBIENCE_SCENE`, default `room_line`
- `VOICE_AGENT_AMBIENCE_VOLUME`, default `0.035`

The ambience bed is generated in the browser with Web Audio after microphone permission is granted. It is connected only to local playback, never sent to Deepgram, never mixed into assistant `audio.chunk` events, and ramps down when the mic or session stops.

Outbound calling:

- `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN`: Twilio REST credentials
- `TWILIO_FROM_NUMBER`: Twilio number or verified outgoing caller ID in E.164 format
- `PUBLIC_HOST`: public hostname without a scheme or path
- `VOICE_AGENT_OUTBOUND_ACCESS_TOKEN`: strong private token that enables and protects the website dialer

The Pipecat page can place an outbound call using its currently selected character,
voice, and model. The access token is submitted as a Bearer token and remains only
in the open page. Twilio call status callbacks update the UI until the call reaches
a terminal state. The in-process call store assumes one application worker; use a
shared database or Redis before running multiple workers.

## Contextual Speech

Live mode preserves raw Deepgram transcripts in `transcript.partial`, `transcript.final`, logs, and stored conversation turns. When `VOICE_AGENT_INTENT_INFERENCE_ENABLED=true`, the model request also receives hidden guidance that the latest user turn may include speech-to-text errors. For a character with `conversation_mode: "social"`, `VOICE_AGENT_CONVERSATION_FLOW_ENABLED=true` also adds a fresh deterministic turn brief immediately before the LLM: match the caller's turn size, avoid consecutive question endings, vary recent openings/fillers, contribute a point of view, and avoid help-desk reflexes. The raw transcript and stored character are unchanged.

`VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS` controls the classic app fallback used when Deepgram has not emitted `speech_final`. The default is `500ms`; the default Pipecat page instead uses Silero VAD plus the local Smart Turn analyzer to distinguish an end-of-thought from a thinking pause.

ElevenLabs `eleven_flash_v2_5` has no SSML/emotion-tag support, so prosody is controlled entirely through the voice settings (`ELEVENLABS_STABILITY`, `ELEVENLABS_SIMILARITY_BOOST`, `ELEVENLABS_STYLE`, `ELEVENLABS_USE_SPEAKER_BOOST`, `ELEVENLABS_SPEED`). Any stray inline markup the model emits (e.g. `<emotion>`, `<break>`) is stripped before TTS so it is never read aloud.

### Choosing a voice

The active ElevenLabs voice can be picked from the UI (the Pipecat page voice bar) instead of the env — either from the dropdown of your account voices or by pasting a voice id. The choice is persisted server-side (`data/active_voice_id`) and applies to every path: browser calls, Twilio phone calls, and the Studio preview. `ELEVENLABS_VOICE_ID` in the env is only the fallback default. The `ELEVENLABS_API_KEY` always stays in the env — it is never exposed to the browser.

Endpoints:
- `GET /voices` — your account's voices (`GET /v1/voices`) plus the currently `active` voice; returns an empty list with a `warning` if the key is missing or the fetch fails.
- `GET /voice` — the resolved `active` voice, the `persisted` UI choice, and the `env_default`.
- `PUT /voice` `{"voice_id": "..."}` — persist the active voice; an empty string clears it (falls back to the env default).

Resolution precedence per session: `?voice=` on the WebSocket connect URL → persisted UI choice → `ELEVENLABS_VOICE_ID`.

### Choosing an LLM model (performance A/B)

The Pipecat page has a **Model** picker so you can switch the LLM on a live call and compare latency/quality. Two providers behind one OpenAI-compatible interface (see `app/llm_models.py` `MODELS`):

- **Groq** — direct, lowest latency (free tier). Needs `GROQ_API_KEY`.
- **OpenRouter** — one key, many models (Claude, GPT-4o-mini, Gemini, Llama…). Needs `OPENROUTER_API_KEY`; without it the OpenRouter entries won't work and the picker says so.

The Groq defaults are now **GPT-OSS 120B** (quality) and **GPT-OSS 20B** (latency), both with low reasoning effort for casual speech. The older Groq Llama presets remain visible only for comparison and are labeled with their retirement status.

Beyond the curated presets, you can pick or paste **any OpenRouter model id** (e.g. `anthropic/claude-sonnet-4.5`) in the model bar's text field — it autocompletes from OpenRouter's live catalog (`GET /openrouter/models`). A value with a `/` is treated as a raw OpenRouter slug; a value without one must be a preset key.

The choice persists server-side (`data/active_model`) and applies to browser and phone calls. `enable_metrics=True` is on, so each model's **TTFB is logged** in the bot console (watch for `OpenAILLMService#0 TTFB: …`) — that's your measurement.

Endpoints:
- `GET /models` — the switchable list + `active` + whether OpenRouter is configured.
- `PUT /model` `{"model": "<key>"}` — persist the active model (empty string clears to the default).

Resolution precedence per session: `?model=` on connect → persisted UI choice → `DEFAULT_MODEL` env (a key in `MODELS`).

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

- `GET /`: redirects to the Pipecat browser client
- `GET /pipecat`: production Pipecat browser client
- `GET /classic`: legacy browser client
- `GET /twiml`: TwiML for inbound Twilio calls
- `GET /health`: deployment health and safe config status
- `GET /events`: JSON event contract
- `POST /api/outbound-calls`: create a protected outbound AI-agent call
- `GET /api/outbound-calls/{call_id}`: read protected outbound call status
- `POST /api/outbound-calls/{call_id}/hangup`: end a protected outbound call
- `POST /twilio/call-status/{call_id}`: signed Twilio call-status callback
- `WS /api/ws`: one Pipecat browser voice session per WebSocket
- `WS /api/twilio-ws`: Twilio Media Streams WebSocket

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
      "tts": "elevenlabs"
    },
    "audio": {
      "input_gain": 2.0
    },
    "conversation": {
      "intent_inference_enabled": true
    },
    "elevenlabs": {
      "model": "eleven_flash_v2_5",
      "sample_rate": 16000,
      "connection": {
        "open_timeout_seconds": 8.0,
        "connect_retries": 1
      },
      "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.8,
        "style": 0.0,
        "use_speaker_boost": false,
        "speed": 1.0
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
