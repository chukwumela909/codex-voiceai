# Plan: Browser-First Conversational Voice Agent

> Source PRD: PRD/browser-first-conversational-voice-agent.md

## Architectural decisions

Durable decisions that apply across all phases:

- **Primary route shape**: backend exposes a health endpoint and a browser conversation WebSocket endpoint for local microphone sessions.
- **Transport boundary**: browser audio and future Twilio audio enter the same conversation core through transport adapters.
- **Session boundary**: one browser WebSocket connection equals one voice session for v1.
- **Audio contract**: the conversation core receives normalized audio frames with encoding, sample rate, timestamp, and source metadata; provider-specific adapters own final conversion.
- **Canonical inbound format**: prefer PCM16 mono frames with sample rate metadata for browser v1, while keeping the core capable of accepting future Twilio mulaw 8000 Hz frames through an adapter.
- **Conversation event model**: client and server exchange typed events for status, transcripts, agent text, latency metrics, interruption, errors, and session lifecycle; binary or base64 audio chunks are associated with typed metadata.
- **Response identity model**: every assistant response has a response ID, and all text, audio, latency, completion, and interruption events for that response include it.
- **Provider boundaries**: Deepgram owns streaming STT, Groq owns streaming LLM inference, and Cartesia owns streaming TTS.
- **Session model**: each WebSocket connection owns one in-memory session with transcript, active task handles, timing data, and provider streams.
- **Memory model**: v1 stores only session-local transcript context and discards it on disconnect.
- **Configuration model**: provider keys, model names, voice ID, persona prompt, CORS origins, and deployment port are environment-driven.
- **Runtime mode model**: mock mode is first-class for local frontend development and deterministic tests; live mode uses Deepgram, Groq, and Cartesia.
- **Latency model**: time-to-first-audio is measured from final user turn detection to first playable audio chunk; sub-1s is best-effort, not a release gate.
- **Playback model**: browser playback uses a queue with response ID filtering so stale audio from cancelled responses can be discarded.
- **Interruption model**: user speech during agent playback cancels active generation, synthesis, backend audio emission, and client playback for the interrupted response, then resumes the conversation from the new user utterance.
- **Deployment model**: backend is container-friendly and Coolify-ready; frontend remains locally runnable for v1.

---

## Phase 1: Connectable app shell and event contract

**User stories**: 1, 13, 14, 21, 23, 24

### What to build

Create the smallest runnable system: a FastAPI backend, a local plain HTML/JavaScript client, a health check, a minimal container shape, and a browser WebSocket connection that exchanges structured status and diagnostic events. No real audio or providers are required yet; this phase proves the app can boot, connect, identify sessions, surface errors, and support Coolify-style configuration.

### Acceptance criteria

- [ ] Browser client can connect and disconnect from the backend WebSocket.
- [ ] Client HUD shows disconnected, connecting, connected, and error states.
- [ ] Backend assigns a session ID and includes it in structured logs and client-visible events.
- [ ] Health endpoint returns a simple healthy response suitable for deployment checks.
- [ ] Missing required provider environment variables are reported clearly without exposing secrets.
- [ ] Core event names and payload shapes are documented in the codebase for later phases.
- [ ] Minimal Dockerfile or equivalent container entrypoint exists for Coolify-oriented deployment.
- [ ] Minimal README documents local startup, WebSocket route, health route, and mock/live mode intent.
- [ ] Core behavior is covered by tests for health, WebSocket connect, status events, and config validation.

---

## Phase 2: Browser microphone uplink with mock conversation loop

**User stories**: 1, 2, 9, 10, 11, 16, 17, 23, 24, 25, 27, 28

### What to build

Add browser microphone capture and send normalized audio frames through the WebSocket into the backend transport adapter. Use a mock conversation engine that emits fake partial transcript, final transcript, agent text, latency metrics, response IDs, and a simple synthetic audio/playback signal so the full browser-to-backend-to-browser loop is demoable before provider APIs are introduced.

### Acceptance criteria

- [ ] Browser requests microphone permission and streams audio frames to the backend.
- [ ] Client can start and stop a local voice session without refreshing the page.
- [ ] Backend receives audio through the browser adapter without coupling the conversation core to browser-specific APIs.
- [ ] HUD displays mock partial transcript, final transcript, agent text, latency metric, and playback state.
- [ ] Mock mode can run without Deepgram, Groq, or Cartesia keys.
- [ ] Browser playback queue handles response IDs and can discard stale queued audio.
- [ ] Audio frame messages include encoding, sample rate, timestamp, and session context needed by the transport contract.
- [ ] Backend cleanup runs on browser disconnect and cancels active mock session work.
- [ ] Tests cover transport adapter behavior, session cleanup, and emitted diagnostic events.

---

## Phase 3: Deepgram streaming transcription slice

**User stories**: 2, 6, 9, 10, 11, 13, 15, 16, 23, 24, 25, 27, 28

### What to build

Replace mock transcription with Deepgram streaming STT while keeping downstream agent and audio behavior mocked. Browser microphone audio should produce live partial and final transcripts in the HUD, and an explicit turn-detection policy should emit timing data that later phases can use to measure response latency.

### Acceptance criteria

- [ ] Backend opens and closes a Deepgram streaming session per browser conversation.
- [ ] Browser microphone audio is converted or framed into a format accepted by the STT adapter.
- [ ] HUD displays Deepgram interim transcript updates as partial transcript events.
- [ ] HUD displays final transcript events separately from partial transcript events.
- [ ] Backend emits turn-detection timing events when a final utterance is ready for response.
- [ ] Turn detection policy is configurable enough to tune endpointing without changing the transport contract.
- [ ] Interim transcript during agent speaking can be surfaced as a candidate barge-in signal for later interruption handling.
- [ ] Deepgram model and endpointing-related options are environment-configurable.
- [ ] Provider failures emit client-visible error/status events.
- [ ] Tests cover STT adapter success, malformed provider event, cancellation, and disconnect cleanup using mocked Deepgram events.

---

## Phase 4: Groq persona response slice

**User stories**: 4, 6, 7, 8, 13, 15, 16, 19, 23, 24, 25, 26, 28

### What to build

Connect final user transcripts to Groq streaming chat completions with a playful persona prompt and session-only conversation memory. Keep TTS mocked, but stream agent text through the event contract and chunk the LLM output into speakable phrases so the next phase can feed Cartesia without redesigning the pipeline.

### Acceptance criteria

- [ ] Final user transcript triggers a Groq streaming response.
- [ ] Agent persona is controlled by backend configuration and defaults to a playful demo character.
- [ ] Session transcript includes user and assistant turns for the current WebSocket session only.
- [ ] Transcript context is discarded on disconnect.
- [ ] HUD shows streaming agent text and final assistant message state.
- [ ] Phrase chunker emits speakable chunks before the full LLM response is complete.
- [ ] Agent text events include response IDs so cancelled or stale responses can be ignored downstream.
- [ ] Groq model and inference settings are environment-configurable.
- [ ] Tests cover persona prompt usage, session memory, phrase chunking, provider failure, and cancellation with mocked Groq streams.

---

## Phase 5: Cartesia streaming speech and browser playback slice

**User stories**: 3, 4, 6, 11, 13, 15, 16, 18, 19, 23, 24, 25, 28

### What to build

Replace mocked speech with Cartesia streaming TTS and Web Audio API browser playback. The backend should feed phrase-level LLM chunks into TTS, stream playable audio events to the browser, and measure time-to-first-audio. The client should play audio smoothly through a response-aware playback queue while updating the HUD with thinking, speaking, latency, and provider status.

### Acceptance criteria

- [ ] Cartesia TTS receives speakable chunks from the Groq response stream.
- [ ] Browser receives and plays streamed audio from the backend.
- [ ] Browser playback uses a queue that can stop active playback and clear queued chunks by response ID.
- [ ] Client state transitions through listening, thinking, speaking, and idle states.
- [ ] Time-to-first-audio is measured and displayed in the HUD.
- [ ] Cartesia model, voice ID, and output format settings are environment-configurable.
- [ ] Provider failures emit client-visible errors and do not leave the session permanently stuck.
- [ ] Tests cover TTS adapter success, streamed audio emission, provider failure, timeout, and cancellation using mocked Cartesia streams.

---

## Phase 6: Natural barge-in and cancellation slice

**User stories**: 5, 6, 11, 12, 16, 18, 20, 23, 24, 25, 28

### What to build

Add interruption behavior across the full voice pipeline. When user speech is detected while the agent is speaking, the system should stop browser playback, clear stale queued audio, cancel active LLM/TTS work, emit interruption diagnostics, and accept the new utterance as the next turn. This phase makes the experience feel conversational rather than queued.

### Acceptance criteria

- [ ] User speech during agent playback triggers an interruption event.
- [ ] Browser playback stops quickly when interruption is confirmed.
- [ ] Browser discards queued audio chunks for the interrupted response ID.
- [ ] Active LLM generation and TTS synthesis are cancelled for the interrupted response.
- [ ] HUD shows interruption events and updated listening/thinking/speaking state.
- [ ] Conversation memory does not record stale, unsent assistant text as a completed response.
- [ ] New user utterance can trigger the next agent response after interruption.
- [ ] Tests cover interrupt during playback, interrupt during LLM generation, interrupt during TTS streaming, and cleanup after cancellation.

---

## Phase 7: Deployment hardening and local operator polish

**User stories**: 11, 13, 14, 15, 21, 23, 24, 25, 28

### What to build

Make the backend fully ready for Coolify deployment and local iteration. Expand container-friendly runtime configuration, startup behavior, CORS/origin controls, structured logging, README instructions, and the local developer workflow for running the browser client against the backend.

### Acceptance criteria

- [ ] Backend can run in a container using environment variables only.
- [ ] Coolify-required port and health check assumptions are documented.
- [ ] Local frontend can be served or opened for browser testing without exposing provider keys.
- [ ] CORS/origin configuration supports local development and deployment settings.
- [ ] Logs include session ID, provider stage, status transitions, errors, and latency timings.
- [ ] README documents required environment variables, local run steps, mock/live mode, and deployment assumptions.
- [ ] Tests cover config loading, safe error reporting, and session shutdown paths.

---

## Phase 8: Twilio-ready seam validation

**User stories**: 16, 17, 22, 27

### What to build

Validate that the architecture can support Twilio later without implementing the production Twilio route. Add a documented transport-adapter contract and a lightweight Twilio adapter design note that maps Twilio Media Stream concepts to the existing conversation core: inbound audio frames, outbound audio frames, mark events, clear events, and audio format conversion.

### Acceptance criteria

- [ ] Conversation core can be driven by transport-neutral inbound audio and control events.
- [ ] Browser adapter uses the same transport contract expected of future adapters.
- [ ] Twilio adapter design note identifies required handling for mulaw 8000 Hz audio, base64 media payloads, mark events, and clear events.
- [ ] No browser-only assumptions are required by the conversation core.
- [ ] Tests cover the transport contract with a fake adapter independent of browser APIs.
