# Browser-First Conversational Voice Agent PRD

## Problem Statement

You want a low-latency conversational voice agent that feels closer to Sesame/OpenAI Realtime than a traditional audio chatbot. The first release should run from a local browser client using the microphone, while the backend is designed so the same conversation core can later power a Twilio Media Streams version.

The main challenge is not only connecting STT, LLM, and TTS APIs, but making the turn-taking feel natural: fast first response, interruption support, smooth playback, and visible developer diagnostics while tuning latency.

## Solution

Build a browser-first voice agent MVP using Python FastAPI, a plain HTML/JavaScript browser client, Deepgram streaming speech-to-text, Groq streaming LLM inference, and Cartesia streaming text-to-speech.

The first release will provide a playful custom persona with session-only memory, natural barge-in, backend-owned provider secrets, and a developer HUD for tuning latency and interruption behavior. The backend should be Coolify-friendly and structured around a transport-agnostic conversation core so Twilio Media Streams can be added later without rewriting the core voice pipeline.

Latency is a best-effort goal, not a hard launch blocker. The system should measure time-to-first-audio and optimize toward sub-1s perceived response, but correctness and a stable conversational loop come first.

## User Stories

1. As a user, I want to open a browser page and connect to the voice agent, so that I can start a conversation without installing extra software.
2. As a user, I want to grant microphone access, so that the agent can hear me speak.
3. As a user, I want the agent to respond with natural spoken audio, so that the interaction feels conversational.
4. As a user, I want the agent to have a playful character, so that the demo feels memorable and expressive.
5. As a user, I want the agent to stop talking when I interrupt, so that I can correct it or move the conversation forward naturally.
6. As a user, I want low perceived delay after I stop speaking, so that the interaction does not feel like a walkie-talkie.
7. As a user, I want the agent to remember context during the current session, so that I do not need to repeat myself.
8. As a user, I do not want the v1 agent to remember me across sessions, so that the first version stays simple and privacy-light.
9. As a developer, I want partial transcripts displayed in the browser, so that I can tune turn detection and barge-in behavior.
10. As a developer, I want final transcripts displayed separately from partial transcripts, so that I can debug when the agent decides to respond.
11. As a developer, I want latency timings in the HUD, so that I can see where delays come from.
12. As a developer, I want interruption events visible in the HUD, so that I can confirm barge-in is working.
13. As a developer, I want backend logs grouped by conversation/session ID, so that I can debug individual conversations.
14. As a developer, I want provider API keys read from environment variables, so that secrets are never exposed in frontend code.
15. As a developer, I want provider model and voice settings configurable by environment variables, so that I can tune quality and latency without code changes.
16. As a developer, I want a reusable conversation core, so that browser and future Twilio integrations share the same STT/LLM/TTS orchestration.
17. As a developer, I want the browser transport isolated from the conversation engine, so that Twilio can be added later as another adapter.
18. As a developer, I want TTS playback to stream instead of waiting for a full response, so that users hear the agent sooner.
19. As a developer, I want Groq responses streamed and chunked into speakable phrases, so that Cartesia can start speaking quickly without awkward token-level speech.
20. As a developer, I want active generation and playback to be cancellable, so that interruption does not leave stale audio or stale LLM output running.
21. As a deployer, I want the backend to be Coolify-friendly, so that I can deploy it later with environment variables and containerized runtime assumptions.
22. As a future deployer, I want the architecture to anticipate Twilio Media Streams, so that the final phone-call version does not require a full rewrite.
23. As a developer, I want provider failures to produce clear client-visible status events, so that I can distinguish microphone, WebSocket, STT, LLM, and TTS failures.
24. As a developer, I want the client to show connection and speaking states, so that I can quickly tell whether the agent is listening, thinking, speaking, interrupted, or disconnected.
25. As a developer, I want clean session shutdown on browser disconnect, so that provider streams and background tasks do not leak.
26. As a developer, I want the system prompt/persona to be configurable on the backend, so that the demo character can be refined without changing frontend code.
27. As a developer, I want audio format handling isolated behind adapters, so that browser audio and future Twilio audio can use different encodings without affecting the conversation core.
28. As a developer, I want manual latency observations during local use, so that I can compare perceived delay against HUD timings.

## Implementation Decisions

- Backend stack: Python FastAPI with async WebSocket endpoints.
- Frontend stack: plain HTML/JavaScript local browser client.
- Provider stack: Deepgram STT, Groq LLM, Cartesia TTS.
- Deployment target: Coolify-native backend deployment using container-friendly environment configuration.
- Memory scope: session-only transcript and conversation state.
- Persona: playful custom demo character.
- UX goal: conversational real-time voice with natural barge-in and measured best-effort sub-1s latency.
- Browser client responsibilities: microphone capture, WebSocket connection, audio playback, developer HUD, and user controls.
- Backend responsibilities: provider connections, conversation orchestration, turn detection, interruption, phrase chunking, TTS streaming, and session cleanup.
- Conversation core: expose a transport-agnostic interface that accepts inbound audio frames/events and emits outbound audio/events.
- Browser adapter: converts browser microphone audio into backend-compatible audio frames and plays backend audio responses.
- Future Twilio adapter: should reuse the conversation core while handling Twilio-specific audio/x-mulaw, 8000 Hz, base64 media messages, mark events, and clear events.
- STT behavior: use Deepgram streaming transcription with interim results and endpointing/utterance-end style signals for turn detection.
- LLM behavior: use Groq streaming responses, but buffer into phrase/sentence-level chunks before TTS.
- TTS behavior: use Cartesia WebSocket streaming with context/continuation support where appropriate.
- Interruption behavior: when user speech is detected during agent playback, cancel active LLM/TTS/playback work and prioritize the new user utterance.
- Secrets: all provider API keys live only on the backend as environment variables.
- Diagnostics: browser HUD and backend logs must expose enough timing data to tune perceived latency.
- Configuration: exact Deepgram model, Groq model, Cartesia model, Cartesia voice ID, and persona prompt should be provided through environment variables or backend config.
- Error handling: backend should emit structured status/error events to the browser rather than silently failing.
- Cleanup: WebSocket disconnect should close provider streams, cancel active generation/synthesis/playback tasks, and release session state.
- Compatibility: do not design browser-specific assumptions into the conversation core that would prevent future Twilio integration.

## Testing Decisions

- Test core modules only for v1.
- Tests should focus on external behavior, not internal implementation details.
- Mock provider APIs for deterministic tests.
- Test conversation orchestration: audio input leads to transcript, agent response, TTS request, and outbound audio event.
- Test turn detection: final/interim transcript events trigger responses at the expected boundary.
- Test interruption: user speech during playback cancels active response work and emits interruption state.
- Test phrase chunking: streamed LLM tokens become speakable chunks without waiting for the full response.
- Test provider adapters: each adapter handles success, timeout, malformed event, and cancellation paths.
- Test session cleanup: WebSocket disconnect closes provider streams and cancels active tasks.
- Test diagnostics: status events and latency measurements are emitted in a client-consumable format.
- Good tests should assert public behavior and emitted events, not private queue structure or internal task names.

## Out of Scope

- Twilio production integration in the first milestone.
- Persistent user memory across sessions.
- Transcript database or analytics dashboard.
- Authentication and multi-user account management.
- Local model inference.
- Mobile app packaging.
- Production-grade monitoring beyond structured logs and basic timing metrics.
- Full visual polish beyond a useful local developer HUD.
- Billing, quotas, or tenant-level rate limiting.
- Admin UI for changing providers or persona settings.
- Long-term storage of audio recordings.

## Further Notes

- Deepgram docs confirm support for interim results and endpointing useful for real-time turn detection: https://developers.deepgram.com/docs/understand-endpointing-interim-results
- Cartesia docs confirm WebSocket TTS, streamed chunks, contexts, continuations, cancellation, and timestamps: https://docs.cartesia.ai/api-reference/tts/websocket
- Cartesia context docs: https://docs.cartesia.ai/use-the-api/tts-websocket/contexts
- Twilio Media Streams require specific audio/message handling for the future phone-call adapter: https://www.twilio.com/docs/voice/media-streams/websocket-messages
- Groq supports OpenAI-compatible chat completions suitable for streaming LLM inference: https://console.groq.com/docs/api-reference
- Assumption: exact Deepgram model, Groq model, and Cartesia voice ID will be configured via environment variables rather than locked into the PRD.
- Assumption: v1 prioritizes a stable browser MVP and latency observability over guaranteeing sub-1s response in every external API condition.
