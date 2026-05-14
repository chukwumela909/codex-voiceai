# Plan: Proactive Conversational Voice Agent

> Source PRD: PRD/proactive-conversational-agent.md

## Architectural Decisions

Durable decisions that apply across all phases:

- **Transport route**: Browser conversations continue to use `WS /ws/browser`; no new browser WebSocket route is needed for proactive behavior.
- **Health and contract routes**: `GET /health` should expose safe proactive configuration status, and `GET /events` should document proactive server events.
- **Runtime modes**: Proactive behavior must work in both `mock` and `live` modes when enabled. Mock/dev can enable it by default; live deployments should opt in explicitly.
- **Configuration**: Proactive behavior is controlled by backend environment settings. The frontend does not need a settings panel in the first version, but `.env.example` should document every setting.
- **Session model**: One WebSocket connection remains one conversation session. Proactive state belongs to the session and is cleaned up when the session closes.
- **Event schema**: All new server events follow the existing typed JSON envelope: `type`, `session_id`, `timestamp`, and `payload`.
- **Proactive event types**: The server should expose monitoring events for trigger, skip, cooldown, cancellation, and policy state changes.
- **Assistant turn shape**: A proactive assistant turn should flow through the same single active assistant response slot, assistant text, audio, response ID, status, latency, and interruption surfaces as a reactive turn.
- **Proactive trigger reasons**: Startup greeting, silence nudge, and contextual follow-up are distinct trigger reasons in emitted metadata.
- **Policy boundary**: A mostly deterministic backend policy decides when the agent may speak, which trigger reason applies, whether cooldown/backoff blocks the turn, and what metadata should be emitted. Session orchestration owns async timers and I/O.
- **Idle trigger model**: The system should use one idle trigger scheduler. At trigger time, policy chooses a generic nudge or contextual follow-up.
- **Readiness rule**: Startup greeting waits for audio readiness and at least one valid inbound audio frame.
- **Silence reset rule**: Silence resets on meaningful user speech or assistant-finished timing, not raw silent microphone frames.
- **Backoff rule**: The agent may emit at most three consecutive proactive prompts before entering backed-off state until the user speaks again.
- **Tone rule**: The agent should not ask "why did you go silent?" Nudges should become softer over repeats.
- **Transcript scope**: Contextual follow-ups use only current-session transcript and state. Persistent memory remains out of scope.
- **Transport compatibility**: The proactive policy should not depend on browser-only microphone or playback details, so a future Twilio transport can reuse it.
- **Interruption semantics**: User speech cancels pending proactive prompts and interrupts active proactive responses using the same rules as normal assistant responses.
- **Failure semantics**: Proactive LLM/TTS failures should not immediately retry the same proactive prompt. Repeated proactive provider failures should back off proactive behavior while preserving reactive turns.

---

## Phase 1: Config And Event Baseline

**User stories**: 16, 17, 18, 19, 21, 22, 28, 32

### What To Build

Add the backend configuration, minimal policy boundary, and event-contract foundation for proactive behavior without making the agent speak proactively yet. The app should report whether proactive behavior is enabled, expose timing/cooldown/backoff defaults safely, and document the proactive monitoring events that later phases will emit.

### Acceptance Criteria

- [ ] Backend config includes enablement, live opt-in behavior, startup greeting delay, silence timeout, repeat cooldown, max consecutive proactive prompts, failure backoff, and follow-up settings.
- [ ] `.env.example` documents all proactive settings.
- [ ] Safe config status reports proactive settings without exposing secrets or implementation internals.
- [ ] The event contract documents proactive trigger, skip, cooldown, cancellation, and state events.
- [ ] The frontend diagnostics HUD can display proactive monitoring events in the existing event log.
- [ ] A minimal proactive policy boundary can evaluate enabled/disabled state, eligibility, cooldown, and backoff without starting timers.
- [ ] Tests verify health/config reporting, event-contract documentation, policy eligibility, and test-friendly timing behavior.
- [ ] Tests avoid real five-second sleeps by using tiny configured durations or deterministic time inputs.
- [ ] No proactive speech is emitted in this phase.

---

## Phase 2: Startup Greeting

**User stories**: 1, 2, 3, 12, 15, 20, 24, 30

### What To Build

Create the first end-to-end proactive turn: after the browser session is connected, audio is configured, the session reaches a listening-ready state, and at least one valid audio frame arrives, the assistant speaks a short proactive startup greeting. Live greetings should be generated by the configured LLM and flow through the same response/audio pipeline as a normal assistant response.

### Acceptance Criteria

- [ ] The startup greeting fires only after the session can receive user speech and has received at least one valid inbound audio frame.
- [ ] The startup greeting fires exactly once per session.
- [ ] The startup greeting is time-of-day aware.
- [ ] The startup greeting uses scripted wording initially.
- [ ] The startup greeting does not mention microphone readiness in normal user-facing speech.
- [ ] User speech before the greeting fires cancels the pending greeting and starts a normal reactive turn.
- [ ] The startup greeting uses the normal assistant text and audio event flow.
- [ ] The greeting includes proactive metadata identifying `startup_greeting` as the trigger reason.
- [ ] Mock mode produces a deterministic greeting without provider keys.
- [ ] Live mode supports the greeting when proactive behavior is explicitly enabled.
- [ ] Pending greeting work is cancelled cleanly when the session closes.
- [ ] Tests verify the greeting timing, event metadata, mock behavior, and cleanup.

---

## Phase 3: Repeating Silence Nudges

**User stories**: 4, 5, 6, 7, 8, 18, 19, 25, 26, 27, 30

### What To Build

Add one idle trigger scheduler after the conversation is listening and idle. If the user remains silent for the configured timeout, defaulting to five seconds, policy chooses a generic silence nudge unless contextual follow-up becomes eligible in a later phase. If the user stays silent, proactive prompts repeat according to cooldown rules, become softer over repeats, and stop after the maximum consecutive count.

### Acceptance Criteria

- [ ] Idle monitoring starts only when the session is idle, listening, and the mic/audio stream is active.
- [ ] A silence nudge fires after the configured timeout.
- [ ] Silence resets on meaningful user speech or assistant-finished timing, not raw silent audio frames.
- [ ] Repeated nudges continue while the user remains silent and backoff has not been reached.
- [ ] Repeated nudges respect cooldown settings.
- [ ] Nudge wording varies across repeats while remaining persona-consistent.
- [ ] Nudge wording never asks "why did you go silent?"
- [ ] Nudge wording becomes softer over repeats.
- [ ] After three consecutive proactive prompts by default, the session enters backed-off state until the user speaks again.
- [ ] `audio.stop`, muted input, or missing audio readiness pauses proactive behavior.
- [ ] The policy skips nudges while the assistant is thinking, speaking, interrupted, processing speech, or closing.
- [ ] User speech cancels pending silence nudges before they fire and resets consecutive nudge count.
- [ ] Proactive monitoring events explain trigger, skip, cooldown, and cancellation decisions.
- [ ] Tests verify first nudge, repeated nudges, cooldown, skip states, user-speech cancellation, and session cleanup.

---

## Phase 4: Contextual Follow-Up Proactivity

**User stories**: 9, 10, 23, 24, 29, 31

### What To Build

Extend the idle trigger policy beyond generic silence recovery so the agent can ask a relevant follow-up question or offer a brief observation when recent conversation context makes that more useful than a generic nudge. This is not a second timer system; the same idle trigger asks the policy which proactive intent should fire.

### Acceptance Criteria

- [ ] The idle trigger can select `contextual_follow_up` instead of `silence_nudge` when recent meaningful context exists.
- [ ] The agent can initiate a relevant follow-up after a completed assistant response.
- [ ] Follow-ups use current-session conversation context only.
- [ ] Follow-up turns include proactive metadata identifying `contextual_follow_up` as the trigger reason.
- [ ] The policy avoids firing follow-ups when a silence nudge or normal response is already pending.
- [ ] Contextual follow-ups do not fire before the user has contributed meaningful context.
- [ ] Contextual follow-ups are skipped when the last assistant response already asked a question or provided a strong next step.
- [ ] Contextual follow-ups count toward the same consecutive proactive backoff limit as generic nudges.
- [ ] LLM-generated follow-ups are constrained to one concise, spoken-friendly question or observation.
- [ ] Follow-ups do not introduce unrelated new topics and do not ask why the user went silent.
- [ ] Mock mode has deterministic fallback prompts that still feel context-aware enough for local demos.
- [ ] Live mode passes the proactive intent and recent context into the normal persona-aware response generation path.
- [ ] Tests verify follow-up eligibility, metadata, fallback wording, provider-backed generation boundaries, and skip behavior.

---

## Phase 5: Interruption And Transport Hardening

**User stories**: 11, 13, 14, 26, 27, 28, 31

### What To Build

Harden proactive behavior so it behaves like a normal conversational turn under interruption, provider failures, disconnects, and future transport reuse. This phase focuses on proving that pending proactive work cancels cleanly, active proactive speech can be interrupted by the user, stale output is ignored, failures back off instead of retrying immediately, and the policy remains independent from browser-only details.

### Acceptance Criteria

- [ ] User speech during an active proactive response emits the same interruption behavior as user speech during a reactive response.
- [ ] Interrupted proactive responses do not append stale assistant turns after cancellation.
- [ ] Stale proactive audio is ignored or cleared using existing response ID behavior.
- [ ] Proactive timers and tasks are cancelled on session shutdown.
- [ ] Proactive LLM/TTS failures emit normal provider errors with proactive metadata.
- [ ] A proactive provider failure does not immediately retry the same proactive prompt.
- [ ] Repeated proactive provider failures enter proactive failure backoff while normal user-driven turns remain available.
- [ ] Reactive and proactive turns share one active assistant response slot.
- [ ] Proactive policy inputs are expressed in transport-neutral session state, not browser-only UI state.
- [ ] Browser behavior remains unchanged except for visible proactive text/audio and diagnostics.
- [ ] Tests cover interruption during proactive text generation, interruption during proactive TTS/playback, stale response handling, and shutdown cleanup.

---

## Phase 6: Tuning And Demo Readiness

**User stories**: 7, 8, 12, 14, 15, 21, 32

### What To Build

Tune the proactive behavior for conversational realism. This phase calibrates default timing, live opt-in defaults, repeat behavior, prompt tone, time-of-day greeting wording, failure/backoff behavior, and observability so the agent feels present without becoming annoying. It also updates developer-facing documentation so future changes know how to reason about proactive policy.

### Acceptance Criteria

- [ ] Default settings produce a startup greeting and five-second silence nudge in local mock mode.
- [ ] Live mode proactive behavior remains opt-in by configuration.
- [ ] Time-of-day greeting copy feels natural across morning, afternoon, evening, and late-night buckets.
- [ ] Repeated nudges feel varied and do not rapidly stack.
- [ ] Backoff behavior is easy to observe and tune.
- [ ] Proactive prompts preserve the configured persona.
- [ ] Diagnostics make it clear why a proactive prompt fired or skipped.
- [ ] Documentation describes proactive configuration, event types, backoff behavior, failure behavior, time-of-day greeting, and tuning guidance.
- [ ] Existing tests continue to pass alongside the new proactive test coverage.
