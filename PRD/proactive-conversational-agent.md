# Proactive Conversational Voice Agent PRD

## Problem Statement

The current voice agent is reactive: it waits for user speech, transcribes it, generates a response, and speaks back. That proves the voice loop, but it does not yet feel like a realistic conversational partner. Real conversations include initiative: someone says hello when the conversation starts, checks in when the other person goes quiet, asks natural follow-up questions, and sometimes offers a relevant thought based on what was just discussed.

You want the agent to feel more alive and socially present without becoming noisy, interruptive, or unpredictable. The agent should be able to greet the user after the system is sure the user can speak, nudge after silence, repeat nudges if the user stays quiet, back off after several unanswered attempts, and proactively ask context-aware follow-up questions that keep the conversation moving.

## Solution

Add a proactive conversation layer to the voice agent. This layer should run inside the session lifecycle and decide when the assistant may initiate a turn without waiting for a new user utterance.

The feature will support three initial proactive behaviors:

1. Time-of-day startup greeting after the backend is listening and the user has proven they can send audio.
2. Idle recovery after roughly five seconds of user silence.
3. Context-aware follow-up prompts selected by the same idle trigger system when recent conversation context makes them useful.

The proactive layer should be available in both mock and live modes. Mock/dev can enable proactive behavior by default, while live deployments should opt in explicitly because this is a major behavior change. The feature should preserve the existing real-time voice pipeline, interruption behavior, event schema style, and future Twilio compatibility. Proactive turns should look like normal assistant turns to the frontend, with additional metadata for debugging and monitoring.

## User Stories

1. As a user, I want the agent to greet me once the conversation is ready, so that the experience starts naturally instead of silently waiting.
2. As a user, I want the startup greeting to reflect the time of day, so that it feels more socially natural.
3. As a user, I want the greeting to happen only after my microphone/session is truly sending audio, so that the agent does not speak before I can answer.
4. As a user, I want the greeting to happen unless I start speaking first, so that I remain in control of the first turn.
5. As a user, I want the agent to notice when I go quiet, so that the conversation does not feel abandoned.
6. As a user, I want the agent to ask an idle nudge after about five seconds, so that the pause feels acknowledged quickly.
7. As a user, I want silence nudges to repeat if I remain quiet, so that the agent continues to feel present.
8. As a user, I want repeated nudges to become softer and eventually back off, so that the agent does not feel needy.
9. As a user, I want repeated nudges to vary naturally, so that the agent does not sound robotic.
10. As a user, I do not want the agent to ask "why did you go silent?", so that the nudge does not feel accusatory.
11. As a user, I want the agent to ask relevant follow-up questions, so that the conversation feels more realistic and guided.
12. As a user, I want proactive prompts to connect to what we were discussing, so that the agent's initiative feels contextual rather than random.
13. As a user, I want the agent to sometimes offer a brief relevant observation instead of always asking a question, so that the conversation feels less mechanical.
14. As a user, I want proactive follow-ups not to stack questions after the agent already asked one, so that I am not overloaded.
15. As a user, I want the agent to stop or cancel a proactive prompt if I start speaking, so that I remain in control of the conversation.
16. As a user, I want proactive speech to sound like the same persona as normal responses, so that the experience feels consistent.
17. As a user, I want proactive behavior to avoid talking over me, so that it does not feel intrusive.
18. As a user, I want the agent to continue listening after a proactive prompt, so that I can respond naturally.
19. As a user, I want the agent to behave consistently in mock and live modes when enabled, so that local development reflects the intended live experience.
20. As a developer, I want proactive behavior controlled by backend environment configuration, so that it can be tuned without frontend changes.
21. As a developer, I want live proactive behavior to be opt-in by configuration, so that deployed agents do not unexpectedly change behavior.
22. As a developer, I want the startup greeting delay to be configurable, so that I can tune it for browser and future phone-call transports.
23. As a developer, I want the silence timeout to be configurable, so that I can adjust the agent's conversational assertiveness.
24. As a developer, I want repeated proactive prompts to have configurable cooldown and max-count behavior, so that I can prevent overly rapid or endless prompts.
25. As a developer, I want audio stop or muted input to pause proactive behavior, so that the agent does not speak when the user cannot naturally answer.
26. As a developer, I want proactive prompts to be represented as first-class conversation turns, so that transcripts, response IDs, audio playback, and interruptions stay coherent.
27. As a developer, I want proactive trigger and skip events in the client HUD, so that I can debug why the agent did or did not speak.
28. As a developer, I want proactive events to include reason metadata, so that I can distinguish startup greeting, silence nudge, and contextual follow-up behavior.
29. As a developer, I want proactive prompts to use a hybrid of deterministic policy and persona-aware wording, so that timing is reliable while language remains natural.
30. As a developer, I want scripted fallback proactive prompts, so that mock mode and provider-failure cases remain demoable.
31. As a developer, I want the proactive policy to know whether the agent is listening, thinking, speaking, interrupted, muted, backed off, or closing, so that it only speaks at safe moments.
32. As a developer, I want proactive timers to cancel cleanly when user speech starts, so that stale prompts never fire after the user has already resumed.
33. As a developer, I want proactive timers to cancel cleanly on session shutdown, so that background tasks do not leak.
34. As a developer, I want proactive behavior to preserve future Twilio compatibility, so that phone calls can use the same policy with a different transport.
35. As a developer, I want proactive prompts to reuse the existing audio and response streaming pipeline, so that the feature does not fork the conversation architecture.
36. As a developer, I want proactive provider failures to trigger backoff instead of immediate retries, so that failures do not cause loops.
37. As a developer, I want tests for startup greeting, silence nudges, repeated nudges, follow-ups, cancellation, failure backoff, and event emission, so that the behavior remains stable as the voice loop evolves.
38. As a developer, I want proactive logic tested through external events and session behavior, so that tests do not depend on private timer implementation details.
39. As a developer, I want the browser HUD to show proactive monitoring data, so that I can tune realism versus intrusiveness during live demos.

## Implementation Decisions

- Proactive behavior will be a backend-owned session capability, not a frontend timer.
- Proactive behavior will be available in both mock and live runtime modes.
- Proactive behavior can be enabled by default for mock/dev, but live deployments must opt in explicitly through backend configuration.
- Proactive behavior will be configurable through backend environment variables only for the first version.
- A small proactive policy boundary should decide eligibility, trigger reason, cooldown, max counts, and backoff; session orchestration should own async timers and I/O.
- The proactive policy should be mostly deterministic and testable with explicit time/state inputs.
- The first proactive startup greeting will fire after the session is connected, audio has been configured, the system has entered a listening-ready state, and at least one valid audio frame has arrived.
- If the user starts speaking before the startup greeting fires, user speech wins and the greeting is cancelled.
- Startup greeting wording should be scripted initially, short, warm, and time-of-day aware.
- The startup greeting should not mention microphone readiness unless debugging.
- The proactive scheduler should use one idle trigger system. At each idle trigger, policy chooses whether to emit a generic silence nudge or a contextual follow-up.
- Idle triggers should use a five-second default timeout.
- Silence should reset on meaningful user speech or assistant-finished timing, not on raw silent microphone frames.
- Audio stop, muted input, missing audio frames, session shutdown, active response work, provider failure, and user speech should pause or cancel proactive work as appropriate.
- Repeated proactive prompts should use cooldown behavior and a maximum of three consecutive prompts before entering backed-off state.
- The nudge count should reset after the user speaks.
- The agent should not ask "why did you go silent?"
- Nudge wording should become softer over repeats and eventually back off calmly.
- Contextual follow-ups require recent meaningful conversation context and should not fire before the user has contributed context.
- Contextual follow-ups should be skipped if the last assistant response already asked a question or provided a strong next step.
- Contextual follow-up LLM generation should be concise, spoken-friendly, and constrained to one brief question or observation.
- Proactive wording will use a hybrid approach: deterministic policy chooses whether and why to speak; persona-aware response generation chooses contextual follow-up wording when useful; scripted fallback prompts keep mock mode and provider-failure cases reliable.
- Greetings and silence nudges should use scripted or templated wording first for speed and reliability.
- Context-aware follow-ups should use the current session transcript only; persistent memory remains out of scope.
- Proactive turns should use the same single active assistant response slot as reactive turns.
- Proactive turns should use the same assistant response path as reactive turns whenever possible, including response IDs, streamed text, streamed audio, latency, interruption, and transcript behavior.
- Proactive turns should add metadata that identifies the proactive trigger reason without requiring a separate frontend rendering path.
- Proactive metadata should appear on assistant text and audio events, not only separate debug events.
- User speech during proactive assistant playback should use the same interruption semantics as speech during any other assistant response.
- Proactive provider failures should emit normal provider errors with proactive metadata, avoid immediate retries, and back off after repeated failures.
- Proactive policy should not fire while the assistant is already thinking, speaking, synthesizing audio, processing a user turn, interrupted, muted, backed off, or closing the session.
- The browser HUD should display proactive trigger, skip, cooldown, cancellation, and state events for debugging.
- The event contract should preserve the existing typed JSON style.
- The design should avoid browser-specific assumptions in the proactive policy so a future Twilio transport can reuse the same session-level behavior.
- The first version will not add a frontend settings panel; tuning happens through backend configuration.

## Testing Decisions

- Tests should assert public behavior through emitted events, transcript state, response metadata, and cancellation outcomes.
- Tests should avoid coupling to private timer implementation details.
- The proactive policy should have focused unit tests for eligibility decisions, cooldown, repeat behavior, trigger selection, max-count backoff, and failure backoff.
- Tests should avoid real five-second sleeps by using tiny test-configured durations or a testable clock/scheduler.
- The session orchestration should have async behavior tests for startup greeting after listening readiness.
- The session orchestration should test that startup greeting waits for at least one valid audio frame.
- The session orchestration should test that user speech cancels a pending startup greeting.
- The session orchestration should have async behavior tests for a silence nudge after the configured timeout.
- The session orchestration should test repeated silence nudges while the user remains quiet.
- The session orchestration should test max consecutive proactive prompts and backed-off state.
- The session orchestration should test that silence does not reset on raw silent audio frames alone.
- The session orchestration should test cancellation of pending proactive prompts when user speech arrives.
- The session orchestration should test that audio stop pauses proactive behavior.
- The session orchestration should test interruption of an active proactive response when user speech arrives.
- The session orchestration should test contextual follow-up prompts after a normal assistant response.
- The session orchestration should test that contextual follow-ups do not fire when the last assistant turn already asked a question.
- The session orchestration should test proactive provider failure backoff.
- The event contract tests should verify proactive monitoring event types are documented.
- Existing WebSocket and live-conversation tests are the closest prior art and should guide the new tests.
- Mock mode tests should verify deterministic fallback wording without provider keys.
- Live mode tests should mock provider adapters and verify that proactive turns still flow through normal agent and TTS boundaries.

## Out of Scope

- Persistent long-term memory across sessions.
- Frontend controls for enabling, disabling, or tuning proactive behavior.
- User-specific proactive personality preferences.
- Analytics dashboards beyond structured session events and HUD visibility.
- New authentication or account-level settings.
- Production Twilio integration in this feature.
- Database storage of proactive turn history.
- A fully autonomous goal-planning agent that changes topics without conversation context.
- Proactive messages that interrupt active user speech.
- Browser UI controls for changing proactive settings during a session.
- Phrase-level "hold on" or "give me a second" pause heuristics in the first implementation.

## Further Notes

- This feature is about conversational realism, not only silence detection.
- The first implementation should be conservative about timing and aggressive about observability.
- The policy should make it easy to tune the agent between "quiet companion" and "highly engaged conversation partner" through backend configuration.
- The key user experience risk is annoyance from over-speaking; the key technical risk is stale timers firing after session state has changed.
- A strong implementation will make proactive turns feel like normal assistant turns with a clear reason attached, rather than a parallel notification system.
- The highest-priority safety rail is avoiding speech over the user. The second-highest is avoiding repeated prompts that feel desperate.
