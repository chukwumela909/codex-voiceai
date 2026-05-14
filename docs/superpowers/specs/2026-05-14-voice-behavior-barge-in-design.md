# Voice Behavior Barge-In Design

## Goal

Improve the voice experience without adding UI surface. The first slice should make the agent feel more interruptible and reliable by stopping spoken playback as soon as the browser detects likely user speech, while keeping the server as the source of truth for confirmed turn-taking.

## Scope

- Add client-local barge-in detection from microphone PCM levels while assistant audio is playing.
- Stop queued and active TTS playback immediately on likely local speech.
- Keep sending microphone frames to the server after local barge-in.
- Keep the existing server `interruption.started` event as authoritative confirmation.
- Separate frontend response text ownership from audio playback ownership.
- Harden provider failure paths that can break or truncate the live voice loop.

Out of scope:

- New visible UI controls or captions settings.
- A full voice activity detection library.
- Replacing Deepgram, Groq, or Cartesia.
- Redesigning proactive conversation behavior.

## Current Pipeline

The browser captures microphone audio with `getUserMedia`, converts it to PCM S16LE in `frontend/pcm-worklet.js`, batches frames in `frontend/app.js`, and sends binary chunks over `/ws/browser`. The server forwards live-mode audio to Deepgram, buffers transcript finalization, streams Groq text, and streams Cartesia or mock PCM audio back through `audio.chunk` events. The browser decodes each chunk and schedules playback with Web Audio.

Interruptions are currently server-confirmed: Deepgram/mock transcript logic decides whether new user speech should interrupt an active assistant response, then the server emits `interruption.started`; the browser responds by clearing playback. This is correct, but can feel slow because audio continues until the server-side signal returns.

## Proposed Behavior

When assistant audio is playing, the browser will monitor outgoing microphone PCM frames before sending them. If recent mic energy crosses a conservative speech threshold for a short sustained window, the browser will:

1. Stop active and queued playback immediately.
2. Mark the current audio response as locally interrupted.
3. Continue sending microphone frames to the server.
4. Ignore additional local barge-in triggers until server state or a new audio response resets the local interruption flag.

The server will still decide whether the speech is meaningful. If confirmed, it emits `interruption.started` and starts the next turn as it does today. If not confirmed, the user still benefits from avoiding assistant-over-user overlap; future chunks from the same locally interrupted response should be ignored to prevent the old answer from resuming over the user.

## Frontend State Changes

Replace the single `activeResponseId` role with separate concepts:

- `activeTextResponseId`: the response currently updating assistant text.
- `activeAudioResponseId`: the response currently owning playback.
- `locallyInterruptedAudioResponseIds`: response IDs whose playback was stopped by local barge-in.

`agent.text.delta` and `agent.text.final` update only `activeTextResponseId`. `audio.chunk` updates only `activeAudioResponseId`. If a chunk arrives for a response in `locallyInterruptedAudioResponseIds`, the browser drops it instead of scheduling playback.

This avoids text events accidentally protecting stale audio from being cleared when a new audio response begins.

## Local Barge-In Detection

The first implementation should use a simple deterministic RMS detector over PCM frames already produced by the worklet:

- Compute RMS and peak from each outbound PCM frame in `queueMicFrame`.
- Only evaluate local barge-in when there are active audio sources or queued playback.
- Require RMS over a configurable constant for a small consecutive frame count.
- Require a minimum peak to avoid low steady noise triggering interruption.
- Reset the counter when levels drop below threshold.

Initial constants should be conservative and local to `frontend/app.js`, for example:

- RMS threshold around `0.025`.
- Peak threshold around `0.08`.
- Consecutive speech frames around `3`.
- Cooldown around `700ms` before another local barge-in can fire.

These values should be easy to tune after real call testing.

## Server Hardening

The first slice should also close two reliability gaps found during review:

- Deepgram `send_audio` failures should not bubble out of `receive_audio` and terminate the browser WebSocket. They should call the existing Deepgram error/fallback path and continue in mock turn detection where possible.
- Cartesia errors after partial audio should not be treated as full success. The session should either fall back for remaining speech or clearly stop the response rather than silently truncating the spoken answer.

The narrow implementation can start by treating any Cartesia error as a failed TTS response, even if previous chunks were sent. This may produce a mock fallback after partial real audio, but it is preferable to silent truncation for the first reliability pass.

## Data Flow

```text
Browser mic frame
  -> compute local PCM levels
  -> if assistant audio active and sustained speech detected:
       clear local playback
       mark active audio response as locally interrupted
  -> send PCM frame to server
  -> server STT confirms or ignores speech
  -> server emits interruption.started when confirmed
  -> browser clears playback again idempotently
```

## Error Handling

Local barge-in must be idempotent. Calling `clearPlaybackQueue()` multiple times should be harmless. Dropping post-interruption chunks for the same response prevents old audio from restarting.

Provider failures should preserve the WebSocket session whenever possible. Deepgram send errors should produce provider error events and fallback state, not unhandled loop exceptions. Cartesia failures should produce an error event and fallback behavior rather than leaving the user with half a spoken answer and no indication.

## Testing

Server tests:

- Deepgram send failure falls back without ending the session.
- Cartesia error after one chunk still emits a fallback/error path and does not report clean `tts_done` success.
- Existing interruption tests continue to pass.

Frontend checks:

- Add small pure helper functions where practical for PCM RMS/peak and local barge-in state transitions.
- If no JavaScript test harness is introduced, keep frontend logic small and verify through manual browser testing with diagnostic events.

Manual verification:

- Start a mock session, let the assistant speak, speak over playback, and confirm playback stops immediately.
- Confirm audio frames continue reaching the server after local barge-in.
- Confirm server-confirmed `interruption.started` still appears for meaningful speech.
- Confirm a new assistant response can play normally after an interruption.

## Success Criteria

- User speech stops assistant playback locally before server transcript finalization.
- Confirmed server interruption behavior still works.
- Old audio chunks do not resume after local interruption.
- Live provider failures do not unnecessarily kill the browser session.
- The change does not add visible UI or require new user controls.
