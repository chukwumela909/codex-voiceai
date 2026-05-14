# Voice Behavior Barge-In Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make assistant speech stop immediately when the user talks over it, while preserving server-confirmed interruption and improving live provider recovery.

**Architecture:** Add a small browser-side audio behavior helper for PCM level detection and local barge-in state. Wire it into the existing static frontend without adding UI. Harden the existing Python session pipeline so Deepgram send failures fall back cleanly and Cartesia partial failures do not look like successful TTS completion.

**Tech Stack:** FastAPI, asyncio, WebSocket PCM S16LE audio, Deepgram STT, Groq streaming LLM, Cartesia streaming TTS, vanilla browser JavaScript, Node built-in test runner through pytest.

---

## File Structure

- Create: `frontend/audio-behavior.js`
  - Pure browser-compatible helper for PCM RMS/peak calculation and local barge-in state transitions.
  - Exposes `window.VoiceAudioBehavior` in the browser and `globalThis.VoiceAudioBehavior` under Node.
- Create: `tests/js/audio-behavior.test.mjs`
  - Node built-in tests for the pure helper.
- Create: `tests/test_frontend_audio_behavior.py`
  - Pytest wrapper that runs the Node helper tests from the existing Python test suite.
- Modify: `frontend/index.html`
  - Load `audio-behavior.js` before `app.js`.
- Modify: `frontend/app.js`
  - Split text and audio response ownership.
  - Detect local barge-in from outbound mic PCM.
  - Drop late chunks for locally interrupted responses.
- Modify: `tests/test_phase1_app_shell.py`
  - Add a static integration check that the helper script is loaded and app state names are present.
- Modify: `tests/test_live_conversation.py`
  - Add server behavior tests for Deepgram send failure fallback and Cartesia partial-audio failure fallback.
- Modify: `app/mock_conversation.py`
  - Catch Deepgram `send_audio` failures inside the session pipeline.
  - Treat Cartesia chunk/error failures as failed TTS so fallback can run.

---

### Task 1: Add Pure Frontend Audio Behavior Helpers

**Files:**
- Create: `frontend/audio-behavior.js`
- Create: `tests/js/audio-behavior.test.mjs`
- Create: `tests/test_frontend_audio_behavior.py`

- [ ] **Step 1: Write the failing Node-backed helper tests**

Create `tests/js/audio-behavior.test.mjs`:

```javascript
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../../frontend/audio-behavior.js", import.meta.url), "utf8");
const context = {};
vm.createContext(context);
vm.runInContext(source, context);

const behavior = context.VoiceAudioBehavior;

function pcmBuffer(samples) {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  samples.forEach((sample, index) => {
    view.setInt16(index * 2, sample, true);
  });
  return buffer;
}

test("pcmLevelFromArrayBuffer reports rms and peak for signed 16-bit PCM", () => {
  const level = behavior.pcmLevelFromArrayBuffer(pcmBuffer([0, 32767, -32768, 0]));

  assert.equal(level.peak, 1);
  assert.equal(level.rms, 0.7071);
});

test("shouldTriggerLocalBargeIn requires active playback and sustained speech", () => {
  let state = { consecutiveSpeechFrames: 0, cooldownUntilMs: 0, playbackActive: true };
  const config = {
    rmsThreshold: 0.025,
    peakThreshold: 0.08,
    consecutiveFrames: 3,
    cooldownMs: 700,
  };
  const speechLevel = { rms: 0.04, peak: 0.12 };

  let result = behavior.shouldTriggerLocalBargeIn(speechLevel, state, config, 1000);
  assert.equal(result.triggered, false);
  state = result.state;

  result = behavior.shouldTriggerLocalBargeIn(speechLevel, state, config, 1010);
  assert.equal(result.triggered, false);
  state = result.state;

  result = behavior.shouldTriggerLocalBargeIn(speechLevel, state, config, 1020);
  assert.equal(result.triggered, true);
  assert.equal(result.state.consecutiveSpeechFrames, 0);
  assert.equal(result.state.cooldownUntilMs, 1720);
});

test("shouldTriggerLocalBargeIn resets speech count for quiet frames", () => {
  const state = { consecutiveSpeechFrames: 2, cooldownUntilMs: 0, playbackActive: true };
  const config = {
    rmsThreshold: 0.025,
    peakThreshold: 0.08,
    consecutiveFrames: 3,
    cooldownMs: 700,
  };

  const result = behavior.shouldTriggerLocalBargeIn({ rms: 0.01, peak: 0.04 }, state, config, 1000);

  assert.equal(result.triggered, false);
  assert.equal(result.state.consecutiveSpeechFrames, 0);
});

test("shouldTriggerLocalBargeIn does not trigger while playback is inactive", () => {
  const state = { consecutiveSpeechFrames: 3, cooldownUntilMs: 0, playbackActive: false };
  const config = {
    rmsThreshold: 0.025,
    peakThreshold: 0.08,
    consecutiveFrames: 3,
    cooldownMs: 700,
  };

  const result = behavior.shouldTriggerLocalBargeIn({ rms: 0.08, peak: 0.2 }, state, config, 1000);

  assert.equal(result.triggered, false);
  assert.equal(result.state.consecutiveSpeechFrames, 0);
});
```

Create `tests/test_frontend_audio_behavior.py`:

```python
import shutil
import subprocess
from pathlib import Path


def test_frontend_audio_behavior_helpers_pass_node_tests():
    node = shutil.which("node")
    assert node, "node is required to run frontend audio behavior tests"

    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [node, "--test", "tests/js/audio-behavior.test.mjs"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
```

- [ ] **Step 2: Run the failing helper test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_frontend_audio_behavior.py -v
```

Expected: FAIL because `frontend/audio-behavior.js` does not exist.

- [ ] **Step 3: Implement the pure helper**

Create `frontend/audio-behavior.js`:

```javascript
(function attachVoiceAudioBehavior(global) {
  const DEFAULT_BARGE_IN_CONFIG = {
    rmsThreshold: 0.025,
    peakThreshold: 0.08,
    consecutiveFrames: 3,
    cooldownMs: 700,
  };

  function roundLevel(value) {
    return Math.round(value * 10000) / 10000;
  }

  function pcmLevelFromArrayBuffer(buffer) {
    if (!buffer || buffer.byteLength < 2) return { rms: 0, peak: 0 };

    const view = new DataView(buffer);
    const sampleCount = Math.floor(buffer.byteLength / 2);
    let peak = 0;
    let sumSquares = 0;

    for (let index = 0; index < sampleCount; index += 1) {
      const sample = view.getInt16(index * 2, true);
      const abs = Math.abs(sample);
      peak = Math.max(peak, abs);
      sumSquares += sample * sample;
    }

    return {
      rms: roundLevel(Math.sqrt(sumSquares / sampleCount) / 32768),
      peak: roundLevel(peak / 32768),
    };
  }

  function shouldTriggerLocalBargeIn(level, state, config = DEFAULT_BARGE_IN_CONFIG, nowMs = 0) {
    const next = {
      consecutiveSpeechFrames: state.consecutiveSpeechFrames || 0,
      cooldownUntilMs: state.cooldownUntilMs || 0,
      playbackActive: state.playbackActive === true,
    };

    if (!next.playbackActive) {
      next.consecutiveSpeechFrames = 0;
      return { triggered: false, state: next };
    }

    if (next.cooldownUntilMs > nowMs) {
      return { triggered: false, state: next };
    }

    const speechDetected = level.rms >= config.rmsThreshold && level.peak >= config.peakThreshold;
    next.consecutiveSpeechFrames = speechDetected ? next.consecutiveSpeechFrames + 1 : 0;

    if (next.consecutiveSpeechFrames >= config.consecutiveFrames) {
      next.consecutiveSpeechFrames = 0;
      next.cooldownUntilMs = nowMs + config.cooldownMs;
      return { triggered: true, state: next };
    }

    return { triggered: false, state: next };
  }

  global.VoiceAudioBehavior = {
    DEFAULT_BARGE_IN_CONFIG,
    pcmLevelFromArrayBuffer,
    shouldTriggerLocalBargeIn,
  };
})(typeof window !== "undefined" ? window : globalThis);
```

- [ ] **Step 4: Run the helper test to verify it passes**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_frontend_audio_behavior.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit the helper**

Run:

```powershell
git add frontend/audio-behavior.js tests/js/audio-behavior.test.mjs tests/test_frontend_audio_behavior.py
git commit -m "Add frontend audio behavior helpers"
```

---

### Task 2: Wire Local Barge-In Into Browser Playback

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/app.js`
- Modify: `tests/test_phase1_app_shell.py`

- [ ] **Step 1: Write the failing static integration test**

Append this test to `tests/test_phase1_app_shell.py`:

```python
def test_frontend_wires_local_barge_in_audio_behavior():
    index = Path("frontend/index.html").read_text(encoding="utf-8")
    app_js = Path("frontend/app.js").read_text(encoding="utf-8")

    assert "/static/audio-behavior.js" in index
    assert "activeTextResponseId" in app_js
    assert "activeAudioResponseId" in app_js
    assert "locallyInterruptedAudioResponseIds" in app_js
    assert "evaluateLocalBargeIn" in app_js
    assert "VoiceAudioBehavior.pcmLevelFromArrayBuffer" in app_js
    assert "VoiceAudioBehavior.shouldTriggerLocalBargeIn" in app_js
```

- [ ] **Step 2: Run the failing integration test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase1_app_shell.py::test_frontend_wires_local_barge_in_audio_behavior -v
```

Expected: FAIL because the script is not loaded and the new state names are absent.

- [ ] **Step 3: Load the helper before the main app script**

In `frontend/index.html`, replace the existing script block at the bottom:

```html
    <script src="https://cdn.jsdelivr.net/npm/p5@1.10.0/lib/p5.min.js"></script>
    <script src="/static/app.js?v=phase6.2"></script>
```

with:

```html
    <script src="https://cdn.jsdelivr.net/npm/p5@1.10.0/lib/p5.min.js"></script>
    <script src="/static/audio-behavior.js?v=phase6.3"></script>
    <script src="/static/app.js?v=phase6.3"></script>
```

- [ ] **Step 4: Replace the single response id state in `frontend/app.js`**

Replace:

```javascript
let activeResponseId = null;
```

with:

```javascript
let activeTextResponseId = null;
let activeAudioResponseId = null;
let localBargeInState = { consecutiveSpeechFrames: 0, cooldownUntilMs: 0, playbackActive: false };
const locallyInterruptedAudioResponseIds = new Set();
```

- [ ] **Step 5: Add local barge-in helper functions in `frontend/app.js`**

Add these functions above `queueMicFrame`:

```javascript
function hasPlaybackActivity() {
  const queuedAheadSeconds = Math.max(0, playbackTime - (audioContext?.currentTime || 0));
  return activeSources.length > 0 || queuedAheadSeconds > 0.03;
}

function rememberLocalAudioInterruption(responseId) {
  if (!responseId) return;
  locallyInterruptedAudioResponseIds.add(responseId);
  while (locallyInterruptedAudioResponseIds.size > 20) {
    const oldest = locallyInterruptedAudioResponseIds.values().next().value;
    locallyInterruptedAudioResponseIds.delete(oldest);
  }
}

function resetLocalBargeInCounter() {
  localBargeInState = {
    consecutiveSpeechFrames: 0,
    cooldownUntilMs: localBargeInState.cooldownUntilMs || 0,
    playbackActive: false,
  };
}

function triggerLocalBargeIn() {
  rememberLocalAudioInterruption(activeAudioResponseId);
  clearPlaybackQueue();
  playbackState.textContent = "interrupted";
}

function evaluateLocalBargeIn(buffer) {
  if (!window.VoiceAudioBehavior) return;
  const playbackActive = hasPlaybackActivity();

  if (!playbackActive) {
    resetLocalBargeInCounter();
    return;
  }

  if (activeAudioResponseId && locallyInterruptedAudioResponseIds.has(activeAudioResponseId)) {
    return;
  }

  const level = window.VoiceAudioBehavior.pcmLevelFromArrayBuffer(buffer);
  const result = window.VoiceAudioBehavior.shouldTriggerLocalBargeIn(
    level,
    { ...localBargeInState, playbackActive },
    window.VoiceAudioBehavior.DEFAULT_BARGE_IN_CONFIG,
    performance.now(),
  );
  localBargeInState = result.state;

  if (result.triggered) {
    triggerLocalBargeIn();
  }
}
```

- [ ] **Step 6: Evaluate barge-in before batching/sending mic frames**

In `queueMicFrame`, replace:

```javascript
function queueMicFrame(buffer) {
  pendingMicBuffers.push(buffer);
```

with:

```javascript
function queueMicFrame(buffer) {
  evaluateLocalBargeIn(buffer);
  pendingMicBuffers.push(buffer);
```

- [ ] **Step 7: Split text response ownership**

In the `agent.text.delta` branch, replace:

```javascript
activeResponseId = event.payload.response_id;
```

with:

```javascript
activeTextResponseId = event.payload.response_id;
```

In the `agent.text.final` branch, replace:

```javascript
activeResponseId = event.payload.response_id;
```

with:

```javascript
activeTextResponseId = event.payload.response_id;
```

- [ ] **Step 8: Split audio response ownership and drop locally interrupted chunks**

At the start of `playPcmChunk`, replace:

```javascript
function playPcmChunk(payload) {
  if (!payload.audio || payload.encoding !== "pcm_s16le") return;

  if (payload.response_id && payload.response_id !== activeResponseId) {
    clearPlaybackQueue();
    activeResponseId = payload.response_id;
  }
```

with:

```javascript
function playPcmChunk(payload) {
  if (!payload.audio || payload.encoding !== "pcm_s16le") return;

  const responseId = payload.response_id || null;
  if (responseId && locallyInterruptedAudioResponseIds.has(responseId)) return;

  if (responseId && responseId !== activeAudioResponseId) {
    clearPlaybackQueue();
    activeAudioResponseId = responseId;
    resetLocalBargeInCounter();
  }
```

- [ ] **Step 9: Mark confirmed server interruptions as locally interrupted too**

In the `interruption.started` branch, replace:

```javascript
clearPlaybackQueue();
```

with:

```javascript
rememberLocalAudioInterruption(event.payload?.interrupted_response_id || activeAudioResponseId);
clearPlaybackQueue();
```

- [ ] **Step 10: Reset response ids on socket close**

In the WebSocket close handler, after:

```javascript
clearPlaybackQueue();
socket = null;
```

insert:

```javascript
activeTextResponseId = null;
activeAudioResponseId = null;
locallyInterruptedAudioResponseIds.clear();
resetLocalBargeInCounter();
```

- [ ] **Step 11: Run frontend behavior and static integration tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_frontend_audio_behavior.py tests/test_phase1_app_shell.py::test_frontend_wires_local_barge_in_audio_behavior -v
```

Expected: PASS.

- [ ] **Step 12: Commit browser wiring**

Run:

```powershell
git add frontend/index.html frontend/app.js tests/test_phase1_app_shell.py
git commit -m "Add local browser barge-in handling"
```

---

### Task 3: Keep Sessions Alive When Deepgram Audio Send Fails

**Files:**
- Modify: `tests/test_live_conversation.py`
- Modify: `app/mock_conversation.py`

- [ ] **Step 1: Write the failing Deepgram send failure test**

Add this test near the other Deepgram fallback tests in `tests/test_live_conversation.py`:

```python
def test_deepgram_send_failure_falls_back_without_closing_session(monkeypatch):
    events = []

    class SendFailingTranscriber:
        def __init__(self, **kwargs):
            self.closed = False

        async def start(self):
            pass

        async def send_audio(self, frame):
            raise RuntimeError("deepgram send closed")

        async def close(self):
            self.closed = True

    monkeypatch.setattr(mock_conversation, "DeepgramStreamingTranscriber", SendFailingTranscriber)

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(
                normalized_mode="live",
                proactive_effective_enabled=False,
                deepgram_api_key="deepgram-key",
                deepgram_model="nova-3",
                deepgram_endpointing_ms=300,
                deepgram_utterance_end_ms=1000,
                cartesia_api_key=None,
                cartesia_voice_id=None,
            ),
        )
        await session.configure_audio(
            {
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration_ms": 20,
            }
        )
        await session.receive_audio(mock_speech_frame())
        assert session.closed is False
        assert session.transcriber is None
        assert session.current_task is not None
        await session.current_task

    asyncio.run(run_session())

    assert any(event["type"] == "error" and event["payload"].get("provider") == "deepgram" for event in events)
    assert any(
        event["type"] == "pipeline.stage"
        and event["payload"].get("stage") == "stt_fallback"
        and event["payload"].get("fallback_from") == "deepgram"
        for event in events
    )
    assert any(
        event["type"] == "transcript.final" and event["payload"].get("provider") == "mock"
        for event in events
    )
```

- [ ] **Step 2: Run the failing Deepgram test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_live_conversation.py::test_deepgram_send_failure_falls_back_without_closing_session -v
```

Expected: FAIL because `receive_audio` lets `send_audio` exceptions escape.

- [ ] **Step 3: Catch Deepgram send failures and fall through to mock detection**

In `app/mock_conversation.py`, replace:

```python
        if self.transcriber:
            await self.transcriber.send_audio(processed_frame)
            return

        if not mock_speech_detected:
            return
```

with:

```python
        if self.transcriber:
            try:
                await self.transcriber.send_audio(processed_frame)
                return
            except Exception as exc:
                await self.handle_live_error(f"Deepgram audio send failed: {exc}")
                mock_speech_detected = is_mock_speech_frame(level)

        if not mock_speech_detected:
            return
```

- [ ] **Step 4: Run the Deepgram test to verify it passes**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_live_conversation.py::test_deepgram_send_failure_falls_back_without_closing_session -v
```

Expected: PASS.

- [ ] **Step 5: Commit Deepgram send fallback**

Run:

```powershell
git add app/mock_conversation.py tests/test_live_conversation.py
git commit -m "Handle Deepgram send failures without ending sessions"
```

---

### Task 4: Treat Cartesia Partial-Audio Errors As Failed TTS

**Files:**
- Modify: `tests/test_live_conversation.py`
- Modify: `app/mock_conversation.py`

- [ ] **Step 1: Write the failing Cartesia partial failure test**

Add this test near the Cartesia TTS tests in `tests/test_live_conversation.py`:

```python
def test_cartesia_error_after_audio_chunk_falls_back_to_mock_audio():
    events = []
    audio = base64.b64encode(b"\x00\x00" * 120).decode("ascii")

    class PartiallyFailingSynthesizer:
        async def stream_speech(self, transcript, *, context_id=None):
            yield {"type": "chunk", "audio": audio, "context_id": context_id}
            yield {"type": "error", "message": "cartesia stream failed", "context_id": context_id, "done": True}

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(
                normalized_mode="live",
                cartesia_api_key="cartesia-key",
                cartesia_voice_id=VALID_CARTESIA_VOICE_ID,
                cartesia_model="sonic-3",
                cartesia_sample_rate=16000,
                cartesia_version="2026-03-01",
            ),
        )
        session.synthesizer = PartiallyFailingSynthesizer()
        await session._stream_speech("resp_test", "Hello from Cartesia.", 0)

    asyncio.run(run_session())

    audio_providers = [event["payload"].get("provider") for event in events if event["type"] == "audio.chunk"]
    assert audio_providers == ["cartesia", "mock"]
    assert any(event["type"] == "error" and event["payload"].get("provider") == "cartesia" for event in events)
    assert any(
        event["type"] == "pipeline.stage"
        and event["payload"].get("stage") == "tts_fallback"
        and event["payload"].get("fallback_from") == "cartesia"
        for event in events
    )
```

- [ ] **Step 2: Run the failing Cartesia test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_live_conversation.py::test_cartesia_error_after_audio_chunk_falls_back_to_mock_audio -v
```

Expected: FAIL because Cartesia returns success after emitting any prior audio chunk.

- [ ] **Step 3: Return failure on Cartesia stream error even after chunks**

In `_stream_cartesia_speech`, replace:

```python
                if chunk["type"] == "error":
                    await self.send_provider_error("cartesia", chunk["message"], metadata=metadata)
                    return audio_chunks > 0
```

with:

```python
                if chunk["type"] == "error":
                    await self.send_provider_error("cartesia", chunk["message"], metadata=metadata)
                    return False
```

In `_stream_cartesia_speech_chunks`, replace:

```python
                if chunk["type"] == "error":
                    await self.send_provider_error("cartesia", chunk["message"], metadata=metadata)
                    return audio_chunks > 0
```

with:

```python
                if chunk["type"] == "error":
                    await self.send_provider_error("cartesia", chunk["message"], metadata=metadata)
                    return False
```

In `_stream_cartesia_speech`, replace:

```python
            await self.send_provider_error("cartesia", str(exc), metadata=metadata)
            return audio_chunks > 0
```

with:

```python
            await self.send_provider_error("cartesia", str(exc), metadata=metadata)
            return False
```

In `_stream_cartesia_speech_chunks`, replace:

```python
            await self.send_provider_error("cartesia", str(exc), metadata=metadata)
            return audio_chunks > 0
```

with:

```python
            await self.send_provider_error("cartesia", str(exc), metadata=metadata)
            return False
```

- [ ] **Step 4: Run the Cartesia test to verify it passes**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_live_conversation.py::test_cartesia_error_after_audio_chunk_falls_back_to_mock_audio -v
```

Expected: PASS.

- [ ] **Step 5: Run existing Cartesia-related tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cartesia_tts.py tests/test_live_conversation.py -k "cartesia or tts or audio" -v
```

Expected: PASS.

- [ ] **Step 6: Commit Cartesia partial failure fallback**

Run:

```powershell
git add app/mock_conversation.py tests/test_live_conversation.py
git commit -m "Fallback when Cartesia fails after partial audio"
```

---

### Task 5: Regression Test The Full Voice Pipeline

**Files:**
- Modify only if a previous task exposed a specific failing test.

- [ ] **Step 1: Run focused voice pipeline tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_live_conversation.py tests/test_deepgram_adapter.py tests/test_cartesia_tts.py tests/test_groq_agent.py tests/test_frontend_audio_behavior.py -v
```

Expected: PASS.

- [ ] **Step 2: Run the full Python suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Expected: PASS. If the existing `.pytest_cache` permission warning appears, record it in the final handoff but do not treat it as a failure when all tests pass.

- [ ] **Step 3: Start the local dev server**

Run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Expected: server starts and serves `http://127.0.0.1:8000`.

- [ ] **Step 4: Manually verify voice behavior in the browser**

Open:

```text
http://127.0.0.1:8000
```

Manual checks:

- Start a mock conversation.
- Let assistant audio begin.
- Speak over the assistant.
- Confirm playback stops immediately before a server transcript final arrives.
- Confirm microphone frames continue updating diagnostics.
- Confirm the server still emits `interruption.started` for meaningful speech.
- Confirm the next assistant response can play normally.

- [ ] **Step 5: Commit final verification notes if code changed during regression**

If no files changed during regression, do not create a commit.

If a regression fix was required, run:

```powershell
git add app frontend tests
git commit -m "Fix voice pipeline regression"
```

---

## Final Handoff Checklist

- [ ] `.\.venv\Scripts\python.exe -m pytest` passes.
- [ ] Local barge-in stops current and queued playback immediately.
- [ ] Confirmed server interruption still clears playback idempotently.
- [ ] Old audio chunks for locally interrupted response IDs are dropped.
- [ ] Deepgram send failures emit provider error/fallback events and do not close the session.
- [ ] Cartesia partial failures produce an error and fallback instead of silently truncating speech.
- [ ] No visible UI controls were added.
