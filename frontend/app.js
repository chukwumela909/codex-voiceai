// ============================================================
// Codex Voice Agent — frontend (phase 6.2)
// Live-diagnostics theater, bottom-right particle-aura FAB,
// glass chat tiles. Uses p5.js (instance mode) for the aura.
// ============================================================

// ---- DOM refs ----
const connectBtn = document.querySelector("#connectBtn");
const disconnectBtn = document.querySelector("#disconnectBtn");
const micBtn = document.querySelector("#micBtn");

const talkBtn = document.querySelector("#talkBtn");
const stageHint = document.querySelector("#stageHint");

const diagnosticsBtn = document.querySelector("#diagnosticsBtn");
const diagnosticsCloseBtn = document.querySelector("#diagnosticsCloseBtn");
const diagnostics = document.querySelector("#diagnostics");
const diagnosticsScrim = document.querySelector("#diagnosticsScrim");
const eventBadge = document.querySelector("#eventBadge");

const connectionChip = document.querySelector("#connectionChip");
const connectionState = document.querySelector("#connectionState");
const sessionId = document.querySelector("#sessionId");
const mode = document.querySelector("#mode");
const micState = document.querySelector("#micState");
const playbackState = document.querySelector("#playbackState");
const latencyValue = document.querySelector("#latencyValue");
const micFrames = document.querySelector("#micFrames");
const micBytes = document.querySelector("#micBytes");
const transcriberState = document.querySelector("#transcriberState");
const micLevel = document.querySelector("#micLevel");
const sampleRate = document.querySelector("#sampleRate");
const frameBytes = document.querySelector("#frameBytes");
const pipelineStage = document.querySelector("#pipelineStage");
const providerStage = document.querySelector("#providerStage");
const lastError = document.querySelector("#lastError");
const lastErrorMetric = lastError?.closest(".metric");
const audioChunks = document.querySelector("#audioChunks");
const outputLevel = document.querySelector("#outputLevel");
const audioContextState = document.querySelector("#audioContextState");
const transcriptText = document.querySelector("#transcriptText");
const userTurn = document.querySelector("#userTurn");
const agentText = document.querySelector("#agentText");
const agentTurn = document.querySelector("#agentTurn");
const latestEvent = document.querySelector("#latestEvent");
const eventLog = document.querySelector("#eventLog");
const meterEl = document.querySelector("#meter");
const orbEl = document.querySelector("#voiceOrb");

// ---- WebSocket / audio state ----
let socket = null;
let micStream = null;
let audioContext = null;
let micSource = null;
let processor = null;
let silentMicGain = null;
let pendingMicBuffers = [];
let pendingMicBytes = 0;
let targetMicFrameBytes = 640;
let playbackTime = 0;
let activeSources = [];
let activeTextResponseId = null;
let activeAudioResponseId = null;
let localBargeInState = { consecutiveSpeechFrames: 0, cooldownUntilMs: 0, playbackActive: false };
const locallyInterruptedAudioResponseIds = new Set();
let ambienceConfig = { enabled: true, scene: "room_line", volume: 0.035 };
let ambienceNodes = null;

// ---- High-level UX state ----
let appState = "idle";
let pendingMicAfterConnect = false;
let unseenEventCount = 0;

// ---- Meter / orb-pulse animation ----
const METER_BARS = 32;
const METER_FALLOFF = 0.86;
const meterBuffer = new Array(METER_BARS).fill(0);
let meterPointer = 0;
const meterBarEls = [];

let currentOrbPulse = 0;
let targetOrbPulse = 0;
let lastSampleAt = 0;
let orbAnimationFrame = null;
const REDUCED_MOTION = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

const PIPELINE_LABELS = {
  idle: "idle",
  disconnected: "offline",
  connecting: "connecting",
  connected: "ready",
  client_ready: "ready",
  transcriber_connected: "ready",
  ready: "ready",
  listening: "listening",
  hearing: "listening",
  thinking: "thinking",
  speaking: "speaking",
  error: "error"
};

function wsUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws/browser`;
}

// ============================================================
// Meter
// ============================================================

function initMeter() {
  if (!meterEl) return;
  for (let i = 0; i < METER_BARS; i += 1) {
    const span = document.createElement("span");
    span.className = "meter__bar";
    span.style.setProperty("--bar-h", "0.06");
    meterEl.appendChild(span);
    meterBarEls.push(span);
  }
}

function pushMeterSample(rms) {
  meterBuffer[meterPointer] = rms;
  meterPointer = (meterPointer + 1) % METER_BARS;
  lastSampleAt = performance.now();
}

function renderMeter() {
  for (let i = 0; i < METER_BARS; i += 1) {
    const idx = (meterPointer + i) % METER_BARS;
    const v = meterBuffer[idx];
    const h = Math.max(0.06, Math.min(1, v * 8));
    meterBarEls[i]?.style.setProperty("--bar-h", h.toFixed(3));
  }
}

// ============================================================
// App-state machine
// ============================================================

function setAppState(next) {
  appState = next;
  document.body.dataset.appState = next;

  switch (next) {
    case "idle":
      if (talkBtn) talkBtn.setAttribute("aria-label", "Start conversation");
      if (stageHint) stageHint.textContent = "Tap to begin";
      break;
    case "connecting":
      if (talkBtn) talkBtn.setAttribute("aria-label", "Cancel connection");
      if (stageHint) stageHint.textContent = "Connecting…";
      break;
    case "active":
      if (talkBtn) talkBtn.setAttribute("aria-label", "End conversation");
      if (stageHint) stageHint.textContent = "Live · tap to end";
      break;
    case "ending":
      if (talkBtn) talkBtn.setAttribute("aria-label", "Ending");
      if (stageHint) stageHint.textContent = "Ending…";
      break;
  }
}

function setPipelineState(state) {
  if (!state) return;
  document.body.dataset.pipeline = state;
  if (connectionChip) connectionChip.dataset.state = state;
  if (pipelineStage) pipelineStage.textContent = PIPELINE_LABELS[state] ?? state;
}

function setConnectionState(state) {
  connectionState.textContent = state;
  if (connectionChip) connectionChip.dataset.state = state;

  connectBtn.disabled = state === "connected" || state === "connecting";
  disconnectBtn.disabled = state !== "connected";
  micBtn.disabled = state !== "connected" || !socket;
}

function setMicState(state) {
  micState.textContent = state;
  micBtn.textContent = state === "streaming" ? "Stop mic" : "Start mic";
}

// ============================================================
// Event handler
// ============================================================

function recordEvent(event) {
  latestEvent.textContent = JSON.stringify(event, null, 2);

  if (event.session_id) sessionId.textContent = event.session_id;

  if (event.type === "session.started" && event.payload?.mode) {
    mode.textContent = event.payload.mode;
    if (event.payload.ambience) {
      configureAmbience(event.payload.ambience);
    }
  }

  if (event.type === "status.changed" && event.payload?.state) {
    const state = event.payload.state;
    const liveStates = ["connected", "client_ready", "transcriber_connected", "listening", "hearing", "thinking", "speaking"];
    if (liveStates.includes(state)) {
      setConnectionState("connected");
    } else {
      setConnectionState(state);
    }
    setPipelineState(state);

    if (state === "client_ready") {
      latestEvent.dataset.clientReady = "true";
      if (pendingMicAfterConnect) {
        pendingMicAfterConnect = false;
        toggleMic().catch(handleMicError);
      }
      if (appState === "connecting") setAppState("active");
    }
    if (state === "transcriber_connected") transcriberState.textContent = "deepgram";
    if (state === "speaking") {
      playbackState.textContent = "speaking";
      if (agentTurn) agentTurn.hidden = false;
    }
    if (state === "listening") {
      playbackState.textContent = "idle";
      transcriberState.textContent = "listening";
    }
    if (state === "hearing") transcriberState.textContent = "hearing";
  }

  if (event.type === "audio.input") {
    micFrames.textContent = event.payload.frames_received || 0;
    micBytes.textContent = event.payload.bytes_received || 0;
    micLevel.textContent = `${event.payload.rms ?? 0}`;
    sampleRate.textContent = event.payload.sample_rate || "unknown";
    frameBytes.textContent = event.payload.frame_bytes || 0;

    const rms = Number(event.payload.rms ?? 0);
    targetOrbPulse = Math.min(1, Math.max(targetOrbPulse, rms * 6));
    pushMeterSample(rms);
  }

  if (event.type === "pipeline.stage") {
    providerStage.textContent = event.payload.provider || "none";
    if (event.payload.message) lastError.textContent = event.payload.message;
  }

  if (event.type === "error") {
    const message = `${event.payload.provider || "app"}: ${event.payload.message || "Unknown error"}`;
    lastError.textContent = message;
    if (lastErrorMetric) lastErrorMetric.dataset.error = "true";
    setPipelineState("error");
  }

  if (event.type === "interruption.started") {
    rememberLocalAudioInterruption(event.payload?.interrupted_response_id || activeAudioResponseId);
    clearPlaybackQueue();
    playbackState.textContent = "interrupted";
    setPipelineState("listening");
    if (event.payload?.next_user_text) {
      transcriptText.textContent = event.payload.next_user_text;
      if (userTurn) userTurn.hidden = false;
    }
  }

  if (event.type === "transcript.partial" || event.type === "transcript.final") {
    const text = event.payload.text || "";
    transcriptText.textContent = text || "Listening…";
    if (userTurn && text) userTurn.hidden = false;
  }

  if (event.type === "agent.text.delta") {
    activeTextResponseId = event.payload.response_id;
    const text = event.payload.text_so_far || event.payload.text || "";
    agentText.textContent = text;
    if (agentTurn && text) agentTurn.hidden = false;
  }

  if (event.type === "agent.text.final") {
    activeTextResponseId = event.payload.response_id;
    const text = event.payload.text || "";
    agentText.textContent = text;
    if (agentTurn && text) agentTurn.hidden = false;
  }

  if (event.type === "latency.metric") {
    latencyValue.textContent = `${event.payload.value_ms}ms`;
  }

  if (event.type === "audio.chunk") playPcmChunk(event.payload);

  const item = document.createElement("li");
  item.textContent = `${new Date().toLocaleTimeString()} ${event.type}`;
  eventLog.prepend(item);
  if (eventLog.children.length > 80) eventLog.removeChild(eventLog.lastChild);

  if (diagnostics?.dataset.open !== "true") {
    unseenEventCount += 1;
    updateEventBadge();
  }
}

function updateEventBadge() {
  if (!eventBadge) return;
  if (unseenEventCount <= 0) {
    eventBadge.hidden = true;
    eventBadge.textContent = "0";
    return;
  }
  eventBadge.hidden = false;
  eventBadge.textContent = unseenEventCount > 99 ? "99+" : String(unseenEventCount);
}

// ============================================================
// WebSocket lifecycle
// ============================================================

function connect() {
  setConnectionState("connecting");
  setPipelineState("connecting");
  socket = new WebSocket(wsUrl());

  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({
      type: "client.hello",
      payload: { client: "browser-codex-ui", version: "phase-6" }
    }));
  });

  socket.addEventListener("message", (message) => {
    try {
      recordEvent(JSON.parse(message.data));
    } catch (error) {
      recordEvent({
        type: "error",
        payload: { message: "Received non-JSON message from server.", raw: String(message.data) }
      });
    }
  });

  socket.addEventListener("close", (event) => {
    const wasExpected = appState === "ending" || event.code === 1000;
    const closeMessage = event.reason || "No close reason provided.";

    recordEvent({
      type: "socket.closed",
      payload: {
        code: event.code,
        reason: closeMessage,
        wasClean: event.wasClean,
        expected: wasExpected
      }
    });

    setConnectionState("disconnected");
    stopMic();
    clearPlaybackQueue();
    socket = null;
    activeTextResponseId = null;
    activeAudioResponseId = null;
    locallyInterruptedAudioResponseIds.clear();
    resetLocalBargeInCounter();
    pendingMicAfterConnect = false;

    if (wasExpected) {
      setPipelineState("idle");
      setAppState("idle");
      return;
    }

    const message = `Connection dropped (${event.code || "unknown"}): ${closeMessage}`;
    lastError.textContent = message;
    if (lastErrorMetric) lastErrorMetric.dataset.error = "true";
    setPipelineState("error");
    setAppState("idle");
    if (stageHint) stageHint.textContent = "Connection dropped";
  });

  socket.addEventListener("error", () => {
    recordEvent({ type: "error", payload: { message: "WebSocket error. Waiting for close details." } });
  });
}

function disconnect() {
  if (!socket) return;
  stopMic();
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "session.stop", payload: {} }));
  }
  socket.close();
}

// ============================================================
// Local ambience bed
// ============================================================

function configureAmbience(config) {
  const nextVolume = Number(config.volume ?? ambienceConfig.volume);
  ambienceConfig = {
    enabled: config.enabled !== false,
    scene: typeof config.scene === "string" && config.scene.trim() ? config.scene.trim() : "room_line",
    volume: Number.isFinite(nextVolume) ? Math.max(0, Math.min(0.2, nextVolume)) : 0.035
  };

  if (!ambienceConfig.enabled) {
    stopAmbience();
    return;
  }

  if (ambienceNodes?.master) {
    const now = ambienceNodes.context.currentTime;
    ambienceNodes.master.gain.cancelScheduledValues(now);
    ambienceNodes.master.gain.setTargetAtTime(ambienceConfig.volume, now, 0.25);
  } else if (micStream && audioContext) {
    startAmbience();
  }
}

function ambienceSeedForScene(scene) {
  let seed = 2166136261;
  for (let i = 0; i < scene.length; i += 1) {
    seed ^= scene.charCodeAt(i);
    seed = Math.imul(seed, 16777619);
  }
  return seed >>> 0;
}

function createAmbienceBuffer(context, scene) {
  const durationSeconds = 2.5;
  const length = Math.max(1, Math.floor(context.sampleRate * durationSeconds));
  const buffer = context.createBuffer(1, length, context.sampleRate);
  const channel = buffer.getChannelData(0);
  let state = ambienceSeedForScene(scene);
  let room = 0;

  for (let i = 0; i < length; i += 1) {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    const white = (state / 4294967295) * 2 - 1;
    room = room * 0.88 + white * 0.12;
    channel[i] = room * 0.5 + white * 0.035;
  }

  return buffer;
}

function startAmbience() {
  const context = audioContext;
  if (!context || ambienceNodes || !ambienceConfig.enabled || ambienceConfig.volume <= 0) return;
  if (context.state === "suspended") context.resume();

  const source = context.createBufferSource();
  const roomHighpass = context.createBiquadFilter();
  const roomLowpass = context.createBiquadFilter();
  const lineBandpass = context.createBiquadFilter();
  const lineGain = context.createGain();
  const master = context.createGain();
  const now = context.currentTime;

  source.buffer = createAmbienceBuffer(context, ambienceConfig.scene);
  source.loop = true;

  roomHighpass.type = "highpass";
  roomHighpass.frequency.value = 70;
  roomHighpass.Q.value = 0.5;

  roomLowpass.type = "lowpass";
  roomLowpass.frequency.value = 950;
  roomLowpass.Q.value = 0.7;

  lineBandpass.type = "bandpass";
  lineBandpass.frequency.value = 1750;
  lineBandpass.Q.value = 2.4;

  lineGain.gain.value = 0.22;
  master.gain.setValueAtTime(0, now);
  master.gain.linearRampToValueAtTime(ambienceConfig.volume, now + 1.2);

  source.connect(roomHighpass);
  roomHighpass.connect(roomLowpass);
  roomLowpass.connect(master);
  source.connect(lineBandpass);
  lineBandpass.connect(lineGain);
  lineGain.connect(master);
  master.connect(context.destination);
  source.start(now);

  ambienceNodes = {
    context,
    source,
    roomHighpass,
    roomLowpass,
    lineBandpass,
    lineGain,
    master
  };
}

function stopAmbience() {
  if (!ambienceNodes) return;
  const nodes = ambienceNodes;
  ambienceNodes = null;

  const now = nodes.context.currentTime;
  nodes.master.gain.cancelScheduledValues(now);
  nodes.master.gain.setTargetAtTime(0, now, 0.12);

  try {
    nodes.source.stop(now + 0.5);
  } catch {
    /* already stopped */
  }

  window.setTimeout(() => {
    Object.values(nodes).forEach((node) => {
      if (node && typeof node.disconnect === "function") {
        try { node.disconnect(); } catch { /* already disconnected */ }
      }
    });
  }, 650);
}

// ============================================================
// Microphone / capture
// ============================================================

async function toggleMic() {
  if (micStream) {
    stopMic();
    return;
  }

  if (!socket || socket.readyState !== WebSocket.OPEN) return;

  audioContext = new AudioContext({ sampleRate: 16000 });
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true
    }
  });
  micSource = audioContext.createMediaStreamSource(micStream);
  await audioContext.audioWorklet.addModule("/static/pcm-worklet.js?v=phase6.2");
  processor = new AudioWorkletNode(audioContext, "pcm-capture-processor");
  silentMicGain = audioContext.createGain();
  silentMicGain.gain.value = 0;

  socket.send(JSON.stringify({
    type: "audio.start",
    payload: {
      encoding: "pcm_s16le",
      sample_rate: audioContext.sampleRate,
      channels: 1,
      frame_duration_ms: Math.round(1000 * 1024 / audioContext.sampleRate)
    }
  }));
  targetMicFrameBytes = Math.max(640, Math.round(audioContext.sampleRate * 0.02 * 2));

  processor.port.onmessage = (event) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    queueMicFrame(event.data);
  };

  micSource.connect(processor);
  processor.connect(silentMicGain);
  silentMicGain.connect(audioContext.destination);
  startAmbience();
  setMicState("streaming");
}

function stopMic() {
  stopAmbience();

  if (socket?.readyState === WebSocket.OPEN && micStream) {
    socket.send(JSON.stringify({ type: "audio.stop", payload: {} }));
  }

  if (processor) {
    processor.disconnect();
    processor.port.onmessage = null;
  }
  if (micSource) micSource.disconnect();
  if (silentMicGain) silentMicGain.disconnect();
  if (micStream) micStream.getTracks().forEach((track) => track.stop());

  processor = null;
  micSource = null;
  silentMicGain = null;
  micStream = null;
  pendingMicBuffers = [];
  pendingMicBytes = 0;
  targetOrbPulse = 0;
  setMicState("off");
}

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

function queueMicFrame(buffer) {
  evaluateLocalBargeIn(buffer);
  pendingMicBuffers.push(buffer);
  pendingMicBytes += buffer.byteLength;
  if (pendingMicBytes < targetMicFrameBytes) return;

  const combined = new Uint8Array(pendingMicBytes);
  let offset = 0;
  pendingMicBuffers.forEach((pending) => {
    combined.set(new Uint8Array(pending), offset);
    offset += pending.byteLength;
  });

  pendingMicBuffers = [];
  pendingMicBytes = 0;
  socket.send(combined.buffer);
}

// ============================================================
// Audio playback
// ============================================================

function playPcmChunk(payload) {
  if (!payload.audio || payload.encoding !== "pcm_s16le") return;

  const responseId = payload.response_id || null;
  if (responseId && locallyInterruptedAudioResponseIds.has(responseId)) return;

  if (responseId && responseId !== activeAudioResponseId) {
    clearPlaybackQueue();
    activeAudioResponseId = responseId;
    resetLocalBargeInCounter();
  }

  const context = audioContext || new AudioContext({ sampleRate: payload.sample_rate || 16000 });
  audioContext = context;
  if (context.state === "suspended") context.resume();
  audioContextState.textContent = context.state;

  const bytes = Uint8Array.from(atob(payload.audio), (char) => char.charCodeAt(0));
  const samples = new Int16Array(bytes.buffer);
  const audioBuffer = context.createBuffer(1, samples.length, payload.sample_rate || 16000);
  const channel = audioBuffer.getChannelData(0);

  for (let i = 0; i < samples.length; i += 1) channel[i] = samples[i] / 32768;

  audioChunks.textContent = Number(audioChunks.textContent || 0) + 1;
  const lvl = payload.rms ?? levelFromSamples(samples).rms;
  const pk = payload.peak ?? levelFromSamples(samples).peak;
  outputLevel.textContent = `${lvl} / ${pk}`;

  const numericLvl = Number(lvl);
  if (Number.isFinite(numericLvl)) {
    targetOrbPulse = Math.min(1, Math.max(targetOrbPulse, numericLvl * 6));
    pushMeterSample(numericLvl);
  }

  const source = context.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(context.destination);
  source.onended = () => {
    activeSources = activeSources.filter((s) => s !== source);
    if (activeSources.length === 0) {
      playbackState.textContent = "idle";
      targetOrbPulse = 0;
    }
  };

  playbackTime = Math.max(context.currentTime, playbackTime);
  source.start(playbackTime);
  playbackTime += audioBuffer.duration;
  activeSources.push(source);
  playbackState.textContent = "speaking";
}

function levelFromSamples(samples) {
  if (!samples.length) return { rms: 0, peak: 0 };
  let peak = 0;
  let sumSquares = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const abs = Math.abs(samples[i]);
    peak = Math.max(peak, abs);
    sumSquares += samples[i] * samples[i];
  }
  return {
    rms: Math.round((Math.sqrt(sumSquares / samples.length) / 32768) * 10000) / 10000,
    peak: Math.round((peak / 32768) * 10000) / 10000
  };
}

function clearPlaybackQueue() {
  activeSources.forEach((source) => {
    try { source.stop(); } catch { /* already stopped */ }
  });
  activeSources = [];
  playbackTime = audioContext?.currentTime || 0;
  playbackState.textContent = "idle";
  targetOrbPulse = 0;
}

// ============================================================
// Animation loop — orb pulse + meter falloff/render
// ============================================================

function animateOrb() {
  currentOrbPulse += (targetOrbPulse - currentOrbPulse) * 0.18;
  targetOrbPulse *= 0.92;
  document.documentElement.style.setProperty("--orb-pulse", currentOrbPulse.toFixed(3));

  const now = performance.now();
  if (now - lastSampleAt > 120) {
    for (let i = 0; i < METER_BARS; i += 1) meterBuffer[i] *= METER_FALLOFF;
  }
  renderMeter();

  orbAnimationFrame = requestAnimationFrame(animateOrb);
}

function startOrbLoop() {
  if (REDUCED_MOTION) {
    renderMeter();
    return;
  }
  if (orbAnimationFrame !== null) return;
  orbAnimationFrame = requestAnimationFrame(animateOrb);
}

// ============================================================
// Particle aura — p5.js instance sketch attached to #voiceOrb
// ============================================================

function hexToRgb(hex) {
  if (!hex) return [200, 200, 200];
  hex = hex.trim().replace("#", "");
  if (hex.length === 3) hex = hex.split("").map((c) => c + c).join("");
  const num = parseInt(hex, 16);
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
}

function initOrbSketch() {
  if (typeof window.p5 === "undefined" || !orbEl) return;

  const sketch = (p) => {
    const PARTICLE_COUNT = 140;
    const particles = [];
    let cachedStops = [
      [167, 229, 211],   // mint
      [168, 200, 232],   // sky
      [200, 184, 224],   // lavender
      [244, 197, 168]    // peach
    ];

    function refreshStops() {
      const cs = getComputedStyle(document.body);
      const next = [];
      for (let i = 1; i <= 4; i += 1) {
        const v = cs.getPropertyValue(`--orb-stop-${i}`);
        if (v) next.push(hexToRgb(v));
      }
      if (next.length === 4) cachedStops = next;
    }

    function fitCanvas() {
      const rect = orbEl.getBoundingClientRect();
      const w = Math.max(64, Math.round(rect.width));
      const h = Math.max(64, Math.round(rect.height));
      p.resizeCanvas(w, h);
    }

    p.setup = () => {
      const rect = orbEl.getBoundingClientRect();
      const w = Math.max(64, Math.round(rect.width));
      const h = Math.max(64, Math.round(rect.height));
      const c = p.createCanvas(w, h);
      c.parent(orbEl);
      p.pixelDensity(Math.min(2, window.devicePixelRatio || 1));

      for (let i = 0; i < PARTICLE_COUNT; i += 1) {
        particles.push({
          angle: p.random(p.TWO_PI),
          angularSpeed: p.random(-0.0035, 0.0035),
          baseRadius: p.random(0.20, 0.40),
          radialPhase: p.random(p.TWO_PI),
          radialSpeed: p.random(0.5, 1.4),
          tone: p.random(),
          sizeBase: p.random(0.9, 2.6),
          twinklePhase: p.random(p.TWO_PI),
          twinkleSpeed: p.random(0.018, 0.05),
          driftPhase: p.random(p.TWO_PI),
          driftSpeed: p.random(0.4, 1.0)
        });
      }
      refreshStops();
    };

    p.windowResized = () => fitCanvas();

    // Re-fit when the FAB might have changed size (state changes can shift padding).
    const ro = new ResizeObserver(fitCanvas);
    ro.observe(orbEl);

    p.draw = () => {
      if (p.frameCount % 30 === 0) refreshStops();
      p.clear();

      const cx = p.width / 2;
      const cy = p.height / 2;
      const baseR = Math.min(p.width, p.height) / 2.6;

      const pulse =
        parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--orb-pulse")) || 0;
      const t = p.frameCount * 0.01;

      p.noStroke();

      for (const part of particles) {
        part.angle += part.angularSpeed;
        part.radialPhase += part.radialSpeed * 0.022;
        part.twinklePhase += part.twinkleSpeed;
        part.driftPhase += part.driftSpeed * 0.012;

        // Wavy aura: multi-frequency oscillation around the ring + per-particle phase.
        const wave1 = Math.sin(part.angle * 3 + t * 1.4) * 0.05;
        const wave2 = Math.sin(part.angle * 5 + t * 0.9 + 2.1) * 0.025;
        const localWave = Math.sin(part.radialPhase) * 0.06;

        const r = (part.baseRadius + wave1 + wave2 + localWave + pulse * 0.22) * baseR;

        // Slight tangential drift for a more organic float.
        const tangential = Math.sin(part.driftPhase) * 0.04;
        const angle = part.angle + tangential;

        const x = cx + Math.cos(angle) * r;
        const y = cy + Math.sin(angle) * r;

        const stop = cachedStops[Math.floor(part.tone * cachedStops.length) % cachedStops.length];
        const twinkle = 0.5 + 0.5 * Math.sin(part.twinklePhase);

        const baseAlpha = (60 + 120 * twinkle) * (0.65 + pulse * 0.35);
        const size = part.sizeBase * (0.85 + 0.4 * twinkle) * (1 + pulse * 0.45);

        // Soft halo (low alpha, larger).
        p.fill(stop[0], stop[1], stop[2], baseAlpha * 0.18);
        p.circle(x, y, size * 5);
        // Bright core.
        p.fill(stop[0], stop[1], stop[2], baseAlpha);
        p.circle(x, y, size);
      }
    };
  };

  // eslint-disable-next-line no-new, new-cap
  new window.p5(sketch);
}

// ============================================================
// Hero "Talk" orchestration
// ============================================================

function handleMicError(error) {
  recordEvent({
    type: "error",
    payload: { message: "Could not start microphone.", detail: error.message }
  });
  setMicState("off");
  if (appState === "connecting" || appState === "active") {
    setAppState("active");
    if (stageHint) stageHint.textContent = "Mic blocked — open Advanced";
  }
}

async function onTalkClick() {
  if (appState === "idle") {
    setAppState("connecting");
    pendingMicAfterConnect = true;
    connect();
    return;
  }

  if (appState === "connecting") {
    pendingMicAfterConnect = false;
    setAppState("ending");
    disconnect();
    return;
  }

  if (appState === "active") {
    setAppState("ending");
    disconnect();
    return;
  }
}

// ============================================================
// Diagnostics drawer
// ============================================================

function setDiagnosticsOpen(open) {
  if (!diagnostics) return;
  diagnostics.dataset.open = open ? "true" : "false";
  diagnosticsBtn?.setAttribute("aria-expanded", open ? "true" : "false");

  if (diagnosticsScrim) {
    if (open) {
      diagnosticsScrim.hidden = false;
      requestAnimationFrame(() => {
        diagnosticsScrim.dataset.show = "true";
      });
    } else {
      diagnosticsScrim.dataset.show = "false";
      setTimeout(() => {
        if (diagnosticsScrim.dataset.show === "false") diagnosticsScrim.hidden = true;
      }, 320);
    }
  }

  if (open) {
    unseenEventCount = 0;
    updateEventBadge();
  }
}

function toggleDiagnostics() {
  setDiagnosticsOpen(diagnostics?.dataset.open !== "true");
}

// ============================================================
// Wire-up
// ============================================================

connectBtn?.addEventListener("click", () => {
  if (appState === "idle") setAppState("connecting");
  connect();
});

disconnectBtn?.addEventListener("click", () => {
  setAppState("ending");
  disconnect();
});

micBtn?.addEventListener("click", () => {
  toggleMic().catch(handleMicError);
});

talkBtn?.addEventListener("click", onTalkClick);
diagnosticsBtn?.addEventListener("click", toggleDiagnostics);
diagnosticsCloseBtn?.addEventListener("click", () => setDiagnosticsOpen(false));
diagnosticsScrim?.addEventListener("click", () => setDiagnosticsOpen(false));

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && diagnostics?.dataset.open === "true") {
    setDiagnosticsOpen(false);
  }
});

// Initial state.
initMeter();
setConnectionState("disconnected");
setPipelineState("idle");
setMicState("off");
setAppState("idle");
startOrbLoop();
initOrbSketch();
