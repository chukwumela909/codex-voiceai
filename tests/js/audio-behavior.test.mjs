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
