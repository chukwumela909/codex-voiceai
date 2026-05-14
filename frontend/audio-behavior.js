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
