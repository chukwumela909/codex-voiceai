class PcmCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;

    const buffer = new ArrayBuffer(input.length * 2);
    const view = new DataView(buffer);
    for (let index = 0; index < input.length; index += 1) {
      const sample = Math.max(-1, Math.min(1, input[index]));
      view.setInt16(index * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    }
    this.port.postMessage(buffer, [buffer]);
    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
