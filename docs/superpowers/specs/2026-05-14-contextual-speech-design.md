# Contextual Speech Design

Date: 2026-05-14

## Goal

Make the live voice agent feel more realistic and resilient when speech-to-text is imperfect. The user-visible transcript should remain the raw Deepgram transcript, while the assistant uses conversation context to infer likely intent and speaks through Cartesia with subtle, context-relevant delivery cues.

## Current Behavior

- Deepgram transcripts are parsed in `app/deepgram.py` and passed into `MockConversationSession.handle_live_transcript`.
- Finalized text is emitted as `transcript.final`, appended to `self.transcript`, and sent to Groq.
- Groq responses are split into speakable chunks by `pop_speakable_chunks`.
- Cartesia receives plain transcript text through `CartesiaStreamingTTS`, with only `generation_config.speed` applied.
- Turn finalization is controlled by Deepgram `speech_final`, Deepgram `UtteranceEnd`, and the app fallback `VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS`.

## Design Principles

- Preserve raw STT output for user trust, observability, and debugging.
- Use hidden reasoning context to make Groq robust to imperfect transcripts.
- Keep latency low by avoiding a separate transcript-correction model call in the first version.
- Use Cartesia SSML-style tags only when they improve delivery for the specific utterance.
- Keep SSML conservative because Cartesia emotion and some control tags are model-dependent or beta.
- Never split SSML tags across streamed TTS chunks.

## Hidden Intent Inference

Add an adapter between transcript finalization and Groq request construction. It will keep `transcript.final` unchanged, but pass a hidden instruction with the latest raw user turn and recent conversation context.

The instruction should tell Groq:

- The latest user message may contain speech-to-text errors.
- Infer the user's likely intent from recent conversation context.
- Do not mention transcription uncertainty unless the ambiguity blocks a useful answer.
- If the utterance is genuinely ambiguous, ask one short clarifying question.
- Keep the response spoken, brief, and phone-call appropriate.

Implementation shape:

- Add a helper such as `agent_transcript_with_intent_inference(transcript)` in `app/mock_conversation.py` or a small new module.
- Use the helper only for live Groq calls where user speech came from Deepgram.
- Keep `self.transcript` as raw conversation history for now.
- Emit optional diagnostics in `pipeline.stage` when intent inference context is attached, without exposing hidden prompt text.

## Conversational Turn Timing

The current `VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS=650` can make the agent answer during a thinking pause. The design should make pauses more conversational without making the agent feel sluggish.

Implementation shape:

- Raise the default partial idle fallback to a more forgiving value, likely `900ms` to `1200ms`.
- Continue honoring Deepgram `speech_final` immediately when it arrives.
- Keep `DEEPGRAM_UTTERANCE_END_MS` clamped to Deepgram's minimum.
- Add README guidance explaining how to tune `DEEPGRAM_ENDPOINTING_MS`, `DEEPGRAM_UTTERANCE_END_MS`, and `VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS`.

## Context-Aware Cartesia Speech Director

Add a speech-direction layer before Cartesia. It receives assistant text chunks plus metadata about the turn, then returns Cartesia-safe transcript text with minimal SSML-style tags.

The first version should support:

- Short pauses with `<break time="150ms"/>` to `<break time="400ms"/>` after natural discourse markers or before careful clarifications.
- Gentle speed changes using `<speed ratio="..."/>` only when the full tag can be emitted in one chunk and when using a model that supports it.
- `<spell>` for obvious IDs, acronyms, or code-like tokens if the text is being read aloud.
- Emotion tags only behind an explicit config flag, because Cartesia marks emotion control as experimental.

The director should avoid:

- Random pauses after every sentence.
- Tags in very short acknowledgements where plain speech is already natural.
- Mid-word or mid-tag chunk boundaries.
- Heavy emotional direction unless the text clearly calls for it.

Implementation shape:

- Add a helper such as `direct_speech_for_cartesia(text, context)` that returns directed text.
- Apply it at TTS chunk boundaries, after `pop_speakable_chunks` has produced complete speakable chunks.
- Escape or leave unchanged any text that could accidentally look like unsupported markup, depending on Cartesia behavior verified by tests.
- Add config flags:
  - `VOICE_AGENT_CARTESIA_SSML_ENABLED`, default `true` for `sonic-3`.
  - `VOICE_AGENT_CARTESIA_EMOTION_TAGS_ENABLED`, default `false`.
  - `VOICE_AGENT_CARTESIA_SPEECH_DIRECTOR_ENABLED`, default `true`.

## Data Flow

1. Browser streams PCM audio to `/ws/browser`.
2. Deepgram emits partial and final transcripts.
3. The app emits raw `transcript.partial` and `transcript.final` events.
4. The app stores the raw user turn.
5. The Groq request helper adds hidden intent-inference context around the latest raw turn.
6. Groq streams assistant text.
7. Speakable chunks are emitted to the frontend as plain assistant text.
8. The Cartesia speech director converts each speakable chunk into TTS-directed text.
9. Cartesia streams audio from the directed text.

## Error Handling

- If speech direction fails, fall back to plain assistant text for Cartesia.
- If Cartesia rejects SSML-style input, disable speech direction for that turn and use the existing mock TTS fallback path if Cartesia produces no audio.
- Hidden intent inference should never block response generation; if helper logic fails, use the raw transcript.
- Diagnostics should identify the failing stage without exposing API keys or hidden prompt content.

## Testing

Add focused tests before implementation:

- Intent inference helper preserves raw turns while adding hidden inference instructions for Groq.
- Groq response path still emits raw `transcript.final` text.
- Partial idle default/config appears correctly in health status.
- Speech director inserts context-relevant pauses only for selected patterns.
- Speech director does not split or emit malformed SSML tags across chunks.
- Cartesia generation requests include directed transcript text when enabled and plain text when disabled.
- Existing interruption and proactive tests continue to pass.

## Open Decisions Resolved

- Use hidden intent inference rather than rewriting the displayed transcript.
- Add context-aware SSML-style delivery as a conservative speech direction layer.
- Keep emotion tags disabled by default.
- Prefer a single-pass response flow over a separate transcript-correction LLM call for the first version.

