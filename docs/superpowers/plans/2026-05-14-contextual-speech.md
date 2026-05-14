# Contextual Speech Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make live conversations more natural by preserving raw Deepgram transcripts, giving Groq hidden intent-inference context, softening pause timing, and directing Cartesia speech with conservative context-aware SSML-style tags.

**Architecture:** Add two small helper modules: one for Groq transcript preparation and one for Cartesia speech direction. Wire them into the existing `MockConversationSession` live response path so frontend text remains plain while Cartesia receives directed speech text. Expose tuning through `Settings`, health status, `.env.example`, and README.

**Tech Stack:** Python 3.11+, FastAPI app, Pydantic Settings, pytest, Deepgram streaming STT, Groq OpenAI-compatible chat streaming, Cartesia WebSocket TTS.

---

## File Structure

- Create `app/conversation_context.py`: pure helpers for hidden intent-inference messages sent to Groq.
- Create `tests/test_conversation_context.py`: unit tests proving raw transcript turns are preserved and hidden guidance is added only when useful.
- Create `app/speech_director.py`: pure helpers for context-aware Cartesia-safe speech direction.
- Create `tests/test_speech_director.py`: unit tests for discourse pauses, clarification pauses, spelling, disabled modes, and safe chunk behavior.
- Modify `app/config.py`: add conversation and Cartesia speech-direction flags, raise partial idle default, and expose safe health status.
- Modify `tests/test_config.py`: update turn timing default and cover new public config fields.
- Modify `app/mock_conversation.py`: use hidden Groq transcript helper for live user-driven responses and direct Cartesia-only speech text.
- Modify `tests/test_live_conversation.py`: prove raw transcript events remain raw, Groq receives hidden intent guidance, frontend text remains plain, and Cartesia receives directed chunks.
- Modify `.env.example`: document new environment variables and updated pause default.
- Modify `README.md`: document intent inference, conversational timing, and Cartesia SSML-style speech direction.

---

### Task 1: Config Defaults And Health Status

**Files:**
- Modify: `app/config.py`
- Modify: `tests/test_config.py`
- Modify: `.env.example`

- [ ] **Step 1: Write failing config tests**

Add this test after `test_balanced_fast_voice_timing_defaults` in `tests/test_config.py`, and update `test_balanced_fast_voice_timing_defaults` to expect `1000` for `partial_idle_finalize_ms`.

```python
def test_balanced_fast_voice_timing_defaults(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_ENDPOINTING_MS", raising=False)
    monkeypatch.delenv("DEEPGRAM_UTTERANCE_END_MS", raising=False)
    monkeypatch.delenv("VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS", raising=False)

    settings = Settings(_env_file=None)
    status = settings.public_config_status()

    assert settings.deepgram_endpointing_ms == 220
    assert settings.deepgram_utterance_end_ms == 1000
    assert settings.partial_idle_finalize_ms == 1000
    assert status["turn_timing"] == {
        "deepgram_endpointing_ms": 220,
        "deepgram_utterance_end_ms": 1000,
        "partial_idle_finalize_ms": 1000,
    }


def test_contextual_speech_public_config_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("VOICE_AGENT_INTENT_INFERENCE_ENABLED", raising=False)
    monkeypatch.delenv("VOICE_AGENT_CARTESIA_SPEECH_DIRECTOR_ENABLED", raising=False)
    monkeypatch.delenv("VOICE_AGENT_CARTESIA_SSML_ENABLED", raising=False)
    monkeypatch.delenv("VOICE_AGENT_CARTESIA_EMOTION_TAGS_ENABLED", raising=False)

    defaults = Settings(_env_file=None).public_config_status()

    monkeypatch.setenv("VOICE_AGENT_INTENT_INFERENCE_ENABLED", "false")
    monkeypatch.setenv("VOICE_AGENT_CARTESIA_SPEECH_DIRECTOR_ENABLED", "false")
    monkeypatch.setenv("VOICE_AGENT_CARTESIA_SSML_ENABLED", "false")
    monkeypatch.setenv("VOICE_AGENT_CARTESIA_EMOTION_TAGS_ENABLED", "true")
    overridden = Settings(_env_file=None).public_config_status()

    assert defaults["conversation"] == {
        "intent_inference_enabled": True,
    }
    assert defaults["cartesia"]["speech_direction"] == {
        "enabled": True,
        "ssml_enabled": True,
        "emotion_tags_enabled": False,
    }
    assert overridden["conversation"] == {
        "intent_inference_enabled": False,
    }
    assert overridden["cartesia"]["speech_direction"] == {
        "enabled": False,
        "ssml_enabled": False,
        "emotion_tags_enabled": True,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
pytest tests/test_config.py::test_balanced_fast_voice_timing_defaults tests/test_config.py::test_contextual_speech_public_config_defaults_and_overrides -v
```

Expected: `test_balanced_fast_voice_timing_defaults` fails because the default is still `650`, and `test_contextual_speech_public_config_defaults_and_overrides` fails because `conversation` or `cartesia` config keys do not exist.

- [ ] **Step 3: Implement config fields**

In `app/config.py`, change the default constant and add fields to `Settings`.

```python
DEFAULT_PARTIAL_IDLE_FINALIZE_MS = 1000
```

Add these fields after `persona`:

```python
    intent_inference_enabled: bool = Field(default=True, alias="VOICE_AGENT_INTENT_INFERENCE_ENABLED")
    cartesia_speech_director_enabled: bool = Field(
        default=True,
        alias="VOICE_AGENT_CARTESIA_SPEECH_DIRECTOR_ENABLED",
    )
    cartesia_ssml_enabled: bool = Field(default=True, alias="VOICE_AGENT_CARTESIA_SSML_ENABLED")
    cartesia_emotion_tags_enabled: bool = Field(
        default=False,
        alias="VOICE_AGENT_CARTESIA_EMOTION_TAGS_ENABLED",
    )
```

Add these sections inside `public_config_status()` after `providers`:

```python
            "conversation": {
                "intent_inference_enabled": self.intent_inference_enabled,
            },
            "cartesia": {
                "speech_direction": {
                    "enabled": self.cartesia_speech_director_enabled,
                    "ssml_enabled": self.cartesia_ssml_enabled,
                    "emotion_tags_enabled": self.cartesia_emotion_tags_enabled,
                },
            },
```

- [ ] **Step 4: Update `.env.example`**

Change:

```dotenv
VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS=650
```

to:

```dotenv
VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS=1000
VOICE_AGENT_INTENT_INFERENCE_ENABLED=true
VOICE_AGENT_CARTESIA_SPEECH_DIRECTOR_ENABLED=true
VOICE_AGENT_CARTESIA_SSML_ENABLED=true
VOICE_AGENT_CARTESIA_EMOTION_TAGS_ENABLED=false
```

- [ ] **Step 5: Run config tests to verify they pass**

Run:

```powershell
pytest tests/test_config.py -v
```

Expected: all config tests pass.

- [ ] **Step 6: Commit config slice**

Run:

```powershell
git add app/config.py tests/test_config.py .env.example
git commit -m "feat: add contextual speech config"
```

---

### Task 2: Hidden Intent Inference For Groq

**Files:**
- Create: `app/conversation_context.py`
- Create: `tests/test_conversation_context.py`
- Modify: `app/mock_conversation.py`
- Modify: `tests/test_live_conversation.py`

- [ ] **Step 1: Write failing helper tests**

Create `tests/test_conversation_context.py`:

```python
from app.conversation_context import INTENT_INFERENCE_INSTRUCTION, agent_transcript_with_intent_inference


def test_intent_inference_preserves_raw_turns_and_inserts_hidden_instruction_before_latest_user():
    transcript = [
        {"role": "user", "content": "I want to test the browser mic."},
        {"role": "assistant", "content": "Sure, say something short."},
        {"role": "user", "content": "You're what's up can you hear me now"},
    ]

    prepared = agent_transcript_with_intent_inference(transcript)

    assert prepared[:-2] == transcript[:-1]
    assert prepared[-2]["role"] == "system"
    assert "speech-to-text errors" in prepared[-2]["content"]
    assert "Infer the user's likely intent" in prepared[-2]["content"]
    assert prepared[-1] == transcript[-1]
    assert transcript[-1]["content"] == "You're what's up can you hear me now"
    assert INTENT_INFERENCE_INSTRUCTION in prepared[-2]["content"]


def test_intent_inference_returns_copy_when_disabled():
    transcript = [{"role": "user", "content": "raw words stay raw"}]

    prepared = agent_transcript_with_intent_inference(transcript, enabled=False)

    assert prepared == transcript
    assert prepared is not transcript


def test_intent_inference_skips_when_latest_turn_is_not_user():
    transcript = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi."},
    ]

    assert agent_transcript_with_intent_inference(transcript) == transcript
```

- [ ] **Step 2: Run helper tests to verify they fail**

Run:

```powershell
pytest tests/test_conversation_context.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'app.conversation_context'`.

- [ ] **Step 3: Implement the helper**

Create `app/conversation_context.py`:

```python
INTENT_INFERENCE_INSTRUCTION = (
    "The latest user message may contain speech-to-text errors. "
    "Infer the user's likely intent from recent conversation context. "
    "Do not mention transcription uncertainty unless the ambiguity blocks a useful answer. "
    "If the utterance is genuinely ambiguous, ask one short clarifying question. "
    "Keep the response spoken, brief, and phone-call friendly."
)


def agent_transcript_with_intent_inference(
    transcript: list[dict[str, str]],
    *,
    enabled: bool = True,
) -> list[dict[str, str]]:
    prepared = [_copy_turn(turn) for turn in transcript]
    if not enabled or not prepared:
        return prepared

    latest = prepared[-1]
    if latest.get("role") != "user" or not latest.get("content", "").strip():
        return prepared

    return [
        *prepared[:-1],
        {"role": "system", "content": INTENT_INFERENCE_INSTRUCTION},
        latest,
    ]


def _copy_turn(turn: dict[str, str]) -> dict[str, str]:
    role = turn.get("role", "")
    content = turn.get("content", "")
    return {"role": role, "content": content}
```

- [ ] **Step 4: Run helper tests to verify they pass**

Run:

```powershell
pytest tests/test_conversation_context.py -v
```

Expected: all helper tests pass.

- [ ] **Step 5: Write failing live wiring test**

Add this test after `test_speech_final_transcript_starts_agent_response_without_cartesia_precheck` in `tests/test_live_conversation.py`:

```python
def test_live_agent_receives_hidden_intent_inference_context_but_events_keep_raw_transcript():
    events = []
    captured_transcripts = []

    class FakeAgent:
        async def stream_response(self, transcript):
            captured_transcripts.append(transcript)
            yield "I understood the intent from context."

    settings = proactive_settings(
        normalized_mode="live",
        groq_api_key="groq-key",
        cartesia_api_key=None,
        cartesia_voice_id=None,
        intent_inference_enabled=True,
    )

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession("sess_test", send_event, settings)
        session.agent = FakeAgent()
        session.transcript = [
            {"role": "user", "content": "I am testing whether the mic hears me."},
            {"role": "assistant", "content": "Say one more phrase and I will listen."},
        ]
        await session.handle_live_transcript(
            {
                "text": "You're what's up can you hear me now",
                "confidence": 0.61,
                "is_final": True,
                "speech_final": True,
                "provider": "deepgram",
            }
        )
        assert session.current_task is not None
        await session.current_task

    asyncio.run(run_session())

    raw_final = next(event for event in events if event["type"] == "transcript.final")
    assert raw_final["payload"]["text"] == "You're what's up can you hear me now"
    assert captured_transcripts
    assert captured_transcripts[0][-2]["role"] == "system"
    assert "Infer the user's likely intent" in captured_transcripts[0][-2]["content"]
    assert captured_transcripts[0][-1] == {
        "role": "user",
        "content": "You're what's up can you hear me now",
    }
    assert any(
        event["type"] == "pipeline.stage"
        and event["payload"].get("stage") == "llm_context"
        and event["payload"].get("intent_inference") is True
        for event in events
    )
    assert session.transcript[-2:] == [
        {"role": "user", "content": "You're what's up can you hear me now"},
        {"role": "assistant", "content": "I understood the intent from context."},
    ]
```

- [ ] **Step 6: Run live wiring test to verify it fails**

Run:

```powershell
pytest tests/test_live_conversation.py::test_live_agent_receives_hidden_intent_inference_context_but_events_keep_raw_transcript -v
```

Expected: test fails because Groq still receives `self.transcript` without hidden intent guidance and no `llm_context` stage is emitted.

- [ ] **Step 7: Wire helper into live user-driven Groq paths**

In `app/mock_conversation.py`, add the import:

```python
from app.conversation_context import agent_transcript_with_intent_inference
```

Add this method near `_ensure_agent`:

```python
    async def _live_user_agent_transcript(self, *, metadata: dict | None = None) -> list[dict[str, str]]:
        metadata = metadata or {}
        enabled = bool(getattr(self.settings, "intent_inference_enabled", True))
        transcript = agent_transcript_with_intent_inference(self.transcript, enabled=enabled)
        if enabled and transcript != self.transcript:
            await self.send_event(
                event(
                    "pipeline.stage",
                    self.session_id,
                    {
                        "stage": "llm_context",
                        "provider": "groq",
                        "intent_inference": True,
                        "turns_sent": len(transcript),
                        **metadata,
                    },
                )
            )
        return transcript
```

In `_stream_live_agent_response_with_streaming_speech`, define the prepared transcript before `speakable_chunks()`:

```python
        agent_transcript = await self._live_user_agent_transcript(metadata=metadata)
```

Then change:

```python
                async for delta in self.agent.stream_response(self.transcript):
```

to:

```python
                async for delta in self.agent.stream_response(agent_transcript):
```

In `_stream_live_agent_response`, define the prepared transcript after `_ensure_agent()`:

```python
        agent_transcript = await self._live_user_agent_transcript()
```

Then change:

```python
            async for delta in self.agent.stream_response(self.transcript):
```

to:

```python
            async for delta in self.agent.stream_response(agent_transcript):
```

Do not change `_stream_live_proactive_response`; proactive turns already use a separate internal instruction path.

- [ ] **Step 8: Run task tests to verify they pass**

Run:

```powershell
pytest tests/test_conversation_context.py tests/test_live_conversation.py::test_live_agent_receives_hidden_intent_inference_context_but_events_keep_raw_transcript -v
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit intent inference slice**

Run:

```powershell
git add app/conversation_context.py app/mock_conversation.py tests/test_conversation_context.py tests/test_live_conversation.py
git commit -m "feat: add hidden intent inference for live turns"
```

---

### Task 3: Context-Aware Cartesia Speech Director

**Files:**
- Create: `app/speech_director.py`
- Create: `tests/test_speech_director.py`
- Modify: `app/mock_conversation.py`
- Modify: `tests/test_live_conversation.py`

- [ ] **Step 1: Write failing speech director unit tests**

Create `tests/test_speech_director.py`:

```python
from app.speech_director import SpeechDirectionConfig, direct_speech_for_cartesia


def test_director_leaves_short_acknowledgement_plain():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True)

    assert direct_speech_for_cartesia("Okay.", config) == "Okay."


def test_director_adds_pause_after_contextual_discourse_marker():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True)

    directed = direct_speech_for_cartesia("Well, I think you were asking about the microphone.", config)

    assert directed == 'Well,<break time="180ms"/> I think you were asking about the microphone.'


def test_director_adds_clarification_pause_when_inferring_intent():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True)

    directed = direct_speech_for_cartesia("I think you meant the Deepgram model is missing words.", config)

    assert directed == 'I think you meant<break time="250ms"/> the Deepgram model is missing words.'


def test_director_spells_code_like_tokens():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=True)

    directed = direct_speech_for_cartesia("Set API mode on port 8000.", config)

    assert directed == "Set <spell>API</spell> mode on port <spell>8000</spell>."


def test_director_can_be_disabled():
    config = SpeechDirectionConfig(enabled=False, ssml_enabled=True)

    assert direct_speech_for_cartesia("Well, this should stay plain.", config) == "Well, this should stay plain."


def test_director_can_disable_ssml_only():
    config = SpeechDirectionConfig(enabled=True, ssml_enabled=False)

    assert direct_speech_for_cartesia("Set API mode.", config) == "Set API mode."
```

- [ ] **Step 2: Run director tests to verify they fail**

Run:

```powershell
pytest tests/test_speech_director.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'app.speech_director'`.

- [ ] **Step 3: Implement the speech director**

Create `app/speech_director.py`:

```python
import re
from dataclasses import dataclass


DISCOURSE_MARKERS = ("Well,", "Hmm,", "Okay,", "Right,", "So,")
MIN_WORDS_FOR_DISCOURSE_PAUSE = 6
SSML_TAG_PATTERN = re.compile(r"<[^>]+>")
SPELLABLE_TOKEN_PATTERN = re.compile(r"\b(?:[A-Z]{2,6}|\d{3,}|\w+-\w+)\b")


@dataclass(frozen=True)
class SpeechDirectionConfig:
    enabled: bool = True
    ssml_enabled: bool = True
    emotion_tags_enabled: bool = False


def direct_speech_for_cartesia(text: str, config: SpeechDirectionConfig) -> str:
    if not config.enabled or not config.ssml_enabled or not text.strip():
        return text
    if SSML_TAG_PATTERN.search(text):
        return text

    directed = _add_discourse_pause(text)
    directed = _add_clarification_pause(directed)
    directed = _spell_code_like_tokens(directed)
    return directed


def _add_discourse_pause(text: str) -> str:
    if len(text.split()) < MIN_WORDS_FOR_DISCOURSE_PAUSE:
        return text
    for marker in DISCOURSE_MARKERS:
        if text.startswith(f"{marker} "):
            return f'{marker}<break time="180ms"/> {text[len(marker) + 1:]}'
    return text


def _add_clarification_pause(text: str) -> str:
    phrase = "I think you meant "
    if text.startswith(phrase):
        return 'I think you meant<break time="250ms"/> ' + text[len(phrase):]
    return text


def _spell_code_like_tokens(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in {"OK"}:
            return token
        return f"<spell>{token}</spell>"

    pieces = re.split(r"(<[^>]+>)", text)
    return "".join(
        piece if piece.startswith("<") and piece.endswith(">") else SPELLABLE_TOKEN_PATTERN.sub(replace, piece)
        for piece in pieces
    )
```

- [ ] **Step 4: Run director tests to verify they pass**

Run:

```powershell
pytest tests/test_speech_director.py -v
```

Expected: all director tests pass.

- [ ] **Step 5: Write failing Cartesia wiring tests**

Add this test after `test_live_cartesia_audio_can_start_before_llm_final_text_event` in `tests/test_live_conversation.py`:

```python
def test_streaming_cartesia_receives_directed_speech_while_frontend_text_stays_plain():
    events = []
    audio = base64.b64encode(b"\x00\x00" * 120).decode("ascii")
    spoken_chunks = []

    class ClarifyingAgent:
        async def stream_response(self, transcript):
            yield "Well, I think you meant the Deepgram model is missing words."

    class StreamingSynthesizer:
        async def stream_speech_chunks(self, chunks, *, context_id):
            async for chunk in chunks:
                spoken_chunks.append(chunk)
                yield {"type": "chunk", "audio": audio, "context_id": context_id}
            yield {"type": "done", "context_id": context_id, "done": True}

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(
                normalized_mode="live",
                groq_api_key="groq-key",
                cartesia_api_key="cartesia-key",
                cartesia_voice_id=VALID_CARTESIA_VOICE_ID,
                cartesia_model="sonic-3",
                cartesia_sample_rate=16000,
                cartesia_version="2026-03-01",
                cartesia_speed=1.2,
                cartesia_speech_director_enabled=True,
                cartesia_ssml_enabled=True,
                cartesia_emotion_tags_enabled=False,
            ),
        )
        session.agent = ClarifyingAgent()
        session.synthesizer = StreamingSynthesizer()
        await session._run_agent_response("You're what's up can you hear me now")

    asyncio.run(run_session())

    final = next(event for event in events if event["type"] == "agent.text.final")
    assert final["payload"]["text"] == "Well, I think you meant the Deepgram model is missing words."
    assert spoken_chunks == [
        'Well,<break time="180ms"/> I think you meant the Deepgram model is missing words. '
    ]
    assert any(
        event["type"] == "pipeline.stage"
        and event["payload"].get("stage") == "tts_speech_direction"
        and event["payload"].get("directed") is True
        for event in events
    )
```

- [ ] **Step 6: Run Cartesia wiring test to verify it fails**

Run:

```powershell
pytest tests/test_live_conversation.py::test_streaming_cartesia_receives_directed_speech_while_frontend_text_stays_plain -v
```

Expected: test fails because `spoken_chunks` still contains plain text and no `tts_speech_direction` stage is emitted.

- [ ] **Step 7: Wire speech direction into Cartesia paths**

In `app/mock_conversation.py`, add imports:

```python
from app.speech_director import SpeechDirectionConfig, direct_speech_for_cartesia
```

Add this method near `_ensure_synthesizer`:

```python
    async def _direct_cartesia_text(self, text: str, *, metadata: dict | None = None) -> str:
        metadata = metadata or {}
        config = SpeechDirectionConfig(
            enabled=bool(getattr(self.settings, "cartesia_speech_director_enabled", True)),
            ssml_enabled=bool(getattr(self.settings, "cartesia_ssml_enabled", True)),
            emotion_tags_enabled=bool(getattr(self.settings, "cartesia_emotion_tags_enabled", False)),
        )
        try:
            directed = direct_speech_for_cartesia(text, config)
        except Exception as exc:
            await self.send_provider_error("cartesia", f"Speech direction failed: {exc}", metadata=metadata)
            return text

        if directed != text:
            await self.send_event(
                event(
                    "pipeline.stage",
                    self.session_id,
                    {
                        "stage": "tts_speech_direction",
                        "provider": "cartesia",
                        "directed": True,
                        **metadata,
                    },
                )
            )
        return directed
```

In `_stream_cartesia_speech`, before `stream_speech`, add:

```python
            directed_message = await self._direct_cartesia_text(message, metadata=metadata)
```

Then change:

```python
            async for chunk in self.synthesizer.stream_speech(
                message,
                context_id=generate_cartesia_context_id(response_id),
            ):
```

to:

```python
            async for chunk in self.synthesizer.stream_speech(
                directed_message,
                context_id=generate_cartesia_context_id(response_id),
            ):
```

In `_stream_cartesia_speech_chunks`, define a directed iterator before calling the synthesizer:

```python
            async def directed_chunks() -> AsyncIterator[str]:
                async for text in chunks:
                    yield await self._direct_cartesia_text(text, metadata=metadata)
```

Then change:

```python
            async for chunk in self.synthesizer.stream_speech_chunks(
                chunks,
                context_id=generate_cartesia_context_id(response_id),
            ):
```

to:

```python
            async for chunk in self.synthesizer.stream_speech_chunks(
                directed_chunks(),
                context_id=generate_cartesia_context_id(response_id),
            ):
```

- [ ] **Step 8: Run task tests to verify they pass**

Run:

```powershell
pytest tests/test_speech_director.py tests/test_live_conversation.py::test_streaming_cartesia_receives_directed_speech_while_frontend_text_stays_plain -v
```

Expected: all selected tests pass.

- [ ] **Step 9: Run Cartesia adapter tests**

Run:

```powershell
pytest tests/test_cartesia_tts.py -v
```

Expected: all Cartesia adapter tests pass because the WebSocket adapter remains focused on request formatting and transport.

- [ ] **Step 10: Commit speech director slice**

Run:

```powershell
git add app/speech_director.py app/mock_conversation.py tests/test_speech_director.py tests/test_live_conversation.py
git commit -m "feat: direct cartesia speech with context-aware ssml"
```

---

### Task 4: Documentation And Full Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README provider tuning**

In `README.md`, change the provider tuning bullets to:

```markdown
- `DEEPGRAM_MODEL`, default `nova-3`
- `DEEPGRAM_ENDPOINTING_MS`, default `220`
- `DEEPGRAM_UTTERANCE_END_MS`, default `1000`
- `VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS`, default `1000`; fallback debounce for useful partial transcripts when Deepgram has not emitted `speech_final`
- `VOICE_AGENT_INTENT_INFERENCE_ENABLED`, default `true`; keeps raw transcripts visible while adding hidden Groq guidance to infer likely intent from recent context
- `GROQ_MODEL`, default `llama-3.1-8b-instant`
- `GROQ_TEMPERATURE`, default `0.7`
- `CARTESIA_MODEL`, default `sonic-3`
- `CARTESIA_SPEED`, default `1.2` (`0.6` to `1.5`; higher is faster)
- `CARTESIA_SAMPLE_RATE`, default `16000`
- `CARTESIA_VERSION`, default `2026-03-01`
- `VOICE_AGENT_CARTESIA_SPEECH_DIRECTOR_ENABLED`, default `true`
- `VOICE_AGENT_CARTESIA_SSML_ENABLED`, default `true`
- `VOICE_AGENT_CARTESIA_EMOTION_TAGS_ENABLED`, default `false`
- `VOICE_AGENT_PERSONA`
```

- [ ] **Step 2: Add README contextual speech section**

Add this section after the ambience section:

```markdown
## Contextual Speech

Live mode preserves raw Deepgram transcripts in `transcript.partial`, `transcript.final`, logs, and stored conversation turns. When `VOICE_AGENT_INTENT_INFERENCE_ENABLED=true`, the Groq request also receives hidden guidance that the latest user turn may include speech-to-text errors, so it should infer likely intent from recent context and ask a short clarifying question only when ambiguity blocks a useful answer.

`VOICE_AGENT_PARTIAL_IDLE_FINALIZE_MS` controls the app fallback used when Deepgram has not emitted `speech_final`. The default is `1000ms` to leave more room for natural thinking pauses. Lower it for snappier demos; raise it when the assistant interrupts too early.

When Cartesia is configured, `VOICE_AGENT_CARTESIA_SPEECH_DIRECTOR_ENABLED=true` applies conservative SSML-style speech direction to TTS input only. Frontend assistant text stays plain. The first version adds short context-relevant pauses after discourse markers, pauses before inferred clarifications, and spells code-like tokens such as API names or numeric IDs. Emotion tags are disabled by default with `VOICE_AGENT_CARTESIA_EMOTION_TAGS_ENABLED=false`.
```

- [ ] **Step 3: Update README health example**

In the health check JSON example, change `"partial_idle_finalize_ms": 650` to `"partial_idle_finalize_ms": 1000` and add these top-level config sections after `"providers"`:

```json
    "conversation": {
      "intent_inference_enabled": true
    },
    "cartesia": {
      "speech_direction": {
        "enabled": true,
        "ssml_enabled": true,
        "emotion_tags_enabled": false
      }
    },
```

- [ ] **Step 4: Run focused tests**

Run:

```powershell
pytest tests/test_config.py tests/test_conversation_context.py tests/test_speech_director.py tests/test_cartesia_tts.py tests/test_live_conversation.py -v
```

Expected: all focused tests pass.

- [ ] **Step 5: Run full test suite**

Run:

```powershell
pytest
```

Expected: all tests pass.

- [ ] **Step 6: Commit documentation and verification slice**

Run:

```powershell
git add README.md .env.example app/config.py app/conversation_context.py app/speech_director.py app/mock_conversation.py tests/test_config.py tests/test_conversation_context.py tests/test_speech_director.py tests/test_cartesia_tts.py tests/test_live_conversation.py
git commit -m "docs: describe contextual speech tuning"
```

If the previous commits already included all code and test changes, this commit should include only `README.md` and any remaining documentation edits.

---

## Final Verification

- [ ] Run `git status --short` and confirm only intentional files are modified.
- [ ] Run `pytest` and confirm the whole suite passes.
- [ ] Start the server with `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`.
- [ ] Open `http://localhost:8000/` and run a live or mock conversation smoke test.
- [ ] In live mode, say a deliberately messy phrase after giving context. Confirm the visible transcript remains raw while the assistant answers the likely intent.
- [ ] Confirm Cartesia audio still streams before `agent.text.final` in the streaming path.
- [ ] Confirm frontend text does not show SSML tags.
