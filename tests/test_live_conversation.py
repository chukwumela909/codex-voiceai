import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace

import app.mock_conversation as mock_conversation
from app.exceptions import ClientConnectionClosed
from app.mock_conversation import MockConversationSession, apply_pcm_gain, calculate_pcm_level

VALID_CARTESIA_VOICE_ID = "6bf6d6c3-9d45-48fb-94a9-4840f83eb385"


def test_apply_pcm_gain_amplifies_quiet_audio_without_changing_frame_size():
    quiet_frame = int(1000).to_bytes(2, "little", signed=True) + int(-1000).to_bytes(2, "little", signed=True)

    boosted = apply_pcm_gain(quiet_frame, 2.5)

    assert boosted == int(2500).to_bytes(2, "little", signed=True) + int(-2500).to_bytes(2, "little", signed=True)
    assert len(boosted) == len(quiet_frame)
    assert calculate_pcm_level(boosted)["rms"] > calculate_pcm_level(quiet_frame)["rms"]


def test_apply_pcm_gain_clips_to_pcm_range():
    loud_frame = int(20000).to_bytes(2, "little", signed=True) + int(-20000).to_bytes(2, "little", signed=True)

    boosted = apply_pcm_gain(loud_frame, 3.0)

    assert boosted == int(32767).to_bytes(2, "little", signed=True) + int(-32768).to_bytes(2, "little", signed=True)


def test_proactive_source_does_not_embed_canned_greeting_terms():
    source = Path("app/mock_conversation.py").read_text(encoding="utf-8")

    for forbidden in (
        "It" + "'s late",
        "No " + "rush",
        "What are we " + "working on",
        "I'm " + "still here",
        "Say the " + "word",
    ):
        assert forbidden not in source


def test_speech_final_transcript_starts_agent_response_without_cartesia_precheck():
    events = []
    settings = SimpleNamespace(
        normalized_mode="live",
        groq_api_key=None,
        groq_model="llama-3.1-8b-instant",
        groq_temperature=0.7,
        persona="Be concise.",
        cartesia_api_key=None,
        cartesia_voice_id=None,
        deepgram_endpointing_ms=300,
        partial_idle_finalize_ms=650,
    )

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession("sess_test", send_event, settings)
        await session.handle_live_transcript(
            {
                "text": "Hello there",
                "confidence": 0.99,
                "is_final": True,
                "speech_final": True,
                "provider": "deepgram",
            }
        )
        assert session.current_task is not None
        await session.current_task

    asyncio.run(run_session())

    event_types = [event["type"] for event in events]
    assert "transcript.final" in event_types
    assert "agent.text.delta" in event_types
    assert "agent.text.final" in event_types
    assert "audio.chunk" in event_types
    assert any(event["payload"].get("stage") == "stt_final" for event in events if event["type"] == "pipeline.stage")
    assert any(
        event["payload"].get("name") == "time_to_first_text_ms"
        for event in events
        if event["type"] == "latency.metric"
    )


def test_live_agent_receives_hidden_intent_inference_context_but_events_keep_raw_transcript():
    events = []
    captured_transcripts = []
    sessions = []

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
        sessions.append(session)
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
    assert sessions[0].transcript[-2:] == [
        {"role": "user", "content": "You're what's up can you hear me now"},
        {"role": "assistant", "content": "I understood the intent from context."},
    ]


def test_live_speech_started_marks_hearing_and_cancels_pending_proactive_prompt():
    events = []

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(
                normalized_mode="live",
                proactive_effective_enabled=True,
                proactive_startup_greeting_delay_ms=100,
            ),
        )
        session.audio_stream_active = True
        session.received_audio_frame = True

        await session._maybe_schedule_startup_greeting()
        assert session.pending_proactive_task is not None

        await session.handle_live_transcript(
            {
                "type": "speech_started",
                "timestamp": 0.64,
                "provider": "deepgram",
            }
        )

        assert session.user_has_spoken is True
        assert session.pending_proactive_task is None
        await session.close()

    asyncio.run(run_session())

    assert any(
        event["type"] == "status.changed" and event["payload"].get("state") == "hearing"
        for event in events
    )
    assert any(
        event["type"] == "pipeline.stage"
        and event["payload"].get("stage") == "stt_speech_started"
        and event["payload"].get("timestamp") == 0.64
        for event in events
    )
    cancellation = next(event for event in events if event["type"] == "proactive.cancelled")
    assert cancellation["payload"]["reason"] == "live_speech_started"


def test_live_speech_started_does_not_interrupt_active_response_before_transcript_text():
    events = []
    response_started = asyncio.Event()
    release_response = asyncio.Event()

    settings = proactive_settings(
        normalized_mode="live",
        groq_api_key="test-key",
        proactive_effective_enabled=False,
    )

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession("sess_test", send_event, settings)

        async def stream_response(response_id):
            response_started.set()
            await release_response.wait()
            await session._emit_agent_text_chunks(response_id, ["Still answering. "])
            return "Still answering."

        session._stream_live_agent_response = stream_response

        await session.handle_live_transcript(
            {
                "text": "First question",
                "confidence": 0.99,
                "is_final": True,
                "speech_final": True,
                "provider": "deepgram",
            }
        )
        await asyncio.wait_for(response_started.wait(), timeout=0.15)

        await session.handle_live_transcript(
            {
                "type": "speech_started",
                "timestamp": 1.2,
                "provider": "deepgram",
            }
        )

        assert session.current_task is not None
        assert not session.current_task.done()
        assert "interruption.started" not in [event["type"] for event in events]

        release_response.set()
        assert session.current_task is not None
        await session.current_task
        await session.close()

    asyncio.run(run_session())


def proactive_settings(**overrides):
    settings = {
        "normalized_mode": "mock",
        "groq_api_key": None,
        "groq_model": "llama-3.1-8b-instant",
        "groq_temperature": 0.7,
        "persona": "Be concise.",
        "cartesia_api_key": None,
        "cartesia_voice_id": None,
        "proactive_effective_enabled": True,
        "partial_idle_finalize_ms": 650,
        "proactive_startup_greeting_delay_ms": 1,
        "proactive_silence_timeout_ms": 5000,
        "proactive_repeat_cooldown_ms": 8000,
        "proactive_max_consecutive_prompts": 3,
        "proactive_failure_backoff_threshold": 2,
        "proactive_failure_backoff_ms": 30000,
        "proactive_contextual_followups_enabled": True,
    }
    settings.update(overrides)
    return SimpleNamespace(**settings)


async def wait_for_pending_proactive_turn(session):
    assert session.pending_proactive_task is not None
    await session.pending_proactive_task
    if session.current_task is not None:
        await session.current_task


def mock_speech_frame(sample_count=12000, amplitude=12000):
    return int(amplitude).to_bytes(2, "little", signed=True) * sample_count


def test_startup_greeting_waits_for_first_audio_frame():
    events = []

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession("sess_test", send_event, proactive_settings())
        await session.configure_audio(
            {
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration_ms": 20,
            }
        )
        await asyncio.sleep(0.01)
        await session.close()

    asyncio.run(run_session())

    event_types = [event["type"] for event in events]
    assert "proactive.triggered" not in event_types
    assert "agent.text.final" not in event_types


def test_startup_greeting_fires_after_first_audio_frame_with_metadata():
    events = []
    transcript = []

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession("sess_test", send_event, proactive_settings())
        await session.configure_audio(
            {
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration_ms": 20,
            }
        )
        await session.receive_audio(b"\x00\x00" * 100)
        await wait_for_pending_proactive_turn(session)
        transcript.extend(session.transcript)

    asyncio.run(run_session())

    event_types = [event["type"] for event in events]
    assert "proactive.triggered" in event_types
    final = next(event for event in events if event["type"] == "agent.text.final")
    audio = next(event for event in events if event["type"] == "audio.chunk")
    assert final["payload"]["proactive"] is True
    assert final["payload"]["trigger_reason"] == "startup_greeting"
    assert final["payload"]["proactive_turn_id"].startswith("pro_")
    assert audio["payload"]["proactive"] is True
    assert audio["payload"]["trigger_reason"] == "startup_greeting"
    assert final["payload"]["text"] == "I'm on the line with you."
    assert transcript == [{"role": "assistant", "content": final["payload"]["text"]}]


def test_live_startup_greeting_uses_agent_path_instead_of_time_bucket_copy():
    events = []
    captured_transcripts = []

    class FakeAgent:
        async def stream_response(self, transcript):
            captured_transcripts.append(transcript)
            yield "Hey, I'm here with you."

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(
                normalized_mode="live",
                groq_api_key="test-key",
                proactive_startup_greeting_delay_ms=1,
            ),
        )
        session.agent = FakeAgent()
        session.audio_stream_active = True
        session.received_audio_frame = True

        await session._maybe_schedule_startup_greeting()
        await wait_for_pending_proactive_turn(session)
        await session.close()

    asyncio.run(run_session())

    assert captured_transcripts
    internal_instruction = captured_transcripts[0][-1]
    assert internal_instruction["role"] == "user"
    assert "open phone call" in internal_instruction["content"]
    assert "still on the line" in internal_instruction["content"]
    assert "Do not mention the time of day" in internal_instruction["content"]

    final = next(event for event in events if event["type"] == "agent.text.final")
    assert final["payload"]["trigger_reason"] == "startup_greeting"
    assert final["payload"]["text"] == "Hey, I'm here with you."
    assert any(
        event["type"] == "pipeline.stage"
        and event["payload"].get("stage") == "llm_streaming"
        and event["payload"].get("provider") == "groq"
        and event["payload"].get("trigger_reason") == "startup_greeting"
        for event in events
    )


def test_user_speech_cancels_pending_startup_greeting():
    events = []

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession("sess_test", send_event, proactive_settings())
        await session.configure_audio(
            {
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration_ms": 20,
            }
        )
        await session.receive_audio(b"\x00\x00" * 100)
        assert session.pending_proactive_task is not None
        await session.receive_audio(mock_speech_frame())
        assert session.current_task is not None
        await session.current_task

    asyncio.run(run_session())

    event_types = [event["type"] for event in events]
    assert "proactive.cancelled" in event_types
    assert "proactive.triggered" not in event_types
    cancellation = next(event for event in events if event["type"] == "proactive.cancelled")
    assert cancellation["payload"]["trigger_reason"] == "startup_greeting"
    assert cancellation["payload"]["reason"] == "mock_audio_threshold"


def test_silent_mock_audio_does_not_create_fake_user_turn_or_beep_loop():
    events = []

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(proactive_startup_greeting_delay_ms=100),
        )
        await session.configure_audio(
            {
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration_ms": 20,
            }
        )
        await session.receive_audio(b"\x00\x00" * 100)
        assert session.pending_proactive_task is not None

        await session.receive_audio(b"\x00\x00" * 12000)
        await asyncio.sleep(0)

        assert session.current_task is None
        assert session.user_has_spoken is False
        await session.close()

    asyncio.run(run_session())

    event_types = [event["type"] for event in events]
    assert "transcript.final" not in event_types
    assert "agent.text.final" not in event_types
    assert "audio.chunk" not in event_types
    assert not any(
        event["type"] == "proactive.cancelled"
        and event["payload"].get("reason") == "mock_audio_threshold"
        for event in events
    )


def test_idle_silence_nudge_fires_after_startup_greeting():
    events = []

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(proactive_silence_timeout_ms=1, proactive_repeat_cooldown_ms=5),
        )
        await session.configure_audio(
            {
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration_ms": 20,
            }
        )
        await session.receive_audio(b"\x00\x00" * 100)
        await wait_for_pending_proactive_turn(session)
        assert session.pending_proactive_task is not None
        await wait_for_pending_proactive_turn(session)
        await session.close()

    asyncio.run(run_session())

    proactive_finals = [
        event["payload"]
        for event in events
        if event["type"] == "agent.text.final" and event["payload"].get("proactive")
    ]
    assert [payload["trigger_reason"] for payload in proactive_finals] == [
        "startup_greeting",
        "silence_nudge",
    ]
    assert "why did you go silent" not in proactive_finals[-1]["text"].lower()
    assert any(
        event["type"] == "proactive.state" and event["payload"].get("state") == "idle_monitoring"
        for event in events
    )


def test_repeated_silence_nudges_stop_at_backoff_limit():
    events = []

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(
                proactive_silence_timeout_ms=1,
                proactive_repeat_cooldown_ms=1,
                proactive_max_consecutive_prompts=3,
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
        await session.receive_audio(b"\x00\x00" * 100)
        await wait_for_pending_proactive_turn(session)
        await wait_for_pending_proactive_turn(session)
        await wait_for_pending_proactive_turn(session)

    asyncio.run(run_session())

    proactive_finals = [
        event["payload"]
        for event in events
        if event["type"] == "agent.text.final" and event["payload"].get("proactive")
    ]
    assert [payload["trigger_reason"] for payload in proactive_finals] == [
        "startup_greeting",
        "silence_nudge",
        "silence_nudge",
    ]
    assert [payload["text"] for payload in proactive_finals if payload["trigger_reason"] == "silence_nudge"] == [
        "I'm here with you on the line.",
        "Still with you when you're ready.",
    ]
    assert any(
        event["type"] == "proactive.skipped"
        and event["payload"].get("skip_reason") == "max_consecutive_prompts_reached"
        for event in events
    )
    assert any(
        event["type"] == "proactive.state" and event["payload"].get("state") == "backed_off"
        for event in events
    )


def test_audio_stop_cancels_pending_silence_nudge():
    events = []

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(proactive_silence_timeout_ms=25),
        )
        await session.configure_audio(
            {
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration_ms": 20,
            }
        )
        await session.receive_audio(b"\x00\x00" * 100)
        await wait_for_pending_proactive_turn(session)
        assert session.pending_proactive_task is not None
        await session.stop_audio()
        await asyncio.sleep(0.03)

    asyncio.run(run_session())

    cancellations = [event for event in events if event["type"] == "proactive.cancelled"]
    assert cancellations[-1]["payload"] == {
        "trigger_reason": "silence_nudge",
        "reason": "audio_stop",
    }
    assert [
        event["payload"]["trigger_reason"]
        for event in events
        if event["type"] == "proactive.triggered"
    ] == ["startup_greeting"]


def test_user_speech_cancels_pending_silence_nudge_and_resets_count():
    events = []
    consecutive_counts = []

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(proactive_silence_timeout_ms=25),
        )
        await session.configure_audio(
            {
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration_ms": 20,
            }
        )
        await session.receive_audio(b"\x00\x00" * 100)
        await wait_for_pending_proactive_turn(session)
        assert session.pending_proactive_task is not None
        await session.receive_audio(mock_speech_frame())
        assert session.current_task is not None
        await session.current_task
        consecutive_counts.append(session.consecutive_proactive_prompts)
        await session.close()

    asyncio.run(run_session())

    assert consecutive_counts == [0]
    assert any(
        event["type"] == "proactive.cancelled"
        and event["payload"] == {
            "trigger_reason": "silence_nudge",
            "reason": "mock_audio_threshold",
        }
        for event in events
    )


def test_raw_silent_audio_does_not_reset_pending_silence_nudge():
    events = []

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(proactive_silence_timeout_ms=1, proactive_repeat_cooldown_ms=5),
        )
        await session.configure_audio(
            {
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration_ms": 20,
            }
        )
        await session.receive_audio(b"\x00\x00" * 100)
        await wait_for_pending_proactive_turn(session)
        assert session.pending_proactive_task is not None
        await session.receive_audio(b"\x00\x00" * 100)
        await wait_for_pending_proactive_turn(session)
        await session.close()

    asyncio.run(run_session())

    assert [
        event["payload"]["trigger_reason"]
        for event in events
        if event["type"] == "proactive.triggered"
    ] == ["startup_greeting", "silence_nudge"]


def test_contextual_followup_fires_after_mock_user_turn_with_recent_context():
    events = []

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(
                proactive_startup_greeting_delay_ms=100,
                proactive_silence_timeout_ms=1,
                proactive_repeat_cooldown_ms=5,
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
        await session.receive_audio(b"\x00\x00" * 100)
        assert session.pending_proactive_task is not None

        await session.receive_audio(mock_speech_frame())
        assert session.current_task is not None
        await session.current_task

        assert session.pending_proactive_trigger_reason == "contextual_follow_up"
        await wait_for_pending_proactive_turn(session)
        await session.close()

    asyncio.run(run_session())

    proactive_finals = [
        event["payload"]
        for event in events
        if event["type"] == "agent.text.final" and event["payload"].get("proactive")
    ]
    assert [payload["trigger_reason"] for payload in proactive_finals] == ["contextual_follow_up"]
    assert proactive_finals[0]["text"] == "A quick thought while we're on the line?"
    assert "why did you go silent" not in proactive_finals[0]["text"].lower()


def test_contextual_followup_live_uses_agent_path_with_proactive_instruction():
    events = []
    captured_transcripts = []

    class FakeAgent:
        async def stream_response(self, transcript):
            captured_transcripts.append(transcript)
            yield "A quick "
            yield "follow-up?"

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(
                normalized_mode="live",
                groq_api_key="test-key",
                proactive_silence_timeout_ms=1,
                proactive_repeat_cooldown_ms=5,
            ),
        )
        session.agent = FakeAgent()
        session.audio_stream_active = True
        session.received_audio_frame = True
        session.user_has_spoken = True
        session.startup_greeting_sent = True
        session.transcript = [
            {"role": "user", "content": "I am planning a customer onboarding script."},
            {"role": "assistant", "content": "We can shape that into a warm opening."},
        ]

        await session._schedule_next_idle_trigger(after_trigger_reason=None)
        assert session.pending_proactive_trigger_reason == "contextual_follow_up"
        await wait_for_pending_proactive_turn(session)
        await session.close()

    asyncio.run(run_session())

    assert captured_transcripts
    internal_instruction = captured_transcripts[0][-1]
    assert internal_instruction["role"] == "user"
    assert "one concise" in internal_instruction["content"]
    assert "open phone call" in internal_instruction["content"]
    assert "configured persona" in internal_instruction["content"]
    assert "Do not ask why the user went silent" in internal_instruction["content"]
    assert captured_transcripts[0][0]["content"] == "I am planning a customer onboarding script."

    proactive_final = next(
        event["payload"]
        for event in events
        if event["type"] == "agent.text.final" and event["payload"].get("proactive")
    )
    assert proactive_final["trigger_reason"] == "contextual_follow_up"
    assert proactive_final["text"] == "A quick follow-up?"
    assert any(
        event["type"] == "pipeline.stage"
        and event["payload"].get("stage") == "llm_streaming"
        and event["payload"].get("provider") == "groq"
        and event["payload"].get("trigger_reason") == "contextual_follow_up"
        for event in events
    )


def test_live_silence_nudge_uses_agent_path_instead_of_scripted_rotation():
    events = []
    captured_transcripts = []

    class FakeAgent:
        async def stream_response(self, transcript):
            captured_transcripts.append(transcript)
            yield "Still with you when you're ready."

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(
                normalized_mode="live",
                groq_api_key="test-key",
                proactive_silence_timeout_ms=1,
                proactive_repeat_cooldown_ms=5,
            ),
        )
        session.agent = FakeAgent()
        session.audio_stream_active = True
        session.received_audio_frame = True
        session.user_has_spoken = True
        session.startup_greeting_sent = True
        session.transcript = []

        await session._schedule_next_idle_trigger(after_trigger_reason=None)
        assert session.pending_proactive_trigger_reason == "silence_nudge"
        await wait_for_pending_proactive_turn(session)
        await session.close()

    asyncio.run(run_session())

    assert captured_transcripts
    internal_instruction = captured_transcripts[0][-1]
    assert internal_instruction["role"] == "user"
    assert "brief check-in" in internal_instruction["content"]
    assert "still on the line" in internal_instruction["content"]
    assert "Avoid repeating" in internal_instruction["content"]

    proactive_final = next(
        event["payload"]
        for event in events
        if event["type"] == "agent.text.final" and event["payload"].get("proactive")
    )
    assert proactive_final["trigger_reason"] == "silence_nudge"
    assert proactive_final["text"] == "Still with you when you're ready."
    assert any(
        event["type"] == "pipeline.stage"
        and event["payload"].get("stage") == "llm_streaming"
        and event["payload"].get("provider") == "groq"
        and event["payload"].get("trigger_reason") == "silence_nudge"
        for event in events
    )


def test_contextual_followup_skips_when_last_assistant_gave_next_step():
    events = []

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(proactive_silence_timeout_ms=25),
        )
        session.audio_stream_active = True
        session.received_audio_frame = True
        session.user_has_spoken = True
        session.startup_greeting_sent = True
        session.transcript = [
            {"role": "user", "content": "I am planning a customer onboarding script."},
            {"role": "assistant", "content": "The next step is drafting the opening line."},
        ]

        await session._schedule_next_idle_trigger(after_trigger_reason=None)
        assert session.pending_proactive_trigger_reason == "silence_nudge"
        await session.close()

    asyncio.run(run_session())

    idle_state = next(event for event in events if event["type"] == "proactive.state")
    assert idle_state["payload"]["state"] == "idle_monitoring"
    assert idle_state["payload"]["trigger_reason"] == "silence_nudge"


def test_proactive_provider_error_has_metadata_and_counts_failure():
    events = []
    failure_counts = []

    class FailingAgent:
        async def stream_response(self, transcript):
            raise RuntimeError("groq went sideways")
            yield ""

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(
                normalized_mode="live",
                groq_api_key="test-key",
                proactive_silence_timeout_ms=1,
            ),
        )
        session.agent = FailingAgent()
        session.audio_stream_active = True
        session.received_audio_frame = True
        session.user_has_spoken = True
        session.startup_greeting_sent = True
        session.transcript = [
            {"role": "user", "content": "I am planning a customer onboarding script."},
            {"role": "assistant", "content": "We can shape that into a warm opening."},
        ]

        await session._schedule_next_idle_trigger(after_trigger_reason=None)
        await wait_for_pending_proactive_turn(session)
        failure_counts.append(session.proactive_failures)
        await session.close()

    asyncio.run(run_session())

    assert failure_counts == [1]
    error_payload = next(event["payload"] for event in events if event["type"] == "error")
    assert error_payload["provider"] == "groq"
    assert error_payload["proactive"] is True
    assert error_payload["trigger_reason"] == "contextual_follow_up"
    assert error_payload["proactive_turn_id"].startswith("pro_")
    assert any(
        event["type"] == "pipeline.stage"
        and event["payload"].get("stage") == "groq_failed"
        and event["payload"].get("trigger_reason") == "contextual_follow_up"
        for event in events
    )
    assert not any(
        event["type"] in {"agent.text.final", "audio.chunk"}
        and event["payload"].get("proactive")
        for event in events
    )


def test_repeated_proactive_provider_failures_enter_backoff_but_reactive_turn_still_runs():
    events = []
    reactive_messages = []

    class FailingAgent:
        async def stream_response(self, transcript):
            raise RuntimeError("groq still down")
            yield ""

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(
                normalized_mode="live",
                groq_api_key="test-key",
                proactive_silence_timeout_ms=1,
                proactive_repeat_cooldown_ms=1,
                proactive_failure_backoff_threshold=2,
            ),
        )
        session.agent = FailingAgent()
        session.audio_stream_active = True
        session.received_audio_frame = True
        session.user_has_spoken = True
        session.startup_greeting_sent = True
        session.transcript = [
            {"role": "user", "content": "I am planning a customer onboarding script."},
            {"role": "assistant", "content": "We can shape that into a warm opening."},
        ]

        await session._schedule_next_idle_trigger(after_trigger_reason=None)
        await wait_for_pending_proactive_turn(session)

        await session._cancel_pending_proactive("test_second_failure_setup")
        session.transcript = [
            {"role": "user", "content": "I am planning a customer onboarding script."},
            {"role": "assistant", "content": "We can shape that into a warm opening."},
        ]
        await session._schedule_next_idle_trigger(after_trigger_reason=None)
        await wait_for_pending_proactive_turn(session)

        async def reactive_response(response_id):
            await session._emit_agent_text_chunks(response_id, ["Reactive still works. "])
            return "Reactive still works."

        session._stream_live_agent_response = reactive_response
        await session._run_agent_response("User asks a fresh question")
        reactive_messages.extend(
            event["payload"]["text"]
            for event in events
            if event["type"] == "agent.text.final" and event["payload"]["text"] == "Reactive still works."
        )
        await session.close()

    asyncio.run(run_session())

    assert any(
        event["type"] == "proactive.skipped"
        and event["payload"].get("skip_reason") == "failure_backoff"
        for event in events
    )
    assert any(
        event["type"] == "proactive.state" and event["payload"].get("state") == "backed_off"
        for event in events
    )
    assert reactive_messages == ["Reactive still works."]


def test_new_speech_during_proactive_tts_interrupts_without_stale_transcript():
    events = []
    proactive_tts_started = None

    class FakeAgent:
        async def stream_response(self, transcript):
            yield "Proactive follow-up?"

    async def send_event(event):
        events.append(event)

    async def run_session():
        nonlocal proactive_tts_started
        proactive_tts_started = asyncio.Event()
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(
                normalized_mode="live",
                groq_api_key="test-key",
                cartesia_api_key="cartesia-key",
                cartesia_voice_id=VALID_CARTESIA_VOICE_ID,
                cartesia_model="sonic-3",
                cartesia_sample_rate=16000,
                cartesia_version="2026-03-01",
                proactive_silence_timeout_ms=1,
            ),
        )
        session.agent = FakeAgent()
        session.audio_stream_active = True
        session.received_audio_frame = True
        session.user_has_spoken = True
        session.startup_greeting_sent = True
        session.transcript = [
            {"role": "user", "content": "I am planning a customer onboarding script."},
            {"role": "assistant", "content": "We can shape that into a warm opening."},
        ]

        async def stream_speech(response_id, message, final_detected_at, *, metadata=None):
            if metadata and metadata.get("proactive"):
                proactive_tts_started.set()
                await asyncio.Event().wait()
            await session._stream_mock_speech(response_id, final_detected_at, metadata=metadata)

        async def reactive_response(response_id):
            await session._emit_agent_text_chunks(response_id, ["Second answer. "])
            return "Second answer."

        session._stream_speech = stream_speech
        session._stream_live_agent_response = reactive_response

        await session._schedule_next_idle_trigger(after_trigger_reason=None)
        await session.pending_proactive_task
        await proactive_tts_started.wait()
        await session.handle_live_transcript(
            {
                "text": "Actually, answer this instead",
                "confidence": 0.98,
                "is_final": True,
                "speech_final": True,
                "provider": "deepgram",
            }
        )
        assert session.current_task is not None
        await session.current_task
        assert session.transcript == [
            {"role": "user", "content": "I am planning a customer onboarding script."},
            {"role": "assistant", "content": "We can shape that into a warm opening."},
            {"role": "user", "content": "Actually, answer this instead"},
            {"role": "assistant", "content": "Second answer."},
        ]
        await session.close()

    asyncio.run(run_session())

    interruption = next(event for event in events if event["type"] == "interruption.started")
    assert interruption["payload"]["reason"] == "user_speech_during_response"
    assert interruption["payload"]["proactive"] is True
    assert interruption["payload"]["trigger_reason"] == "contextual_follow_up"
    assert [event["payload"].get("provider") for event in events if event["type"] == "audio.chunk"] == ["mock"]


def test_proactive_tts_error_has_metadata_and_counts_failure():
    events = []
    failure_counts = []

    class FailingSynthesizer:
        async def stream_speech(self, message, *, context_id):
            yield {"type": "error", "message": "cartesia went sideways"}

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
        session.synthesizer = FailingSynthesizer()
        await session._stream_cartesia_speech(
            "resp_test",
            "Hello.",
            0,
            metadata={
                "proactive": True,
                "trigger_reason": "silence_nudge",
                "proactive_turn_id": "pro_test",
            },
        )
        failure_counts.append(session.proactive_failures)

    asyncio.run(run_session())

    assert failure_counts == [1]
    error_payload = next(event["payload"] for event in events if event["type"] == "error")
    assert error_payload["provider"] == "cartesia"
    assert error_payload["proactive"] is True
    assert error_payload["trigger_reason"] == "silence_nudge"
    assert error_payload["proactive_turn_id"] == "pro_test"


def test_tts_error_falls_back_to_audible_mock_audio():
    events = []

    class FailingSynthesizer:
        async def stream_speech(self, message, *, context_id):
            yield {"type": "error", "message": "bad voice"}

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(
                normalized_mode="live",
                cartesia_api_key="cartesia-key",
                cartesia_voice_id="6bf6d6c3-9d45-48fb-94a9-4840f83eb385",
                cartesia_model="sonic-3",
                cartesia_sample_rate=16000,
                cartesia_version="2026-03-01",
            ),
        )
        session.synthesizer = FailingSynthesizer()
        await session._stream_speech("resp_test", "Hello.", 0)

    asyncio.run(run_session())

    assert any(event["payload"].get("provider") == "cartesia" for event in events if event["type"] == "error")
    assert any(
        event["payload"].get("stage") == "tts_fallback"
        and event["payload"].get("fallback_from") == "cartesia"
        for event in events
        if event["type"] == "pipeline.stage"
    )
    audio_events = [event for event in events if event["type"] == "audio.chunk"]
    assert audio_events
    assert audio_events[-1]["payload"]["provider"] == "mock"
    assert audio_events[-1]["payload"]["audio"]


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


def test_live_cartesia_audio_can_start_before_llm_final_text_event():
    events = []
    audio = base64.b64encode(b"\x00\x00" * 120).decode("ascii")
    spoken_chunks = []

    class SlowAgent:
        async def stream_response(self, transcript):
            yield "This first sentence can speak now. "
            await asyncio.sleep(0)
            yield "Second sentence arrives later."

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
            ),
        )
        session.agent = SlowAgent()
        session.synthesizer = StreamingSynthesizer()
        await session._run_agent_response("Hello?")

    asyncio.run(run_session())

    event_types = [event["type"] for event in events]
    assert event_types.index("audio.chunk") < event_types.index("agent.text.final")
    assert spoken_chunks == ["This first sentence can speak now. ", "Second sentence arrives later. "]


def test_cartesia_streaming_failure_does_not_truncate_final_agent_text():
    events = []
    spoken_chunks = []

    class TwoChunkAgent:
        async def stream_response(self, transcript):
            yield "This first sentence can speak now. "
            await asyncio.sleep(0)
            yield "Second sentence should still reach the final text."

    class FailingStreamingSynthesizer:
        async def stream_speech_chunks(self, chunks, *, context_id):
            async for chunk in chunks:
                spoken_chunks.append(chunk)
                yield {"type": "error", "message": "bad ssml", "context_id": context_id, "done": True}
                return

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
        session.agent = TwoChunkAgent()
        session.synthesizer = FailingStreamingSynthesizer()
        await session._run_agent_response("Hello?")

    asyncio.run(run_session())

    final = next(event for event in events if event["type"] == "agent.text.final")
    assert final["payload"]["text"] == (
        "This first sentence can speak now. "
        "Second sentence should still reach the final text."
    )
    assert "<break" not in final["payload"]["text"]
    assert spoken_chunks == ["This first sentence can speak now. "]
    assert any(event["type"] == "error" and event["payload"].get("provider") == "cartesia" for event in events)
    assert any(
        event["type"] == "pipeline.stage"
        and event["payload"].get("stage") == "tts_fallback"
        and event["payload"].get("fallback_from") == "cartesia"
        for event in events
    )


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
        'Well,<break time="180ms"/> I think you meant<break time="250ms"/> the Deepgram model is missing words. '
    ]
    assert any(
        event["type"] == "pipeline.stage"
        and event["payload"].get("stage") == "tts_speech_direction"
        and event["payload"].get("directed") is True
        for event in events
    )


def test_cartesia_uses_plain_text_when_speech_direction_fails(monkeypatch):
    events = []
    audio = base64.b64encode(b"\x00\x00" * 120).decode("ascii")
    spoken_messages = []

    class StreamingSynthesizer:
        async def stream_speech(self, message, *, context_id):
            spoken_messages.append(message)
            yield {"type": "chunk", "audio": audio, "context_id": context_id}
            yield {"type": "done", "context_id": context_id, "done": True}

    def failing_director(text, config):
        raise RuntimeError("direction unavailable")

    monkeypatch.setattr(mock_conversation, "direct_speech_for_cartesia", failing_director)

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
        session.synthesizer = StreamingSynthesizer()
        result = await session._stream_cartesia_speech("resp_test", "Set API mode.", 0)
        assert result is True

    asyncio.run(run_session())

    assert spoken_messages == ["Set API mode."]
    assert any(
        event["type"] == "error"
        and event["payload"].get("provider") == "cartesia"
        and "Speech direction failed: direction unavailable" in event["payload"].get("message", "")
        for event in events
    )


def test_live_audio_input_gain_boosts_frames_before_deepgram(monkeypatch):
    events = []
    transcribers = []

    class CapturingTranscriber:
        def __init__(self, **kwargs):
            self.sent_frames = []
            transcribers.append(self)

        async def start(self):
            pass

        async def send_audio(self, frame):
            self.sent_frames.append(frame)

        async def close(self):
            pass

    monkeypatch.setattr(mock_conversation, "DeepgramStreamingTranscriber", CapturingTranscriber)

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
                input_gain=2.5,
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
        await session.receive_audio(int(1000).to_bytes(2, "little", signed=True) * 320)
        await session.close()

    asyncio.run(run_session())

    assert transcribers
    assert transcribers[0].sent_frames
    assert transcribers[0].sent_frames[0] == int(2500).to_bytes(2, "little", signed=True) * 320

    audio_input = next(event for event in events if event["type"] == "audio.input")
    assert audio_input["payload"]["input_gain"] == 2.5
    assert audio_input["payload"]["raw_rms"] < audio_input["payload"]["rms"]


def test_deepgram_start_failure_falls_back_to_mock_turn_detection(monkeypatch):
    events = []

    class FailingTranscriber:
        def __init__(self, **kwargs):
            pass

        async def start(self):
            raise RuntimeError("deepgram unavailable")

        async def close(self):
            pass

    monkeypatch.setattr(mock_conversation, "DeepgramStreamingTranscriber", FailingTranscriber)

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
        assert session.transcriber is None
        await session.receive_audio(mock_speech_frame())
        assert session.current_task is not None
        await session.current_task

    asyncio.run(run_session())

    assert any(event["payload"].get("provider") == "deepgram" for event in events if event["type"] == "error")
    assert any(
        event["payload"].get("stage") == "stt_listening" and event["payload"].get("provider") == "mock"
        for event in events
        if event["type"] == "pipeline.stage"
    )
    assert any(
        event["payload"].get("provider") == "mock"
        for event in events
        if event["type"] == "transcript.final"
    )


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


def test_close_cancels_pending_proactive_timer_without_triggering_turn():
    events = []

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession(
            "sess_test",
            send_event,
            proactive_settings(proactive_startup_greeting_delay_ms=50),
        )
        await session.configure_audio(
            {
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration_ms": 20,
            }
        )
        await session.receive_audio(b"\x00\x00" * 100)
        assert session.pending_proactive_task is not None
        await session.close()
        await asyncio.sleep(0.06)
        assert session.pending_proactive_task is None

    asyncio.run(run_session())

    assert "proactive.triggered" not in [event["type"] for event in events]
    assert "agent.text.final" not in [event["type"] for event in events]


def test_final_transcript_without_speech_final_buffers_until_utterance_end():
    events = []
    settings = SimpleNamespace(
        normalized_mode="live",
        groq_api_key=None,
        groq_model="llama-3.1-8b-instant",
        groq_temperature=0.7,
        persona="Be concise.",
        cartesia_api_key=None,
        cartesia_voice_id=None,
        deepgram_endpointing_ms=300,
        partial_idle_finalize_ms=650,
    )

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession("sess_test", send_event, settings)
        await session.handle_live_transcript(
            {
                "text": "Can you hear me now?",
                "confidence": 0.95,
                "is_final": True,
                "speech_final": False,
                "provider": "deepgram",
            }
        )
        assert session.current_task is None
        await session.handle_live_transcript(
            {
                "type": "utterance_end",
                "last_word_end": 1.25,
                "provider": "deepgram",
            }
        )
        assert session.current_task is not None
        await session.current_task

    asyncio.run(run_session())

    event_types = [event["type"] for event in events]
    assert "agent.text.delta" in event_types
    assert "agent.text.final" in event_types
    assert "audio.chunk" in event_types
    assert any(
        event["payload"].get("reason") == "utterance_end"
        for event in events
        if event["type"] == "pipeline.stage"
    )


def test_partial_transcript_idle_timeout_starts_agent_response():
    events = []
    settings = SimpleNamespace(
        normalized_mode="live",
        groq_api_key=None,
        groq_model="llama-3.1-8b-instant",
        groq_temperature=0.7,
        persona="Be concise.",
        cartesia_api_key=None,
        cartesia_voice_id=None,
        deepgram_endpointing_ms=1000,
        partial_idle_finalize_ms=1,
    )

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession("sess_test", send_event, settings)
        await session.handle_live_transcript(
            {
                "text": "You're what's up? Can you hear me now?",
                "confidence": 0.91,
                "is_final": False,
                "speech_final": False,
                "provider": "deepgram",
            }
        )
        assert session.pending_transcript_task is not None
        await asyncio.wait_for(session.pending_transcript_task, timeout=0.15)
        if session.current_task is not None:
            await session.current_task

    asyncio.run(run_session())

    event_types = [event["type"] for event in events]
    assert "transcript.final" in event_types
    assert "agent.text.delta" in event_types
    assert "agent.text.final" in event_types
    assert "audio.chunk" in event_types
    assert any(
        event["payload"].get("reason") == "partial_idle_timeout"
        for event in events
        if event["type"] == "pipeline.stage"
    )
    assert any(
        event["payload"].get("name") == "partial_idle_finalize_wait_ms"
        and event["payload"].get("value_ms") == 1
        for event in events
        if event["type"] == "latency.metric"
    )


def test_duplicate_active_turn_transcript_does_not_interrupt_response():
    events = []
    settings = SimpleNamespace(
        normalized_mode="live",
        groq_api_key="test-key",
        groq_model="llama-3.1-8b-instant",
        groq_temperature=0.7,
        persona="Be concise.",
        cartesia_api_key=None,
        cartesia_voice_id=None,
        deepgram_endpointing_ms=300,
    )
    response_started = asyncio.Event()
    release_response = asyncio.Event()

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession("sess_test", send_event, settings)

        async def response_for_duplicate_test(response_id):
            response_started.set()
            await release_response.wait()
            await session._emit_agent_text_chunks(response_id, ["Holding the line. "])
            return "Holding the line."

        session._stream_live_agent_response = response_for_duplicate_test

        await session.handle_live_transcript(
            {
                "text": "Can you hear me now?",
                "confidence": 0.99,
                "is_final": True,
                "speech_final": True,
                "provider": "deepgram",
            }
        )
        await response_started.wait()
        await session.handle_live_transcript(
            {
                "text": "Can you hear me now?",
                "confidence": 0.99,
                "is_final": True,
                "speech_final": True,
                "provider": "deepgram",
            }
        )
        release_response.set()
        assert session.current_task is not None
        await session.current_task

    asyncio.run(run_session())

    assert "interruption.started" not in [event["type"] for event in events]


def test_partial_transcript_during_active_response_does_not_interrupt():
    events = []
    settings = SimpleNamespace(
        normalized_mode="live",
        groq_api_key="test-key",
        groq_model="llama-3.1-8b-instant",
        groq_temperature=0.7,
        persona="Be concise.",
        cartesia_api_key=None,
        cartesia_voice_id=None,
        deepgram_endpointing_ms=300,
    )
    first_response_started = asyncio.Event()
    release_response = asyncio.Event()

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession("sess_test", send_event, settings)

        async def stream_response(response_id):
            first_response_started.set()
            await release_response.wait()
            await session._emit_agent_text_chunks(response_id, ["Still answering. "])
            return "Still answering."

        session._stream_live_agent_response = stream_response

        await session.handle_live_transcript(
            {
                "text": "First question",
                "confidence": 0.99,
                "is_final": True,
                "speech_final": True,
                "provider": "deepgram",
            }
        )
        await first_response_started.wait()
        await session.handle_live_transcript(
            {
                "text": "background noise maybe",
                "confidence": 0.98,
                "is_final": False,
                "speech_final": False,
                "provider": "deepgram",
            }
        )
        release_response.set()
        assert session.current_task is not None
        await session.current_task

    asyncio.run(run_session())

    assert "interruption.started" not in [event["type"] for event in events]


def test_assistant_echo_during_active_response_does_not_interrupt():
    events = []
    settings = SimpleNamespace(
        normalized_mode="live",
        groq_api_key="test-key",
        groq_model="llama-3.1-8b-instant",
        groq_temperature=0.7,
        persona="Be concise.",
        cartesia_api_key=None,
        cartesia_voice_id=None,
        deepgram_endpointing_ms=300,
    )
    response_text_started = asyncio.Event()
    release_response = asyncio.Event()

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession("sess_test", send_event, settings)

        async def stream_response(response_id):
            await session._emit_agent_text_chunks(response_id, ["I am here and listening closely. "])
            response_text_started.set()
            await release_response.wait()
            return "I am here and listening closely."

        session._stream_live_agent_response = stream_response

        await session.handle_live_transcript(
            {
                "text": "First question",
                "confidence": 0.99,
                "is_final": True,
                "speech_final": True,
                "provider": "deepgram",
            }
        )
        await response_text_started.wait()
        await session.handle_live_transcript(
            {
                "text": "I am here and listening closely",
                "confidence": 0.98,
                "is_final": True,
                "speech_final": True,
                "provider": "deepgram",
            }
        )
        release_response.set()
        assert session.current_task is not None
        await session.current_task

    asyncio.run(run_session())

    assert "interruption.started" not in [event["type"] for event in events]


def test_new_speech_during_active_response_interrupts_and_starts_next_turn():
    events = []
    settings = SimpleNamespace(
        normalized_mode="live",
        groq_api_key="test-key",
        groq_model="llama-3.1-8b-instant",
        groq_temperature=0.7,
        persona="Be concise.",
        cartesia_api_key=None,
        cartesia_voice_id=None,
        deepgram_endpointing_ms=300,
    )
    first_response_started = asyncio.Event()

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession("sess_test", send_event, settings)

        async def stream_response(response_id):
            active_user_text = session.transcript[-1]["content"]
            if active_user_text == "First question":
                first_response_started.set()
                await asyncio.Event().wait()
            await session._emit_agent_text_chunks(response_id, ["Second answer. "])
            return "Second answer."

        session._stream_live_agent_response = stream_response

        await session.handle_live_transcript(
            {
                "text": "First question",
                "confidence": 0.99,
                "is_final": True,
                "speech_final": True,
                "provider": "deepgram",
            }
        )
        await first_response_started.wait()
        await session.handle_live_transcript(
            {
                "text": "Actually, different question",
                "confidence": 0.98,
                "is_final": True,
                "speech_final": True,
                "provider": "deepgram",
            }
        )
        assert session.current_task is not None
        await session.current_task
        assert session.transcript == [
            {"role": "user", "content": "First question"},
            {"role": "user", "content": "Actually, different question"},
            {"role": "assistant", "content": "Second answer."},
        ]

    asyncio.run(run_session())

    event_types = [event["type"] for event in events]
    assert "interruption.started" in event_types
    assert event_types.count("agent.text.final") == 1
    interruption = next(event for event in events if event["type"] == "interruption.started")
    assert interruption["payload"]["interrupted_response_id"].startswith("resp_")
    assert interruption["payload"]["next_user_text"] == "Actually, different question"


def test_new_speech_during_tts_does_not_record_interrupted_assistant_turn():
    events = []
    settings = SimpleNamespace(
        normalized_mode="live",
        groq_api_key="test-key",
        groq_model="llama-3.1-8b-instant",
        groq_temperature=0.7,
        persona="Be concise.",
        cartesia_api_key="cartesia-key",
        cartesia_voice_id=VALID_CARTESIA_VOICE_ID,
        cartesia_model="sonic-3",
        cartesia_sample_rate=16000,
        cartesia_version="2026-03-01",
        deepgram_endpointing_ms=300,
    )
    first_tts_started = asyncio.Event()

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession("sess_test", send_event, settings)

        async def stream_response(response_id):
            active_user_text = session.transcript[-1]["content"]
            if active_user_text == "First question":
                await session._emit_agent_text_chunks(response_id, ["First answer. "])
                return "First answer."
            await session._emit_agent_text_chunks(response_id, ["Second answer. "])
            return "Second answer."

        async def stream_speech(response_id, message, final_detected_at, *, metadata=None):
            if message == "First answer.":
                first_tts_started.set()
                await asyncio.Event().wait()
            await session._stream_mock_speech(response_id, final_detected_at, metadata=metadata)
            return True

        session._stream_live_agent_response = stream_response
        session._stream_cartesia_speech = stream_speech

        await session.handle_live_transcript(
            {
                "text": "First question",
                "confidence": 0.99,
                "is_final": True,
                "speech_final": True,
                "provider": "deepgram",
            }
        )
        await first_tts_started.wait()
        await session.handle_live_transcript(
            {
                "text": "Second question",
                "confidence": 0.98,
                "is_final": True,
                "speech_final": True,
                "provider": "deepgram",
            }
        )
        assert session.current_task is not None
        await session.current_task
        assert session.transcript == [
            {"role": "user", "content": "First question"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer."},
        ]

    asyncio.run(run_session())

    event_types = [event["type"] for event in events]
    assert "interruption.started" in event_types
    assert event_types.count("agent.text.final") == 2
    assert event_types.count("audio.chunk") == 1


def test_dead_browser_send_stops_cartesia_without_provider_error():
    events = []
    settings = SimpleNamespace(
        normalized_mode="live",
        cartesia_api_key="cartesia-key",
        cartesia_voice_id=VALID_CARTESIA_VOICE_ID,
        cartesia_model="sonic-3",
        cartesia_sample_rate=16000,
        cartesia_version="2026-03-01",
    )

    async def send_event(event):
        if event["type"] == "audio.chunk":
            raise ClientConnectionClosed()
        events.append(event)

    class FakeSynthesizer:
        async def stream_speech(self, transcript, *, context_id=None):
            yield {
                "type": "chunk",
                "audio": base64.b64encode(b"\x00\x00" * 8).decode("ascii"),
            }

    async def run_session():
        session = MockConversationSession("sess_test", send_event, settings)
        session.synthesizer = FakeSynthesizer()
        try:
            await session._stream_cartesia_speech("resp_test", "Hello.", 0)
        except ClientConnectionClosed:
            pass
        assert session.closed is True

    asyncio.run(run_session())

    event_types = [event["type"] for event in events]
    assert "error" not in event_types
    assert not any(
        event["payload"].get("stage") == "cartesia_failed"
        for event in events
        if event["type"] == "pipeline.stage"
    )
