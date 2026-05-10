import asyncio
from types import SimpleNamespace

from app.mock_conversation import MockConversationSession


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


def test_final_transcript_without_speech_final_starts_agent_response():
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
        assert session.current_task is not None
        await session.current_task

    asyncio.run(run_session())

    event_types = [event["type"] for event in events]
    assert "agent.text.delta" in event_types
    assert "agent.text.final" in event_types
    assert "audio.chunk" in event_types
    assert any(
        event["payload"].get("reason") == "provider_final"
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
        deepgram_endpointing_ms=1,
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
        await session.pending_transcript_task
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
    release_response = asyncio.Event()

    async def send_event(event):
        events.append(event)

    async def run_session():
        session = MockConversationSession("sess_test", send_event, settings)

        async def slow_response(response_id):
            await release_response.wait()
            await session._emit_agent_text_chunks(response_id, ["Holding the line. "])
            return "Holding the line."

        session._stream_live_agent_response = slow_response

        await session.handle_live_transcript(
            {
                "text": "Can you hear me now?",
                "confidence": 0.99,
                "is_final": True,
                "speech_final": False,
                "provider": "deepgram",
            }
        )
        await asyncio.sleep(0)
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
        cartesia_voice_id="voice-id",
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

        async def stream_speech(response_id, message, final_detected_at):
            if message == "First answer.":
                first_tts_started.set()
                await asyncio.Event().wait()
            await session._stream_mock_speech(response_id, final_detected_at)

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
