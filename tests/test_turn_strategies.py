"""Tests for the Pipecat turn-taking configuration builders.

Bug context: pipecat 1.2 moved VAD and interruption config from the transport
to the user aggregator. The old ``vad_analyzer=`` transport kwarg and
``allow_interruptions=`` PipelineParams kwarg were silently dropped by
pydantic, so VAD never ran — every final transcript ended the turn on a fixed
~0.35s fallback timer and any transcribed noise interrupted the bot. These
tests pin the new aggregator-based wiring in app/pipeline.py.
"""

import pytest
from pipecat.turns.user_start import (
    MinWordsUserTurnStartStrategy,
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)

import app.characters as characters_mod
from app.characters import Character, save_character
from app.config import Settings
from app.pipeline import (
    build_user_turn_strategies,
    build_vad_params,
    resolve_session_character_id,
)


def make_settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_build_vad_params_maps_settings(monkeypatch: pytest.MonkeyPatch):
    settings = make_settings(
        monkeypatch,
        VOICE_AGENT_VAD_CONFIDENCE="0.8",
        VOICE_AGENT_VAD_START_SECS="0.3",
        VOICE_AGENT_VAD_STOP_SECS="0.25",
        VOICE_AGENT_VAD_MIN_VOLUME="0.5",
    )
    params = build_vad_params(settings)
    assert params.confidence == 0.8
    assert params.start_secs == 0.3
    assert params.stop_secs == 0.25
    assert params.min_volume == 0.5


def test_vad_params_default_to_pipecat_recommended_values(monkeypatch: pytest.MonkeyPatch):
    params = build_vad_params(make_settings(monkeypatch))
    assert params.confidence == 0.7
    assert params.start_secs == 0.2
    assert params.stop_secs == 0.2
    assert params.min_volume == 0.6


def test_smart_turn_enabled_uses_turn_analyzer_stop_strategy(monkeypatch: pytest.MonkeyPatch):
    settings = make_settings(monkeypatch, VOICE_AGENT_SMART_TURN_STOP_SECS="2.5")
    strategies = build_user_turn_strategies(settings)
    assert len(strategies.stop) == 1
    stop = strategies.stop[0]
    assert isinstance(stop, TurnAnalyzerUserTurnStopStrategy)
    assert stop._turn_analyzer._params.stop_secs == 2.5


def test_smart_turn_disabled_falls_back_to_speech_timeout(monkeypatch: pytest.MonkeyPatch):
    settings = make_settings(
        monkeypatch,
        VOICE_AGENT_SMART_TURN_ENABLED="false",
        VOICE_AGENT_SPEECH_TIMEOUT_STOP_SECS="0.7",
    )
    strategies = build_user_turn_strategies(settings)
    assert len(strategies.stop) == 1
    stop = strategies.stop[0]
    assert isinstance(stop, SpeechTimeoutUserTurnStopStrategy)
    assert stop._user_speech_timeout == 0.7


def test_min_words_is_the_sole_start_strategy(monkeypatch: pytest.MonkeyPatch):
    # MinWords must be alone: a VAD start strategy would open the turn first and
    # the turn controller ignores later start triggers, so MinWords would never
    # gate interruptions while the bot speaks.
    settings = make_settings(monkeypatch, VOICE_AGENT_INTERRUPT_MIN_WORDS="3")
    strategies = build_user_turn_strategies(settings)
    assert len(strategies.start) == 1
    start = strategies.start[0]
    assert isinstance(start, MinWordsUserTurnStartStrategy)
    assert start._min_words == 3


def test_zero_min_words_restores_default_start_strategies(monkeypatch: pytest.MonkeyPatch):
    settings = make_settings(monkeypatch, VOICE_AGENT_INTERRUPT_MIN_WORDS="0")
    strategies = build_user_turn_strategies(settings)
    types = {type(s) for s in strategies.start}
    assert VADUserTurnStartStrategy in types
    assert TranscriptionUserTurnStartStrategy in types


def test_default_interruption_is_one_word(monkeypatch: pytest.MonkeyPatch):
    # Default gates barge-in on a single word: catches one-word interjections
    # ("stop", "wait") that a 2-word threshold would let the bot talk over, while
    # staying robust to non-speech noise.
    settings = make_settings(monkeypatch)
    strategies = build_user_turn_strategies(settings)
    assert len(strategies.start) == 1
    assert isinstance(strategies.start[0], MinWordsUserTurnStartStrategy)
    assert strategies.start[0]._min_words == 1


def _seed_characters(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(characters_mod, "CHARACTERS_DIR", tmp_path)
    save_character(Character(id="alpha", name="Alpha", role="tester"))
    save_character(Character(id="beta", name="Beta", role="tester"))


def test_resolve_session_character_honors_known_id(tmp_path, monkeypatch: pytest.MonkeyPatch):
    _seed_characters(tmp_path, monkeypatch)
    settings = make_settings(monkeypatch, VOICE_AGENT_DEFAULT_CHARACTER="alpha")
    assert resolve_session_character_id("beta", settings) == "beta"
    assert resolve_session_character_id("  BETA ", settings) == "beta"


def test_resolve_session_character_falls_back_to_default(tmp_path, monkeypatch: pytest.MonkeyPatch):
    _seed_characters(tmp_path, monkeypatch)
    settings = make_settings(monkeypatch, VOICE_AGENT_DEFAULT_CHARACTER="alpha")
    assert resolve_session_character_id(None, settings) == "alpha"
    assert resolve_session_character_id("", settings) == "alpha"
    assert resolve_session_character_id("ghost", settings) == "alpha"
