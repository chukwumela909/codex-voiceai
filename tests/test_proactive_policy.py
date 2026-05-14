from types import SimpleNamespace

from app.proactive import (
    TRIGGER_CONTEXTUAL_FOLLOW_UP,
    TRIGGER_SILENCE_NUDGE,
    TRIGGER_STARTUP_GREETING,
    ProactiveContext,
    ProactivePolicy,
    ProactivePolicyConfig,
)


def ready_context(**overrides):
    defaults = {
        "session_state": "listening",
        "audio_stream_active": True,
        "received_audio_frame": True,
    }
    defaults.update(overrides)
    return ProactiveContext(**defaults)


def test_policy_config_can_be_built_from_settings():
    settings = SimpleNamespace(
        proactive_effective_enabled=True,
        proactive_startup_greeting_delay_ms=250,
        proactive_silence_timeout_ms=1500,
        proactive_repeat_cooldown_ms=2500,
        proactive_max_consecutive_prompts=2,
        proactive_failure_backoff_threshold=4,
        proactive_failure_backoff_ms=45000,
        proactive_contextual_followups_enabled=False,
    )

    config = ProactivePolicyConfig.from_settings(settings)

    assert config == ProactivePolicyConfig(
        enabled=True,
        startup_greeting_delay_ms=250,
        silence_timeout_ms=1500,
        repeat_cooldown_ms=2500,
        max_consecutive_prompts=2,
        failure_backoff_threshold=4,
        failure_backoff_ms=45000,
        contextual_followups_enabled=False,
    )


def test_disabled_policy_skips_without_retry():
    policy = ProactivePolicy(ProactivePolicyConfig(enabled=False))

    decision = policy.evaluate(TRIGGER_STARTUP_GREETING, ready_context(), now_ms=0)

    assert decision.allowed is False
    assert decision.event_type == "proactive.skipped"
    assert decision.skip_reason == "disabled"
    assert decision.will_retry is False


def test_startup_greeting_waits_for_audio_frame():
    policy = ProactivePolicy(ProactivePolicyConfig(enabled=True))
    context = ready_context(received_audio_frame=False)

    decision = policy.evaluate(TRIGGER_STARTUP_GREETING, context, now_ms=0)

    assert decision.allowed is False
    assert decision.skip_reason == "waiting_for_audio_frame"
    assert decision.will_retry is True


def test_startup_greeting_skips_when_user_spoke_first():
    policy = ProactivePolicy(ProactivePolicyConfig(enabled=True))

    decision = policy.evaluate(TRIGGER_STARTUP_GREETING, ready_context(user_has_spoken=True), now_ms=0)

    assert decision.allowed is False
    assert decision.skip_reason == "user_spoke_first"
    assert decision.will_retry is False


def test_allowed_decision_uses_trigger_event_and_payload():
    policy = ProactivePolicy(ProactivePolicyConfig(enabled=True))

    decision = policy.evaluate(TRIGGER_SILENCE_NUDGE, ready_context(consecutive_prompts=1), now_ms=1000)

    assert decision.allowed is True
    assert decision.event_type == "proactive.triggered"
    assert decision.payload["trigger_reason"] == TRIGGER_SILENCE_NUDGE
    assert decision.payload["consecutive_prompts"] == 1
    assert decision.payload["max_consecutive_prompts"] == 3


def test_cooldown_skip_reports_remaining_time():
    policy = ProactivePolicy(ProactivePolicyConfig(enabled=True))
    context = ready_context(cooldown_until_ms=5000)

    decision = policy.evaluate(TRIGGER_SILENCE_NUDGE, context, now_ms=3000)

    assert decision.allowed is False
    assert decision.skip_reason == "cooldown"
    assert decision.will_retry is True
    assert decision.payload["next_eligible_at_ms"] == 5000
    assert decision.payload["remaining_cooldown_ms"] == 2000


def test_max_consecutive_prompts_enters_backoff():
    policy = ProactivePolicy(ProactivePolicyConfig(enabled=True, max_consecutive_prompts=3))
    context = ready_context(consecutive_prompts=3)

    decision = policy.evaluate(TRIGGER_SILENCE_NUDGE, context, now_ms=0)

    assert decision.allowed is False
    assert decision.skip_reason == "max_consecutive_prompts_reached"
    assert decision.will_retry is False


def test_failure_backoff_blocks_proactive_prompt_without_blocking_policy_object():
    policy = ProactivePolicy(ProactivePolicyConfig(enabled=True, failure_backoff_threshold=2))
    context = ready_context(proactive_failures=2)

    decision = policy.evaluate(TRIGGER_SILENCE_NUDGE, context, now_ms=0)

    assert decision.allowed is False
    assert decision.skip_reason == "failure_backoff"
    assert decision.will_retry is False


def test_select_idle_trigger_prefers_contextual_followup_only_when_safe():
    policy = ProactivePolicy(ProactivePolicyConfig(enabled=True))

    assert policy.select_idle_trigger(ready_context(has_recent_user_context=True)) == TRIGGER_CONTEXTUAL_FOLLOW_UP
    assert (
        policy.select_idle_trigger(
            ready_context(has_recent_user_context=True, last_assistant_asked_question=True)
        )
        == TRIGGER_SILENCE_NUDGE
    )
    assert policy.select_idle_trigger(ready_context(has_recent_user_context=False)) == TRIGGER_SILENCE_NUDGE


def test_contextual_followup_skips_when_question_is_already_pending():
    policy = ProactivePolicy(ProactivePolicyConfig(enabled=True))
    context = ready_context(has_recent_user_context=True, last_assistant_asked_question=True)

    decision = policy.evaluate(TRIGGER_CONTEXTUAL_FOLLOW_UP, context, now_ms=0)

    assert decision.allowed is False
    assert decision.skip_reason == "question_already_pending"
    assert decision.will_retry is True
