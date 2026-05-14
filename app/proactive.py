from dataclasses import dataclass, field


TRIGGER_STARTUP_GREETING = "startup_greeting"
TRIGGER_SILENCE_NUDGE = "silence_nudge"
TRIGGER_CONTEXTUAL_FOLLOW_UP = "contextual_follow_up"

PROACTIVE_TRIGGER_REASONS = {
    TRIGGER_STARTUP_GREETING,
    TRIGGER_SILENCE_NUDGE,
    TRIGGER_CONTEXTUAL_FOLLOW_UP,
}


def setting_value(settings, effective_name: str, legacy_name: str, default: int) -> int:
    value = getattr(settings, effective_name, None)
    if value is not None:
        return value
    value = getattr(settings, legacy_name, None)
    if value is not None:
        return value
    return default


@dataclass(frozen=True)
class ProactivePolicyConfig:
    enabled: bool
    startup_greeting_delay_ms: int = 500
    silence_timeout_ms: int = 5000
    repeat_cooldown_ms: int = 8000
    max_consecutive_prompts: int = 3
    failure_backoff_threshold: int = 2
    failure_backoff_ms: int = 30000
    contextual_followups_enabled: bool = True

    @classmethod
    def from_settings(cls, settings) -> "ProactivePolicyConfig":
        return cls(
            enabled=getattr(settings, "proactive_effective_enabled", False),
            startup_greeting_delay_ms=getattr(settings, "proactive_startup_greeting_delay_ms", 500),
            silence_timeout_ms=setting_value(
                settings,
                "proactive_effective_silence_timeout_ms",
                "proactive_silence_timeout_ms",
                5000,
            ),
            repeat_cooldown_ms=setting_value(
                settings,
                "proactive_effective_repeat_cooldown_ms",
                "proactive_repeat_cooldown_ms",
                8000,
            ),
            max_consecutive_prompts=setting_value(
                settings,
                "proactive_effective_max_consecutive_prompts",
                "proactive_max_consecutive_prompts",
                3,
            ),
            failure_backoff_threshold=getattr(settings, "proactive_failure_backoff_threshold", 2),
            failure_backoff_ms=getattr(settings, "proactive_failure_backoff_ms", 30000),
            contextual_followups_enabled=getattr(settings, "proactive_contextual_followups_enabled", True),
        )


@dataclass(frozen=True)
class ProactiveContext:
    session_state: str = "listening"
    audio_stream_active: bool = False
    received_audio_frame: bool = False
    muted: bool = False
    closing: bool = False
    active_response: bool = False
    user_has_spoken: bool = False
    startup_greeting_sent: bool = False
    consecutive_prompts: int = 0
    proactive_failures: int = 0
    cooldown_until_ms: int | None = None
    has_recent_user_context: bool = False
    last_assistant_asked_question: bool = False
    last_assistant_has_next_step: bool = False


@dataclass(frozen=True)
class ProactiveDecision:
    event_type: str
    trigger_reason: str
    allowed: bool
    source_state: str
    skip_reason: str | None = None
    will_retry: bool = False
    payload: dict[str, object] = field(default_factory=dict)


class ProactivePolicy:
    def __init__(self, config: ProactivePolicyConfig) -> None:
        self.config = config

    def select_idle_trigger(self, context: ProactiveContext) -> str:
        if (
            self.config.contextual_followups_enabled
            and context.has_recent_user_context
            and not context.last_assistant_asked_question
            and not context.last_assistant_has_next_step
        ):
            return TRIGGER_CONTEXTUAL_FOLLOW_UP
        return TRIGGER_SILENCE_NUDGE

    def evaluate(self, trigger_reason: str, context: ProactiveContext, *, now_ms: int) -> ProactiveDecision:
        if trigger_reason not in PROACTIVE_TRIGGER_REASONS:
            return self._skip(trigger_reason, context, "unsupported_trigger", will_retry=False)

        common_skip = self._common_skip_reason(context, now_ms)
        if common_skip:
            skip_reason, will_retry = common_skip
            return self._skip(trigger_reason, context, skip_reason, will_retry=will_retry, now_ms=now_ms)

        if trigger_reason == TRIGGER_STARTUP_GREETING:
            if context.startup_greeting_sent:
                return self._skip(trigger_reason, context, "startup_greeting_already_sent", will_retry=False)
            if context.user_has_spoken:
                return self._skip(trigger_reason, context, "user_spoke_first", will_retry=False)
        else:
            if context.consecutive_prompts >= self.config.max_consecutive_prompts:
                return self._skip(trigger_reason, context, "max_consecutive_prompts_reached", will_retry=False)

        if trigger_reason == TRIGGER_CONTEXTUAL_FOLLOW_UP:
            contextual_skip = self._contextual_followup_skip_reason(context)
            if contextual_skip:
                skip_reason, will_retry = contextual_skip
                return self._skip(trigger_reason, context, skip_reason, will_retry=will_retry)

        return self._trigger(trigger_reason, context)

    def _common_skip_reason(self, context: ProactiveContext, now_ms: int) -> tuple[str, bool] | None:
        if not self.config.enabled:
            return ("disabled", False)
        if context.closing:
            return ("closing", False)
        if context.muted:
            return ("muted", True)
        if not context.audio_stream_active:
            return ("audio_stream_inactive", True)
        if not context.received_audio_frame:
            return ("waiting_for_audio_frame", True)
        if context.active_response:
            return ("active_response", True)
        if context.session_state != "listening":
            return ("unsafe_state", True)
        if context.proactive_failures >= self.config.failure_backoff_threshold:
            return ("failure_backoff", False)
        if context.cooldown_until_ms is not None and now_ms < context.cooldown_until_ms:
            return ("cooldown", True)
        return None

    def _contextual_followup_skip_reason(self, context: ProactiveContext) -> tuple[str, bool] | None:
        if not self.config.contextual_followups_enabled:
            return ("contextual_followups_disabled", False)
        if not context.has_recent_user_context:
            return ("no_recent_user_context", True)
        if context.last_assistant_asked_question:
            return ("question_already_pending", True)
        if context.last_assistant_has_next_step:
            return ("next_step_already_pending", True)
        return None

    def _trigger(self, trigger_reason: str, context: ProactiveContext) -> ProactiveDecision:
        payload = self._base_payload(trigger_reason, context)
        return ProactiveDecision(
            event_type="proactive.triggered",
            trigger_reason=trigger_reason,
            allowed=True,
            source_state=context.session_state,
            payload=payload,
        )

    def _skip(
        self,
        trigger_reason: str,
        context: ProactiveContext,
        skip_reason: str,
        *,
        will_retry: bool,
        now_ms: int | None = None,
    ) -> ProactiveDecision:
        payload = self._base_payload(trigger_reason, context)
        payload.update(
            {
                "skip_reason": skip_reason,
                "will_retry": will_retry,
            }
        )
        if skip_reason == "cooldown" and context.cooldown_until_ms is not None and now_ms is not None:
            payload["next_eligible_at_ms"] = context.cooldown_until_ms
            payload["remaining_cooldown_ms"] = max(0, context.cooldown_until_ms - now_ms)

        return ProactiveDecision(
            event_type="proactive.skipped",
            trigger_reason=trigger_reason,
            allowed=False,
            source_state=context.session_state,
            skip_reason=skip_reason,
            will_retry=will_retry,
            payload=payload,
        )

    def _base_payload(self, trigger_reason: str, context: ProactiveContext) -> dict[str, object]:
        return {
            "trigger_reason": trigger_reason,
            "source_state": context.session_state,
            "consecutive_prompts": context.consecutive_prompts,
            "max_consecutive_prompts": self.config.max_consecutive_prompts,
            "proactive_failures": context.proactive_failures,
            "failure_backoff_threshold": self.config.failure_backoff_threshold,
        }
