"""In-process state for website-initiated outbound Twilio calls.

The store intentionally keeps destination numbers out of API responses and logs.
It is sufficient for the app's current single-process deployment model. If the
service moves to multiple workers, replace this module with a shared Redis or
database-backed implementation while preserving the public methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import secrets
from threading import RLock
from uuid import uuid4


TERMINAL_CALL_STATUSES = {
    "busy",
    "canceled",
    "completed",
    "failed",
    "no-answer",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _masked_number(number: str) -> str:
    return f"••••{number[-4:]}"


@dataclass
class OutboundCall:
    id: str
    to_number: str
    character_id: str | None
    voice_id: str | None
    model_id: str | None
    stream_token: str = field(repr=False)
    status: str = "creating"
    call_sid: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def public(self) -> dict:
        return {
            "id": self.id,
            "to": _masked_number(self.to_number),
            "status": self.status,
            "character_id": self.character_id,
            "voice_id": self.voice_id,
            "model_id": self.model_id,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class OutboundCallStore:
    def __init__(self) -> None:
        self._calls: dict[str, OutboundCall] = {}
        self._lock = RLock()

    def create(
        self,
        *,
        to_number: str,
        character_id: str | None,
        voice_id: str | None,
        model_id: str | None,
    ) -> OutboundCall:
        call = OutboundCall(
            id=str(uuid4()),
            to_number=to_number,
            character_id=character_id,
            voice_id=voice_id,
            model_id=model_id,
            stream_token=secrets.token_urlsafe(32),
        )
        with self._lock:
            self._calls[call.id] = call
        return call

    def get(self, call_id: str) -> OutboundCall | None:
        with self._lock:
            return self._calls.get(call_id)

    def attach_twilio_call(self, call_id: str, call_sid: str) -> OutboundCall | None:
        with self._lock:
            call = self._calls.get(call_id)
            if not call:
                return None
            call.call_sid = call_sid
            # A very fast Twilio status callback can arrive before the REST
            # create response returns. Do not regress initiated/ringing state.
            if call.status == "creating":
                call.status = "queued"
            call.updated_at = _now()
            return call

    def update_status(
        self,
        call_id: str,
        status: str,
        *,
        call_sid: str | None = None,
        error: str | None = None,
    ) -> OutboundCall | None:
        with self._lock:
            call = self._calls.get(call_id)
            if not call:
                return None
            if call.call_sid and call_sid and not secrets.compare_digest(call.call_sid, call_sid):
                return None
            if call_sid and not call.call_sid:
                call.call_sid = call_sid
            call.status = status
            call.error = error
            call.updated_at = _now()
            return call

    def authenticate_stream(
        self,
        *,
        call_id: str,
        stream_token: str,
        call_sid: str,
    ) -> OutboundCall | None:
        with self._lock:
            call = self._calls.get(call_id)
            if not call or not call.call_sid:
                return None
            if not secrets.compare_digest(call.stream_token, stream_token):
                return None
            if not secrets.compare_digest(call.call_sid, call_sid):
                return None
            call.status = "in-progress"
            call.updated_at = _now()
            return call

    def reset(self) -> None:
        """Clear state for isolated tests."""
        with self._lock:
            self._calls.clear()


outbound_call_store = OutboundCallStore()
