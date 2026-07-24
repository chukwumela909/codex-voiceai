"""Twilio REST and request-validation helpers for outbound voice calls."""

from __future__ import annotations

from xml.etree.ElementTree import Element, SubElement, tostring

from app.outbound_calls import OutboundCall


def public_https_url(settings, path: str) -> str:
    host = (settings.public_host or "").strip().rstrip("/")
    if not host:
        raise RuntimeError("PUBLIC_HOST must be configured for outbound calls.")
    return f"https://{host}{path}"


def build_outbound_twiml(call: OutboundCall, settings) -> str:
    host = (settings.public_host or "").strip().rstrip("/")
    if not host:
        raise RuntimeError("PUBLIC_HOST must be configured for outbound calls.")

    response = Element("Response")
    connect = SubElement(response, "Connect")
    stream = SubElement(connect, "Stream", {"url": f"wss://{host}/api/twilio-ws"})
    SubElement(stream, "Parameter", {"name": "call_id", "value": call.id})
    SubElement(stream, "Parameter", {"name": "stream_token", "value": call.stream_token})
    return '<?xml version="1.0" encoding="UTF-8"?>' + tostring(
        response, encoding="unicode", short_empty_elements=True
    )


def place_outbound_call(call: OutboundCall, settings):
    """Create an outbound call and return Twilio's Call resource."""
    from twilio.rest import Client

    client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    callback_url = public_https_url(settings, f"/twilio/call-status/{call.id}")
    return client.calls.create(
        to=call.to_number,
        from_=settings.twilio_from_number,
        twiml=build_outbound_twiml(call, settings),
        status_callback=callback_url,
        status_callback_method="POST",
        status_callback_event=["initiated", "ringing", "answered", "completed"],
    )


def end_outbound_call(call: OutboundCall, settings):
    """Cancel a pending call or hang up an active call."""
    from twilio.rest import Client

    if not call.call_sid:
        raise RuntimeError("The call has not been created at Twilio yet.")
    client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    target_status = (
        "canceled"
        if call.status in {"creating", "queued", "initiated", "ringing"}
        else "completed"
    )
    return client.calls(call.call_sid).update(status=target_status)


def validate_twilio_request(
    *,
    url: str,
    params: dict[str, str],
    signature: str | None,
    auth_token: str | None,
) -> bool:
    if not signature or not auth_token:
        return False
    from twilio.request_validator import RequestValidator

    return RequestValidator(auth_token).validate(url, params, signature)


def callback_url_for_request(request, settings) -> str:
    """Reconstruct the exact public callback URL Twilio was given."""
    path = request.url.path
    query = request.url.query
    if query:
        path = f"{path}?{query}"
    return public_https_url(settings, path)
