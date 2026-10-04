"""Log one final inference outcome after HTTP delivery and owned cleanup."""

import asyncio
import codecs
import json
from time import monotonic

from loguru import logger
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from code_relay.core.anthropic.stream_contracts import SSEEvent
from code_relay.core.anthropic.streaming.decoder import AnthropicSSEDecoder
from code_relay.core.request_outcomes import (
    RequestOutcome,
    current_request_outcome,
    record_request_exception,
    record_request_failure,
)

from .usage_tracking import TokenUsage, usage_from_payload, usage_tracker

_FAILURE_EVENTS = frozenset({"error", "response.error", "response.failed"})
_USAGE_EVENTS = frozenset({"message_start", "message_delta", "response.completed"})
_OBSERVED_EVENTS = _FAILURE_EVENTS | _USAGE_EVENTS
_MAX_JSON_BODY = 4 * 1024 * 1024


def _observe_usage(event: SSEEvent, tokens: TokenUsage) -> None:
    kind = event.event or event.data.get("type")
    if kind == "message_start":
        message = event.data.get("message")
        if isinstance(message, dict):
            usage_from_payload(message.get("usage"), merge_into=tokens)
    elif kind == "message_delta":
        usage_from_payload(event.data.get("usage"), merge_into=tokens)
    elif kind == "response.completed":
        response = event.data.get("response")
        if isinstance(response, dict):
            usage_from_payload(response.get("usage"), merge_into=tokens)


def _observe_event(event: SSEEvent, tokens: TokenUsage) -> None:
    _observe_usage(event, tokens)
    kind = event.event or event.data.get("type")
    if not isinstance(kind, str) or kind not in _FAILURE_EVENTS:
        return
    response = event.data.get("response")
    payload = response if isinstance(response, dict) else event.data
    error = payload.get("error")
    error = error if isinstance(error, dict) else payload
    reason = error.get("code") or error.get("type")
    record_request_failure(reason if isinstance(reason, str) else str(kind))


class RequestOutcomeMiddleware:
    """Observe inference delivery without changing its body or control flow."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") not in {"/v1/messages", "/v1/responses"}
        ):
            await self._app(scope, receive, send)
            return

        started = monotonic()
        outcome = RequestOutcome()
        status_code: int | None = None
        completed = disconnected = cancelled = False
        decoder: AnthropicSSEDecoder | None = None
        tokens = TokenUsage()
        json_body: bytearray | None = None
        text_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        async def receive_observed() -> Message:
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def send_observed(message: Message) -> None:
            nonlocal status_code, completed, decoder, disconnected, json_body
            try:
                await send(message)
            except OSError:
                disconnected = True
                raise
            if message["type"] == "http.response.start":
                status_code = message["status"]
                content_type = Headers(raw=message.get("headers", [])).get(
                    "content-type", ""
                )
                if content_type.startswith("text/event-stream"):
                    decoder = AnthropicSSEDecoder(event_names=_OBSERVED_EVENTS)
                elif content_type.startswith("application/json"):
                    json_body = bytearray()
            elif message["type"] == "http.response.body":
                completed = not message.get("more_body", False)
                if decoder is not None:
                    text = text_decoder.decode(
                        message.get("body", b""), final=completed
                    )
                    for event in decoder.feed(text):
                        _observe_event(event, tokens)
                    if completed:
                        for event in decoder.finish():
                            _observe_event(event, tokens)
                elif json_body is not None:
                    json_body += message.get("body", b"")
                    if len(json_body) > _MAX_JSON_BODY:
                        json_body = None
                    elif completed:
                        try:
                            payload = json.loads(json_body)
                        except ValueError:
                            payload = None
                        if isinstance(payload, dict):
                            usage_from_payload(payload.get("usage"), merge_into=tokens)

        token = current_request_outcome.set(outcome)
        try:
            await self._app(scope, receive_observed, send_observed)
        except asyncio.CancelledError:
            cancelled = True
            raise
        except BaseException as error:
            if not disconnected:
                record_request_exception(error)
                if status_code is None:
                    status_code = 500
            raise
        finally:
            current_request_outcome.reset(token)
            reason = outcome.failure_reason
            if reason is not None or (status_code is not None and status_code >= 400):
                result = "failure"
                reason = reason or f"http_{status_code}"
            elif (cancelled or disconnected) and not completed:
                result = "cancelled"
            elif not completed:
                result, reason = "failure", "incomplete_response"
            else:
                result = "success"
            duration_ms = round((monotonic() - started) * 1000, 2)
            usage_tracker.record(
                provider_id=outcome.provider_id,
                model=outcome.model,
                outcome=result,
                failure_reason=reason,
                status_code=status_code,
                duration_ms=duration_ms,
                tokens=tokens,
            )
            logger.bind(
                event="request.completed",
                wire_api="responses"
                if scope["path"] == "/v1/responses"
                else "messages",
                provider_id=outcome.provider_id,
                model=outcome.model,
                status_code=status_code,
                outcome=result,
                duration_ms=duration_ms,
                failure_reason=reason,
                input_tokens=tokens.input_tokens,
                output_tokens=tokens.output_tokens,
            ).info("Inference request {}", result)
