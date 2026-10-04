"""Anthropic credentials, HTTP lifetime, and atomic model capability snapshots."""

import asyncio
from collections.abc import AsyncIterator, Mapping
from types import MappingProxyType
from typing import cast
from urllib.parse import quote

import httpx

from code_relay.application.errors import InvalidRequestError
from code_relay.application.model_metadata import ProviderModelInfo
from code_relay.core.anthropic.models import MessagesRequest
from code_relay.core.anthropic.native import NativeMessagesError
from code_relay.core.anthropic.passthrough import (
    NativeMessagesRequest,
    restore_native_history,
)
from code_relay.core.history_replay import HistoryReplayError
from code_relay.core.json_types import JsonObject
from code_relay.core.openai_responses import OpenAIResponsesRequest
from code_relay.core.reasoning import DEFAULT_REASONING_POLICY, ReasoningPolicy
from code_relay.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from code_relay.providers.anthropic_messages.discovery import list_messages_models
from code_relay.providers.anthropic_messages.passthrough import (
    stream_native_messages,
)
from code_relay.providers.anthropic_messages.request_policy import (
    MessagesModelCapabilities,
)
from code_relay.providers.anthropic_messages.transport import (
    AnthropicMessagesTransport,
)
from code_relay.providers.base import BaseProvider, ProviderConfig
from code_relay.providers.endpoint_types import HttpEndpoint
from code_relay.providers.failure_policy import ProviderRecoveryExhausted
from code_relay.providers.history_replay import replay_origin
from code_relay.providers.http import maybe_await_aclose
from code_relay.providers.model_listing import ModelListResponseError

from .headers import api_headers, native_request
from .models import AnthropicModelRecord, model_record


async def _omit_cookies(request: httpx.Request) -> None:
    request.headers.pop("cookie", None)


class AnthropicProvider(BaseProvider):
    def __init__(
        self,
        config: ProviderConfig,
        *,
        admission: ProviderAdmissionController,
        workspace_id: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__(config)
        self._admission = admission
        self._snapshot = HttpEndpoint(
            base_url=config.base_url,
            api_key=config.api_key,
            headers=MappingProxyType(api_headers(config.api_key or "", workspace_id)),
        )
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                write=config.http_write_timeout,
            ),
            proxy=config.proxy,
            transport=transport,
            follow_redirects=False,
            event_hooks={"request": [_omit_cookies]},
        )
        self._records: Mapping[str, AnthropicModelRecord] = MappingProxyType({})
        self._model_lock = asyncio.Lock()

    async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
        return self._snapshot

    async def cleanup(self) -> None:
        await self._http.aclose()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        async with self._model_lock:
            items = await list_messages_models(
                self._http,
                self._admission,
                base_url=self._snapshot.base_url,
                headers=self._snapshot.headers,
            )
            records = tuple(model_record(item) for item in items)
            self._records = MappingProxyType(
                {record.info.model_id: record for record in records}
            )
            return frozenset(record.info for record in records)

    async def model_record(self, model: str) -> AnthropicModelRecord:
        record = self._records.get(model)
        if record is not None:
            return record
        async with self._model_lock:
            record = self._records.get(model)
            if record is not None:
                return record

            async def fetch() -> object:
                response = await self._http.get(
                    self._snapshot.base_url.rstrip("/")
                    + "/models/"
                    + quote(model, safe=""),
                    headers=self._snapshot.headers,
                )
                response.raise_for_status()
                return response.json()

            try:
                item = await self._admission.start_execution().run_call(
                    fetch, operation_kind=ProviderOperationKind.MODEL_DISCOVERY
                )
                if not isinstance(item, dict):
                    raise ModelListResponseError("Invalid Anthropic model record")
                record = model_record(cast(JsonObject, item))
            except (
                httpx.HTTPError,
                ModelListResponseError,
                ProviderRecoveryExhausted,
                ValueError,
            ):
                return AnthropicModelRecord(
                    ProviderModelInfo(model), MessagesModelCapabilities()
                )
            self._records = MappingProxyType({**self._records, model: record})
            return record

    def _transport(self, record: AnthropicModelRecord) -> AnthropicMessagesTransport:
        return AnthropicMessagesTransport(
            client=self._http,
            admission=self._admission,
            provider_name="anthropic",
            replay_scope="anthropic",
            read_timeout_s=self._config.http_read_timeout,
            capabilities=record.messages,
        )

    def stream_native_messages(
        self,
        request: NativeMessagesRequest,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        request_headers: Mapping[str, str] | None = None,
    ) -> AsyncIterator[str]:
        body, headers = native_request(request.body, request_headers)
        try:
            body = restore_native_history(
                body,
                replay_origin(
                    "anthropic", "messages", request.model, endpoint=self._snapshot
                ),
            )
        except (NativeMessagesError, HistoryReplayError) as error:
            raise InvalidRequestError(str(error)) from error
        return stream_native_messages(
            self._http,
            self._admission,
            base_url=self._snapshot.base_url,
            headers={**self._snapshot.headers, **headers},
            body=body,
            public_model=response_model or request.model,
            provider_name="anthropic",
            read_timeout_s=self._config.http_read_timeout,
            request_id=request_id,
        )

    async def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        record = await self.model_record(request.model)
        stream = self._transport(record).stream_messages(
            request,
            endpoint_context=self,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
            model_info=record.info,
        )
        try:
            async for event in stream:
                yield event
        finally:
            await maybe_await_aclose(stream)

    async def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        request_headers: Mapping[str, str] | None = None,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        record = await self.model_record(request.model)
        stream = self._transport(record).stream_responses(
            request,
            endpoint_context=self,
            request_id=request_id,
            response_model=response_model,
            reasoning=reasoning,
            model_info=record.info,
        )
        try:
            async for event in stream:
                yield event
        finally:
            await maybe_await_aclose(stream)
