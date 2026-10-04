"""Explicit test composition for the API adapter."""

import asyncio
from collections.abc import Mapping, MutableMapping

from fastapi import FastAPI

from code_relay.api.app import create_app
from code_relay.api.ports import ApiServices
from code_relay.application.code_sessions import CodeApplicationPort
from code_relay.application.connected_accounts import ConnectedAccountPort
from code_relay.application.model_metadata import ProviderModelRefreshResult
from code_relay.config.loader import ManagedConfigStore
from code_relay.config.settings import Settings
from code_relay.providers.base import BaseProvider
from code_relay.providers.runtime import ProviderRuntime
from code_relay.runtime.application import ApplicationRuntime, RestartCallback
from code_relay.runtime.configuration import ConfigurationService
from code_relay.runtime.provider_manager import ProviderRuntimeManager
from tests.web_tools_support import StubWebToolsClient


class ApiTestRuntime(ProviderRuntimeManager):
    """API-only tests supply metadata explicitly; startup/discovery has separate tests."""

    def _catalog_task(self, generation, provider_id, *, refresh=False):
        if refresh:
            return super()._catalog_task(generation, provider_id, refresh=True)
        task = generation.catalog_tasks.get(provider_id)
        if task is None:

            async def supplied_catalog():
                generation.initialized.add(provider_id)
                return ProviderModelRefreshResult()

            task = asyncio.create_task(supplied_catalog())
            generation.catalog_tasks[provider_id] = task
        return task


def create_test_app(
    settings: Settings | None = None,
    *,
    providers: MutableMapping[str, BaseProvider] | None = None,
    restart_callback: RestartCallback | None = None,
    connected_accounts: Mapping[str, ConnectedAccountPort] | None = None,
    code: CodeApplicationPort | None = None,
) -> FastAPI:
    """Build an API app with explicit in-memory runtime services."""
    store = ManagedConfigStore()
    store.initialize()  # API-only tests do not run the production ASGI lifespan.
    settings = settings or store.read().settings
    connected_accounts = dict(connected_accounts or {})

    def connected_provider_ids() -> tuple[str, ...]:
        return tuple(
            provider_id
            for provider_id, account in connected_accounts.items()
            if account.is_connected()
        )

    if providers is None:
        manager = ApiTestRuntime(
            settings,
            connected_provider_ids=connected_provider_ids,
        )
    else:
        manager = ApiTestRuntime(
            settings,
            runtime_factory=lambda snapshot, admission_registry: ProviderRuntime(
                snapshot,
                admission_registry,
                dict(providers),
            ),
            connected_provider_ids=connected_provider_ids,
        )
    runtime = ApplicationRuntime(
        manager,
        configuration=ConfigurationService(store),
        transcriber=None,
        restart_callback=restart_callback,
        connected_accounts=connected_accounts,
    )
    return create_app(
        ApiServices(
            requests=manager,
            admin=runtime,
            tasks=runtime,
            code=code,
            web_tools=StubWebToolsClient(),
        )
    )


def runtime_for_app(app: FastAPI) -> ApplicationRuntime:
    """Return the runtime supplied by :func:`create_test_app`."""
    runtime = app.state.services.admin
    if not isinstance(runtime, ApplicationRuntime):
        raise TypeError("Test app does not use ApplicationRuntime")
    return runtime


def provider_manager_for_app(app: FastAPI) -> ProviderRuntimeManager:
    """Return the provider manager supplied by :func:`create_test_app`."""
    manager = app.state.services.requests
    if not isinstance(manager, ProviderRuntimeManager):
        raise TypeError("Test app does not use ProviderRuntimeManager")
    return manager
