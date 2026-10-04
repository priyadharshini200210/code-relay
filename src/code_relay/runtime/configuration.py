"""Worker boundary for managed configuration; runtime state stays on the loop."""

from collections.abc import Mapping

from anyio import CapacityLimiter, to_thread

from code_relay.config.admin.custom_providers import CustomProviderMutation
from code_relay.config.admin.persistence import (
    PreparedAdminUpdate,
    prepare_admin_update,
)
from code_relay.config.admin.state import ConfigInputValue, ValueState
from code_relay.config.admin.values import load_config_response, load_value_state
from code_relay.config.loader import ManagedConfigStore
from code_relay.config.settings import Settings
from code_relay.core.json_types import JsonObject


class ConfigurationService:
    def __init__(self, store: ManagedConfigStore) -> None:
        self._store = store
        # Storage is serialized; its waiters must not exhaust FastAPI's worker pool.
        self._worker_limiter = CapacityLimiter(1)

    async def initialize(self) -> None:
        await to_thread.run_sync(self._store.initialize, limiter=self._worker_limiter)

    async def admin_config(self) -> JsonObject:
        snapshot = await to_thread.run_sync(
            self._store.read, limiter=self._worker_limiter
        )
        return load_config_response(snapshot)

    async def admin_values(self) -> ValueState:
        snapshot = await to_thread.run_sync(
            self._store.read, limiter=self._worker_limiter
        )
        return load_value_state(snapshot)

    async def prepare(
        self,
        updates: Mapping[str, ConfigInputValue],
        active_settings: Settings,
        custom_provider: CustomProviderMutation | None = None,
    ) -> PreparedAdminUpdate:
        snapshot = await to_thread.run_sync(
            self._store.read, limiter=self._worker_limiter
        )
        return prepare_admin_update(updates, snapshot, active_settings, custom_provider)

    async def commit(self, prepared: PreparedAdminUpdate) -> JsonObject:
        if not prepared.valid:
            raise ValueError("Cannot commit an invalid Admin update")
        await to_thread.run_sync(
            self._store.commit, prepared.target_values, limiter=self._worker_limiter
        )
        return prepared.applied_response()
