"""Fresh-process contracts for the work excluded from HTTP/tray startup."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("module", "forbidden"),
    [
        (
            "code_relay.cli.desktop",
            ("uvicorn", "fastapi", "openai", "copilot", "tiktoken"),
        ),
        (
            "code_relay.runtime.bootstrap",
            ("openai", "copilot", "tiktoken", "telegram", "discord"),
        ),
    ],
)
def test_entrypoint_import_does_not_load_optional_runtime(module, forbidden):
    script = f"""
import importlib
import sys
importlib.import_module({module!r})
assert not set({forbidden!r}).intersection(sys.modules), sorted(set({forbidden!r}).intersection(sys.modules))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr


def test_sdk_preparation_runs_in_worker_and_provider_construction_runs_on_owner_loop():
    script = """
import asyncio
import sys
import threading
from unittest.mock import MagicMock
from code_relay.config.settings import Settings
from code_relay.providers.base import BaseProvider
from code_relay.providers.runtime.runtime import create_provider
from code_relay.providers.admission_policy import ProviderAdmissionLimits
from code_relay.providers.admission_registry import ProviderAdmissionRegistry

async def main():
    loop = asyncio.get_running_loop()
    thread = threading.get_ident()
    provider = MagicMock(spec=BaseProvider)
    def load():
        assert threading.get_ident() != thread
        assert "openai.resources" in sys.modules
        def construct(*args):
            assert asyncio.get_running_loop() is loop
            assert threading.get_ident() == thread
            return provider
        return construct
    assert await create_provider("groq", Settings(groq_api_key="test"), ProviderAdmissionRegistry(ProviderAdmissionLimits(1, 2, 2)), provider_loaders={"groq": load}) is provider
    await provider.cleanup()

asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr


def test_tokenizer_initializes_offline_with_an_empty_cache():
    script = """
import os
import tempfile
import requests

def network_forbidden(*args, **kwargs):
    raise AssertionError("tokenizer attempted a network request")

requests.get = network_forbidden
with tempfile.TemporaryDirectory() as cache:
    os.environ["TIKTOKEN_CACHE_DIR"] = cache
    from code_relay.core.token_estimation import initialize_token_estimation, estimate_text_tokens
    encoder = initialize_token_estimation()
    assert encoder is not None
    assert estimate_text_tokens("hello world") == 2
    assert estimate_text_tokens("abcdefgh") == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
