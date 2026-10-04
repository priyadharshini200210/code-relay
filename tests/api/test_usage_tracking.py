"""Usage tracking aggregates tokens and outcomes per provider/model."""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from code_relay.api import admin_routes, request_outcomes
from code_relay.api.handlers import messages as messages_handler
from code_relay.api.usage_tracking import (
    RATE_LIMIT_COOLDOWN_SECONDS,
    TokenUsage,
    UsageTracker,
    normalize_usage_config,
    usage_from_payload,
)
from code_relay.application.execution import ProviderExecutor
from code_relay.application.routing import ProviderModelTarget
from code_relay.config.settings import Settings
from tests.api.support import create_test_app


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    tracker = UsageTracker(tmp_path)
    monkeypatch.setattr(request_outcomes, "usage_tracker", tracker)
    monkeypatch.setattr(admin_routes, "usage_tracker", tracker)
    monkeypatch.setattr(messages_handler, "usage_tracker", tracker)
    return tracker


class UsageProvider:
    def __init__(self, result="success"):
        self.result = result

    async def stream_messages(self, request, **kwargs):
        start = {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 120, "cache_read_input_tokens": 30},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(start)}\n\n"
        delta = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 45},
        }
        frame = f"event: message_delta\ndata: {json.dumps(delta)}\n\n"
        yield frame[:20]
        yield frame[20:]
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    async def stream_responses(self, request, **kwargs):
        yield 'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_test","object":"response","status":"in_progress","output":[]}}\n\n'
        done = {
            "type": "response.completed",
            "response": {
                "id": "resp_test",
                "object": "response",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 80,
                    "output_tokens": 20,
                    "input_tokens_details": {"cached_tokens": 10},
                },
            },
        }
        yield f"event: response.completed\ndata: {json.dumps(done)}\n\n"


def _post(client, wire_api, stream=True):
    payload = {"model": "nvidia_nim/test-model", "stream": stream}
    if wire_api == "messages":
        payload.update(max_tokens=64, messages=[{"role": "user", "content": "hi"}])
    else:
        payload["input"] = "hi"
    return client.post(f"/v1/{wire_api}", json=payload)


def test_usage_from_payload_reads_both_wire_formats():
    tokens = TokenUsage()
    usage_from_payload(
        {"input_tokens": 5, "cache_creation_input_tokens": 2}, merge_into=tokens
    )
    usage_from_payload({"output_tokens": 7}, merge_into=tokens)
    usage_from_payload(
        {"input_tokens_details": {"cached_tokens": 3}}, merge_into=tokens
    )
    usage_from_payload("not a dict", merge_into=tokens)
    assert tokens == TokenUsage(5, 7, 3, 2)


@pytest.mark.parametrize(
    "wire_api,expected",
    [
        ("messages", (120, 45, 30)),
        ("responses", (80, 20, 10)),
    ],
)
def test_streamed_usage_is_recorded(tracker, wire_api, expected):
    with (
        patch(
            "code_relay.api.routes.resolve_provider",
            return_value=UsageProvider(),
        ),
        TestClient(create_test_app(Settings())) as client,
    ):
        assert _post(client, wire_api).status_code == 200

    model = tracker.snapshot()["models"]["nvidia_nim/test-model"]
    assert (
        model["input_tokens"],
        model["output_tokens"],
        model["cache_read_input_tokens"],
    ) == expected
    assert (model["requests"], model["successes"], model["failures"]) == (1, 1, 0)
    assert sum(model["daily_tokens"].values()) == sum(expected)


def test_non_streaming_usage_is_recorded(tracker):
    with (
        patch(
            "code_relay.api.routes.resolve_provider",
            return_value=UsageProvider(),
        ),
        TestClient(create_test_app(Settings())) as client,
    ):
        assert _post(client, "messages", stream=False).status_code == 200

    model = tracker.snapshot()["models"]["nvidia_nim/test-model"]
    assert (model["input_tokens"], model["output_tokens"]) == (120, 45)


def test_failures_and_rate_limits_are_counted(tracker):
    tokens = TokenUsage()
    for outcome, reason, status in [
        ("success", None, 200),
        ("failure", "rate_limit", 429),
        ("cancelled", None, 200),
    ]:
        tracker.record(
            provider_id="groq",
            model="m",
            outcome=outcome,
            failure_reason=reason,
            status_code=status,
            duration_ms=10,
            tokens=tokens,
        )
    snapshot = tracker.snapshot()
    provider = snapshot["providers"]["groq"]
    assert (
        provider["requests"],
        provider["successes"],
        provider["failures"],
        provider["cancelled"],
        provider["rate_limited"],
    ) == (3, 1, 1, 1, 1)
    assert snapshot["overall"]["requests"] == 3


def test_usage_persists_and_resets(tmp_path):
    path = tmp_path / "usage.json"
    UsageTracker(tmp_path).record(
        provider_id="groq",
        model="m",
        outcome="success",
        failure_reason=None,
        status_code=200,
        duration_ms=5,
        tokens=TokenUsage(input_tokens=3, output_tokens=4),
    )
    reloaded = UsageTracker(tmp_path)
    assert reloaded.snapshot()["overall"]["input_tokens"] == 3
    reloaded.reset()
    assert UsageTracker(tmp_path).snapshot()["models"] == {}


def test_corrupt_usage_file_is_ignored(tmp_path):
    path = tmp_path / "usage.json"
    path.write_text("{not json", encoding="utf-8")
    assert UsageTracker(tmp_path).snapshot()["models"] == {}


def test_admin_usage_endpoints(tracker):
    tracker.record(
        provider_id="groq",
        model="m",
        outcome="success",
        failure_reason=None,
        status_code=200,
        duration_ms=5,
        tokens=TokenUsage(input_tokens=1),
    )
    with TestClient(create_test_app(Settings())) as client:
        usage = client.get("/admin/api/usage")
        assert usage.status_code == 200
        assert usage.json()["overall"]["requests"] == 1
        reset = client.post("/admin/api/usage/reset")
        assert reset.status_code == 200
        assert reset.json()["models"] == {}


def _target(ref):
    provider_id, model = ref.split("/", 1)
    return ProviderModelTarget(provider_id, model, ref)


def _record(tracker, ref, *, outcome="success", status=200, ms=100.0, tokens=None):
    provider_id, model = ref.split("/", 1)
    tracker.record(
        provider_id=provider_id,
        model=model,
        outcome=outcome,
        failure_reason="rate_limit" if status == 429 else None,
        status_code=status,
        duration_ms=ms,
        tokens=tokens or TokenUsage(),
    )


def test_config_is_normalized_and_persisted(tmp_path):
    tracker = UsageTracker(tmp_path)
    saved = tracker.set_config(
        {
            "baseline": {"input": 3, "output": "15", "cache_read": -1},
            "provider_prices": {"groq": {"input": 0.5}, "bad": "x"},
            "daily_token_limits": {"groq": 1000, "nim": 0, "zai": True},
            "unknown": 1,
        }
    )
    assert saved == {
        "baseline": {"input": 3.0},
        "provider_prices": {"groq": {"input": 0.5}},
        "model_prices": {},
        "daily_token_limits": {"groq": 1000},
    }
    assert UsageTracker(tmp_path).get_config() == saved
    assert normalize_usage_config("nonsense")["baseline"] == {}


def test_costs_savings_and_quota(tmp_path):
    tracker = UsageTracker(tmp_path)
    tracker.set_config(
        {
            "baseline": {"input": 3, "output": 15},
            "model_prices": {"anthropic/sonnet": {"input": 3, "output": 15}},
            "daily_token_limits": {"groq": 1_500_000},
        }
    )
    million = TokenUsage(input_tokens=1_000_000, output_tokens=0)
    _record(tracker, "groq/llama", tokens=million)
    _record(tracker, "anthropic/sonnet", tokens=million)
    snapshot = tracker.snapshot()
    assert snapshot["models"]["groq/llama"]["cost_usd"] == 0
    assert snapshot["models"]["groq/llama"]["baseline_cost_usd"] == 3
    assert snapshot["models"]["anthropic/sonnet"]["cost_usd"] == 3
    assert snapshot["overall"]["savings_usd"] == 3
    groq = snapshot["providers"]["groq"]
    assert (groq["tokens_today"], groq["quota_remaining_today"]) == (
        1_000_000,
        500_000,
    )
    assert snapshot["providers"]["anthropic"]["quota_remaining_today"] is None


def test_ranking_prefers_healthy_fast_under_quota(tmp_path):
    tracker = UsageTracker(tmp_path)
    tracker.set_config({"daily_token_limits": {"quota": 10}})
    slow, fast, failing, over, fresh = (
        _target(ref)
        for ref in ("a/slow", "b/fast", "c/failing", "quota/m", "d/fresh")
    )
    _record(tracker, "a/slow", ms=900)
    _record(tracker, "b/fast", ms=50)
    _record(tracker, "c/failing", outcome="failure", status=500)
    _record(tracker, "quota/m", ms=1, tokens=TokenUsage(output_tokens=10))
    ranked = tracker.rank_candidates((failing, over, slow, fresh, fast))
    assert ranked == (fast, slow, fresh, failing, over)


def test_rate_limited_model_is_down_until_cooldown(tmp_path, monkeypatch):
    from code_relay.api import usage_tracking

    clock = [1000.0]
    monkeypatch.setattr(usage_tracking, "monotonic", lambda: clock[0])
    tracker = UsageTracker(tmp_path)
    for _ in range(5):
        _record(tracker, "a/m", ms=10)
    _record(tracker, "a/m", outcome="failure", status=429)
    assert tracker.snapshot()["models"]["a/m"]["health"]["status"] == "down"
    clock[0] += RATE_LIMIT_COOLDOWN_SECONDS + 1
    assert tracker.snapshot()["models"]["a/m"]["health"]["status"] == "up"


def test_executor_ignores_rankers_that_change_targets():
    a, b = _target("a/m"), _target("b/m")
    executor = ProviderExecutor(
        lambda _provider: None,
        progress_timeout_seconds=1,
        candidate_ranker=lambda candidates: (b,),
    )
    assert executor._ranked_candidates((a, b)) == (a, b)
    executor._candidate_ranker = lambda candidates: (b, a)
    assert executor._ranked_candidates((a, b)) == (b, a)
    executor._candidate_ranker = lambda candidates: 1 / 0
    assert executor._ranked_candidates((a, b)) == (a, b)


class RoutingProvider:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    async def stream_messages(self, request, **kwargs):
        self.calls.append((self.name, request.model))
        yield 'event: message_start\ndata: {"type":"message_start","message":{"id":"msg","type":"message","role":"assistant","content":[]}}\n\n'
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'


@pytest.mark.parametrize("smart", [False, True])
def test_smart_routing_tries_healthy_fallback_first(tracker, smart):
    _record(tracker, "nvidia_nim/primary-model", outcome="failure", status=429)
    _record(tracker, "open_router/fallback-model", ms=20)
    calls = []
    providers = {
        "nvidia_nim": RoutingProvider("nvidia_nim", calls),
        "open_router": RoutingProvider("open_router", calls),
    }
    settings = Settings(
        model_fallbacks=["open_router/fallback-model"], smart_routing=smart
    )
    with (
        patch(
            "code_relay.api.routes.resolve_provider",
            side_effect=lambda name, **kwargs: providers[name],
        ),
        TestClient(create_test_app(settings)) as client,
    ):
        response = client.post(
            "/v1/messages",
            json={
                "model": "nvidia_nim/primary-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
    assert response.status_code == 200
    expected = (
        ("open_router", "fallback-model") if smart else ("nvidia_nim", "primary-model")
    )
    assert calls == [expected]


def test_admin_usage_config_and_page(tracker):
    with TestClient(create_test_app(Settings())) as client:
        saved = client.put(
            "/admin/api/usage/config",
            json={"daily_token_limits": {"groq": 5}},
        )
        assert saved.status_code == 200
        assert saved.json()["daily_token_limits"] == {"groq": 5}
        assert client.get("/admin/api/usage/config").json() == saved.json()
        assert client.put(
            "/admin/api/usage/config", content="[1]"
        ).status_code == 400
        page = client.get("/admin/usage")
        assert page.status_code == 200
        assert "/admin/api/usage" in page.text
