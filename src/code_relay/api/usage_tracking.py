"""Per-provider/model usage, cost estimates, and smart candidate ranking."""

import json
import math
import os
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from loguru import logger

from code_relay.application.routing import ProviderModelTarget
from code_relay.config import paths

USAGE_FILENAME = "usage.json"
USAGE_CONFIG_FILENAME = "usage_config.json"
_COUNTERS = (
    "requests",
    "successes",
    "failures",
    "cancelled",
    "rate_limited",
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "duration_ms",
)
_PRICE_FIELDS = ("input", "output", "cache_read", "cache_write")
# Recent outcomes drive smart routing; older history only feeds the dashboard.
RECENT_WINDOW = 20
RECENT_SECONDS = 15 * 60
UNHEALTHY_FAILURE_RATIO = 0.5
RATE_LIMIT_COOLDOWN_SECONDS = 60


@dataclass
class TokenUsage:
    """Token counts reported by one response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )


@dataclass
class UsageTotals:
    """Aggregated counters for one provider/model pair."""

    requests: int = 0
    successes: int = 0
    failures: int = 0
    cancelled: int = 0
    rate_limited: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    duration_ms: float = 0.0
    last_used: str | None = None
    daily_tokens: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _RecentOutcome:
    at: float
    ok: bool
    rate_limited: bool
    duration_ms: float
    output_tokens: int


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _price(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def usage_from_payload(usage: object, *, merge_into: TokenUsage) -> None:
    """Merge Anthropic or OpenAI Responses usage fields; later values win."""
    if not isinstance(usage, dict):
        return
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        if _int(usage.get(key)):
            setattr(merge_into, key, _int(usage[key]))
    details = usage.get("input_tokens_details")
    if isinstance(details, dict) and _int(details.get("cached_tokens")):
        merge_into.cache_read_input_tokens = _int(details["cached_tokens"])


def normalize_usage_config(raw: object) -> dict[str, object]:
    """Keep only well-formed prices (USD per 1M tokens) and daily token limits."""
    raw = raw if isinstance(raw, dict) else {}

    def prices(value: object) -> dict[str, float]:
        value = value if isinstance(value, dict) else {}
        return {
            name: price
            for name in _PRICE_FIELDS
            if (price := _price(value.get(name))) is not None
        }

    def price_table(value: object) -> dict[str, dict[str, float]]:
        value = value if isinstance(value, dict) else {}
        return {
            str(key): table
            for key, entry in value.items()
            if (table := prices(entry))
        }

    limits = raw.get("daily_token_limits")
    limits = limits if isinstance(limits, dict) else {}
    return {
        "baseline": prices(raw.get("baseline")),
        "provider_prices": price_table(raw.get("provider_prices")),
        "model_prices": price_table(raw.get("model_prices")),
        "daily_token_limits": {
            str(key): _int(value) for key, value in limits.items() if _int(value)
        },
    }


def _cost(totals: Mapping[str, object], prices: Mapping[str, float]) -> float:
    def tokens(name: str) -> int:
        value = totals.get(name, 0)
        return value if isinstance(value, int) else 0

    return (
        tokens("input_tokens") * prices.get("input", 0.0)
        + tokens("output_tokens") * prices.get("output", 0.0)
        + tokens("cache_read_input_tokens")
        * prices.get("cache_read", prices.get("input", 0.0))
        + tokens("cache_creation_input_tokens")
        * prices.get("cache_write", prices.get("input", 0.0))
    ) / 1_000_000


class UsageTracker:
    """Thread-safe usage store backed by JSON files in the FCC config dir."""

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory
        self._lock = threading.Lock()
        self._totals: dict[str, UsageTotals] | None = None
        self._since: str | None = None
        self._config: dict[str, object] | None = None
        self._recent: dict[str, deque[_RecentOutcome]] = {}

    @property
    def path(self) -> Path:
        return (self._directory or paths.config_dir_path()) / USAGE_FILENAME

    @property
    def config_path(self) -> Path:
        return (self._directory or paths.config_dir_path()) / USAGE_CONFIG_FILENAME

    def record(
        self,
        *,
        provider_id: str | None,
        model: str | None,
        outcome: str,
        failure_reason: str | None,
        status_code: int | None,
        duration_ms: float,
        tokens: TokenUsage,
    ) -> None:
        key = f"{provider_id or 'unknown'}/{model or 'unknown'}"
        now = datetime.now(UTC)
        day = now.date().isoformat()
        rate_limited = status_code == 429 or failure_reason in {
            "rate_limit",
            "rate_limit_error",
        }
        with self._lock:
            totals = self._load().setdefault(key, UsageTotals())
            totals.requests += 1
            if outcome == "success":
                totals.successes += 1
            elif outcome == "cancelled":
                totals.cancelled += 1
            else:
                totals.failures += 1
            if rate_limited:
                totals.rate_limited += 1
            totals.input_tokens += tokens.input_tokens
            totals.output_tokens += tokens.output_tokens
            totals.cache_read_input_tokens += tokens.cache_read_input_tokens
            totals.cache_creation_input_tokens += tokens.cache_creation_input_tokens
            totals.duration_ms += duration_ms
            totals.last_used = now.isoformat(timespec="seconds")
            totals.daily_tokens[day] = totals.daily_tokens.get(day, 0) + tokens.total
            if outcome != "cancelled":
                self._recent.setdefault(key, deque(maxlen=RECENT_WINDOW)).append(
                    _RecentOutcome(
                        at=monotonic(),
                        ok=outcome == "success",
                        rate_limited=rate_limited,
                        duration_ms=duration_ms,
                        output_tokens=tokens.output_tokens,
                    )
                )
            self._save()

    def snapshot(self) -> dict[str, object]:
        """Return totals, cost estimates, quota, and health per model/provider."""
        today = datetime.now(UTC).date().isoformat()
        with self._lock:
            models = {key: asdict(value) for key, value in self._load().items()}
            config = self._load_config()
            since = self._since
            health = {key: self._health(key) for key in models}
        baseline = config["baseline"]
        provider_prices = config["provider_prices"]
        model_prices = config["model_prices"]
        limits = config["daily_token_limits"]
        assert isinstance(baseline, dict)
        assert isinstance(provider_prices, dict)
        assert isinstance(model_prices, dict)
        assert isinstance(limits, dict)

        providers: dict[str, dict[str, object]] = {}
        overall: dict[str, float] = dict.fromkeys(
            (*_COUNTERS, "cost_usd", "baseline_cost_usd"), 0
        )
        for key, totals in models.items():
            provider_id = key.split("/", 1)[0]
            prices = model_prices.get(key) or provider_prices.get(provider_id) or {}
            totals["cost_usd"] = round(_cost(totals, prices), 6)
            totals["baseline_cost_usd"] = round(_cost(totals, baseline), 6)
            totals["tokens_today"] = totals["daily_tokens"].get(today, 0)
            totals["health"] = health[key]
            provider = providers.setdefault(
                provider_id,
                dict.fromkeys(
                    (*_COUNTERS, "cost_usd", "baseline_cost_usd", "tokens_today"), 0
                ),
            )
            for counter in (*_COUNTERS, "cost_usd", "baseline_cost_usd"):
                provider[counter] += totals[counter]
                overall[counter] += totals[counter]
            provider["tokens_today"] += totals["tokens_today"]
        for provider_id, provider in providers.items():
            limit = limits.get(provider_id)
            provider["daily_token_limit"] = limit
            provider["quota_remaining_today"] = (
                max(0, limit - provider["tokens_today"]) if limit else None
            )
        overall["savings_usd"] = max(
            0.0, overall["baseline_cost_usd"] - overall["cost_usd"]
        )
        return {
            "since": since,
            "today": today,
            "config": config,
            "overall": overall,
            "providers": providers,
            "models": models,
        }

    def get_config(self) -> dict[str, object]:
        with self._lock:
            return self._load_config()

    def set_config(self, raw: object) -> dict[str, object]:
        config = normalize_usage_config(raw)
        with self._lock:
            self._config = config
            self._write_json(self.config_path, config)
        return config

    def reset(self) -> None:
        with self._lock:
            self._totals = {}
            self._recent.clear()
            self._since = datetime.now(UTC).isoformat(timespec="seconds")
            self._save()

    def rank_candidates(
        self, candidates: tuple[ProviderModelTarget, ...]
    ) -> tuple[ProviderModelTarget, ...]:
        """Healthy, under-quota targets first, then fastest recent latency.

        Targets with no recent data keep their configured order after measured
        healthy ones, so an unmeasured model is still tried before a failing one.
        """
        today = datetime.now(UTC).date().isoformat()
        with self._lock:
            totals = self._load()
            limits = self._load_config()["daily_token_limits"]
            assert isinstance(limits, dict)
            provider_today: dict[str, int] = {}
            for key, value in totals.items():
                provider_id = key.split("/", 1)[0]
                provider_today[provider_id] = provider_today.get(
                    provider_id, 0
                ) + value.daily_tokens.get(today, 0)

            def sort_key(item: tuple[int, ProviderModelTarget]) -> tuple:
                index, target = item
                health = self._health(target.provider_model_ref)
                limit = limits.get(target.provider_id)
                over_quota = bool(
                    limit and provider_today.get(target.provider_id, 0) >= limit
                )
                penalty = 2 if over_quota else 1 if health["status"] == "down" else 0
                latency = health["avg_latency_ms"]
                return (
                    penalty,
                    latency if isinstance(latency, float) else math.inf,
                    index,
                )

            return tuple(
                target for _, target in sorted(enumerate(candidates), key=sort_key)
            )

    def _health(self, key: str) -> dict[str, object]:
        now = monotonic()
        recent = [
            outcome
            for outcome in self._recent.get(key, ())
            if now - outcome.at <= RECENT_SECONDS
        ]
        if not recent:
            return {"status": "unknown", "recent": 0, "avg_latency_ms": None}
        failures = sum(not outcome.ok for outcome in recent)
        rate_limited_now = any(
            outcome.rate_limited and now - outcome.at <= RATE_LIMIT_COOLDOWN_SECONDS
            for outcome in recent
        )
        latencies = [outcome.duration_ms for outcome in recent if outcome.ok]
        down = rate_limited_now or failures / len(recent) >= UNHEALTHY_FAILURE_RATIO
        return {
            "status": "down" if down else "up",
            "recent": len(recent),
            "recent_failures": failures,
            "avg_latency_ms": (
                round(sum(latencies) / len(latencies), 2) if latencies else None
            ),
        }

    def _load(self) -> dict[str, UsageTotals]:
        if self._totals is not None:
            return self._totals
        self._totals = {}
        raw = self._read_json(self.path)
        self._since = raw.get("since") or datetime.now(UTC).isoformat(
            timespec="seconds"
        )
        models = raw.get("models")
        for key, value in (models if isinstance(models, dict) else {}).items():
            if isinstance(value, dict):
                known = {
                    name: value[name]
                    for name in UsageTotals.__dataclass_fields__
                    if name in value
                }
                self._totals[key] = UsageTotals(**known)
        return self._totals

    def _load_config(self) -> dict[str, object]:
        if self._config is None:
            self._config = normalize_usage_config(self._read_json(self.config_path))
        return self._config

    def _save(self) -> None:
        self._write_json(
            self.path,
            {
                "since": self._since,
                "models": {
                    key: asdict(value) for key, value in (self._totals or {}).items()
                },
            },
        )

    @staticmethod
    def _read_json(path: Path) -> dict:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable {}: {}", path.name, type(exc).__name__)
            return {}
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _write_json(path: Path, data: object) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("Could not persist {}: {}", path.name, type(exc).__name__)


usage_tracker = UsageTracker()
