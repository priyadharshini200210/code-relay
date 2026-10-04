<div align="center">

# Code Relay

**An AI model gateway for coding agents, with usage tracking, cost estimates and smart routing.**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-2e7d32.svg)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/python-3.14-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

</div>

Coding agents like Claude Code, Codex and Aider each talk to one model
provider. When that provider rate-limits you or goes down, you're stuck, and
you can't see how much you're using.

Code Relay sits between your coding agent and 59+ model providers (free tiers,
paid APIs, subscriptions and local models). It shows exactly what you use,
estimates what it costs, and sends each request to the healthiest, fastest
model that still has free quota.

## Features

### 📊 Usage dashboard
A live page at `/admin/usage` that refreshes every 5 seconds:
- Requests, successes, failures and rate limits (HTTP 429) per provider and model
- Input, output and cached tokens, plus a daily token history
- Health (up, down or unknown) and average response time for each model
- Totals saved in `~/.fcc/usage.json`, so they survive restarts

### 💰 Cost estimates
- Enter your own prices (USD per 1M tokens) per model or per provider
- Set a `baseline` price (what you'd pay without free providers) to see
  estimated savings
- Saved in `~/.fcc/usage_config.json`

### 🎯 Free-quota tracking
- Set a daily token limit for each provider
- See how much free quota is left today, with a bar that turns amber at 80%
  and red at 100%

### 🔀 Smart routing
Turn on **Smart Routing** (`SMART_ROUTING`) in the Admin UI's advanced settings.
For each request, your primary model and fallbacks are tried in this order:
1. Healthy models under their daily limit, fastest first
2. Models with no recent data, in your configured order
3. Models that are failing (half or more of recent requests) or were
   rate-limited in the last 60 seconds
4. Providers over today's limit

It's off by default. If ranking ever fails, Code Relay keeps your configured
order, so routing never makes a request worse.

### Built in from the base gateway
- One endpoint for 59+ providers, including NVIDIA NIM, Groq, Gemini,
  OpenRouter, DeepSeek, Anthropic, GitHub Copilot, Ollama and LM Studio
- Automatic fallback to the next model during provider outages
- Separate models for each Claude tier (Opus, Sonnet, Haiku), for example a
  strong model for planning and a fast one for execution
- Works with Claude Code, Codex, Aider, Cline, OpenCode, VS Code and JetBrains

## Quick Start

Requires [uv](https://docs.astral.sh/uv/); it installs the right Python
version automatically.

```bash
cd code-relay
uv sync
uv run fcc-server
```

The server log prints the Admin UI address.

1. Open the Admin UI and go to **Providers**.
2. Add an API key for at least one provider, for example a free
   [NVIDIA NIM key](https://build.nvidia.com/settings/api-keys).
3. Choose a `MODEL` (and optionally `MODEL_OPUS`, `MODEL_SONNET`,
   `MODEL_HAIKU` and fallbacks), then click **Apply**.
4. Open `/admin/usage` to set prices and daily limits.

Then start your coding agent through the gateway:

```bash
uv run fcc-claude     # Claude Code
uv run fcc-codex      # Codex
uv run fcc-aider      # Aider
```

Enter API keys only in the Admin UI or as environment variables. Never commit
them.

## Usage API

All endpoints are local-only, like the rest of the Admin UI.

| Method | Path | Returns |
| --- | --- | --- |
| `GET` | `/admin/api/usage` | Totals, costs, quota and health per provider and model |
| `POST` | `/admin/api/usage/reset` | Clears usage counters (prices and limits are kept) |
| `GET` | `/admin/api/usage/config` | Current prices and daily limits |
| `PUT` | `/admin/api/usage/config` | Saves prices and daily limits |

Example settings:

```json
{
  "baseline": { "input": 3, "output": 15 },
  "provider_prices": { "groq": { "input": 0, "output": 0 } },
  "model_prices": {},
  "daily_token_limits": { "groq": 1000000 }
}
```

## How It Works

- **Usage capture:** one middleware sees every `/v1/messages` and
  `/v1/responses` request as it completes. It reads token counts from both
  streaming and normal responses, in Anthropic and OpenAI formats, so no
  provider code needed to change.
  (`src/code_relay/api/request_outcomes.py`, `src/code_relay/api/usage_tracking.py`)
- **Smart routing:** the provider executor asks a ranker to reorder the
  candidate models before trying them. The ranker uses each model's last 20
  results from the past 15 minutes and today's quota. If a ranker adds, drops
  or duplicates a model, it's ignored. (`src/code_relay/application/execution.py`)
- **Dashboard:** a single self-contained HTML page with no external
  dependencies, in light and dark mode.
  (`src/code_relay/api/admin_static/usage.html`)

## Development

```bash
uv sync
uv run pytest tests/api/test_usage_tracking.py -p no:xdist
uv run ruff check . && uv run ruff format --check .
```

## Credits And License

Code Relay is built on the open-source
Free claude code project,
which provides the base multi-provider gateway. Code Relay adds
the usage dashboard, cost estimates, free-quota tracking and smart routing.

Licensed under the [GNU AGPL v3]

Not affiliated with or endorsed by Anthropic. Claude and Claude Code are
trademarks of Anthropic.
