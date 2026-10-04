"""Installed Codex launcher and external-client credential handoff."""

import sys
from collections.abc import Sequence

from code_relay.config.loader import get_settings
from code_relay.harnesses.codex import (
    CODEX_INSTALL_HINT,
    PRINT_PROXY_AUTH_TOKEN_FLAG,
    prepare_codex_launch,
)
from code_relay.harnesses.launch import PreparedLaunch
from code_relay.harnesses.resources import LaunchResources

from .runner import HarnessSpec, LaunchContext, launch_harness


def _configure(
    ctx: LaunchContext, args: list[str], files: LaunchResources
) -> PreparedLaunch:
    catalog = ctx.require_catalog()
    return prepare_codex_launch(
        binary_path=ctx.binary_path,
        proxy_root_url=ctx.proxy_root_url,
        model=ctx.settings.model,
        models=catalog.models,
        base_env=ctx.base_env,
        args=args,
        files=files,
    )


SPEC = HarnessSpec(
    binary_name="codex",
    display_name="Codex CLI",
    install_hint=CODEX_INSTALL_HINT,
    configure=_configure,
    catalog_view="responses",
)


def launch(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == [PRINT_PROXY_AUTH_TOKEN_FLAG]:
        print(get_settings().proxy_auth_token)
        return
    launch_harness(SPEC, args)
