"""Lightweight entry points for installed Code Relay commands."""

import sys
from collections.abc import Sequence

from code_relay.core.version import package_version


def serve(argv: Sequence[str] | None = None) -> None:
    """Start the FastAPI server (registered as ``fcc-server``)."""
    if _print_version_if_requested(argv):
        return

    # Keep the server composition root off metadata-only command paths.
    from code_relay.cli.commands import serve as run_server

    run_server()


def _print_version_if_requested(argv: Sequence[str] | None) -> bool:
    args = sys.argv[1:] if argv is None else argv
    if "--version" not in args:
        return False
    print(f"code-relay {package_version()}")
    return True


def doctor(argv: Sequence[str] | None = None) -> None:
    """Print and copy a local diagnostic report (registered as ``fcc-doctor``)."""
    if _print_version_if_requested(argv):
        return
    from code_relay.cli.doctor import main

    main(argv)
