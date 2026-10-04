"""Lightweight entrypoint for the optional FCC desktop shell."""

import sys
from collections.abc import Sequence
from pathlib import Path

from code_relay.cli.desktop_assets import export_app_icon
from code_relay.core.windows_dpi import enable_dpi_awareness


def launch(argv: Sequence[str] | None = None) -> None:
    """Export installer assets or launch the supported native tray adapter."""

    args = tuple(sys.argv[1:] if argv is None else argv)
    if len(args) == 2 and args[0] == "--export-icon":
        export_app_icon(Path(args[1]))
        return
    if args:
        print("Usage: fcc-desktop [--export-icon PATH]", file=sys.stderr)
        raise SystemExit(2)
    if sys.platform not in {"darwin", "win32"}:
        print("FCC Desktop is supported on Windows and macOS.", file=sys.stderr)
        raise SystemExit(1)

    enable_dpi_awareness()

    from code_relay.cli.desktop_tray import launch as launch_tray

    launch_tray()
