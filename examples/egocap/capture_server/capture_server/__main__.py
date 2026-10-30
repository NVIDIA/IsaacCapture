"""CLI entry point. Importing this module never starts a listener --
only calling :func:`main` does. The actual argument parsing and listener
live in :mod:`capture_server.server`; this module just re-exports them as
the console-script target declared in pyproject.toml."""

from __future__ import annotations

from .server import main

__all__ = ["main"]

if __name__ == "__main__":  # pragma: no cover - exercised by running the module, not pytest
    raise SystemExit(main())
