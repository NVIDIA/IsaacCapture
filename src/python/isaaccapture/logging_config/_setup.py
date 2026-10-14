# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-time bootstrap of this process's half of the logger tree."""

from __future__ import annotations

from . import _console, _file, _forwarding, _native_fd
from ._core import logging_enabled, root_logger

_installed = False


def install() -> None:
    """Install this process once as the session leader or a forwarding child."""
    global _installed
    if _installed:
        return
    _installed = True
    if not logging_enabled():
        return

    socket_path = _forwarding.socket_path()
    if socket_path is not None:
        # Attach forwarding before setup can emit a warning.
        _forwarding.ensure_handler(socket_path)
        # Capture scopes still need an unattached console stream.
        console = _console.ensure_handler()
        _native_fd.ensure_sink()
        _native_fd.follow_console_level(console.level)
        return

    console = _console.ensure_handler()
    try:
        _file.ensure_handler()
    except OSError as exc:
        # File failure must not break ``import isaaccapture``.
        root_logger.warning(
            "File logging disabled: %s. Records will reach the console only. "
            "Set ISAACCAPTURE_LOG_DIR to a directory you can write.",
            exc,
        )
    # Attach handlers before native capture can warn.
    _native_fd.ensure_sink()
    # Honor TRACE mirroring from the first native byte.
    _native_fd.follow_console_level(console.level)
    _forwarding.ensure_receiver()


def set_propagate_to_root(enabled: bool) -> None:
    """Disable root propagation when host handlers would duplicate output."""
    if logging_enabled():
        root_logger.propagate = bool(enabled)
