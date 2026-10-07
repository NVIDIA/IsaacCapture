# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one console handler on the ``isaaccapture`` root logger, and its knobs."""

from __future__ import annotations

import logging
import os
import re
import threading

from . import _forwarding, _native_fd
from ._core import (
    _LEVEL_NAME_BY_VALUE,
    DATE_FORMAT,
    LINE_FORMAT,
    TRACE,
    env_console_level,
    logging_enabled,
    resolve_level,
    root_logger,
)


class KeywordFilter(logging.Filter):
    """Keep only records whose logger name and/or message match *pattern*."""

    def __init__(self, pattern: str, target: str = "both") -> None:
        super().__init__()
        if target not in ("logger_name", "content", "both"):
            raise ValueError(
                f"target must be 'logger_name', 'content', or 'both', got {target!r}"
            )
        self._regex = re.compile(pattern)
        self._target = target

    def filter(self, record: logging.LogRecord) -> bool:
        return (
            self._target in ("logger_name", "both")
            and bool(self._regex.search(record.name))
        ) or (
            self._target in ("content", "both")
            and bool(self._regex.search(record.getMessage()))
        )


_ANSI_RESET = "\033[0m"

# Accept only SGR color escapes such as ``\x1b[36m``.
_SGR_ESCAPE = re.compile(r"(?:\x1b\[[0-9;]*m)+")

# Exact logger name to ANSI emphasis.
_logger_colors: dict[str, str] = {
    # cloudxr.oob_teleop_env contains [setup-oob] / [usb-local] progress.
    "isaaccapture.cloudxr.oob_teleop_env": "\033[36m",
}


#: Color only WARNING and ERROR+ so severity remains distinctive.
_LEVEL_WARNING_COLOR = "\033[33m"
_LEVEL_ERROR_COLOR = "\033[31m"


def _level_color(levelno: int) -> str | None:
    """The whole-line emphasis for *levelno*, or ``None`` to leave it plain."""
    if levelno >= logging.ERROR:
        return _LEVEL_ERROR_COLOR
    if levelno >= logging.WARNING:
        return _LEVEL_WARNING_COLOR
    return None


class _LoggerNameColorFormatter(logging.Formatter):
    """Color terminal lines by level and optionally emphasize the logger name."""

    def __init__(self, *args, handler: logging.StreamHandler, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Capture scopes can replace the handler stream after construction.
        self._handler = handler

    def _is_terminal(self) -> bool:
        stream = getattr(self._handler, "stream", None)
        try:
            return bool(stream.isatty())
        except (AttributeError, OSError, ValueError):
            return False

    def format(self, record: logging.LogRecord) -> str:
        if not self._is_terminal():
            return super().format(record)

        level = _level_color(record.levelno)
        emphasis = _logger_colors.get(record.name)
        if level is None and emphasis is None:
            return super().format(record)

        # Restore the shared record before the file handler formats it.
        original = record.name
        if emphasis is not None:
            record.name = f"{emphasis}{original}{level or _ANSI_RESET}"
        try:
            line = super().format(record)
        finally:
            record.name = original
        return f"{level}{line}{_ANSI_RESET}" if level is not None else line


_lock = threading.Lock()
_handler: logging.StreamHandler | None = None
_active_filter: KeywordFilter | None = None


def ensure_handler() -> logging.StreamHandler:
    """Create the console handler once; attach it only in an enabled leader."""
    global _handler
    if _handler is not None:
        return _handler
    with _lock:
        if _handler is not None:
            return _handler
        handler = logging.StreamHandler()
        handler.setFormatter(
            _LoggerNameColorFormatter(LINE_FORMAT, datefmt=DATE_FORMAT, handler=handler)
        )
        handler.setLevel(env_console_level())
        if not logging_enabled():
            _handler = handler
            return _handler
        root_logger.setLevel(
            TRACE
        )  # handlers filter; the logger itself must stay maximally permissive
        if _forwarding.socket_path() is None:
            root_logger.addHandler(handler)
        _handler = handler
        return _handler


def set_console_level(level: str) -> None:
    """Set the console threshold by level name and mirror raw output only at ``TRACE``.

    *level* is trace, debug, info, warning, error or critical, in any case; any
    other name is logged as an error and the current threshold is kept.
    """
    if level.lower() not in _LEVEL_NAME_BY_VALUE.values():
        root_logger.error(
            "Unknown console level %r, expected one of %s; keeping the current one.",
            level,
            sorted(_LEVEL_NAME_BY_VALUE.values()),
        )
        return
    resolved = resolve_level(level)
    handler = ensure_handler()
    handler.setLevel(resolved)
    _native_fd.follow_console_level(resolved)
    # C++ processes that fall back to local sinks read their threshold here.
    os.environ["ISAACCAPTURE_LOG_LEVEL"] = _LEVEL_NAME_BY_VALUE[resolved]


def set_console_filter(pattern: str | None, target: str = "both") -> None:
    """Set the console handler's keyword filter, or clear it if *pattern* is ``None``."""
    handler = ensure_handler()
    # Validate the replacement before removing the active filter.
    replacement = KeywordFilter(pattern, target=target) if pattern is not None else None

    global _active_filter
    if _active_filter is not None:
        handler.removeFilter(_active_filter)
    _active_filter = replacement
    if replacement is not None:
        handler.addFilter(replacement)


def set_logger_colors(colors: dict[str, str | None]) -> None:
    """Set terminal-only SGR emphasis by exact logger name; ``None`` removes it.

    A value that is not purely SGR escapes is logged as an error and skipped.
    """
    ensure_handler()
    for name, color in colors.items():
        if color is None:
            _logger_colors.pop(name, None)
            continue
        if not _SGR_ESCAPE.fullmatch(color):
            root_logger.error(
                "Colour for logger %r must be one or more SGR escapes, such as "
                "'\\033[36m' or '\\033[38;2;255;136;0m', got %r; ignoring it.",
                name,
                color,
            )
            continue
        _logger_colors[name] = color
