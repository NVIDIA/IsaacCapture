# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one console handler on the ``isaaccapture`` root logger, and its knobs."""

from __future__ import annotations

import functools
import logging
import os
import re
import threading
from collections.abc import Callable, Mapping
from typing import TypedDict, TypeVar

from . import _forwarding, _native_fd
from ._core import (
    _LEVEL_NAME_BY_VALUE,
    TRACE,
    env_console_level,
    logging_enabled,
    resolve_level,
    root_logger,
)


_ANSI_RESET = "\033[0m"

# Accept only SGR color escapes such as ``\x1b[36m``.
_SGR_ESCAPE = re.compile(r"(?:\x1b\[[0-9;]*m)+")

# Logger name to ANSI emphasis; a name's colour also covers its dotted descendants.
_logger_colors: dict[str, str] = {
    # cloudxr.oob_teleop_env contains [setup-oob] / [usb-local] progress.
    "isaaccapture.cloudxr.oob_teleop_env": "\033[36m",
}


_V = TypeVar("_V")


def _prefix_lookup(name: str, table: Mapping[str, _V]) -> _V | None:
    """*table*'s value for *name* or its nearest dotted ancestor, else ``None``."""
    while name not in table and "." in name:
        name = name.rpartition(".")[0]
    return table.get(name)


def _cached_prefix_lookup(table: Mapping[str, _V]) -> Callable[[str], _V | None]:
    """``_prefix_lookup`` over a copy of *table*, cached per logger name.

    The copy keeps every cached result valid, so a change to *table* needs a new lookup.
    """
    table = dict(table)
    return functools.cache(lambda name: _prefix_lookup(name, table))


# set_logger_colors() replaces this lookup after every change, so it never goes stale.
_logger_color = _cached_prefix_lookup(_logger_colors)


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
    """Color terminal lines by level; optionally emphasize or shorten logger names."""

    def __init__(
        self,
        *args,
        handler: logging.StreamHandler,
        use_short_name: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Capture scopes can replace the handler stream after construction.
        self._handler = handler
        self._use_short_name = use_short_name

    def _is_terminal(self) -> bool:
        stream = getattr(self._handler, "stream", None)
        try:
            return bool(stream.isatty())
        except (AttributeError, OSError, ValueError):
            return False

    def format(self, record: logging.LogRecord) -> str:
        terminal = self._is_terminal()
        level = _level_color(record.levelno) if terminal else None
        emphasis = _logger_color(record.name) if terminal else None
        if level is None and emphasis is None and not self._use_short_name:
            return super().format(record)

        # Restore the shared record before the file handler formats it.
        original = record.name
        name = original.rpartition(".")[2] if self._use_short_name else original
        if emphasis is not None:
            name = f"{emphasis}{name}{level or _ANSI_RESET}"
        record.name = name
        try:
            line = super().format(record)
        finally:
            record.name = original
        return f"{level}{line}{_ANSI_RESET}" if level is not None else line


_lock = threading.Lock()
_handler: logging.StreamHandler | None = None

# The console filters as cached matchers: below WARNING a record must pass both, and
# None passes all. A change replaces a matcher, so its cached results never go stale.
_match_logger_name: Callable[[str], bool | None] | None = None
_match_content: Callable[[str], bool] | None = None


def _console_filter(record: logging.LogRecord) -> bool:
    # Read each once; a setter on another thread may replace it in between.
    match_name, match_content = _match_logger_name, _match_content
    return (match_name is None or match_name(record.name) is not None) and (
        match_content is None or match_content(record.getMessage())
    )


def ensure_handler() -> logging.StreamHandler:
    """Create the console handler once; attach it only in an enabled leader."""
    global _handler
    if _handler is not None:
        return _handler
    with _lock:
        if _handler is not None:
            return _handler
        handler = logging.StreamHandler()
        handler.setFormatter(_console_formatter(handler))
        handler.setLevel(env_console_level())
        handler.addFilter(_console_filter)
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
    other value, an integer included, is logged as an error and the current
    threshold is kept.
    """
    if not isinstance(level, str) or level.lower() not in _LEVEL_NAME_BY_VALUE.values():
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


def set_console_logger_name_filter(names: set[str] | None) -> None:
    """Keep console records below WARNING only from *names* and their dotted descendants.

    Each call replaces the whole set, so a name left out no longer passes; an empty
    set passes no logger, and ``None`` removes this filter. Warnings and errors
    always pass; other records must also pass ``set_console_content_filter()``.
    """
    global _match_logger_name
    # Listed names map to True, so the lookup shared with logger colours matches them.
    _match_logger_name = (
        None if names is None else _cached_prefix_lookup(dict.fromkeys(names, True))
    )


def set_console_content_filter(pattern: str | None) -> bool:
    """Keep console records below WARNING only if their message matches regex *pattern*.

    ``None`` removes this filter. Returns ``True`` once the filter is applied or
    removed, and ``False`` for an invalid *pattern*, which is logged as an error while
    the current filter stays. Warnings and errors always pass; other records must
    also pass ``set_console_logger_name_filter()``.
    """
    global _match_content
    if pattern is None:
        _match_content = None
        return True
    # Only compiling a regex can tell whether it is valid, so that one error is caught.
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        root_logger.error(
            "Invalid console content filter %r (%s); keeping the current one.",
            pattern,
            exc,
        )
        return False
    # Messages vary without bound, so lru_cache keeps only the most recent results.
    _match_content = functools.lru_cache(lambda message: bool(regex.search(message)))
    return True


class _ConsoleFormat(TypedDict, total=False):
    show_time: bool
    show_level: bool
    show_logger_name: bool
    show_pid: bool

    use_short_name: bool  # True: only the last dotted segment


# Console only: log files keep DATE_FORMAT's date, which sessions crossing midnight,
# rotated logs and cross-process correlation need.
_CONSOLE_DATE_FORMAT = "%H:%M:%S"

# The console's current columns, starting from the defaults the handler is built with.
_console_format: dict[str, bool] = {
    "show_time": True,
    "show_level": True,
    "show_logger_name": True,
    "show_pid": False,
    "use_short_name": True,
}


def _console_formatter(handler: logging.StreamHandler) -> _LoggerNameColorFormatter:
    """Build the console formatter from the current ``_console_format`` columns."""
    columns = [
        "[%(asctime)s.%(msecs)03d]" if _console_format["show_time"] else "",
        "[%(levelname)-5s]" if _console_format["show_level"] else "",
        "[%(name)s]" if _console_format["show_logger_name"] else "",
        "[pid:%(process)d]" if _console_format["show_pid"] else "",
        "%(message)s",
    ]
    return _LoggerNameColorFormatter(
        " ".join(column for column in columns if column),
        datefmt=_CONSOLE_DATE_FORMAT,
        handler=handler,
        use_short_name=_console_format["use_short_name"],
    )


def set_console_format(changes: _ConsoleFormat) -> None:
    """Update the console columns, changing only the given keys.

    Keys left out keep their current setting, which starts as time, level and the
    last logger-name segment, without the pid; an unknown key or a non-bool value
    is logged as an error and skipped. Columns keep the ``LINE_FORMAT`` order with
    the time as ``HH:MM:SS.mmm``; log files from both halves keep the full
    ``LINE_FORMAT``, date included, and logger colours still match the full name.
    """
    for key, value in changes.items():
        if key not in _console_format:
            root_logger.error(
                "Unknown console format key %r, expected one of %s; ignoring it.",
                key,
                sorted(_console_format),
            )
            continue
        if not isinstance(value, bool):
            root_logger.error(
                "Console format %r must be a bool, got %r; ignoring it.", key, value
            )
            continue
        _console_format[key] = value

    handler = ensure_handler()
    handler.setFormatter(_console_formatter(handler))


def set_logger_colors(colors: dict[str, str | None]) -> None:
    """Set terminal-only SGR emphasis by logger name; ``None`` removes it.

    A colour also covers the name's dotted descendants; the most specific name wins.
    A value that is not purely SGR escapes is logged as an error and skipped.
    """
    global _logger_color
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
    _logger_color = _cached_prefix_lookup(_logger_colors)
