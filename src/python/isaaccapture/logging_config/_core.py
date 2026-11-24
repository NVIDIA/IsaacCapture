# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Names, line format, levels and log directory shared by every handler here."""

from __future__ import annotations

import logging
import os
import stat
import tempfile
from pathlib import Path

# Guard POSIX-only APIs because Windows builds import this module.
_POSIX = os.name == "posix"

root_logger = logging.getLogger("isaaccapture")


def logging_enabled() -> bool:
    """``ISAACCAPTURE_LOGGING=off`` switches this whole system off; anything else leaves it on."""
    return (os.environ.get("ISAACCAPTURE_LOGGING") or "").strip().lower() != "off"


LINE_FORMAT = "[%(asctime)s.%(msecs)03d] [%(levelname)-5s] [%(name)s] [pid:%(process)d] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# POSIX needs per-uid isolation; the Windows temp directory is already per-user.
_DEFAULT_LOG_DIR = (
    Path(f"/tmp/isaaccapture-{os.getuid()}/logs")
    if _POSIX
    else Path(tempfile.gettempdir()) / "isaaccapture" / "logs"
)

# Custom level below DEBUG; emit with ``logger.log(TRACE, ...)``.
TRACE = 5
logging.addLevelName(TRACE, "TRACE")

_LEVEL_NAMES = {
    "trace": TRACE,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# Shared name mapping for Python and C++ console thresholds.
_LEVEL_NAME_BY_VALUE = {value: name for name, value in _LEVEL_NAMES.items()}


def _move_above_std(fd: int) -> int:
    """Relocate *fd* clear of 0/1/2, leaving any closed std fd closed."""
    low: list[int] = []
    try:
        while fd <= 2:
            low.append(fd)
            fd = os.dup(fd)
    except OSError:
        for spare in low:
            try:
                os.close(spare)
            except OSError:
                pass
        raise
    for spare in low:
        os.close(spare)
    return fd


def resolve_level(level: str) -> int:
    """Map one of the names in ``_LEVEL_NAMES``, in any case, to its level."""
    try:
        return _LEVEL_NAMES[level.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown log level {level!r}; expected one of {sorted(_LEVEL_NAMES)}"
        ) from None


def env_console_level() -> int:
    """Read the shared console threshold; invalid values fall back to ``INFO``."""
    raw = os.environ.get("ISAACCAPTURE_LOG_LEVEL")
    if not raw:
        return logging.INFO
    raw = raw.strip()
    digits = raw[1:] if raw[:1] in ("+", "-") else raw
    if digits.isascii() and digits.isdecimal():
        return int(raw)
    return _LEVEL_NAMES.get(raw.lower(), logging.INFO)


def log_dir() -> Path:
    """Resolve and publish the session log directory shared with C++."""
    override = os.environ.get("ISAACCAPTURE_LOG_DIR")
    try:
        directory = Path(override).expanduser() if override else _DEFAULT_LOG_DIR
    except RuntimeError:
        # Containers may have neither HOME nor a passwd entry for ``~``.
        directory = _DEFAULT_LOG_DIR

    try:
        directory = directory.absolute()
    except OSError:
        pass
    # Publish before a child changes directory or creates its first C++ logger.
    os.environ["ISAACCAPTURE_LOG_DIR"] = str(directory)
    return directory


def ensure_private_dir(directory: Path) -> Path:
    """Create owner-only directories without following replaceable POSIX paths."""
    if not _POSIX:
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    parts = directory.absolute().parts[1:]
    if not parts or ".." in parts:
        raise PermissionError(f"refusing ambiguous log directory path: {directory}")

    # Pin parents against pathname swaps; O_PATH needs search permission, not read.
    flags = getattr(os, "O_PATH", os.O_RDONLY)
    flags |= os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    parent = os.open("/", flags)
    try:
        for index, name in enumerate(parts):
            try:
                os.mkdir(name, mode=0o700, dir_fd=parent)
            except FileExistsError:
                pass  # Existing entries still undergo the checks below.
            child = os.open(name, flags, dir_fd=parent)
            os.close(parent)
            parent = child

            info = os.fstat(parent)
            target = index == len(parts) - 1
            # Sticky shared ancestors (such as /tmp) are safe to traverse, but
            # the final directory must belong to us and forbid other writers.
            if info.st_uid not in (0, os.getuid()) or (
                target and info.st_uid != os.getuid()
            ):
                raise PermissionError(f"refusing foreign-owned log path: {directory}")
            writable = info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            if writable and (target or not info.st_mode & stat.S_ISVTX):
                raise PermissionError(f"refusing replaceable log path: {directory}")
    finally:
        os.close(parent)
    return directory


def ensure_log_dir() -> Path:
    """:func:`log_dir`, created if needed."""
    return ensure_private_dir(log_dir())
