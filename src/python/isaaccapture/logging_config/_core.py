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


def default_runtime_dir() -> Path:
    """Resolve the platform temp-root alias before adding our per-user directory."""
    if not _POSIX:
        return Path(tempfile.gettempdir()) / "isaaccapture"
    root = Path("/tmp")
    try:
        root = root.resolve(strict=True)
    except (OSError, RuntimeError):
        pass  # Directory validation will report an unusable root without breaking import.
    return root / f"isaaccapture-{os.getuid()}"


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
        directory = (
            Path(override).expanduser() if override else default_runtime_dir() / "logs"
        )
    except RuntimeError:
        # Containers may have neither HOME nor a passwd entry for ``~``.
        directory = default_runtime_dir() / "logs"

    try:
        directory = directory.absolute()
    except OSError:
        pass
    # Publish before a child changes directory or creates its first C++ logger.
    os.environ["ISAACCAPTURE_LOG_DIR"] = str(directory)
    return directory


def _validate_directory(fd: int, directory: Path, *, target: bool) -> None:
    info = os.fstat(fd)
    # Sticky shared ancestors protect entries; the target must forbid other writers.
    if info.st_uid not in (0, os.geteuid()) or (target and info.st_uid != os.geteuid()):
        raise PermissionError(f"refusing foreign-owned directory path: {directory}")
    writable = info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    if writable and (target or not info.st_mode & stat.S_ISVTX):
        raise PermissionError(f"refusing replaceable directory path: {directory}")


def _private_dir(directory: Path, *, create_missing: bool) -> Path:
    if not _POSIX:
        if create_missing:
            directory.mkdir(parents=True, exist_ok=True)
        elif not directory.is_dir():
            raise NotADirectoryError(str(directory))
        return directory

    parts = directory.absolute().parts[1:]
    if not parts or ".." in parts:
        raise PermissionError(f"refusing ambiguous directory path: {directory}")

    # Pin parents against pathname swaps; O_PATH needs search permission, not read.
    flags = getattr(os, "O_PATH", os.O_RDONLY)
    flags |= os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    parent = os.open("/", flags)
    try:
        # The initial root is also a parent and must be checked before mkdirat.
        component = Path("/")
        _validate_directory(parent, component, target=False)
        for index, name in enumerate(parts):
            if create_missing:
                try:
                    os.mkdir(name, mode=0o700, dir_fd=parent)
                except FileExistsError:
                    pass  # Existing entries still undergo the checks below.
            child = os.open(name, flags, dir_fd=parent)
            os.close(parent)
            parent = child

            component /= name
            _validate_directory(parent, component, target=index == len(parts) - 1)
    finally:
        os.close(parent)
    return directory


def ensure_private_dir(directory: Path) -> Path:
    """Create owner-only directories without following replaceable POSIX paths."""
    return _private_dir(directory, create_missing=True)


def is_private_dir(directory: Path) -> bool:
    """Apply the same ownership and path checks without creating anything."""
    try:
        _private_dir(directory, create_missing=False)
    except OSError:
        return False
    return True


def ensure_log_dir() -> Path:
    """:func:`log_dir`, created if needed."""
    return ensure_private_dir(log_dir())
