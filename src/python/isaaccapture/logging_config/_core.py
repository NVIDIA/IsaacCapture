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


def _validate_directory_chain(directory: Path, *, require_owner: bool) -> None:
    current = directory
    target = True
    while True:
        info = current.lstat()
        mode = info.st_mode
        if not stat.S_ISDIR(mode):
            raise NotADirectoryError(current)
        if target and require_owner and info.st_uid != os.getuid():
            raise PermissionError(
                f"refusing log directory not owned by this user: {current}"
            )
        publicly_writable = mode & (stat.S_IWGRP | stat.S_IWOTH)
        if publicly_writable and (target or not mode & stat.S_ISVTX):
            raise PermissionError(f"refusing replaceable log directory path: {current}")
        if current == current.parent:
            break
        current = current.parent
        target = False


def ensure_private_dir(directory: Path) -> Path:
    """Create *directory* owner-only and reject replaceable POSIX paths."""
    if not _POSIX:
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    missing: list[Path] = []
    probe = directory
    while True:
        try:
            probe.lstat()
            break
        except FileNotFoundError:
            missing.append(probe)
            if probe == probe.parent:
                raise
            probe = probe.parent

    # Validate the existing prefix before creating through it; otherwise a
    # symlinked parent could redirect even an owner-only child mkdir.
    _validate_directory_chain(probe, require_owner=False)
    for component in reversed(missing):
        try:
            # mkdir applies the restrictive mode atomically; chmod after a
            # recursive mkdir would leave a world-writable race window.
            component.mkdir(mode=0o700)
        except FileExistsError:
            pass

    _validate_directory_chain(directory, require_owner=True)
    return directory


def ensure_log_dir() -> Path:
    """:func:`log_dir`, created if needed."""
    return ensure_private_dir(log_dir())
