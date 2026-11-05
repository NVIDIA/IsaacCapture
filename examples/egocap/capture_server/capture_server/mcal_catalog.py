# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The MCAL files in the server's managed directory, and the one policy deciding which are selectable.

The device query and the begin-time check of a selected file both go through :func:`scan`, so a file that is
listed is a file that can be selected, and the reverse. The directory is just a folder of ``.mcal`` files:
anything placed there, by hand or by a calibration tool, is offered. Scanning reads the directory, so callers
run it off the event loop.

A file is selectable when it is a nonempty, readable ``.mcal`` regular file directly in the directory. Symbolic
links are never offered; copy files into the directory instead. Whether a glove accepts a file is the Manus
plugin's decision, not this module's.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import protocol as p

__all__ = ["Catalog", "resolve", "scan", "selection_refusal"]


@dataclass(frozen=True)
class Catalog:
    """One scan's result: its status, the selectable files sorted by name, and the reason when UNAVAILABLE."""

    status: p.ManusCalibrationCatalogStatus
    files: tuple[str, ...] = ()
    error_message: str | None = None


def scan(directory: Path | None) -> Catalog:
    """Read the managed directory once and return everything selectable, sorted by filename."""
    # No directory configured: nothing can be listed or selected.
    if directory is None:
        return Catalog(p.ManusCalibrationCatalogStatus.DISABLED)
    try:
        # The directory is created by whatever first writes a file into it, so absence is an empty catalog.
        if not directory.exists():
            return Catalog(p.ManusCalibrationCatalogStatus.READY)
        root = directory.resolve(strict=True)
        if not root.is_dir():
            return _unavailable("the calibration directory is not a directory")
        files = _selectable(root)
    except PermissionError:
        return _unavailable("the calibration directory cannot be read (permission denied)")
    except OSError as exc:
        return _unavailable(f"the calibration directory cannot be read ({exc.strerror or 'I/O error'})")
    # A listing too large for one reply is an error, never a truncated list.
    if len(files) > p.MAX_MCAL_FILES:
        return _unavailable(f"the calibration directory has more than {p.MAX_MCAL_FILES} selectable files")
    return Catalog(p.ManusCalibrationCatalogStatus.READY, tuple(sorted(files)))


def _unavailable(message: str) -> Catalog:
    """An UNAVAILABLE catalog carrying the operator-facing reason."""
    return Catalog(p.ManusCalibrationCatalogStatus.UNAVAILABLE, (), message)


def _selectable(root: Path) -> list[str]:
    """Files directly in ``root`` that pass the selection policy."""
    files = []
    with os.scandir(root) as listing:
        for entry in listing:
            # Names the protocol cannot carry are never offered.
            if not p._MCAL_NAME_RE.fullmatch(entry.name):
                continue
            try:
                # Links are refused even when their target stays inside the directory.
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    continue
                if entry.stat(follow_symlinks=False).st_size == 0 or not os.access(entry.path, os.R_OK):
                    continue
            except OSError:
                # An entry that vanished mid-scan is simply not offered.
                continue
            files.append(entry.name)
    return files


def selection_refusal(catalog: Catalog, names: list[str]) -> str | None:
    """Why a requested file name cannot be used against this catalog, or None when all can."""
    if catalog.status is p.ManusCalibrationCatalogStatus.DISABLED:
        return "calibration files can only be selected when the server has a --manus-calibration-dir"
    if catalog.status is p.ManusCalibrationCatalogStatus.UNAVAILABLE:
        return f"calibration files cannot be selected: {catalog.error_message}"
    available = set(catalog.files)
    for name in names:
        if name not in available:
            return f"calibration file {name!r} is not an available file in the calibration directory"
    return None


def resolve(directory: Path, name: str) -> Path:
    """The host path of one selected file; the server has already checked it against a scan."""
    # ``directory`` is the real path ServerConfig resolved at construction, so this does no filesystem access.
    return directory / name
