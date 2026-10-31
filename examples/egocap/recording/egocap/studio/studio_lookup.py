# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared Studio / task-description lookup built from the UMI task_library CSVs.

The CSVs under the egocap package's ``task_library/`` directory follow this layout:

    row 0: title row (ignored)
    row 1: blank
    row 2: headers  -> Studio, Scenario, Task, Description
    row 3+: data rows

This module exposes a single ``build_description_to_studio()`` helper that
returns a ``{normalized_description: studio_name}`` dict, plus the
``_normalize_desc()`` helper used to compare descriptions robustly.

Consumed by ``egocap.recording.route`` for live routing, so the routing logic stays
identical across every caller.

Resolution order for the CSV directory path:

    1. Explicit ``task_library_dir`` argument (highest priority)
    2. ``UMI_TASK_LIBRARY_DIR`` environment variable
    3. ``<egocap_pkg>/task_library`` (default — sibling of this file)
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

from egocap.studio.studio_path_names import sanitize_studio_dir_name

_SUMMARY_CSV = "Summary.csv"


def default_task_library_dir() -> Path:
    """Return the resolved task library dir (see module docstring)."""
    env = os.environ.get("UMI_TASK_LIBRARY_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "task_library"


def _normalize_desc(text: str) -> str:
    """Normalize a task description for robust matching across files.

    Collapses runs of whitespace and strips trailing periods / whitespace,
    then lowercases so minor text differences (extra spaces, missing
    period, case) don't break the CSV lookup.
    """
    if not text:
        return ""
    collapsed = re.sub(r"\s+", " ", text.strip())
    return collapsed.rstrip(" .").lower()


def build_description_to_studio(
    task_library_dir: Path | str | None = None,
) -> dict[str, str]:
    """Return ``{normalized_description: studio}`` from every studio CSV.

    The ``Summary.csv`` file is skipped since it doesn't describe tasks.
    A CSV missing the expected header row (``Studio`` + ``Description``)
    is skipped with no error.

    Args:
        task_library_dir: Directory holding the per-studio CSVs. If
            ``None``, resolves via :func:`default_task_library_dir`.

    Returns:
        Mapping from ``_normalize_desc(description)`` to ``studio``.
    """
    if task_library_dir is None:
        root = default_task_library_dir()
    else:
        root = Path(task_library_dir).expanduser()

    lookup: dict[str, str] = {}
    if not root.is_dir():
        return lookup

    for csv_path in sorted(root.glob("*.csv")):
        if csv_path.name == _SUMMARY_CSV:
            continue
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
        except OSError:
            continue
        if len(rows) < 4:
            continue

        headers = [h.strip().lower() for h in rows[2]]
        try:
            studio_idx = headers.index("studio")
            desc_idx = headers.index("description")
        except ValueError:
            continue

        for row in rows[3:]:
            if len(row) <= max(studio_idx, desc_idx):
                continue
            studio = sanitize_studio_dir_name(row[studio_idx].strip())
            desc = row[desc_idx].strip()
            if not studio or not desc:
                continue
            lookup[_normalize_desc(desc)] = studio

    return lookup


def known_studios_from_task_library_dir(
    task_library_dir: Path | str | None = None,
) -> frozenset[str]:
    """Return unique canonical studio names from task CSVs (same rules as :func:`build_description_to_studio`)."""
    if task_library_dir is None:
        root = default_task_library_dir()
    else:
        root = Path(task_library_dir).expanduser()

    studios: set[str] = set()
    if not root.is_dir():
        return frozenset()

    for csv_path in sorted(root.glob("*.csv")):
        if csv_path.name == _SUMMARY_CSV:
            continue
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
        except OSError:
            continue
        if len(rows) < 4:
            continue

        headers = [h.strip().lower() for h in rows[2]]
        try:
            studio_idx = headers.index("studio")
        except ValueError:
            continue

        for row in rows[3:]:
            if len(row) <= studio_idx:
                continue
            studio = sanitize_studio_dir_name(row[studio_idx].strip())
            if studio:
                studios.add(studio)

    return frozenset(studios)


def lookup_studio(description: str, mapping: dict[str, str]) -> str | None:
    """Convenience: normalize *description* and look it up in *mapping*.

    Returns ``None`` if the description is empty or has no match.
    """
    if not description:
        return None
    return mapping.get(_normalize_desc(description))
