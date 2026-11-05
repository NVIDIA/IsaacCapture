# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The submission archive for a hand take.

``cts_common.panel.bundle`` builds it from the hand groups and each hand's frame series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from cts_common.panel import bundle as _bundle
from cts_common.panel.track import Track

from ..labels import StepTimeline
from ..report import Report
from . import status
from .track import SIDES

# The package whose commit and working-tree state the archive records.
PACKAGE = Path(__file__).resolve().parent.parent


def build(
    report: Report,
    tracks: dict[str, Track],
    recording: Path | str,
    timeline: StepTimeline | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[str, bytes]:
    """The archive's filename and its bytes. ``on_progress`` is called with 0.0–1.0."""
    return _bundle.build(
        report,
        recording,
        timeline,
        groups=status.groups(report),
        series=series(tracks),
        package=PACKAGE,
        on_progress=on_progress,
    )


def series(tracks: dict[str, Track]) -> dict[str, dict[str, list[Any]]]:
    """Each hand's series under its side; a hand with no channel has empty columns."""
    return {
        side: _bundle.series(tracks[side].samples if side in tracks else ())
        for side in SIDES
    }
