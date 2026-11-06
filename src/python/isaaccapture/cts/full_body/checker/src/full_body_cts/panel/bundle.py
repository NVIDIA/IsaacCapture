# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The submission archive for a full-body take.

``cts_common.panel.bundle`` builds it from the full-body groups and frame series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from cts_common.panel import bundle as _bundle

from ..labels import StepTimeline
from ..report import Report
from . import status
from .track import Track

# The package whose commit and working-tree state the archive records.
PACKAGE = Path(__file__).resolve().parent.parent


def build(
    report: Report,
    track: Track,
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
        series=_bundle.series(track.samples),
        package=PACKAGE,
        on_progress=on_progress,
    )
