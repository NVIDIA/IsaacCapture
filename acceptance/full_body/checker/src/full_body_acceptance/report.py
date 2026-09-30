# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running the full-body checks over a frame source.

The report model and the verdict live in ``acceptance_common.report``; this adds what
only a full-body recording needs said about itself.
"""

from __future__ import annotations

from typing import Sequence

from acceptance_common.report import (
    CheckResult,
    Mark,
    Report,
    Verdict,
    aggregate,
    assemble,
    evaluate,
    json_safe,
    suppress_dependents,
)

from .checks import Check, build_all
from .frames import FrameSource
from .labels import StepTimeline

__all__ = [
    "CheckResult",
    "Mark",
    "Report",
    "Verdict",
    "aggregate",
    "json_safe",
    "run",
    "suppress_dependents",
]


def run(
    source: FrameSource,
    checks: Sequence[Check] | None = None,
    timeline: StepTimeline | None = None,
) -> Report:
    path = getattr(source, "path", None)
    if timeline is None and path is not None:
        timeline = StepTimeline.beside(path)
    active = list(checks) if checks is not None else build_all(timeline)
    frames, results = evaluate(source, active)

    metadata = source.metadata
    notes: list[str] = []
    if timeline is None:
        notes.append(
            "no motion-step labels beside the recording, so the G4 window measurements "
            "are reported as unanswered rather than guessed"
        )
    elif timeline.provisional:
        notes.append(f"motion labels read from {timeline.source}, marked provisional")
    if not metadata.channel_found:
        notes.append(
            "no channel declares schema core.FullBodyPoseRecord; nothing to check"
        )
    elif frames == 0:
        notes.append(
            "the full-body channel is registered but carries no messages, which is what "
            "a tracker in limp mode produces"
        )
    return assemble(source, frames, results, notes)
