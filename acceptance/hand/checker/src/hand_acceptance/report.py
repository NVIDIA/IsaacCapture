# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running the hand checks over a frame source."""

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
)

from .checks import Check, build_all
from .frames import SCHEMA_NAME, FrameSource
from .labels import StepTimeline
from .mcap_source import side_of

__all__ = ["CheckResult", "Mark", "Report", "Verdict", "aggregate", "run"]


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

    notes: list[str] = []
    if timeline is None:
        notes.append(
            "no motion-step labels beside the recording, so the window measurements "
            "are reported as unanswered rather than guessed"
        )
    elif timeline.provisional:
        notes.append(f"motion labels read from {timeline.source}, marked provisional")
    topics = getattr(source, "topics", ())
    unsided = [t for t in topics if side_of(t) is None]
    if unsided:
        notes.append(f"ignored channels with no left/right suffix: {unsided}")
    if not source.metadata.channel_found:
        notes.append(f"no channel declares schema {SCHEMA_NAME}; nothing to check")
    elif frames == 0:
        notes.append("the hand channels are registered but carry no messages")
    return assemble(source, frames, results, notes)
