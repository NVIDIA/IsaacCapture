# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check registry, in report order: signal, hand shape, placement, label windows.

Check names must match ``expected_failing_check`` in the oracle fixtures index.
"""

from __future__ import annotations

from cts_common.checks.base import (
    Attribution,
    Check,
    Outcome,
    Severity,
    Status,
)

from ..labels import StepTimeline
from . import envelope, placement, shape, windows

CHECKS: tuple[type[Check], ...] = (
    *envelope.CHECKS,
    *shape.CHECKS,
    *placement.CHECKS,
    *windows.CHECKS,
)


def _construct(cls: type[Check], timeline: StepTimeline | None) -> Check:
    return cls(timeline) if getattr(cls, "needs_timeline", False) else cls()


def build_all(timeline: StepTimeline | None = None) -> list[Check]:
    return [_construct(cls, timeline) for cls in CHECKS]


def build(names: list[str], timeline: StepTimeline | None = None) -> list[Check]:
    by_name = {cls.name: cls for cls in CHECKS}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise KeyError(f"unknown checks: {unknown}")
    return [_construct(by_name[name], timeline) for name in names]


__all__ = [
    "CHECKS",
    "Attribution",
    "Check",
    "Outcome",
    "Severity",
    "Status",
    "build",
    "build_all",
]
