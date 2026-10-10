# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check registry.

Check names must match the ``expected_failing_check`` strings in the fixture index.
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
from .consistency import AllJointPosesTracked
from .envelope import (
    AvailableNotBeforeSample,
    DeviceClockDistinct,
    Finite,
    FrameGaps,
    IntervalRegularity,
    JointsFieldPresent,
    MaxJointVelocity,
    Monotonic,
    PayloadPresenceRate,
    UnitNormOnValidJoints,
    ValidityTrend,
    ZeroPoseOnValidJoint,
)
from .geometry import (
    AnthropometricPlausibility,
    BoneLengthConstancy,
    Handedness,
    JointIndexAssignment,
    LeftRightLabelling,
    PositionScaleMetres,
    UpAxis,
)
from .orientation import ComponentOrder, PositionOrientationSameFrame
from .posture import (
    ArmRaiseRangeOfMotion,
    ArmRaiseTorsoStability,
    ContralateralCrosstalkSingleLegRaise,
    CumulativeDriftBetweenTposeWindows,
    MarchAnkleAntiphase,
    MarchCadenceSteady,
    SquatDepthSufficient,
    SquatKneeSymmetry,
    SquatRepRepeatability,
    TposeArmDroop,
    TposeLeftRightAsymmetry,
)
from .segmentation import (
    FallbackWithoutLabels,
    LabelAlignment,
    LabelledStepActuallyPerformed,
    LabelWindowsWellformed,
    StepOrderMatchesLabels,
)

CHECKS: tuple[type[Check], ...] = (
    JointsFieldPresent,
    Finite,
    ZeroPoseOnValidJoint,
    UnitNormOnValidJoints,
    Monotonic,
    AvailableNotBeforeSample,
    DeviceClockDistinct,
    PayloadPresenceRate,
    ValidityTrend,
    AllJointPosesTracked,
    IntervalRegularity,
    FrameGaps,
    MaxJointVelocity,
    UpAxis,
    PositionScaleMetres,
    BoneLengthConstancy,
    AnthropometricPlausibility,
    PositionOrientationSameFrame,
    ComponentOrder,
    Handedness,
    LeftRightLabelling,
    JointIndexAssignment,
    # Posture checks
    LabelWindowsWellformed,
    LabelAlignment,
    FallbackWithoutLabels,
    LabelledStepActuallyPerformed,
    StepOrderMatchesLabels,
    TposeArmDroop,
    TposeLeftRightAsymmetry,
    CumulativeDriftBetweenTposeWindows,
    ContralateralCrosstalkSingleLegRaise,
    ArmRaiseRangeOfMotion,
    MarchAnkleAntiphase,
    SquatKneeSymmetry,
    SquatRepRepeatability,
    SquatDepthSufficient,
    MarchCadenceSteady,
    ArmRaiseTorsoStability,
)


def _construct(cls: type[Check], timeline: StepTimeline | None) -> Check:
    # Only checks with needs_timeline take the timeline.
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
