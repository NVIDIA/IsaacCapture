# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared envelope and payload checks, run once per hand."""

from __future__ import annotations

from cts_common.checks import (
    continuity,
    coverage,
    quaternion,
    rate,
    schema,
    timestamps,
    values,
)

from .base import per_hand

GROUP = "signal"


class _MinimumRate(rate.MinimumRate):
    # Lowest median frame rate accepted, hertz. Provisional.
    MIN_HZ = 30.0


JointsFieldPresent = per_hand(schema.JointsFieldPresent, GROUP)
Finite = per_hand(values.Finite, GROUP)
ZeroPoseOnValidJoint = per_hand(values.ZeroPoseOnValidJoint, GROUP)
UnitNormOnValidJoints = per_hand(quaternion.UnitNormOnValidJoints, GROUP)
Monotonic = per_hand(timestamps.Monotonic, GROUP)
AvailableNotBeforeSample = per_hand(timestamps.AvailableNotBeforeSample, GROUP)
PayloadPresenceRate = per_hand(coverage.PayloadPresenceRate, GROUP)
IntervalRegularity = per_hand(rate.IntervalRegularity, GROUP)
MinimumRate = per_hand(_MinimumRate, GROUP)
FrameGaps = per_hand(rate.FrameGaps, GROUP)
MaxJointVelocity = per_hand(continuity.MaxJointVelocity, GROUP)

CHECKS = (
    JointsFieldPresent,
    Finite,
    ZeroPoseOnValidJoint,
    UnitNormOnValidJoints,
    Monotonic,
    AvailableNotBeforeSample,
    PayloadPresenceRate,
    IntervalRegularity,
    MinimumRate,
    FrameGaps,
    MaxJointVelocity,
)
