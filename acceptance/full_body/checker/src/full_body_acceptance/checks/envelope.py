# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared envelope and payload checks, reported under G1.

The measurements live in ``acceptance_common.checks``; only the group is decided here.
"""

from __future__ import annotations

from acceptance_common.checks import (
    continuity,
    coverage,
    quaternion,
    rate,
    schema,
    timestamps,
    values,
)


class JointsFieldPresent(schema.JointsFieldPresent):
    gate = "G1"


class Finite(values.Finite):
    gate = "G1"


class ZeroPoseOnValidJoint(values.ZeroPoseOnValidJoint):
    gate = "G1"


class UnitNormOnValidJoints(quaternion.UnitNormOnValidJoints):
    gate = "G1"


class Monotonic(timestamps.Monotonic):
    gate = "G1"


class AvailableNotBeforeSample(timestamps.AvailableNotBeforeSample):
    gate = "G1"


class DeviceClockDistinct(timestamps.DeviceClockDistinct):
    gate = "G1"


class PayloadPresenceRate(coverage.PayloadPresenceRate):
    gate = "G1"


class ValidityTrend(coverage.ValidityTrend):
    gate = "G1"


class IntervalRegularity(rate.IntervalRegularity):
    gate = "G1"


class FrameGaps(rate.FrameGaps):
    gate = "G1"


class MaxJointVelocity(continuity.MaxJointVelocity):
    gate = "G1"
