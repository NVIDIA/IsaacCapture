# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared envelope and payload checks, reported under the ``signal`` group."""

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


class JointsFieldPresent(schema.JointsFieldPresent):
    group = "signal"


class Finite(values.Finite):
    group = "signal"


class ZeroPoseOnValidJoint(values.ZeroPoseOnValidJoint):
    group = "signal"


class UnitNormOnValidJoints(quaternion.UnitNormOnValidJoints):
    group = "signal"


class Monotonic(timestamps.Monotonic):
    group = "signal"


class AvailableNotBeforeSample(timestamps.AvailableNotBeforeSample):
    group = "signal"


class DeviceClockDistinct(timestamps.DeviceClockDistinct):
    group = "signal"


class PayloadPresenceRate(coverage.PayloadPresenceRate):
    group = "signal"


class ValidityTrend(coverage.ValidityTrend):
    group = "signal"


class IntervalRegularity(rate.IntervalRegularity):
    group = "signal"


class FrameGaps(rate.FrameGaps):
    group = "signal"


class MaxJointVelocity(continuity.MaxJointVelocity):
    group = "signal"
