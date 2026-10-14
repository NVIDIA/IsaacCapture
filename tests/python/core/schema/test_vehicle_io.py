# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for steering wheel schema bindings."""

import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    SteeringWheelOutput,
    SteeringWheelOutputRecord,
)


def test_steering_wheel_output_defaults():
    output = SteeringWheelOutput()

    assert output.steering == 0.0
    assert output.throttle == 0.0
    assert output.brake == 0.0
    assert output.clutch == 0.0
    assert output.buttons == []
    assert output.hat_x == 0
    assert output.hat_y == 0
    assert output.connected is False
    assert output.sample_time_monotonic_ns == 0


def test_steering_wheel_output_constructs_with_values():
    output = SteeringWheelOutput(-0.25, 0.8, 0.1, 0.0, [1, 0, 1], -1, 1)

    assert output.steering == pytest.approx(-0.25)
    assert output.throttle == pytest.approx(0.8)
    assert output.brake == pytest.approx(0.1)
    assert output.buttons == [1, 0, 1]
    assert output.hat_x == -1
    assert output.hat_y == 1


def test_steering_wheel_wrappers_hold_data():
    output = SteeringWheelOutput(
        0.0, 1.0, 0.0, 0.0, connected=True, sample_time_monotonic_ns=123
    )
    timestamp = DeviceDataTimestamp(10, 20, 30)

    record = SteeringWheelOutputRecord(output, timestamp)

    assert record.data.throttle == pytest.approx(1.0)
    assert record.data.connected is True
    assert record.data.sample_time_monotonic_ns == 123
    assert record.timestamp.sample_time_local_common_clock == 20
