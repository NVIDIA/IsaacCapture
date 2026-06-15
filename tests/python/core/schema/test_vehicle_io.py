# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for steering wheel and vehicle control schema bindings."""

import pytest

from isaaccapture.schema import (
    DeviceDataTimestamp,
    SteeringWheelOutput,
    SteeringWheelOutputRecord,
    VehicleControlCommand,
    VehicleControlCommandRecord,
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


def test_steering_wheel_output_constructs_with_values():
    output = SteeringWheelOutput(-0.25, 0.8, 0.1, 0.0, [1, 0, 1], -1, 1)

    assert output.steering == pytest.approx(-0.25)
    assert output.throttle == pytest.approx(0.8)
    assert output.brake == pytest.approx(0.1)
    assert output.buttons == [1, 0, 1]
    assert output.hat_x == -1
    assert output.hat_y == 1


def test_steering_wheel_wrappers_hold_data():
    output = SteeringWheelOutput(0.0, 1.0, 0.0, 0.0)
    timestamp = DeviceDataTimestamp(10, 20, 30)

    record = SteeringWheelOutputRecord(output, timestamp)

    assert record.data.throttle == pytest.approx(1.0)
    assert record.timestamp.sample_time_local_common_clock == 20


def test_vehicle_control_command_constructs_with_values():
    command = VehicleControlCommand(42, -0.5, 0.75, 0.75, 0.0)

    assert command.sequence == 42
    assert command.steer == pytest.approx(-0.5)
    assert command.accel == pytest.approx(0.75)
    assert command.throttle == pytest.approx(0.75)
    assert command.brake == pytest.approx(0.0)


def test_vehicle_control_record_holds_data():
    command = VehicleControlCommand(7, 0.25, -0.5, 0.0, 0.5)
    timestamp = DeviceDataTimestamp(10, 20, 30)

    record = VehicleControlCommandRecord(command, timestamp)

    assert record.data.sequence == 7
    assert record.data.brake == pytest.approx(0.5)
    assert record.timestamp.sample_time_raw_device_clock == 30
