# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sim-free unit tests for steering wheel vehicle-control retargeting."""

from dataclasses import astuple, dataclass, FrozenInstanceError
import math

import pytest

from isaaccapture.retargeters import (
    VehicleControlCommand,
    VehicleControlRetargeter,
    VehicleControlRetargeterConfig,
    axis_to_pedal,
)


@dataclass(frozen=True)
class SteeringSample:
    steering: float = 0.0
    throttle: float = 1.0
    brake: float = 1.0


@pytest.mark.parametrize(
    "axis, expected", [(1.0, 0.0), (0.0, 0.5), (-1.0, 1.0), (1.5, 0.0), (-1.5, 1.0)]
)
def test_axis_to_pedal(axis, expected):
    assert axis_to_pedal(axis) == pytest.approx(expected)


@pytest.mark.parametrize(
    "sample, expected",
    [
        pytest.param((0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0), id="released"),
        pytest.param((0.0, -0.40, 0.20), (0.0, 0.30, 0.30, 0.0), id="throttle"),
        pytest.param((0.0, 0.50, -0.50), (0.0, -0.50, 0.0, 0.50), id="brake"),
        pytest.param((0.0, -0.20, -0.20), (0.0, 0.0, 0.0, 0.0), id="cancel"),
    ],
)
def test_retarget_pedals(sample, expected):
    command = VehicleControlRetargeter().retarget(SteeringSample(*sample), sequence=42)
    assert isinstance(command, VehicleControlCommand)
    assert command.sequence == 42
    assert astuple(command)[1:] == pytest.approx(expected)


@pytest.mark.parametrize(
    "sample, expected",
    [
        pytest.param((0.05, 0.80, 0.80), (0.0, 0.0, 0.0, 0.0), id="at-deadzone"),
        pytest.param(
            (-0.06, 0.78, 1.0), (-0.06, 0.11, 0.11, 0.0), id="outside-deadzone"
        ),
    ],
)
def test_deadzone_boundaries(sample, expected):
    retargeter = VehicleControlRetargeter(
        VehicleControlRetargeterConfig(steering_deadzone=0.05, pedal_deadzone=0.10)
    )
    command = retargeter.retarget(SteeringSample(*sample), sequence=1)
    assert astuple(command)[1:] == pytest.approx(expected)


def test_calibrate_neutral_rebases_steering():
    retargeter = VehicleControlRetargeter()
    retargeter.calibrate_neutral(SteeringSample(steering=0.25))
    centered = retargeter.retarget(SteeringSample(steering=0.25), sequence=1)
    turned = retargeter.retarget(SteeringSample(steering=-0.25), sequence=2)
    assert retargeter.steering_neutral == pytest.approx(0.25)
    assert centered.steer == pytest.approx(0.0)
    assert turned.steer == pytest.approx(-0.5)


@pytest.mark.parametrize(
    "sample, expected",
    [
        ((0.50, -1.0, 1.0), (1.0, 1.0, 1.0, 0.0)),
        ((-0.50, 1.0, -1.0), (-1.0, -1.0, 0.0, 1.0)),
    ],
)
def test_scaled_outputs_clamp_to_command_range(sample, expected):
    retargeter = VehicleControlRetargeter(
        VehicleControlRetargeterConfig(
            steer_scale=3.0, throttle_scale=2.0, brake_scale=2.0
        )
    )
    command = retargeter.retarget(SteeringSample(*sample), sequence=1)
    assert astuple(command)[1:] == pytest.approx(expected)


@pytest.mark.parametrize(
    "sample",
    [
        SteeringSample(steering=math.nan),
        SteeringSample(throttle=math.inf),
        SteeringSample(brake=-math.inf),
    ],
)
def test_rejects_non_finite_input(sample):
    with pytest.raises(ValueError, match="non-finite"):
        VehicleControlRetargeter().retarget(sample, sequence=1)


def test_vehicle_command_fields_are_readonly():
    assert astuple(VehicleControlCommand()) == (0, 0.0, 0.0, 0.0, 0.0)
    command = VehicleControlCommand(42, -0.5, 0.75, 0.75, 0.0)
    assert astuple(command) == (42, -0.5, 0.75, 0.75, 0.0)
    with pytest.raises(FrozenInstanceError):
        command.steer = 1.0
