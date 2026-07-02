# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sim-free unit tests for steering wheel vehicle-control retargeting."""

from dataclasses import dataclass
import math

import pytest

from isaacteleop.retargeters import (
    VehicleControlRetargeter,
    VehicleControlRetargeterConfig,
    axis_to_pedal,
)


@dataclass(frozen=True)
class SteeringSample:
    """Minimal SteeringWheelOutput-compatible sample for pure retargeter tests."""

    steering: float = 0.0
    throttle: float = 1.0
    brake: float = 1.0


class TestAxisToPedalMath:
    """The pure inverted-axis mapping used for throttle and brake pedals."""

    def test_released_axis_maps_to_zero_fraction(self):
        """A fully released inverted pedal axis maps to no pedal application."""
        assert axis_to_pedal(1.0) == pytest.approx(0.0)

    def test_center_axis_maps_to_half_fraction(self):
        """A centered full-range axis maps to half pedal application."""
        assert axis_to_pedal(0.0) == pytest.approx(0.5)

    def test_pressed_axis_maps_to_full_fraction(self):
        """A fully pressed inverted pedal axis maps to full pedal application."""
        assert axis_to_pedal(-1.0) == pytest.approx(1.0)

    def test_clamps_out_of_range_axes(self):
        """Hardware overshoot clamps to the normalized pedal fraction endpoints."""
        assert axis_to_pedal(1.5) == pytest.approx(0.0)
        assert axis_to_pedal(-1.5) == pytest.approx(1.0)


class TestVehicleControlRetargeter:
    """End-to-end math behavior of VehicleControlRetargeter.retarget."""

    def test_released_pedals_and_centered_steering_are_neutral(self):
        """Released pedals and centered wheel emit zero steer, accel, throttle, and brake."""
        command = VehicleControlRetargeter().retarget(
            SteeringSample(steering=0.0, throttle=1.0, brake=1.0),
            sequence=42,
        )

        assert command.sequence == 42
        assert command.steer == pytest.approx(0.0)
        assert command.accel == pytest.approx(0.0)
        assert command.throttle == pytest.approx(0.0)
        assert command.brake == pytest.approx(0.0)

    def test_deadzone_thresholds_are_inclusive(self):
        """Steering and pedal values exactly on their deadzone thresholds are suppressed."""
        retargeter = VehicleControlRetargeter(
            VehicleControlRetargeterConfig(
                steering_deadzone=0.05,
                pedal_deadzone=0.10,
            )
        )

        command = retargeter.retarget(
            SteeringSample(steering=0.05, throttle=0.80, brake=0.80),
            sequence=1,
        )

        assert command.steer == pytest.approx(0.0)
        assert command.accel == pytest.approx(0.0)
        assert command.throttle == pytest.approx(0.0)
        assert command.brake == pytest.approx(0.0)

    def test_values_outside_deadzone_pass_through(self):
        """Values just outside deadzone are not rescaled; they pass through directly."""
        retargeter = VehicleControlRetargeter(
            VehicleControlRetargeterConfig(
                steering_deadzone=0.05,
                pedal_deadzone=0.10,
            )
        )

        command = retargeter.retarget(
            SteeringSample(steering=-0.06, throttle=0.78, brake=1.0),
            sequence=1,
        )

        assert command.steer == pytest.approx(-0.06)
        assert command.accel == pytest.approx(0.11)
        assert command.throttle == pytest.approx(0.11)
        assert command.brake == pytest.approx(0.0)

    def test_calibrate_neutral_rebases_steering(self):
        """Neutral calibration subtracts the sampled steering offset before scaling/deadzone."""
        retargeter = VehicleControlRetargeter()
        retargeter.calibrate_neutral(SteeringSample(steering=0.25))

        centered = retargeter.retarget(
            SteeringSample(steering=0.25, throttle=1.0, brake=1.0),
            sequence=1,
        )
        turned = retargeter.retarget(
            SteeringSample(steering=-0.25, throttle=1.0, brake=1.0),
            sequence=2,
        )

        assert retargeter.steering_neutral == pytest.approx(0.25)
        assert centered.steer == pytest.approx(0.0)
        assert turned.steer == pytest.approx(-0.5)

    def test_positive_accel_splits_to_throttle_only(self):
        """When throttle fraction exceeds brake fraction, accel is positive throttle."""
        command = VehicleControlRetargeter().retarget(
            SteeringSample(steering=0.0, throttle=-0.40, brake=0.20),
            sequence=1,
        )

        assert command.accel == pytest.approx(0.30)
        assert command.throttle == pytest.approx(0.30)
        assert command.brake == pytest.approx(0.0)

    def test_negative_accel_splits_to_brake_only(self):
        """When brake fraction exceeds throttle fraction, accel is negative brake."""
        command = VehicleControlRetargeter().retarget(
            SteeringSample(steering=0.0, throttle=0.50, brake=-0.50),
            sequence=1,
        )

        assert command.accel == pytest.approx(-0.50)
        assert command.throttle == pytest.approx(0.0)
        assert command.brake == pytest.approx(0.50)

    def test_equal_pedal_fractions_cancel_to_neutral(self):
        """Equal throttle and brake fractions cancel before command splitting."""
        command = VehicleControlRetargeter().retarget(
            SteeringSample(steering=0.0, throttle=-0.20, brake=-0.20),
            sequence=1,
        )

        assert command.accel == pytest.approx(0.0)
        assert command.throttle == pytest.approx(0.0)
        assert command.brake == pytest.approx(0.0)

    def test_scaled_outputs_clamp_to_command_range(self):
        """Configured scale can amplify inputs, but steer and accel remain bounded."""
        retargeter = VehicleControlRetargeter(
            VehicleControlRetargeterConfig(
                steer_scale=3.0,
                throttle_scale=2.0,
                brake_scale=2.0,
            )
        )

        forward = retargeter.retarget(
            SteeringSample(steering=0.50, throttle=-1.0, brake=1.0),
            sequence=1,
        )
        reverse = retargeter.retarget(
            SteeringSample(steering=-0.50, throttle=1.0, brake=-1.0),
            sequence=2,
        )

        assert forward.steer == pytest.approx(1.0)
        assert forward.accel == pytest.approx(1.0)
        assert forward.throttle == pytest.approx(1.0)
        assert forward.brake == pytest.approx(0.0)
        assert reverse.steer == pytest.approx(-1.0)
        assert reverse.accel == pytest.approx(-1.0)
        assert reverse.throttle == pytest.approx(0.0)
        assert reverse.brake == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "sample",
        [
            SteeringSample(steering=math.nan),
            SteeringSample(throttle=math.inf),
            SteeringSample(brake=-math.inf),
        ],
    )
    def test_rejects_non_finite_input(self, sample):
        """Non-finite wheel samples fail fast instead of leaking NaN/Inf commands."""
        with pytest.raises(ValueError, match="non-finite"):
            VehicleControlRetargeter().retarget(sample, sequence=1)
