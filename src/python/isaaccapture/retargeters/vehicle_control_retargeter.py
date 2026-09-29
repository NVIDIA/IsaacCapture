# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Retarget steering wheel input into normalized vehicle control commands."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


@dataclass(frozen=True)
class VehicleControlCommand:
    """Retargeting result; the application owns wire and MCAP encoding."""

    sequence: int = 0
    steer: float = 0.0
    accel: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0


@dataclass(frozen=True)
class VehicleControlRetargeterConfig:
    """Configuration for steering wheel to vehicle command retargeting."""

    steering_deadzone: float = 0.01
    pedal_deadzone: float = 0.01
    steer_scale: float = 1.0
    throttle_scale: float = 1.0
    brake_scale: float = 1.0


class SteeringWheelLike(Protocol):
    """Structural type for objects with SteeringWheelOutput-compatible fields."""

    steering: float
    throttle: float
    brake: float


class VehicleControlRetargeter:
    """Maps normalized steering wheel state into ``VehicleControlCommand``."""

    def __init__(
        self,
        config: VehicleControlRetargeterConfig | None = None,
        *,
        steering_neutral: float = 0.0,
    ) -> None:
        self._config = config or VehicleControlRetargeterConfig()
        self._steering_neutral = steering_neutral

    @property
    def steering_neutral(self) -> float:
        return self._steering_neutral

    def calibrate_neutral(self, sample: SteeringWheelLike) -> None:
        self._steering_neutral = sample.steering

    def retarget(
        self, sample: SteeringWheelLike, *, sequence: int
    ) -> VehicleControlCommand:
        if (
            not math.isfinite(sample.steering)
            or not math.isfinite(sample.throttle)
            or not math.isfinite(sample.brake)
        ):
            raise ValueError("Steering wheel sample data contains non-finite values.")

        steer = self._apply_deadzone(
            (sample.steering - self._steering_neutral) * self._config.steer_scale,
            self._config.steering_deadzone,
        )
        throttle = self._apply_deadzone(
            axis_to_pedal(sample.throttle) * self._config.throttle_scale,
            self._config.pedal_deadzone,
        )
        brake = self._apply_deadzone(
            axis_to_pedal(sample.brake) * self._config.brake_scale,
            self._config.pedal_deadzone,
        )
        accel = _clamp(throttle - brake, -1.0, 1.0)

        return VehicleControlCommand(
            sequence=sequence,
            steer=_clamp(steer, -1.0, 1.0),
            accel=accel,
            throttle=accel if accel > 0.0 else 0.0,
            brake=-accel if accel < 0.0 else 0.0,
        )

    @staticmethod
    def _apply_deadzone(value: float, threshold: float) -> float:
        return 0.0 if abs(value) <= threshold else value


def axis_to_pedal(axis_value: float) -> float:
    """Map inverted full-range pedal axes (1 released, -1 pressed) into [0, 1]."""

    return _clamp((-float(axis_value) + 1.0) / 2.0, 0.0, 1.0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))
