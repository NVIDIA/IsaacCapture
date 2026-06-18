# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VehicleControlCommand:
    sequence: int
    timestamp_ns: int
    steer: float
    accel: float
    throttle: float
    brake: float

    @classmethod
    def neutral(
        cls, sequence: int = 0, timestamp_ns: int | None = None
    ) -> "VehicleControlCommand":
        return cls(
            sequence=sequence,
            timestamp_ns=time.time_ns() if timestamp_ns is None else timestamp_ns,
            steer=0.0,
            accel=0.0,
            throttle=0.0,
            brake=0.0,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "VehicleControlCommand":
        return cls(
            sequence=int(value["sequence"]),
            timestamp_ns=int(value["timestamp_ns"]),
            steer=float(value["steer"]),
            accel=float(value["accel"]),
            throttle=float(value.get("throttle", 0.0)),
            brake=float(value.get("brake", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))
