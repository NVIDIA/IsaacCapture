# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Natural RealHand FFG Glove sensor-angle retargeting for L6, O6, and L20.

USB transport and hand-side discovery live in the RealHand FFG Glove C++ device plugin. This
module consumes one standard ``JointStateSource`` and maps only that glove's 21 sensors to one
mechanical hand. Thumb outputs depend only on thumb sensors, and each non-thumb digit depends only
on that digit's sensors; there is no contact snapping or cross-finger pose activation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from isaaccapture.retargeting_engine.interface import BaseRetargeter
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    RetargeterIO,
    RetargeterIOType,
)
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    OptionalType,
    TensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import FloatType

from .profiles import RealHandJointSpec, get_realhand_profile

REALHAND_FFG_GLOVE_SENSOR_NAMES = tuple(f"sensor_{index}" for index in range(21))

_SENSOR_CHANNELS = {
    "thumb_rotation": (1,),
    "thumb_opposition": (1, 2),
    "thumb_flex": (2,),
    "thumb_base_flex": (2,),
    "thumb_distal_flex": (2,),
    "index_curl": (6, 8),
    "index_spread": (5,),
    "index_base_flex": (6, 8),
    "index_distal_flex": (6, 8),
    "middle_curl": (10, 12),
    "middle_spread": (9,),
    "middle_base_flex": (10, 12),
    "middle_distal_flex": (10, 12),
    "ring_curl": (14, 16),
    "ring_spread": (13,),
    "ring_base_flex": (14, 16),
    "ring_distal_flex": (14, 16),
    "pinky_curl": (18, 20),
    "pinky_spread": (17,),
    "pinky_base_flex": (18, 20),
    "pinky_distal_flex": (18, 20),
}

_L20_TOUCH_TARGETS = {
    ("left", "index"): (0.483, 1.534, 0.407, 0.706),
    ("left", "middle"): (0.777, 1.116, 0.413, 0.684),
    ("left", "ring"): (0.943, 1.175, 0.415, 0.674),
    ("left", "pinky"): (1.191, 1.135, 0.495, 0.611),
    ("right", "index"): (0.615, 1.015, 0.453, 0.655),
    ("right", "middle"): (0.740, 1.232, 0.533, 1.222),
    ("right", "ring"): (1.003, 1.084, 0.379, 0.738),
    ("right", "pinky"): (1.116, 1.245, 0.459, 0.657),
}
_L20_THUMB_SEMANTICS = (
    "thumb_rotation",
    "thumb_opposition",
    "thumb_base_flex",
    "thumb_distal_flex",
)


def load_realhand_ffg_glove_calibration(
    path: str | Path, side: str
) -> dict[str, object]:
    """Load one side from a RealHand FFG Glove raw-calibration YAML file."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be left or right, got {side!r}")
    calibration_path = Path(path).expanduser()
    with calibration_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(
            f"RealHand FFG Glove calibration must be a YAML mapping: {calibration_path}"
        )

    suffix = "l" if side == "left" else "r"
    other_suffix = "r" if suffix == "l" else "l"

    def extract(current_suffix: str) -> dict[str, object]:
        oposes: dict[str, Sequence[float]] = {}
        legacy = data.get(f"jointangleopose_{current_suffix}")
        for finger in ("index", "middle", "ring", "pinky"):
            value = data.get(f"jointangleopose_{finger}_{current_suffix}")
            if value is None and finger == "index":
                value = legacy
            if value is not None:
                oposes[finger] = value
        return {
            "open": data.get(f"jointangleoriginal_{current_suffix}"),
            "fist": data.get(f"jointanglefist_{current_suffix}"),
            "thumb_curl": data.get(f"jointanglethumb_curl_{current_suffix}"),
            "oposes": oposes,
        }

    calibration = extract(suffix)
    if calibration["open"] is None or calibration["fist"] is None:
        calibration = extract(other_suffix)
    if calibration["open"] is None or calibration["fist"] is None:
        raise ValueError(
            "RealHand FFG Glove calibration has no complete open/fist pair "
            f"for {side}: {calibration_path}"
        )
    return calibration


@dataclass
class RealHandFFGGloveRetargeterConfig:
    """Configuration for one RealHand FFG Glove and one RealHand mechanical hand."""

    input_device: str
    joint_names: list[str]
    side: str
    hand_model: str = "l6"
    calibration_path: str | None = None
    smoothing_alpha: float = 0.65
    joint_deadband_rad: float = 0.01
    max_close_step_rad: float = 0.05
    max_open_step_rad: float = 0.20
    finger_close_gain: float = 1.0
    thumb_flex_gain: float = 1.0
    thumb_roll_gain: float = 1.0
    spread_gain: float = 1.0
    hold_last_on_missing: bool = True


class _NaturalRealHandFFGGloveMapper:
    def __init__(self, config: RealHandFFGGloveRetargeterConfig) -> None:
        self.profile = get_realhand_profile(config.hand_model)
        self.side = config.side
        self.config = config
        self.open: np.ndarray | None = None
        self.fist: np.ndarray | None = None
        self.thumb_curl: np.ndarray | None = None
        self.oposes: dict[str, np.ndarray] = {}
        self.baseline: np.ndarray | None = None
        if config.calibration_path:
            calibration = load_realhand_ffg_glove_calibration(
                config.calibration_path, config.side
            )
            self.open = self._coerce(calibration["open"])
            self.fist = self._coerce(calibration["fist"])
            if calibration["thumb_curl"] is not None:
                self.thumb_curl = self._coerce(calibration["thumb_curl"])
            oposes = calibration["oposes"]
            if isinstance(oposes, Mapping):
                self.oposes = {
                    str(name): self._coerce(value)
                    for name, value in oposes.items()
                    if value is not None
                }

    @staticmethod
    def _coerce(values: object) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size != len(REALHAND_FFG_GLOVE_SENSOR_NAMES) or not np.all(
            np.isfinite(array)
        ):
            raise ValueError(
                "RealHand FFG Glove samples must contain 21 finite sensor angles"
            )
        return array

    def reset(self) -> None:
        self.baseline = None

    def map(self, sensors: Sequence[float], joint_names: Sequence[str]) -> np.ndarray:
        values = self._coerce(sensors)
        if self.open is None or self.fist is None:
            if self.baseline is None:
                self.baseline = values.copy()
                return np.zeros(len(joint_names), dtype=np.float64)
            return self._uncalibrated(values - self.baseline, joint_names)
        return np.asarray(
            [
                self._joint_target(values, self.profile.joint_spec(self.side, name))
                for name in joint_names
            ],
            dtype=np.float64,
        )

    def _gain(self, spec: RealHandJointSpec) -> float:
        if not spec.semantic.startswith("thumb"):
            return self.config.finger_close_gain
        if spec.semantic in ("thumb_rotation", "thumb_opposition"):
            return self.config.thumb_roll_gain
        return self.config.thumb_flex_gain

    def _closed_target(self, spec: RealHandJointSpec) -> float:
        return float(
            np.clip(spec.trigger_closed * self._gain(spec), spec.lower, spec.upper)
        )

    def _uncalibrated(
        self, delta: np.ndarray, joint_names: Sequence[str]
    ) -> np.ndarray:
        targets = []
        for name in joint_names:
            spec = self.profile.joint_spec(self.side, name)
            channels = _SENSOR_CHANNELS.get(spec.semantic, ())
            response = max(
                (abs(float(delta[index])) for index in channels), default=0.0
            )
            targets.append(
                np.clip(
                    response * 2.0 * self._closed_target(spec), spec.lower, spec.upper
                )
            )
        return np.asarray(targets, dtype=np.float64)

    def _joint_target(self, values: np.ndarray, spec: RealHandJointSpec) -> float:
        assert self.open is not None and self.fist is not None
        if self.profile.model == "l20" and spec.semantic.startswith("thumb"):
            return self._l20_thumb_target(values, spec)

        channels = _SENSOR_CHANNELS.get(spec.semantic, ())
        if spec.semantic.endswith("_spread"):
            response = self._signed_response(values, self.fist, channels)
            magnitude = spec.upper if response >= 0.0 else abs(spec.lower)
            return float(
                np.clip(
                    response * magnitude * self.config.spread_gain,
                    spec.lower,
                    spec.upper,
                )
            )
        response = np.clip(self._response(values, self.fist, channels), 0.0, 1.0)
        return float(
            np.clip(response * self._closed_target(spec), spec.lower, spec.upper)
        )

    def _response(
        self, values: np.ndarray, target: np.ndarray, channels: Sequence[int]
    ) -> float:
        assert self.open is not None
        ratios = []
        weights = []
        for channel in channels:
            span = float(target[channel] - self.open[channel])
            if abs(span) < 0.05:
                continue
            ratios.append(float((values[channel] - self.open[channel]) / span))
            weights.append(abs(span))
        if not ratios:
            return 0.0
        positive = [ratio for ratio in ratios if ratio > 0.0]
        if positive:
            return max(positive)
        return float(np.average(ratios, weights=weights))

    def _signed_response(
        self, values: np.ndarray, target: np.ndarray, channels: Sequence[int]
    ) -> float:
        return float(np.clip(self._response(values, target, channels), -1.0, 1.0))

    def _l20_thumb_target(self, values: np.ndarray, spec: RealHandJointSpec) -> float:
        assert self.open is not None and self.fist is not None
        raw_anchors = [self.open, self.fist]
        target_anchors = [0.0, self._closed_target(spec)]
        if self.thumb_curl is not None:
            raw_anchors.append(self.thumb_curl)
            target_anchors.append(self._closed_target(spec))

        semantic_index = _L20_THUMB_SEMANTICS.index(spec.semantic)
        for finger in ("index", "middle", "ring", "pinky"):
            pose = self.oposes.get(finger)
            targets = _L20_TOUCH_TARGETS.get((self.side, finger))
            if pose is not None and targets is not None:
                raw_anchors.append(pose)
                target_anchors.append(
                    float(np.clip(targets[semantic_index], spec.lower, spec.upper))
                )

        # Only thumb sensors participate. Moving any other finger cannot alter a thumb target.
        channels = np.asarray((0, 1, 2, 3, 4), dtype=np.int64)
        points = np.asarray(raw_anchors, dtype=np.float64)[:, channels]
        query = values[channels]
        scale = np.ptp(points, axis=0)
        valid = scale >= 0.05
        if not np.any(valid):
            return 0.0
        endpoint = self._outside_calibrated_endpoint(
            points[:, valid], query[valid], scale[valid]
        )
        if endpoint is not None:
            return target_anchors[endpoint]
        distances = np.sqrt(
            np.mean(((points[:, valid] - query[valid]) / scale[valid]) ** 2, axis=1)
        )
        nearest = int(np.argmin(distances))
        if distances[nearest] < 1.0e-5:
            return target_anchors[nearest]
        weights = 1.0 / np.maximum(distances, 1.0e-4) ** 2
        target = float(np.average(np.asarray(target_anchors), weights=weights))
        return float(np.clip(target, spec.lower, spec.upper))

    @staticmethod
    def _outside_calibrated_endpoint(
        points: np.ndarray, query: np.ndarray, scale: np.ndarray
    ) -> int | None:
        """Return an endpoint when the sample continues beyond its calibrated ray."""
        normalized = (points - points[0]) / scale
        normalized_query = (query - points[0]) / scale
        candidates: list[tuple[float, int]] = []
        for index in range(1, len(normalized)):
            direction = normalized[index]
            denominator = float(np.dot(direction, direction))
            if denominator < 1.0e-8:
                continue
            progress = float(np.dot(normalized_query, direction) / denominator)
            if progress < 1.0:
                continue
            perpendicular = normalized_query - progress * direction
            distance = float(np.sqrt(np.mean(perpendicular**2)))
            if distance <= 0.20:
                candidates.append((distance, index))
        return min(candidates)[1] if candidates else None


class RealHandFFGGloveRetargeter(BaseRetargeter):
    """Map one RealHand FFG Glove ``JointStateSource`` to an L6, O6, or L20."""

    JOINTS = "joints"
    OUTPUT = "hand_joints"

    def __init__(self, config: RealHandFFGGloveRetargeterConfig, name: str) -> None:
        if config.side not in ("left", "right"):
            raise ValueError(f"side must be left or right, got {config.side!r}")
        self._config = config
        self._profile = get_realhand_profile(config.hand_model)
        expected = set(self._profile.joint_names(config.side))
        if set(config.joint_names) != expected:
            raise ValueError(
                f"{self._profile.display_name} {config.side} joint names do not match its profile"
            )
        self._specs = [
            self._profile.joint_spec(config.side, name) for name in config.joint_names
        ]
        self._mapper = _NaturalRealHandFFGGloveMapper(config)
        self._last = np.zeros(len(config.joint_names), dtype=np.float64)
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {
            self._config.input_device: OptionalType(
                TensorGroupType(
                    self.JOINTS,
                    [FloatType(name) for name in REALHAND_FFG_GLOVE_SENSOR_NAMES],
                )
            )
        }

    def output_spec(self) -> RetargeterIOType:
        return {
            self.OUTPUT: TensorGroupType(
                f"realhand_ffg_glove_{self._profile.model}_{self._config.side}_hand_joints",
                [FloatType(name) for name in self._config.joint_names],
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        source = inputs[self._config.input_device]
        if source.is_none:
            target = (
                self._last.copy()
                if self._config.hold_last_on_missing
                else np.zeros_like(self._last)
            )
        else:
            sensors = [
                float(source[index])
                for index in range(len(REALHAND_FFG_GLOVE_SENSOR_NAMES))
            ]
            target = self._mapper.map(sensors, self._config.joint_names)

        if context.execution_events.reset:
            self._mapper.reset()
            self._last = target
        else:
            target = self._stabilize(target)
            alpha = float(np.clip(self._config.smoothing_alpha, 0.0, 1.0))
            self._last = alpha * target + (1.0 - alpha) * self._last
        for index, value in enumerate(self._last):
            outputs[self.OUTPUT][index] = float(value)

    def _stabilize(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float64).copy()
        delta = target - self._last
        deadband = max(float(self._config.joint_deadband_rad), 0.0)
        target[np.abs(delta) < deadband] = self._last[np.abs(delta) < deadband]
        for index, spec in enumerate(self._specs):
            closing = abs(target[index]) >= abs(self._last[index])
            limit = (
                self._config.max_close_step_rad
                if closing
                else self._config.max_open_step_rad
            )
            target[index] = self._last[index] + np.clip(
                target[index] - self._last[index], -limit, limit
            )
            target[index] = np.clip(target[index], spec.lower, spec.upper)
        return target


__all__ = [
    "RealHandFFGGloveRetargeter",
    "RealHandFFGGloveRetargeterConfig",
    "REALHAND_FFG_GLOVE_SENSOR_NAMES",
    "load_realhand_ffg_glove_calibration",
]
