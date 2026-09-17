# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-aware OpenXR hand and controller-trigger retargeters."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from isaaccapture.retargeting_engine.interface import BaseRetargeter
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    RetargeterIO,
    RetargeterIOType,
)
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    OptionalType,
    TensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import (
    ControllerInput,
    ControllerInputIndex,
    FloatType,
    HandInput,
    HandInputIndex,
    HandJointIndex,
)

from .profiles import RealHandJointSpec, get_realhand_profile

_FINGER_JOINTS = {
    "index": (
        HandJointIndex.INDEX_METACARPAL,
        HandJointIndex.INDEX_PROXIMAL,
        HandJointIndex.INDEX_INTERMEDIATE,
        HandJointIndex.INDEX_DISTAL,
        HandJointIndex.INDEX_TIP,
    ),
    "middle": (
        HandJointIndex.MIDDLE_METACARPAL,
        HandJointIndex.MIDDLE_PROXIMAL,
        HandJointIndex.MIDDLE_INTERMEDIATE,
        HandJointIndex.MIDDLE_DISTAL,
        HandJointIndex.MIDDLE_TIP,
    ),
    "ring": (
        HandJointIndex.RING_METACARPAL,
        HandJointIndex.RING_PROXIMAL,
        HandJointIndex.RING_INTERMEDIATE,
        HandJointIndex.RING_DISTAL,
        HandJointIndex.RING_TIP,
    ),
    "pinky": (
        HandJointIndex.LITTLE_METACARPAL,
        HandJointIndex.LITTLE_PROXIMAL,
        HandJointIndex.LITTLE_INTERMEDIATE,
        HandJointIndex.LITTLE_DISTAL,
        HandJointIndex.LITTLE_TIP,
    ),
}


def _as_np(value) -> np.ndarray:
    return np.from_dlpack(value)


def _safe_normalized(vector: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(vector)
    if norm < 1.0e-6:
        return None
    return vector / norm


def _angle_between(first: np.ndarray, second: np.ndarray) -> float:
    first_n = _safe_normalized(first)
    second_n = _safe_normalized(second)
    if first_n is None or second_n is None:
        return 0.0
    return float(np.arccos(np.clip(np.dot(first_n, second_n), -1.0, 1.0)))


def _env_debug_enabled() -> bool:
    value = os.getenv("P7_REALHAND_DEBUG_CONTROLLER", "")
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_debug_every() -> int:
    value = os.getenv("P7_REALHAND_DEBUG_CONTROLLER_EVERY", "30")
    try:
        return max(1, int(value))
    except ValueError:
        return 30


@dataclass
class RealHandHandTrackingRetargeterConfig:
    input_device: str
    joint_names: list[str]
    side: str
    hand_model: str = "l6"
    smoothing_alpha: float = 0.65
    finger_curl_scale: float = 0.70
    thumb_flex_scale: float = 0.55
    finger_close_gain: float = 1.0
    thumb_close_gain: float = 1.1
    thumb_opposition_gain: float = 1.0
    finger_spread_gain: float = 1.0
    curl_response_exponent: float = 1.0
    thumb_curl_response_exponent: float = 1.0
    thumb_opposition_to_flex: float = 0.85
    calibrate_open_on_first_frame: bool = True
    adaptive_open_baseline: bool = True


def realhand_handtracking_thumb_defaults(hand_model: str) -> dict[str, float]:
    """Return model-specific thumb response defaults for OpenXR handtracking."""
    model = get_realhand_profile(hand_model).model
    if model in ("l6", "o6"):
        return {
            "thumb_flex_scale": 0.55,
            "thumb_close_gain": 1.10,
            "thumb_opposition_gain": 1.00,
            "thumb_curl_response_exponent": 1.00,
            "thumb_opposition_to_flex": 0.85,
        }
    return {
        "thumb_flex_scale": 1.15,
        "thumb_close_gain": 1.80,
        "thumb_opposition_gain": 1.80,
        "thumb_curl_response_exponent": 0.75,
        "thumb_opposition_to_flex": 0.85,
    }


class RealHandHandTrackingRetargeter(BaseRetargeter):
    """Retarget OpenXR joint geometry to an L6, O6, or L20 hand."""

    def __init__(self, config: RealHandHandTrackingRetargeterConfig, name: str) -> None:
        if config.side not in ("left", "right"):
            raise ValueError(f"side must be left or right, got {config.side!r}")
        self._config = config
        self._profile = get_realhand_profile(config.hand_model)
        expected_names = self._profile.joint_names(config.side)
        if set(config.joint_names) != set(expected_names):
            missing = sorted(set(expected_names) - set(config.joint_names))
            extra = sorted(set(config.joint_names) - set(expected_names))
            raise ValueError(
                f"{self._profile.display_name} {config.side} joint set mismatch; missing={missing}, extra={extra}"
            )
        self._specs = [
            self._profile.joint_spec(config.side, name) for name in config.joint_names
        ]
        self._last_output = np.zeros(len(config.joint_names), dtype=np.float64)
        self._open_baseline: np.ndarray | None = None
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {self._config.input_device: OptionalType(HandInput())}

    def output_spec(self) -> RetargeterIOType:
        return {
            "hand_joints": TensorGroupType(
                f"{self._profile.model}_{self._config.side}_hand_joints",
                [FloatType(name) for name in self._config.joint_names],
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        output = outputs["hand_joints"]
        hand_group = inputs[self._config.input_device]

        if context.execution_events.reset:
            self._last_output[:] = 0.0
            self._open_baseline = None

        if hand_group.is_none:
            self._write_output(output)
            return

        positions = _as_np(hand_group[HandInputIndex.JOINT_POSITIONS]).astype(
            np.float64
        )
        valid = _as_np(hand_group[HandInputIndex.JOINT_VALID])

        raw_target = np.array(
            [self._raw_joint_target(spec, positions, valid) for spec in self._specs],
            dtype=np.float64,
        )
        if self._config.calibrate_open_on_first_frame and self._open_baseline is None:
            self._open_baseline = raw_target.copy()
        elif self._config.adaptive_open_baseline and self._open_baseline is not None:
            for index, spec in enumerate(self._specs):
                if not spec.semantic.endswith("_spread"):
                    self._open_baseline[index] = min(
                        self._open_baseline[index], raw_target[index]
                    )

        target = (
            raw_target.copy()
            if self._open_baseline is None
            else raw_target - self._open_baseline
        )
        for index, spec in enumerate(self._specs):
            target[index] = self._shape_target(target[index], spec)

        alpha = float(np.clip(self._config.smoothing_alpha, 0.0, 1.0))
        self._last_output = alpha * target + (1.0 - alpha) * self._last_output
        self._write_output(output)

    def _write_output(self, output) -> None:
        for index, value in enumerate(self._last_output):
            output[index] = float(value)

    @staticmethod
    def _valid(valid: np.ndarray, *indices: int) -> bool:
        return all(valid[index] > 0 for index in indices)

    def _raw_joint_target(
        self, spec: RealHandJointSpec, positions: np.ndarray, valid: np.ndarray
    ) -> float:
        semantic = spec.semantic
        if semantic == "thumb_opposition" or semantic == "thumb_rotation":
            value = self._thumb_opposition_close(positions, valid) * spec.upper
        elif semantic == "thumb_flex":
            flexion = self._thumb_flexion_angles(positions, valid)
            value = sum(flexion) * self._config.thumb_flex_scale
            value = max(
                value,
                self._thumb_opposition_close(positions, valid)
                * spec.upper
                * self._config.thumb_opposition_to_flex,
            )
        elif semantic == "thumb_base_flex":
            base_flexion, _ = self._thumb_flexion_angles(positions, valid)
            value = base_flexion * self._config.thumb_flex_scale
        elif semantic == "thumb_distal_flex":
            _, distal_flexion = self._thumb_flexion_angles(positions, valid)
            value = distal_flexion * self._config.thumb_flex_scale
            value = max(
                value,
                self._thumb_opposition_close(positions, valid)
                * spec.upper
                * self._config.thumb_opposition_to_flex,
            )
        else:
            finger = semantic.split("_", 1)[0]
            if semantic.endswith("_curl"):
                _, distal = self._finger_flexion_angles(finger, positions, valid)
                value = distal * self._config.finger_curl_scale
            elif semantic.endswith("_base_flex"):
                base, _ = self._finger_flexion_angles(finger, positions, valid)
                value = base * self._config.finger_curl_scale
            elif semantic.endswith("_distal_flex"):
                _, distal = self._finger_flexion_angles(finger, positions, valid)
                value = distal * self._config.finger_curl_scale
            elif semantic.endswith("_spread"):
                value = self._finger_spread(finger, positions, valid)
            else:
                raise ValueError(f"unsupported hand-joint semantic {semantic!r}")
        return float(np.clip(value, spec.lower, spec.upper))

    def _shape_target(self, value: float, spec: RealHandJointSpec) -> float:
        if spec.semantic.endswith("_spread"):
            return float(
                np.clip(value * self._config.finger_spread_gain, spec.lower, spec.upper)
            )

        upper = max(spec.upper, 1.0e-6)
        normalized = np.clip(value / upper, 0.0, 1.0)
        is_thumb = spec.semantic.startswith("thumb")
        if spec.semantic in ("thumb_opposition", "thumb_rotation"):
            gain = self._config.thumb_opposition_gain
        else:
            gain = (
                self._config.thumb_close_gain
                if is_thumb
                else self._config.finger_close_gain
            )
        shaped = np.clip(normalized * gain, 0.0, 1.0)
        exponent = (
            max(float(self._config.thumb_curl_response_exponent), 1.0e-3)
            if is_thumb
            else max(float(self._config.curl_response_exponent), 1.0e-3)
        )
        return float(np.clip((shaped**exponent) * upper, spec.lower, spec.upper))

    def _finger_flexion_angles(
        self, finger: str, positions: np.ndarray, valid: np.ndarray
    ) -> tuple[float, float]:
        joints = _FINGER_JOINTS[finger]
        if not self._valid(valid, *joints):
            return 0.0, 0.0
        metacarpal, proximal, intermediate, distal, tip = (
            positions[index] for index in joints
        )
        metacarpal_segment = proximal - metacarpal
        proximal_segment = intermediate - proximal
        intermediate_segment = distal - intermediate
        distal_segment = tip - distal
        base = _angle_between(metacarpal_segment, proximal_segment)
        distal_curl = _angle_between(
            proximal_segment, intermediate_segment
        ) + _angle_between(intermediate_segment, distal_segment)
        return base, distal_curl

    def _finger_spread(
        self, finger: str, positions: np.ndarray, valid: np.ndarray
    ) -> float:
        metacarpal, proximal, _, _, _ = _FINGER_JOINTS[finger]
        required = (
            HandJointIndex.WRIST,
            HandJointIndex.PALM,
            HandJointIndex.INDEX_METACARPAL,
            HandJointIndex.LITTLE_METACARPAL,
            metacarpal,
            proximal,
        )
        if not self._valid(valid, *required):
            return 0.0
        forward = _safe_normalized(
            positions[HandJointIndex.PALM] - positions[HandJointIndex.WRIST]
        )
        lateral = _safe_normalized(
            positions[HandJointIndex.INDEX_METACARPAL]
            - positions[HandJointIndex.LITTLE_METACARPAL]
        )
        segment = _safe_normalized(positions[proximal] - positions[metacarpal])
        if forward is None or lateral is None or segment is None:
            return 0.0
        return float(np.arctan2(np.dot(segment, lateral), np.dot(segment, forward)))

    def _thumb_flexion_angles(
        self, positions: np.ndarray, valid: np.ndarray
    ) -> tuple[float, float]:
        joints = (
            HandJointIndex.THUMB_METACARPAL,
            HandJointIndex.THUMB_PROXIMAL,
            HandJointIndex.THUMB_DISTAL,
            HandJointIndex.THUMB_TIP,
        )
        if not self._valid(valid, *joints):
            return 0.0, 0.0
        metacarpal, proximal, distal, tip = (positions[index] for index in joints)
        first = proximal - metacarpal
        second = distal - proximal
        third = tip - distal
        return _angle_between(first, second), _angle_between(second, third)

    def _thumb_opposition_close(
        self, positions: np.ndarray, valid: np.ndarray
    ) -> float:
        required = (
            HandJointIndex.THUMB_TIP,
            HandJointIndex.INDEX_PROXIMAL,
            HandJointIndex.LITTLE_PROXIMAL,
        )
        if not self._valid(valid, *required):
            return 0.0

        thumb_tip = positions[HandJointIndex.THUMB_TIP]
        palm_width = max(
            np.linalg.norm(
                positions[HandJointIndex.INDEX_PROXIMAL]
                - positions[HandJointIndex.LITTLE_PROXIMAL]
            ),
            1.0e-3,
        )
        closes = []
        for tip_index in (
            HandJointIndex.INDEX_TIP,
            HandJointIndex.MIDDLE_TIP,
            HandJointIndex.RING_TIP,
            HandJointIndex.LITTLE_TIP,
        ):
            if not self._valid(valid, tip_index):
                continue
            distance = np.linalg.norm(thumb_tip - positions[tip_index])
            closes.append(
                np.clip((1.8 * palm_width - distance) / (1.5 * palm_width), 0.0, 1.0)
            )
        return float(max(closes)) if closes else 0.0


@dataclass
class ControllerTriggerRealHandRetargeterConfig:
    input_device: str
    joint_names: list[str]
    side: str
    hand_model: str = "l6"
    close_at_trigger_one: bool = True
    smoothing_alpha: float = 0.8
    trigger_source: str = "max"
    max_close_step: float = 1.0
    max_open_step: float = 1.0


class ControllerTriggerRealHandRetargeter(BaseRetargeter):
    """Map one controller's trigger to its matching RealHand posture."""

    def __init__(
        self, config: ControllerTriggerRealHandRetargeterConfig, name: str
    ) -> None:
        if config.side not in ("left", "right"):
            raise ValueError(f"side must be left or right, got {config.side!r}")
        if config.trigger_source not in ("trigger", "squeeze", "max"):
            raise ValueError(
                f"trigger_source must be trigger, squeeze, or max, got {config.trigger_source!r}"
            )
        if config.max_close_step <= 0.0 or config.max_open_step <= 0.0:
            raise ValueError("max_close_step and max_open_step must both be positive")
        self._config = config
        self._profile = get_realhand_profile(config.hand_model)
        self._specs = [
            self._profile.joint_spec(config.side, name) for name in config.joint_names
        ]
        self._last_output = np.zeros(len(config.joint_names), dtype=np.float64)
        self._last_close_amount = 0.0
        self._debug_enabled = _env_debug_enabled()
        self._debug_every = _env_debug_every()
        self._debug_counter = 0
        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        return {self._config.input_device: OptionalType(ControllerInput())}

    def output_spec(self) -> RetargeterIOType:
        return {
            "hand_joints": TensorGroupType(
                f"{self._profile.model}_{self._config.side}_trigger_joints",
                [FloatType(name) for name in self._config.joint_names],
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        output = outputs["hand_joints"]
        controller_group = inputs[self._config.input_device]
        if controller_group.is_none or not bool(
            controller_group[ControllerInputIndex.GRIP_IS_VALID]
        ):
            self._last_output[:] = 0.0
            self._last_close_amount = 0.0
            self._write_output(output)
            return

        trigger_value = float(
            np.clip(controller_group[ControllerInputIndex.TRIGGER_VALUE], 0.0, 1.0)
        )
        squeeze_value = float(
            np.clip(controller_group[ControllerInputIndex.SQUEEZE_VALUE], 0.0, 1.0)
        )
        if self._config.trigger_source == "trigger":
            close_input = trigger_value
        elif self._config.trigger_source == "squeeze":
            close_input = squeeze_value
        else:
            close_input = max(trigger_value, squeeze_value)
        requested = (
            close_input if self._config.close_at_trigger_one else 1.0 - close_input
        )

        if context.execution_events.reset:
            self._last_output[:] = 0.0
            self._last_close_amount = 0.0
        delta = requested - self._last_close_amount
        max_step = (
            self._config.max_close_step if delta >= 0.0 else self._config.max_open_step
        )
        self._last_close_amount = float(
            np.clip(
                self._last_close_amount + np.clip(delta, -max_step, max_step), 0.0, 1.0
            )
        )
        target = np.array(
            [self._last_close_amount * spec.trigger_closed for spec in self._specs],
            dtype=np.float64,
        )
        alpha = float(np.clip(self._config.smoothing_alpha, 0.0, 1.0))
        self._last_output = alpha * target + (1.0 - alpha) * self._last_output
        self._write_output(output)

        if self._debug_enabled:
            self._debug_counter += 1
            if self._debug_counter == 1 or self._debug_counter % self._debug_every == 0:
                print(
                    f"[P7 {self._profile.display_name} controller trigger]"
                    f" side={self._config.side} trigger={trigger_value:.3f}"
                    f" squeeze={squeeze_value:.3f} close={self._last_close_amount:.3f}",
                    flush=True,
                )

    def _write_output(self, output) -> None:
        for index, value in enumerate(self._last_output):
            output[index] = float(value)


__all__ = [
    "ControllerTriggerRealHandRetargeter",
    "ControllerTriggerRealHandRetargeterConfig",
    "RealHandHandTrackingRetargeter",
    "RealHandHandTrackingRetargeterConfig",
    "realhand_handtracking_thumb_defaults",
]
