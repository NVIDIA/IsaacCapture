# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal: the tensor-group contract and synthetic data a mocked ``step()`` returns.

Not part of the public shim surface (nothing here is in Guman's import list) -- shared by
``retargeting_engine.deviceio_source_nodes`` (output-name constants) and
``teleop_session_manager`` (``TeleopSession.step()``'s synthesis dispatch).
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np


class TensorGroup:
    """Minimal stand-in for the real ``OptionalTensorGroup``.

    Supports exactly what every Guman decode helper uses (see
    ``GumanIsaacTeleopUsage.md`` section 6): ``.is_none``, and ``__getitem__`` keyed by an
    ``IntEnum`` member (by its integer value) or a plain ``int`` for positional joint-state
    groups.
    """

    def __init__(self, fields: dict[int, Any], *, is_none: bool = False) -> None:
        self._fields = fields
        self.is_none = is_none

    def __getitem__(self, index: Any) -> Any:
        key = int(index)
        if key not in self._fields:
            raise KeyError(f"no field at index {key}")
        return self._fields[key]

    def __repr__(self) -> str:
        return f"TensorGroup(is_none={self.is_none}, fields={self._fields})"


def _none_group() -> TensorGroup:
    return TensorGroup({}, is_none=True)


def _identity_quat_xyzw() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _yaw_quat_xyzw(radians: float) -> np.ndarray:
    half = radians / 2.0
    return np.array([0.0, math.sin(half), 0.0, math.cos(half)], dtype=np.float32)


def head_pose_group(t: float) -> TensorGroup:
    """§5.1 ``HeadPose`` shape: a slowly circling head, always valid."""
    from .retargeting_engine.tensor_types import HeadPoseIndex

    angle = 2.0 * math.pi * t / 8.0
    position = np.array(
        [0.03 * math.cos(angle), 1.6, 0.03 * math.sin(angle)], dtype=np.float32
    )
    return TensorGroup(
        {
            HeadPoseIndex.POSITION: position,
            HeadPoseIndex.ORIENTATION: _yaw_quat_xyzw(angle),
            HeadPoseIndex.IS_VALID: True,
        }
    )


def controller_input_group(t: float, *, is_left: bool) -> TensorGroup:
    """§5.2 ``ControllerInput`` shape: one hand reaching, breathing trigger value."""
    from .retargeting_engine.tensor_types import ControllerInputIndex

    phase = 0.0 if is_left else math.pi
    angle = 2.0 * math.pi * t / 3.0 + phase
    lateral = -0.2 if is_left else 0.2
    position = np.array(
        [
            lateral + 0.15 * math.cos(angle),
            1.2 + 0.15 * math.sin(angle),
            -0.3,
        ],
        dtype=np.float32,
    )
    trigger = 0.5 + 0.5 * math.sin(angle)
    quat = _identity_quat_xyzw()
    return TensorGroup(
        {
            ControllerInputIndex.GRIP_POSITION: position,
            ControllerInputIndex.GRIP_ORIENTATION: quat,
            ControllerInputIndex.GRIP_IS_VALID: True,
            ControllerInputIndex.AIM_POSITION: position,
            ControllerInputIndex.AIM_ORIENTATION: quat,
            ControllerInputIndex.AIM_IS_VALID: True,
            ControllerInputIndex.PRIMARY_CLICK: 0.0,
            ControllerInputIndex.SECONDARY_CLICK: 0.0,
            ControllerInputIndex.THUMBSTICK_X: 0.0,
            ControllerInputIndex.THUMBSTICK_Y: 0.0,
            ControllerInputIndex.THUMBSTICK_CLICK: 0.0,
            ControllerInputIndex.MENU_CLICK: 0.0,
            ControllerInputIndex.SQUEEZE_VALUE: trigger,
            ControllerInputIndex.TRIGGER_VALUE: trigger,
        }
    )


_HAND_JOINT_COUNT = 26


def hand_input_group(t: float, *, is_left: bool) -> TensorGroup:
    """§5.3 ``HandInput`` shape: 26 OpenXR joints, all valid, gently curling."""
    from .retargeting_engine.tensor_types import HandInputIndex

    phase = 0.0 if is_left else math.pi
    curl = 0.5 + 0.5 * math.sin(2.0 * math.pi * t / 3.0 + phase)
    lateral = -0.2 if is_left else 0.2
    # Simple radial fan, not an anatomical model -- see HandGenerator in
    # src/plugins/controller_synthetic_hands for a real one, if ever ported here.
    positions = np.zeros((_HAND_JOINT_COUNT, 3), dtype=np.float32)
    orientations = np.zeros((_HAND_JOINT_COUNT, 4), dtype=np.float32)
    for i in range(_HAND_JOINT_COUNT):
        spread = i / _HAND_JOINT_COUNT
        positions[i] = (
            lateral + 0.03 * spread * (1.0 - curl),
            1.1 - 0.02 * spread * curl,
            -0.3 - 0.01 * i,
        )
        orientations[i] = _identity_quat_xyzw()
    radii = np.full((_HAND_JOINT_COUNT,), 0.008, dtype=np.float32)
    valid = np.ones((_HAND_JOINT_COUNT,), dtype=np.uint8)
    return TensorGroup(
        {
            HandInputIndex.JOINT_POSITIONS: positions,
            HandInputIndex.JOINT_ORIENTATIONS: orientations,
            HandInputIndex.JOINT_RADII: radii,
            HandInputIndex.JOINT_VALID: valid,
        }
    )


def joint_state_group(joint_count: int) -> TensorGroup:
    """§5.5 generic ``JointStateSource`` shape: one float per joint, positionally indexed."""
    values = dict.fromkeys(range(joint_count), 0.0)
    return TensorGroup(values)


class SyntheticPoseGenerator:
    """Dispatches an output name to the right synthetic shape (``GumanIsaacTeleopUsage.md`` §10).

    ``joint_counts`` maps a non-head/controller/hand output name to its positional field count
    (§5.5) -- ``TeleopSession`` builds this from each ``JointStateSource``'s ``joint_names`` at
    construction time, since the name alone doesn't carry a count.
    """

    def __init__(
        self, joint_counts: dict[str, int], *, start_time: float | None = None
    ) -> None:
        self._joint_counts = joint_counts
        self._start_time = start_time if start_time is not None else time.monotonic()

    def sample(self, name: str) -> TensorGroup:
        t = time.monotonic() - self._start_time
        if name == "head":
            return head_pose_group(t)
        if name in ("controller_left", "controller_right"):
            return controller_input_group(t, is_left=name.endswith("left"))
        if name in ("hand_left", "hand_right"):
            return hand_input_group(t, is_left=name.endswith("left"))
        count = self._joint_counts.get(name)
        if count is not None:
            return joint_state_group(count)
        return _none_group()
