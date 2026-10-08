# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sharpa Hand Retargeter Module.

Thin wrapper around `robotic_grounding.retarget.hand_kinematics.SharpaHandKinematics`
that adapts Teleop's OpenXR hand-tracking input format (26-joint HandInput,
xyzw quats) to the MANO 21-joint / wxyz layout the kinematics class expects,
runs IK, and writes the resulting finger DOFs into Teleop's TensorGroup
output.

The IK loop and Pinocchio/Pink implementation live in `robotic_grounding`.
This adapter owns the OpenXR-to-MANO normalization and the live-input task
constraints needed when only finger DOFs are emitted.

The Teleop wheel includes the public V2D retargeting source and generated
mesh-free Sharpa MJCFs. Install `isaaccapture[grounding]` for its runtime
dependencies.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pink.limits import VelocityLimit
from robotic_grounding.retarget.hand_kinematics import SharpaHandKinematics
from scipy.spatial.transform import Rotation

from isaaccapture.retargeting_engine.interface import (
    BaseRetargeter,
    RetargeterIOType,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    TensorGroupType,
    OptionalType,
)
from isaaccapture.retargeting_engine.tensor_types import (
    HandInput,
    FloatType,
    HandInputIndex,
    HandJointIndex,
)

logger = logging.getLogger(__name__)

# OpenXR (26 joints) -> MANO (21 joints) index mapping.
# Skips OpenXR palm(0) and metacarpal joints for non-thumb fingers.
# Order matches MANO_JOINTS_ORDER in robotic_grounding.retarget.params.
_OPENXR_TO_MANO_INDICES: list[int] = [
    HandJointIndex.WRIST,  # wrist
    HandJointIndex.THUMB_METACARPAL,  # thumb1
    HandJointIndex.THUMB_PROXIMAL,  # thumb2
    HandJointIndex.THUMB_DISTAL,  # thumb3
    HandJointIndex.THUMB_TIP,  # thumb4
    HandJointIndex.INDEX_PROXIMAL,  # index1
    HandJointIndex.INDEX_INTERMEDIATE,  # index2
    HandJointIndex.INDEX_DISTAL,  # index3
    HandJointIndex.INDEX_TIP,  # index4
    HandJointIndex.MIDDLE_PROXIMAL,  # middle1
    HandJointIndex.MIDDLE_INTERMEDIATE,  # middle2
    HandJointIndex.MIDDLE_DISTAL,  # middle3
    HandJointIndex.MIDDLE_TIP,  # middle4
    HandJointIndex.RING_PROXIMAL,  # ring1
    HandJointIndex.RING_INTERMEDIATE,  # ring2
    HandJointIndex.RING_DISTAL,  # ring3
    HandJointIndex.RING_TIP,  # ring4
    HandJointIndex.LITTLE_PROXIMAL,  # pinky1
    HandJointIndex.LITTLE_INTERMEDIATE,  # pinky2
    HandJointIndex.LITTLE_DISTAL,  # pinky3
    HandJointIndex.LITTLE_TIP,  # pinky4
]

# Number of FreeFlyer DOFs Pinocchio prepends to qpos when SharpaHandKinematics
# loads the MJCF with a JointModelFreeFlyer root joint.
_FREEFLYER_NQ = 7

# OpenXR hand-space (X right, Y up, -Z forward) to the V2D MANO convention
# (X palm-normal, Y lateral, Z finger-forward).
_MANO_FROM_OPENXR = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
_OPENXR_FINGER_SCALES = (
    (1, 2, 5, 1.4),  # thumb
    (5, 6, 9, 1.3),  # index
    (9, 10, 13, 1.1),  # middle
    (13, 14, 17, 1.2),  # ring
    (17, 18, 21, 1.6),  # pinky
)


@dataclass
class SharpaHandRetargeterConfig:
    """Configuration for the Sharpa hand retargeter.

    Attributes:
        robot_asset_path: Path to the Sharpa MJCF (e.g. right_sharpawave.xml,
            or right_sharpawave_nomesh.xml for tests). Resolve via
            `importlib.resources.files("robotic_grounding") / "assets" / "xmls" / "sharpawave"`.
        hand_side: "left" or "right".
        hand_joint_names: Output joint ordering override. If None, uses the
            finger joint names discovered from the MJCF model.
        input_convention: Input coordinate convention. OpenXR input is
            canonicalized into wrist-local MANO coordinates before IK.
        source_to_robot_scale: Scale factor from MANO to robot coordinates.
        solver: QP solver backend for Pink IK.
        max_iter: Maximum IK iterations per frame.
        frequency: IK rate-limiter frequency [Hz], used to compute dt.
        frame_tasks_converged_threshold: Per-task position-error convergence
            threshold [m] for early termination.
        parameter_config_path: Optional path to a JSON file for saving/loading
            tunable parameters.
    """

    robot_asset_path: str
    hand_side: str = "right"
    hand_joint_names: list[str] | None = None
    input_convention: Literal["openxr", "mano"] = "openxr"
    source_to_robot_scale: float = 1.0
    solver: str = "daqp"
    max_iter: int = 200
    frequency: float = 200.0
    frame_tasks_converged_threshold: float = 1e-6
    parameter_config_path: str | None = None


class SharpaHandRetargeter(BaseRetargeter):
    """Retargets OpenXR hand tracking to Sharpa hand joint angles via Pink IK.

    Inputs:
        - "hand_{side}": OpenXR hand tracking data (26 joints), optional.

    Outputs:
        - "hand_joints": Sharpa finger joint angles.
    """

    def __init__(self, config: SharpaHandRetargeterConfig, name: str) -> None:
        self._config = config
        self._hand_side = config.hand_side.lower()
        if self._hand_side not in ("left", "right"):
            raise ValueError(
                f"hand_side must be 'left' or 'right', got: {self._hand_side}"
            )
        if config.input_convention not in ("openxr", "mano"):
            raise ValueError(
                "input_convention must be 'openxr' or 'mano', "
                f"got: {config.input_convention}"
            )

        self._kinematics = SharpaHandKinematics(
            side=self._hand_side,
            robot_asset_path=config.robot_asset_path,
            source_model="mano",
            use_relative_frames=False,
            solver=config.solver,
            max_iter=config.max_iter,
            frequency=config.frequency,
            frame_tasks_converged_threshold=config.frame_tasks_converged_threshold,
        )
        if config.input_convention == "openxr":
            # OpenXR joint orientation axes do not follow MANO link frames.
            for task in self._kinematics.frame_tasks.values():
                task.set_orientation_cost(0.0)

            # The discarded FreeFlyer must not absorb tracked finger motion.
            velocity_limit = self._kinematics.robot.model.velocityLimit.copy()
            velocity_limit[:6] = 1e-9
            self._kinematics.configuration_limits[-1] = VelocityLimit(
                self._kinematics.robot.model, velocity_limit
            )

            # Sharpa MP sites and OpenXR intermediate joints are both PIP frames.
            for finger in ("index", "middle", "ring", "pinky"):
                frame = f"{self._hand_side}_{finger}_MP_site"
                _, position_cost, orientation_cost = self._kinematics.target_to_source[
                    frame
                ]
                self._kinematics.target_to_source[frame] = (
                    f"{finger}2",
                    position_cost,
                    orientation_cost,
                )

        # Finger joint names = everything Pinocchio reports past the FreeFlyer.
        self._finger_joint_names: list[str] = list(
            self._kinematics.robot_finger_joint_names.values()
        )

        if config.hand_joint_names is None:
            self._hand_joint_names = list(self._finger_joint_names)
        else:
            override = list(config.hand_joint_names)
            if len(override) != len(set(override)):
                seen: set[str] = set()
                dupes = [n for n in override if n in seen or seen.add(n)]  # type: ignore[func-returns-value]
                raise ValueError(f"hand_joint_names contains duplicates: {dupes}")
            finger_set = set(self._finger_joint_names)
            unknown = [n for n in override if n not in finger_set]
            if unknown:
                raise ValueError(
                    f"hand_joint_names contains names not found in the MJCF "
                    f"model's finger joints: {unknown}. "
                    f"Valid names: {self._finger_joint_names}"
                )
            self._hand_joint_names = override

        self._hand_joint_name_to_idx = {
            name: idx for idx, name in enumerate(self._hand_joint_names)
        }
        self._source_to_robot_scale = config.source_to_robot_scale

        # Warm-start qpos persists across frames.
        self._qpos_prev: np.ndarray | None = None

        super().__init__(name=name)

    def input_spec(self) -> RetargeterIOType:
        """Define input: optional hand tracking for the configured side."""
        key = f"hand_{self._hand_side}"
        return {key: OptionalType(HandInput())}

    def output_spec(self) -> RetargeterIOType:
        """Define output: Sharpa finger joint angles."""
        return {
            "hand_joints": TensorGroupType(
                f"hand_joints_{self._hand_side}",
                [FloatType(name) for name in self._hand_joint_names],
            )
        }

    def _emit_zeros(self, output_group) -> None:
        for i in range(len(self._hand_joint_names)):
            output_group[i] = 0.0
        self._qpos_prev = None

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        output_group = outputs["hand_joints"]
        hand_group = inputs[f"hand_{self._hand_side}"]

        if hand_group.is_none:
            self._emit_zeros(output_group)
            return

        joint_positions = np.from_dlpack(
            hand_group[HandInputIndex.JOINT_POSITIONS]
        )  # (26, 3)
        joint_orientations = np.from_dlpack(
            hand_group[HandInputIndex.JOINT_ORIENTATIONS]
        )  # (26, 4) xyzw
        joint_valid = np.from_dlpack(hand_group[HandInputIndex.JOINT_VALID])  # (26,)

        # SharpaHandKinematics has no per-task validity gate; if any joint
        # we depend on is invalid, fall back to zeros + reset warm-start
        # rather than feed the IK bogus targets.
        if not all(joint_valid[xr_idx] for xr_idx in _OPENXR_TO_MANO_INDICES):
            self._emit_zeros(output_group)
            return

        # Repack OpenXR (26 joints, xyzw quats) -> MANO (21 joints, wxyz quats).
        mano_positions = joint_positions[_OPENXR_TO_MANO_INDICES].astype(
            np.float64, copy=False
        )
        xyzw = joint_orientations[_OPENXR_TO_MANO_INDICES]
        mano_quats_wxyz = np.empty((21, 4), dtype=np.float64)
        mano_quats_wxyz[:, 0] = xyzw[:, 3]
        mano_quats_wxyz[:, 1:4] = xyzw[:, 0:3]

        if self._config.input_convention == "openxr":
            wrist_rotation = Rotation.from_quat(
                mano_quats_wxyz[0], scalar_first=True
            ).as_matrix()
            joint_rotations = Rotation.from_quat(
                mano_quats_wxyz, scalar_first=True
            ).as_matrix()
            mano_positions = (
                _MANO_FROM_OPENXR
                @ (wrist_rotation.T @ (mano_positions - mano_positions[0]).T)
            ).T
            joint_rotations = (
                _MANO_FROM_OPENXR
                @ (wrist_rotation.T @ joint_rotations)
                @ _MANO_FROM_OPENXR.T
            )
            mano_quats_wxyz = Rotation.from_matrix(joint_rotations).as_quat(
                scalar_first=True
            )
            for anchor, start, stop, scale in _OPENXR_FINGER_SCALES:
                mano_positions[start:stop] = (
                    mano_positions[anchor]
                    + (mano_positions[start:stop] - mano_positions[anchor]) * scale
                )

        # Initial qpos: warm-start from previous frame (with wrist re-anchored
        # to the new tracker reading) if available, else from q0.
        wrist_pos = mano_positions[0]
        wrist_wxyz = mano_quats_wxyz[0]
        wrist_xyzw = np.array(
            [wrist_wxyz[1], wrist_wxyz[2], wrist_wxyz[3], wrist_wxyz[0]]
        )

        if self._qpos_prev is None:
            qpos = self._kinematics.robot.q0.copy()
        else:
            qpos = self._qpos_prev.copy()
        qpos[:3] = wrist_pos
        qpos[3:7] = wrist_xyzw

        result = self._kinematics.compute(
            mano_positions,
            mano_quats_wxyz,
            source_to_robot_scale=self._source_to_robot_scale,
            qpos=qpos,
        )
        new_qpos = result["q"]
        self._qpos_prev = new_qpos.copy()

        # Slice finger DOFs out of the FreeFlyer-prefixed qpos and write
        # them through the (potentially reordered) hand_joint_names mapping.
        finger_angles = new_qpos[_FREEFLYER_NQ:]
        for i, jname in enumerate(self._finger_joint_names):
            out_idx = self._hand_joint_name_to_idx.get(jname)
            if out_idx is not None:
                output_group[out_idx] = float(finger_angles[i])


class SharpaBiManualRetargeter(BaseRetargeter):
    """Combines left and right Sharpa hand joint angles into a single vector.

    Inputs:
        - "left_hand_joints": Joint angles from a left SharpaHandRetargeter
        - "right_hand_joints": Joint angles from a right SharpaHandRetargeter

    Outputs:
        - "hand_joints": Combined joint angles ordered by target_joint_names
    """

    def __init__(
        self,
        left_joint_names: list[str],
        right_joint_names: list[str],
        target_joint_names: list[str],
        name: str,
    ) -> None:
        self._target_joint_names = target_joint_names
        self._left_joint_names = left_joint_names
        self._right_joint_names = right_joint_names

        super().__init__(name=name)

        self._left_indices: list[int] = []
        self._right_indices: list[int] = []
        self._output_indices_left: list[int] = []
        self._output_indices_right: list[int] = []

        for i, jname in enumerate(target_joint_names):
            if jname in left_joint_names:
                self._output_indices_left.append(i)
                self._left_indices.append(left_joint_names.index(jname))
            elif jname in right_joint_names:
                self._output_indices_right.append(i)
                self._right_indices.append(right_joint_names.index(jname))

        mapped = len(self._output_indices_left) + len(self._output_indices_right)
        if mapped != len(target_joint_names):
            known = set(left_joint_names) | set(right_joint_names)
            missing = [n for n in target_joint_names if n not in known]
            raise ValueError(
                f"target_joint_names contains {len(missing)} name(s) not found "
                f"in left or right joint lists: {missing}"
            )

    def input_spec(self) -> RetargeterIOType:
        """Define input collections for both hands."""
        return {
            "left_hand_joints": TensorGroupType(
                "left_hand_joints",
                [FloatType(name) for name in self._left_joint_names],
            ),
            "right_hand_joints": TensorGroupType(
                "right_hand_joints",
                [FloatType(name) for name in self._right_joint_names],
            ),
        }

    def output_spec(self) -> RetargeterIOType:
        """Define output collections for combined hand joints."""
        return {
            "hand_joints": TensorGroupType(
                "hand_joints_bimanual",
                [FloatType(name) for name in self._target_joint_names],
            )
        }

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        left_input = inputs["left_hand_joints"]
        right_input = inputs["right_hand_joints"]
        combined = outputs["hand_joints"]

        for src, dst in zip(self._left_indices, self._output_indices_left):
            combined[dst] = float(left_input[src])

        for src, dst in zip(self._right_indices, self._output_indices_right):
            combined[dst] = float(right_input[src])
