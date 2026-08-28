# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic joint-space device retargeter (leader arms, exoskeletons, ...).

A single retargeter that maps a name-keyed joint-state input (produced by
:class:`~isaaccapture.retargeting_engine.deviceio_source_nodes.JointStateSource`, which converts
the ``JointStateOutput`` FlatBuffer schema) into an Isaac Lab action, in one of two modes:

* ``mode="joint"`` -- pass the device joints straight through to robot joint targets, remapped
  by name with an optional per-joint affine (``offset + sign * scale * value``). This is the
  lossless leader -> follower mirror used for same-kinematics teleoperation. No extra
  dependencies.
* ``mode="ee_pose"`` -- forward-kinematics the device joints (via a URDF, using ``pinocchio``)
  into a 7-D end-effector pose ``[x, y, z, qx, qy, qz, qw]`` plus a scalar gripper command,
  for task-space / cross-embodiment teleoperation. ``pinocchio`` is imported lazily so that
  ``mode="joint"`` never requires it.

The output element names are chosen so a downstream
:class:`~isaaccapture.retargeters.TensorReorderer` can flatten them into the exact action layout
an Isaac Lab environment expects. See ``examples/teleop/python/isaaccapture_examples/teleop/joint_space_device_example.py``
for an end-to-end reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from isaaccapture.retargeting_engine.interface import (
    BaseRetargeter,
    RetargeterIOType,
)
from isaaccapture.retargeting_engine.interface.execution_events import ExecutionState
from isaaccapture.retargeting_engine.interface.retargeter_core_types import RetargeterIO
from isaaccapture.retargeting_engine.interface.tensor_group_type import (
    OptionalType,
    TensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import (
    DLDataType,
    FloatType,
    NDArrayType,
    TransformMatrix,
)

# Output group / element keys (single source of truth for the pipeline wiring).
JOINT_TARGETS_KEY = "joint_targets"
EE_POSE_KEY = "ee_pose"
GRIPPER_COMMAND_KEY = "gripper_command"
GRIPPER_ELEMENT_LABEL = "gripper_value"

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
_MATRIX_INDEX = 0


@dataclass
class JointStateRetargeterConfig:
    """Configuration for :class:`JointStateRetargeter`.

    Args:
        device_joints: Ordered device DOF names, matching ``JointStateOutput.joints[*].name``
            and the names declared by the upstream ``JointStateSource``.
        target_joints: ``joint`` mode only -- ordered robot joint names to emit, one output
            element each. Defaults to ``device_joints`` (identity mirror).
        joint_map: ``joint`` mode only -- ``{device_name: target_name}`` overrides; any target
            not covered maps from the device joint of the same name.
        scale: ``joint`` mode only -- per-target multiplicative gain (e.g. gear ratio / unit
            conversion). Defaults to 1.0.
        offset: ``joint`` mode only -- per-target additive offset [rad or m]. Defaults to 0.0.
        sign: ``joint`` mode only -- per-target sign (+1 / -1). Defaults to +1.0.
        urdf_path: ``ee_pose`` mode only -- path to the device URDF used for forward kinematics.
        ee_link: ``ee_pose`` mode only -- URDF frame/link name of the end-effector (tool) frame.
        gripper_joint: ``ee_pose`` mode only -- device DOF name treated as the gripper.
        gripper_open: ``ee_pose`` mode only -- gripper DOF value at fully open. When both
            ``gripper_open`` and ``gripper_close`` are set, the emitted gripper command is the
            normalized closedness in ``[0, 1]``; otherwise the raw value is passed through.
        gripper_close: ``ee_pose`` mode only -- gripper DOF value at fully closed.
        clutch: ``ee_pose`` mode only -- when true, rebase the EE position around an origin
            captured on the first ``RUNNING`` frame so engaging teleop does not jump the robot;
            the home is the live ``robot_ee_pos`` input when connected, else the FK pose at
            engage. When false (default) the absolute FK pose is emitted.
    """

    device_joints: list[str]
    target_joints: list[str] = field(default_factory=list)
    joint_map: dict[str, str] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)
    offset: dict[str, float] = field(default_factory=dict)
    sign: dict[str, float] = field(default_factory=dict)
    urdf_path: str | None = None
    ee_link: str | None = None
    gripper_joint: str = "gripper"
    gripper_open: float | None = None
    gripper_close: float | None = None
    clutch: bool = False


class JointStateRetargeter(BaseRetargeter):
    """Maps a name-keyed joint-state input to a robot action in ``joint`` or ``ee_pose`` mode.

    Input (both modes):
        - :data:`JOINTS` -- Optional name-keyed group with one ``FloatType`` per
          ``config.device_joints`` entry (the joint positions).

    Input (``ee_pose`` mode, only when ``config.clutch``):
        - :data:`ROBOT_EE_POS_INPUT` -- Optional ``world_T_ee`` 4x4 transform of the robot's
          current end-effector, used to latch the clutch home on engage.

    Output (``joint`` mode):
        - :data:`JOINT_TARGETS_KEY` -- one ``FloatType`` per ``config.target_joints`` entry.

    Output (``ee_pose`` mode):
        - :data:`EE_POSE_KEY` -- a single 7-D ``NDArray`` ``[x, y, z, qx, qy, qz, qw]``.
        - :data:`GRIPPER_COMMAND_KEY` -- a single ``FloatType`` gripper command.

    Note:
        The :data:`JOINTS` input is read positionally in ``config.device_joints`` order, so the
        upstream producer (e.g. ``JointStateSource``) must declare the same names in the same
        order; a name mismatch is rejected by the graph's type check at ``connect`` time.
        ``ee_pose`` mode ignores the schema's ``ee_pose`` field and always computes forward
        kinematics from the joint positions.
    """

    JOINTS = "joints"
    ROBOT_EE_POS_INPUT = "robot_ee_pos"

    def __init__(
        self, name: str, mode: str, config: JointStateRetargeterConfig
    ) -> None:
        """Initialize the joint-space retargeter.

        Args:
            name: Name identifier for this retargeter node.
            mode: ``"joint"`` or ``"ee_pose"``.
            config: Device / mode configuration.
        """
        if mode not in ("joint", "ee_pose"):
            raise ValueError(f"mode must be 'joint' or 'ee_pose', got: {mode!r}")
        self._mode = mode
        self._cfg = config

        if mode == "joint":
            self._target_joints = list(config.target_joints) or list(
                config.device_joints
            )
            # Per target joint, the device joint that feeds it (inverse of joint_map, which is
            # device -> target). Targets not covered map from the device joint of the same name.
            self._device_for_target: dict[str, str] = {
                tgt: next((d for d, t in config.joint_map.items() if t == tgt), tgt)
                for tgt in self._target_joints
            }
            self._last_targets = np.zeros(len(self._target_joints), dtype=np.float32)
        else:
            if not config.urdf_path or not config.ee_link:
                raise ValueError(
                    "ee_pose mode requires config.urdf_path and config.ee_link"
                )
            self._fk = _UrdfForwardKinematics(config.urdf_path, config.ee_link)
            self._origin: np.ndarray | None = None
            self._home = np.zeros(3, dtype=np.float64)
            self._last_pose = np.concatenate([np.zeros(3), _IDENTITY_QUAT]).astype(
                np.float32
            )
            self._last_gripper = 0.0

        super().__init__(name=name)

    # ------------------------------------------------------------------ specs

    def input_spec(self) -> RetargeterIOType:
        joints_type = TensorGroupType(
            self.JOINTS, [FloatType(n) for n in self._cfg.device_joints]
        )
        spec: RetargeterIOType = {self.JOINTS: OptionalType(joints_type)}
        if self._mode == "ee_pose" and self._cfg.clutch:
            spec[self.ROBOT_EE_POS_INPUT] = OptionalType(TransformMatrix())
        return spec

    def output_spec(self) -> RetargeterIOType:
        if self._mode == "joint":
            return {
                JOINT_TARGETS_KEY: TensorGroupType(
                    JOINT_TARGETS_KEY, [FloatType(n) for n in self._target_joints]
                )
            }
        return {
            EE_POSE_KEY: TensorGroupType(
                EE_POSE_KEY,
                [
                    NDArrayType(
                        "pose", shape=(7,), dtype=DLDataType.FLOAT, dtype_bits=32
                    )
                ],
            ),
            GRIPPER_COMMAND_KEY: TensorGroupType(
                GRIPPER_COMMAND_KEY, [FloatType(GRIPPER_ELEMENT_LABEL)]
            ),
        }

    # ---------------------------------------------------------------- compute

    def _compute_fn(self, inputs: RetargeterIO, outputs: RetargeterIO, context) -> None:
        if self._mode == "joint":
            self._compute_joint(inputs, outputs, context)
        else:
            self._compute_ee(inputs, outputs, context)

    def _read_positions(self, joints_group) -> dict[str, float]:
        """Read the name-keyed joint group into a ``{device_name: position}`` dict."""
        return {
            name: float(joints_group[i])
            for i, name in enumerate(self._cfg.device_joints)
        }

    def _compute_joint(self, inputs, outputs, context) -> None:
        if context.execution_events.reset:
            self._last_targets = np.zeros(len(self._target_joints), dtype=np.float32)

        out = outputs[JOINT_TARGETS_KEY]
        jin = inputs[self.JOINTS]
        if jin.is_none:
            for i in range(len(self._target_joints)):
                out[i] = float(self._last_targets[i])
            return

        positions = self._read_positions(jin)
        for i, tgt in enumerate(self._target_joints):
            raw = positions.get(self._device_for_target[tgt], 0.0)
            value = (
                self._cfg.offset.get(tgt, 0.0)
                + self._cfg.sign.get(tgt, 1.0) * self._cfg.scale.get(tgt, 1.0) * raw
            )
            self._last_targets[i] = value
            out[i] = float(value)

    def _compute_ee(self, inputs, outputs, context) -> None:
        running = context.execution_events.execution_state == ExecutionState.RUNNING
        if context.execution_events.reset or (self._cfg.clutch and not running):
            self._origin = None
            self._last_pose = np.concatenate([np.zeros(3), _IDENTITY_QUAT]).astype(
                np.float32
            )
            self._last_gripper = 0.0

        ee_out = outputs[EE_POSE_KEY]
        grip_out = outputs[GRIPPER_COMMAND_KEY]
        jin = inputs[self.JOINTS]
        if jin.is_none:
            ee_out[0] = self._last_pose
            grip_out[0] = self._last_gripper
            return

        positions = self._read_positions(jin)
        fk_pose = self._fk.solve(positions)  # [x, y, z, qx, qy, qz, qw]

        if self._cfg.clutch:
            if self._origin is None:
                if not running:
                    ee_out[0] = self._last_pose
                    grip_out[0] = self._compute_gripper(positions)
                    return
                self._origin = fk_pose[:3].copy()
                self._home = self._latch_home(inputs, fk_pose)
            position = self._home + (fk_pose[:3] - self._origin)
        else:
            position = fk_pose[:3]

        self._last_pose = np.concatenate([position, fk_pose[3:7]]).astype(np.float32)
        self._last_gripper = self._compute_gripper(positions)
        ee_out[0] = self._last_pose
        grip_out[0] = self._last_gripper

    def _latch_home(self, inputs: RetargeterIO, fk_pose: np.ndarray) -> np.ndarray:
        """Home for clutch rebasing: live robot EE position if connected, else the FK pose."""
        ee_inp = inputs.get(self.ROBOT_EE_POS_INPUT)
        if ee_inp is not None and not ee_inp.is_none:
            world_T_ee = np.from_dlpack(ee_inp[_MATRIX_INDEX]).astype(np.float64)
            return world_T_ee[:3, 3].copy()
        return fk_pose[:3].copy()

    def _compute_gripper(self, positions: dict[str, float]) -> float:
        raw = positions.get(self._cfg.gripper_joint, 0.0)
        lo, hi = self._cfg.gripper_open, self._cfg.gripper_close
        if lo is None or hi is None or hi == lo:
            return float(raw)
        c = (raw - lo) / (hi - lo)
        return float(min(1.0, max(0.0, c)))


class _UrdfForwardKinematics:
    """Lazy ``pinocchio`` forward-kinematics helper for a URDF end-effector frame."""

    def __init__(self, urdf_path: str, ee_link: str) -> None:
        try:
            import pinocchio as pin  # noqa: F401
        except ImportError as exc:
            raise ModuleNotFoundError(
                "JointStateRetargeter(mode='ee_pose') requires pinocchio.\n"
                "Install it with:  pip install 'isaaccapture[retargeters]'  (or: pip install pin)"
            ) from exc
        self._pin = pin
        self._model = pin.buildModelFromUrdf(urdf_path)
        self._data = self._model.createData()
        if not self._model.existFrame(ee_link):
            raise ValueError(f"ee_link {ee_link!r} not found in URDF {urdf_path!r}")
        self._frame_id = self._model.getFrameId(ee_link)

    def solve(self, positions: dict[str, float]) -> np.ndarray:
        """Forward-kinematics the named joint positions to a 7-D ``[x,y,z,qx,qy,qz,qw]`` pose.

        Assumes a fixed-base model of single-DOF joints (revolute/prismatic) -- the common case
        for leader arms and exoskeletons -- writing one configuration value per named joint.
        Names not present in the URDF (e.g. the gripper) are ignored for the EE pose.
        """
        pin = self._pin
        q = pin.neutral(self._model)
        for name, value in positions.items():
            if self._model.existJointName(name):
                jid = self._model.getJointId(name)
                q[self._model.joints[jid].idx_q] = value
        pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)
        return np.asarray(
            pin.SE3ToXYZQUAT(self._data.oMf[self._frame_id]), dtype=np.float32
        )
