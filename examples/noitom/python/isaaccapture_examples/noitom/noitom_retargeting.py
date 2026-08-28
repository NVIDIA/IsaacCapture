# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Noitom full-body to G1 wrist SE3 retargeting for locomanipulation teleop.

Uses **posture-based** arm retargeting: mocap bone *directions* with G1 link
lengths (not scaled human joint positions). Wrist SE(3) targets feed Pink IK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from isaaccapture.retargeting_engine.interface import (
    BaseRetargeter,
    ParameterState,
    RetargeterIOType,
)
from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
    ComputeContext,
    RetargeterIO,
)
from isaaccapture.retargeting_engine.interface.tunable_parameter import FloatParameter
from isaaccapture.retargeting_engine.interface.tensor_group_type import TensorGroupType
from isaaccapture.retargeting_engine.tensor_types import DLDataType, NDArrayType
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    DeviceIOFullBodyPoseTracked,
)
from isaaccapture.schema import BodyJoint

# PNS/Noitom Y-up -> Isaac Z-up. PNS forward is -Z; axis remap maps it to Isaac -Y.
# Retarget / debug draw then apply operator_faces_robot (+180 deg Z) to align with G1 (+Y).
_NOITOM_TO_ISAAC = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_COORD_ROT = Rotation.from_matrix(_NOITOM_TO_ISAAC)

_DEFAULT_LEFT_WRIST_POS = np.array([-0.18, 0.1, 0.8], dtype=np.float64)
_DEFAULT_RIGHT_WRIST_POS = np.array([0.18, 0.1, 0.8], dtype=np.float64)
# G1 locomanipulation VR teleop wrist orientations (see locomanipulation_g1_env_cfg).
_DEFAULT_LEFT_WRIST_QUAT = np.array([-0.2706, 0.6533, 0.2706, 0.6533], dtype=np.float64)
_DEFAULT_RIGHT_WRIST_QUAT = np.array([-0.7071, 0.0, 0.7071, 0.0], dtype=np.float64)
_DEFAULT_ROBOT_PELVIS = np.array([0.0, 0.0, 0.72], dtype=np.float64)
# Approximate G1 shoulder origins in pelvis frame (Isaac Z-up, +Y left).
_ROBOT_LEFT_SHOULDER_OFFSET = np.array([0.05, 0.19, 0.30], dtype=np.float64)
_ROBOT_RIGHT_SHOULDER_OFFSET = np.array([0.05, -0.19, 0.30], dtype=np.float64)
# Torso segment lengths for posture-based reference skeleton (meters, pelvis frame).
_ROBOT_TORSO_SEGMENT_Z = 0.07
_ROBOT_NECK_SEGMENT = 0.08
_ROBOT_HEAD_SEGMENT = 0.12
_ROBOT_HAND_EXTENSION = 0.05
_DELTA_LIMIT = np.array([0.65, 0.65, 0.65], dtype=np.float64)


@dataclass
class NoitomRetargetingSettings:
    """Tunable retargeting parameters for Noitom-driven G1 wrists."""

    # Fraction of mocap pose applied relative to calibrated neutral (not link length).
    motion_scale: float = 0.55
    # Reduce motion when the solved arm chain is nearly straight (elbow singularity).
    arm_extension_soft_limit: float = 0.72
    # Minimum interior elbow angle (rad) when reconstructing the forearm direction.
    min_elbow_interior_angle: float = 0.65
    # Scale motion when the operator arm span exceeds the G1 link lengths.
    human_reach_margin: float = 0.96
    # Cap how much operator torso twist rotates arm targets (radians).
    max_torso_yaw_delta: float = 0.22
    # Fraction of torso yaw applied to arms (lower = arms ignore body twist).
    torso_yaw_arm_influence: float = 0.35
    position_smoothing: float = 0.85
    rotation_smoothing: float = 0.75
    robot_upper_arm_length: float = 0.28
    robot_forearm_length: float = 0.26
    arm_scale_min: float = 0.5
    arm_scale_max: float = 1.5
    robot_pelvis_world: np.ndarray = field(
        default_factory=lambda: _DEFAULT_ROBOT_PELVIS.copy()
    )
    robot_left_shoulder_offset: np.ndarray = field(
        default_factory=lambda: _ROBOT_LEFT_SHOULDER_OFFSET.copy()
    )
    robot_right_shoulder_offset: np.ndarray = field(
        default_factory=lambda: _ROBOT_RIGHT_SHOULDER_OFFSET.copy()
    )
    nominal_left_wrist_pos: np.ndarray = field(
        default_factory=lambda: _DEFAULT_LEFT_WRIST_POS.copy()
    )
    nominal_right_wrist_pos: np.ndarray = field(
        default_factory=lambda: _DEFAULT_RIGHT_WRIST_POS.copy()
    )
    nominal_left_wrist_quat_xyzw: np.ndarray = field(
        default_factory=lambda: _DEFAULT_LEFT_WRIST_QUAT.copy()
    )
    nominal_right_wrist_quat_xyzw: np.ndarray = field(
        default_factory=lambda: _DEFAULT_RIGHT_WRIST_QUAT.copy()
    )
    # Mocap wrist orientation does not match G1 wrist_yaw_link; lock by default.
    track_wrist_orientation: bool = False
    # Operator stands facing the robot (mirror L/R in horizontal plane).
    operator_faces_robot: bool = True
    # Drive Pink IK wrists to the cyan skeleton wrist joints (shared placement frame).
    track_aligned_mocap_wrists: bool = True
    # Rebuild cyan skeleton with G1 link lengths (bone directions, not human joint spacing).
    reference_use_robot_link_lengths: bool = True
    # Global multiplier on G1 segment lengths for the shared reference skeleton.
    reference_length_scale: float = 1.0
    # Additional scale applied only to upper-arm/forearm segments in reference skeleton.
    reference_arm_length_scale: float = 0.7
    # Scale left-right shoulder span in the reference skeleton (lower = narrower shoulders).
    reference_shoulder_span_scale: float = 0.82
    # Retain mocap pose via bone directions + robot link lengths (not joint positions).
    use_posture_based_arms: bool = True
    # Blend G1 nominal wrist quat with forearm-aligned frame (helps Pink IK converge).
    wrist_orientation_forearm_blend: float = 0.35
    # Feed posture-based elbow positions into Pink IK (narrows shoulder/elbow null space).
    track_elbow_ik_targets: bool = True
    # Feed posture-based shoulder positions into Pink IK (locks shoulder placement).
    track_shoulder_ik_targets: bool = True
    delta_limit: np.ndarray = field(default_factory=lambda: _DELTA_LIMIT.copy())
    sync_nominal_at_calibration: bool = True


@dataclass
class SE3Pose:
    """Pose in Isaac Z-up world frame (position meters, quaternion xyzw)."""

    position: np.ndarray
    quaternion_xyzw: np.ndarray

    def as_action_pose(self) -> np.ndarray:
        return np.concatenate(
            [
                self.position.astype(np.float32),
                self.quaternion_xyzw.astype(np.float32),
            ]
        )

    @staticmethod
    def from_nominal(position: np.ndarray, quaternion_xyzw: np.ndarray) -> SE3Pose:
        return SE3Pose(position.copy(), quaternion_xyzw.copy())


@dataclass
class ArmIkTargets:
    """Pink IK frame targets for one update (wrist, elbow, shoulder per arm)."""

    left_wrist: SE3Pose
    right_wrist: SE3Pose
    left_elbow: SE3Pose
    right_elbow: SE3Pose
    left_shoulder: SE3Pose
    right_shoulder: SE3Pose


@dataclass
class NoitomCalibrationView:
    """Read-only calibration snapshot for debug visualization alignment."""

    pelvis_world: np.ndarray
    body_yaw_isaac: float
    arm_length_scale: float
    body_height_scale: float


@dataclass
class _TorsoFrame:
    origin: np.ndarray
    rotation: Rotation


@dataclass
class _ArmCalibration:
    shoulder_torso: np.ndarray
    shoulder_world: np.ndarray
    elbow_world: np.ndarray
    wrist_pos_torso: np.ndarray
    wrist_rot_torso: Rotation
    wrist_rel_pelvis: np.ndarray
    wrist_world: np.ndarray
    upper_arm_length: float
    forearm_length: float


@dataclass
class _CalibrationState:
    torso: _TorsoFrame
    left: _ArmCalibration
    right: _ArmCalibration
    arm_length_scale: float
    body_height_scale: float
    body_yaw_isaac: float
    pelvis_world: np.ndarray
    nominal_left: SE3Pose
    nominal_right: SE3Pose
    nominal_left_elbow: SE3Pose
    nominal_right_elbow: SE3Pose
    nominal_left_shoulder: SE3Pose
    nominal_right_shoulder: SE3Pose


class NoitomG1Retargeter(BaseRetargeter):
    """Retarget Noitom upper-body motion to G1 arm SE3 targets for Pink IK."""

    def __init__(
        self,
        settings: NoitomRetargetingSettings | None = None,
        name: str = "noitom_g1_retargeter",
    ) -> None:
        self._settings = settings or NoitomRetargetingSettings()
        self._nominal_left = SE3Pose.from_nominal(
            self._settings.nominal_left_wrist_pos,
            self._settings.nominal_left_wrist_quat_xyzw,
        )
        self._nominal_right = SE3Pose.from_nominal(
            self._settings.nominal_right_wrist_pos,
            self._settings.nominal_right_wrist_quat_xyzw,
        )
        self._calibration: _CalibrationState | None = None
        self._latest_torso: _TorsoFrame | None = None
        self._smoothed_left = SE3Pose.from_nominal(
            self._nominal_left.position, self._nominal_left.quaternion_xyzw
        )
        self._smoothed_right = SE3Pose.from_nominal(
            self._nominal_right.position, self._nominal_right.quaternion_xyzw
        )
        self._smoothed_left_elbow = self._default_elbow_pose(is_left=True)
        self._smoothed_right_elbow = self._default_elbow_pose(is_left=False)
        self._smoothed_left_shoulder = self._default_shoulder_pose(is_left=True)
        self._smoothed_right_shoulder = self._default_shoulder_pose(is_left=False)

        param_state = ParameterState(
            name,
            parameters=[
                FloatParameter(
                    "motion_scale",
                    "Pose amplitude vs calibrated neutral (0=hold, 1=full mocap posture).",
                    default_value=self._settings.motion_scale,
                    min_value=0.0,
                    max_value=1.0,
                    step_size=0.05,
                    sync_fn=lambda v: setattr(self._settings, "motion_scale", v),
                ),
                FloatParameter(
                    "position_smoothing",
                    "Position smoothing alpha (0=hold last, 1=instant).",
                    default_value=self._settings.position_smoothing,
                    min_value=0.0,
                    max_value=1.0,
                    step_size=0.05,
                    sync_fn=lambda v: setattr(self._settings, "position_smoothing", v),
                ),
                FloatParameter(
                    "rotation_smoothing",
                    "Rotation smoothing alpha (0=hold last, 1=instant).",
                    default_value=self._settings.rotation_smoothing,
                    min_value=0.0,
                    max_value=1.0,
                    step_size=0.05,
                    sync_fn=lambda v: setattr(self._settings, "rotation_smoothing", v),
                ),
                FloatParameter(
                    "robot_upper_arm_length",
                    "Robot upper arm length for reach clamping [m].",
                    default_value=self._settings.robot_upper_arm_length,
                    min_value=0.05,
                    max_value=0.5,
                    step_size=0.01,
                    sync_fn=lambda v: setattr(
                        self._settings, "robot_upper_arm_length", v
                    ),
                ),
                FloatParameter(
                    "robot_forearm_length",
                    "Robot forearm length for reach clamping [m].",
                    default_value=self._settings.robot_forearm_length,
                    min_value=0.05,
                    max_value=0.5,
                    step_size=0.01,
                    sync_fn=lambda v: setattr(
                        self._settings, "robot_forearm_length", v
                    ),
                ),
                FloatParameter(
                    "arm_scale_min",
                    "Minimum arm length scale (robot/human ratio floor).",
                    default_value=self._settings.arm_scale_min,
                    min_value=0.1,
                    max_value=1.0,
                    step_size=0.05,
                    sync_fn=lambda v: setattr(self._settings, "arm_scale_min", v),
                ),
                FloatParameter(
                    "arm_scale_max",
                    "Maximum arm length scale (robot/human ratio ceiling).",
                    default_value=self._settings.arm_scale_max,
                    min_value=1.0,
                    max_value=3.0,
                    step_size=0.05,
                    sync_fn=lambda v: setattr(self._settings, "arm_scale_max", v),
                ),
            ],
        )

        super().__init__(name=name, parameter_state=param_state)

    @property
    def is_calibrated(self) -> bool:
        return self._calibration is not None

    @property
    def awaiting_calibration(self) -> bool:
        return self._calibration is None

    @property
    def current_left(self) -> SE3Pose:
        return self._smoothed_left

    @property
    def current_right(self) -> SE3Pose:
        return self._smoothed_right

    @property
    def current_left_elbow(self) -> SE3Pose:
        return self._smoothed_left_elbow

    @property
    def current_right_elbow(self) -> SE3Pose:
        return self._smoothed_right_elbow

    @property
    def current_left_shoulder(self) -> SE3Pose:
        return self._smoothed_left_shoulder

    @property
    def current_right_shoulder(self) -> SE3Pose:
        return self._smoothed_right_shoulder

    @property
    def current_arm_targets(self) -> ArmIkTargets:
        return ArmIkTargets(
            left_wrist=self._smoothed_left,
            right_wrist=self._smoothed_right,
            left_elbow=self._smoothed_left_elbow,
            right_elbow=self._smoothed_right_elbow,
            left_shoulder=self._smoothed_left_shoulder,
            right_shoulder=self._smoothed_right_shoulder,
        )

    @property
    def calibration_view(self) -> NoitomCalibrationView | None:
        if self._calibration is None:
            return None
        calib = self._calibration
        return NoitomCalibrationView(
            pelvis_world=calib.pelvis_world.copy(),
            body_yaw_isaac=calib.body_yaw_isaac,
            arm_length_scale=calib.arm_length_scale,
            body_height_scale=calib.body_height_scale,
        )

    @property
    def body_yaw_isaac(self) -> float:
        if self._latest_torso is None:
            return 0.0
        return _compute_torso_yaw(self._latest_torso)

    @property
    def body_yaw_delta(self) -> float:
        if self._calibration is None:
            return 0.0
        return self.body_yaw_isaac - self._calibration.body_yaw_isaac

    @property
    def retargeting_settings(self) -> NoitomRetargetingSettings:
        return self._settings

    @property
    def neutral_arms(self) -> tuple[_ArmCalibration, _ArmCalibration] | None:
        if self._calibration is None:
            return None
        return self._calibration.left, self._calibration.right

    def clear_calibration(self) -> None:
        self._calibration = None
        self._latest_torso = None
        self._smoothed_left = SE3Pose.from_nominal(
            self._nominal_left.position, self._nominal_left.quaternion_xyzw
        )
        self._smoothed_right = SE3Pose.from_nominal(
            self._nominal_right.position, self._nominal_right.quaternion_xyzw
        )
        self._smoothed_left_elbow = self._default_elbow_pose(is_left=True)
        self._smoothed_right_elbow = self._default_elbow_pose(is_left=False)
        self._smoothed_left_shoulder = self._default_shoulder_pose(is_left=True)
        self._smoothed_right_shoulder = self._default_shoulder_pose(is_left=False)

    def _default_shoulder_pose(self, is_left: bool) -> SE3Pose:
        shoulder = _shoulder_world_robot(self._settings, 0.0, is_left)
        upper_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        nominal_quat = (
            self._settings.nominal_left_wrist_quat_xyzw
            if is_left
            else self._settings.nominal_right_wrist_quat_xyzw
        )
        quat = _elbow_quat_for_ik(upper_dir, nominal_quat, self._settings)
        return SE3Pose(shoulder, quat)

    def _default_elbow_pose(self, is_left: bool) -> SE3Pose:
        shoulder = _shoulder_world_robot(self._settings, 0.0, is_left)
        upper_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        elbow = shoulder + upper_dir * self._settings.robot_upper_arm_length
        nominal_quat = (
            self._settings.nominal_left_wrist_quat_xyzw
            if is_left
            else self._settings.nominal_right_wrist_quat_xyzw
        )
        quat = _elbow_quat_for_ik(upper_dir, nominal_quat, self._settings)
        return SE3Pose(elbow, quat)

    def calibrate(self, frame: Any) -> bool:
        self._sync_parameters_from_state()
        parsed = _parse_upper_body(frame)
        if parsed is None:
            return False
        torso, left, right, pelvis_world = parsed
        self._latest_torso = torso
        human_arm = (
            left.upper_arm_length
            + left.forearm_length
            + right.upper_arm_length
            + right.forearm_length
        ) * 0.5
        robot_arm = (
            self._settings.robot_upper_arm_length + self._settings.robot_forearm_length
        )
        if human_arm < 1e-4:
            return False
        raw_scale = robot_arm / human_arm
        arm_scale = float(
            np.clip(
                raw_scale, self._settings.arm_scale_min, self._settings.arm_scale_max
            )
        )
        shoulder_world = torso.origin + torso.rotation.apply(left.shoulder_torso)
        shoulder_rel_z = float((shoulder_world - pelvis_world)[2])
        robot_shoulder_z = float(self._settings.robot_left_shoulder_offset[2])
        if shoulder_rel_z > 0.1:
            raw_body_scale = robot_shoulder_z / shoulder_rel_z
        else:
            raw_body_scale = arm_scale
        body_height_scale = float(np.clip(raw_body_scale, 0.45, 1.15))

        if self._settings.sync_nominal_at_calibration:
            if self._settings.track_aligned_mocap_wrists:
                nominal_left, nominal_right = _nominal_wrists_from_aligned_frame(
                    frame,
                    self._settings,
                    arm_scale,
                    body_height_scale,
                    pelvis_world,
                )
                if nominal_left is None or nominal_right is None:
                    return False
            elif self._settings.use_posture_based_arms:
                nominal_left = _wrist_pose_from_posture(
                    left,
                    self._settings,
                    is_left=True,
                    yaw_delta=0.0,
                    nominal_quat=self._settings.nominal_left_wrist_quat_xyzw,
                )
                nominal_right = _wrist_pose_from_posture(
                    right,
                    self._settings,
                    is_left=False,
                    yaw_delta=0.0,
                    nominal_quat=self._settings.nominal_right_wrist_quat_xyzw,
                )
            else:
                nominal_left = SE3Pose.from_nominal(
                    self._nominal_left.position, self._nominal_left.quaternion_xyzw
                )
                nominal_right = SE3Pose.from_nominal(
                    self._nominal_right.position, self._nominal_right.quaternion_xyzw
                )
        else:
            nominal_left = SE3Pose.from_nominal(
                self._nominal_left.position, self._nominal_left.quaternion_xyzw
            )
            nominal_right = SE3Pose.from_nominal(
                self._nominal_right.position, self._nominal_right.quaternion_xyzw
            )

        nominal_left_elbow = self._default_elbow_pose(is_left=True)
        nominal_right_elbow = self._default_elbow_pose(is_left=False)
        if (
            self._settings.track_elbow_ik_targets
            and self._settings.sync_nominal_at_calibration
        ):
            elbow_pair = _nominal_elbows_from_aligned_frame(
                frame,
                self._settings,
                arm_scale,
                body_height_scale,
                pelvis_world,
            )
            if elbow_pair is not None:
                nominal_left_elbow, nominal_right_elbow = elbow_pair

        nominal_left_shoulder = self._default_shoulder_pose(is_left=True)
        nominal_right_shoulder = self._default_shoulder_pose(is_left=False)
        if (
            self._settings.track_shoulder_ik_targets
            and self._settings.sync_nominal_at_calibration
        ):
            shoulder_pair = _nominal_shoulders_from_aligned_frame(
                frame,
                self._settings,
                arm_scale,
                body_height_scale,
                pelvis_world,
            )
            if shoulder_pair is not None:
                nominal_left_shoulder, nominal_right_shoulder = shoulder_pair

        self._calibration = _CalibrationState(
            torso=torso,
            left=left,
            right=right,
            arm_length_scale=arm_scale,
            body_height_scale=body_height_scale,
            body_yaw_isaac=_compute_torso_yaw(torso),
            pelvis_world=pelvis_world,
            nominal_left=nominal_left,
            nominal_right=nominal_right,
            nominal_left_elbow=nominal_left_elbow,
            nominal_right_elbow=nominal_right_elbow,
            nominal_left_shoulder=nominal_left_shoulder,
            nominal_right_shoulder=nominal_right_shoulder,
        )
        self._smoothed_left = nominal_left
        self._smoothed_right = nominal_right
        self._smoothed_left_elbow = nominal_left_elbow
        self._smoothed_right_elbow = nominal_right_elbow
        self._smoothed_left_shoulder = nominal_left_shoulder
        self._smoothed_right_shoulder = nominal_right_shoulder
        return True

    def retarget(self, frame: Any) -> ArmIkTargets | None:
        self._sync_parameters_from_state()
        if self._calibration is None:
            return None
        parsed = _parse_upper_body(frame)
        if parsed is None:
            return None

        torso, left, right, pelvis_world = parsed
        self._latest_torso = torso
        calib = self._calibration
        if self._settings.track_aligned_mocap_wrists:
            left_target = _wrist_target_from_aligned_skeleton(
                frame,
                calib,
                self._settings,
                is_left=True,
            )
            right_target = _wrist_target_from_aligned_skeleton(
                frame,
                calib,
                self._settings,
                is_left=False,
            )
            if left_target is None or right_target is None:
                return None
        else:
            left_target = _solve_wrist_target(
                torso=torso,
                arm=left,
                pelvis_world=pelvis_world,
                neutral=calib.left,
                neutral_torso=calib.torso,
                nominal=calib.nominal_left,
                calib_yaw=calib.body_yaw_isaac,
                arm_length_scale=calib.arm_length_scale,
                settings=self._settings,
                is_left=True,
            )
            right_target = _solve_wrist_target(
                torso=torso,
                arm=right,
                pelvis_world=pelvis_world,
                neutral=calib.right,
                neutral_torso=calib.torso,
                nominal=calib.nominal_right,
                calib_yaw=calib.body_yaw_isaac,
                arm_length_scale=calib.arm_length_scale,
                settings=self._settings,
                is_left=False,
            )
        self._smoothed_left = _smooth_pose(
            self._smoothed_left,
            left_target,
            self._settings.position_smoothing,
            self._settings.rotation_smoothing,
        )
        self._smoothed_right = _smooth_pose(
            self._smoothed_right,
            right_target,
            self._settings.position_smoothing,
            self._settings.rotation_smoothing,
        )
        left_elbow_target = self._smoothed_left_elbow
        right_elbow_target = self._smoothed_right_elbow
        if self._settings.track_elbow_ik_targets:
            if self._settings.track_aligned_mocap_wrists:
                left_elbow_target = _elbow_target_from_aligned_skeleton(
                    frame, calib, self._settings, is_left=True
                )
                right_elbow_target = _elbow_target_from_aligned_skeleton(
                    frame, calib, self._settings, is_left=False
                )
            else:
                yaw_delta = _resolve_yaw_delta(
                    _compute_torso_yaw(torso) - calib.body_yaw_isaac, self._settings
                )
                left_elbow_target = _solve_elbow_target(
                    arm=left,
                    neutral=calib.left,
                    nominal=calib.nominal_left_elbow,
                    settings=self._settings,
                    yaw_delta=yaw_delta,
                    is_left=True,
                )
                right_elbow_target = _solve_elbow_target(
                    arm=right,
                    neutral=calib.right,
                    nominal=calib.nominal_right_elbow,
                    settings=self._settings,
                    yaw_delta=yaw_delta,
                    is_left=False,
                )
            if left_elbow_target is None or right_elbow_target is None:
                return None
            self._smoothed_left_elbow = _smooth_pose(
                self._smoothed_left_elbow,
                left_elbow_target,
                self._settings.position_smoothing,
                self._settings.rotation_smoothing,
            )
            self._smoothed_right_elbow = _smooth_pose(
                self._smoothed_right_elbow,
                right_elbow_target,
                self._settings.position_smoothing,
                self._settings.rotation_smoothing,
            )
        left_shoulder_target = self._smoothed_left_shoulder
        right_shoulder_target = self._smoothed_right_shoulder
        if self._settings.track_shoulder_ik_targets:
            if self._settings.track_aligned_mocap_wrists:
                left_shoulder_target = _shoulder_target_from_aligned_skeleton(
                    frame, calib, self._settings, is_left=True
                )
                right_shoulder_target = _shoulder_target_from_aligned_skeleton(
                    frame, calib, self._settings, is_left=False
                )
            else:
                yaw_delta = _resolve_yaw_delta(
                    _compute_torso_yaw(torso) - calib.body_yaw_isaac, self._settings
                )
                left_shoulder_target = _solve_shoulder_target(
                    arm=left,
                    neutral=calib.left,
                    nominal=calib.nominal_left_shoulder,
                    settings=self._settings,
                    yaw_delta=yaw_delta,
                    is_left=True,
                )
                right_shoulder_target = _solve_shoulder_target(
                    arm=right,
                    neutral=calib.right,
                    nominal=calib.nominal_right_shoulder,
                    settings=self._settings,
                    yaw_delta=yaw_delta,
                    is_left=False,
                )
            if left_shoulder_target is None or right_shoulder_target is None:
                return None
            self._smoothed_left_shoulder = _smooth_pose(
                self._smoothed_left_shoulder,
                left_shoulder_target,
                self._settings.position_smoothing,
                self._settings.rotation_smoothing,
            )
            self._smoothed_right_shoulder = _smooth_pose(
                self._smoothed_right_shoulder,
                right_shoulder_target,
                self._settings.position_smoothing,
                self._settings.rotation_smoothing,
            )
        return self.current_arm_targets

    def input_spec(self) -> RetargeterIOType:
        return {"full_body_tracked": DeviceIOFullBodyPoseTracked()}

    def output_spec(self) -> RetargeterIOType:
        wrist_type = NDArrayType(
            "pose", shape=(7,), dtype=DLDataType.FLOAT, dtype_bits=32
        )
        return {
            "left_wrist": TensorGroupType("left_wrist", [wrist_type]),
            "right_wrist": TensorGroupType("right_wrist", [wrist_type]),
            "body_yaw_delta": TensorGroupType(
                "body_yaw_delta",
                [NDArrayType("yaw", shape=(1,), dtype=DLDataType.FLOAT, dtype_bits=32)],
            ),
        }

    def _compute_fn(
        self,
        inputs: RetargeterIO,
        outputs: RetargeterIO,
        context: ComputeContext,
    ) -> None:
        if context.execution_events.reset:
            self.clear_calibration()

        frame = inputs["full_body_tracked"][0]

        if frame is not None:
            if self._calibration is None:
                self.calibrate(frame)
            else:
                self.retarget(frame)

        outputs["left_wrist"][0] = self._smoothed_left.as_action_pose()
        outputs["right_wrist"][0] = self._smoothed_right.as_action_pose()
        outputs["body_yaw_delta"][0] = np.float32(self.body_yaw_delta)


def noitom_position_to_isaac(position: np.ndarray) -> np.ndarray:
    """Convert a Noitom Y-up position vector to Isaac Z-up."""
    return (_NOITOM_TO_ISAAC @ np.asarray(position, dtype=np.float64)).astype(
        np.float64
    )


def noitom_quaternion_to_isaac(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Convert a Noitom Y-up quaternion (xyzw) to Isaac Z-up."""
    rot = Rotation.from_quat(_normalize_quat(quaternion_xyzw))
    return _normalize_quat((_COORD_ROT * rot * _COORD_ROT.inv()).as_quat())


def map_point_to_robot_frame(
    noitom_point_yup: np.ndarray,
    pelvis_yup: np.ndarray,
    calib: NoitomCalibrationView | None,
    current_yaw: float,
    robot_pelvis: np.ndarray,
    draw_scale: float = 1.0,
    *,
    operator_faces_robot: bool = True,
    length_scale: float | None = None,
) -> np.ndarray:
    """Map a Noitom joint position into the robot simulation frame for debug draw."""
    point = noitom_position_to_isaac(noitom_point_yup)
    pelvis = noitom_position_to_isaac(pelvis_yup)
    rel = point - pelvis
    if calib is None:
        offset = rel * draw_scale
        if operator_faces_robot:
            offset = Rotation.from_euler("z", np.pi).apply(offset)
        return robot_pelvis + offset
    yaw_delta = current_yaw - calib.body_yaw_isaac
    scale = calib.arm_length_scale if length_scale is None else length_scale
    offset = _map_mocap_rel_to_robot_offset(
        rel,
        scale,
        draw_scale,
        yaw_delta,
        operator_faces_robot,
    )
    return robot_pelvis + offset


def _normalize_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


def _point_to_array(point: Any) -> np.ndarray:
    return np.array([point.x, point.y, point.z], dtype=np.float64)


def _quat_to_array(point: Any) -> np.ndarray:
    return _normalize_quat(
        np.array([point.x, point.y, point.z, point.w], dtype=np.float64)
    )


def _joint_pose(frame: Any, joint_index: BodyJoint | int) -> SE3Pose | None:
    if frame.joints is None:
        return None
    joint = frame.joints.joints(int(joint_index))
    if not joint.is_valid:
        return None
    pos = _point_to_array(joint.pose.position)
    quat = _quat_to_array(joint.pose.orientation)
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)):
        return None
    return SE3Pose(
        noitom_position_to_isaac(pos),
        noitom_quaternion_to_isaac(quat),
    )


def _build_torso_frame(frame: Any) -> _TorsoFrame | None:
    pelvis = _joint_pose(frame, BodyJoint.PELVIS)
    spine = _joint_pose(frame, BodyJoint.SPINE3)
    left_shoulder = _joint_pose(frame, BodyJoint.LEFT_SHOULDER)
    right_shoulder = _joint_pose(frame, BodyJoint.RIGHT_SHOULDER)
    if (
        pelvis is None
        or spine is None
        or left_shoulder is None
        or right_shoulder is None
    ):
        return None

    up = spine.position - pelvis.position
    right = right_shoulder.position - left_shoulder.position
    if np.linalg.norm(up) < 1e-5 or np.linalg.norm(right) < 1e-5:
        return None
    up /= np.linalg.norm(up)
    right /= np.linalg.norm(right)
    forward = np.cross(up, right)
    if np.linalg.norm(forward) < 1e-5:
        return None
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    rotation = Rotation.from_matrix(np.column_stack([right, forward, up]))
    return _TorsoFrame(origin=spine.position.copy(), rotation=rotation)


def _compute_torso_yaw(torso: _TorsoFrame) -> float:
    forward = torso.rotation.as_matrix()[:, 1]
    return float(np.arctan2(forward[1], forward[0]))


def _pose_to_torso(pose: SE3Pose, torso: _TorsoFrame) -> tuple[np.ndarray, Rotation]:
    pos_torso = torso.rotation.inv().apply(pose.position - torso.origin)
    rot_torso = torso.rotation.inv() * Rotation.from_quat(pose.quaternion_xyzw)
    return pos_torso, rot_torso


def _parse_arm(
    frame: Any,
    torso: _TorsoFrame,
    pelvis_world: np.ndarray,
    shoulder_index: BodyJoint,
    elbow_index: BodyJoint,
    wrist_index: BodyJoint,
) -> _ArmCalibration | None:
    shoulder = _joint_pose(frame, shoulder_index)
    elbow = _joint_pose(frame, elbow_index)
    wrist = _joint_pose(frame, wrist_index)
    if shoulder is None or elbow is None or wrist is None:
        return None

    upper_arm_length = float(np.linalg.norm(elbow.position - shoulder.position))
    forearm_length = float(np.linalg.norm(wrist.position - elbow.position))
    if upper_arm_length < 1e-4 or forearm_length < 1e-4:
        return None

    wrist_pos_torso, wrist_rot_torso = _pose_to_torso(wrist, torso)
    shoulder_torso = torso.rotation.inv().apply(shoulder.position - torso.origin)
    wrist_rel_pelvis = wrist.position - pelvis_world
    return _ArmCalibration(
        shoulder_torso=shoulder_torso,
        shoulder_world=shoulder.position.copy(),
        elbow_world=elbow.position.copy(),
        wrist_pos_torso=wrist_pos_torso,
        wrist_rot_torso=wrist_rot_torso,
        wrist_rel_pelvis=wrist_rel_pelvis,
        wrist_world=wrist.position.copy(),
        upper_arm_length=upper_arm_length,
        forearm_length=forearm_length,
    )


def _parse_upper_body(
    frame: Any,
) -> tuple[_TorsoFrame, _ArmCalibration, _ArmCalibration, np.ndarray] | None:
    pelvis_pose = _joint_pose(frame, BodyJoint.PELVIS)
    torso = _build_torso_frame(frame)
    if pelvis_pose is None or torso is None:
        return None
    pelvis_world = pelvis_pose.position.copy()
    left = _parse_arm(
        frame,
        torso,
        pelvis_world,
        BodyJoint.LEFT_SHOULDER,
        BodyJoint.LEFT_ELBOW,
        BodyJoint.LEFT_WRIST,
    )
    right = _parse_arm(
        frame,
        torso,
        pelvis_world,
        BodyJoint.RIGHT_SHOULDER,
        BodyJoint.RIGHT_ELBOW,
        BodyJoint.RIGHT_WRIST,
    )
    if left is None or right is None:
        return None
    return torso, left, right, pelvis_world


def _resolve_yaw_delta(yaw_delta: float, settings: NoitomRetargetingSettings) -> float:
    """Limit torso twist fed into arm FK (prevents waist+arm IK deadlock)."""
    influenced = yaw_delta * settings.torso_yaw_arm_influence
    limit = settings.max_torso_yaw_delta
    return float(np.clip(influenced, -limit, limit))


def _map_mocap_direction_to_robot(
    direction_isaac: np.ndarray,
    yaw_delta: float,
    operator_faces_robot: bool,
) -> np.ndarray:
    """Rotate a unit bone direction from mocap into the robot frame (no length scale)."""
    direction = np.asarray(direction_isaac, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    direction = direction / norm
    if operator_faces_robot:
        direction = Rotation.from_euler("z", np.pi).apply(direction)
    return Rotation.from_euler("z", yaw_delta).apply(direction)


def _shoulder_world_robot(
    settings: NoitomRetargetingSettings,
    yaw_delta: float,
    is_left: bool,
) -> np.ndarray:
    anchor = settings.robot_pelvis_world.astype(np.float64)
    return anchor + _shoulder_offset_robot(settings, yaw_delta, is_left)


def _arm_fk_robot(
    arm: _ArmCalibration,
    settings: NoitomRetargetingSettings,
    yaw_delta: float,
    is_left: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FK along mocap bone directions using G1 upper-arm and forearm lengths."""
    shoulder_robot = _shoulder_world_robot(settings, yaw_delta, is_left)
    upper_dir = _map_mocap_direction_to_robot(
        arm.elbow_world - arm.shoulder_world,
        yaw_delta,
        settings.operator_faces_robot,
    )
    forearm_dir = _map_mocap_direction_to_robot(
        arm.wrist_world - arm.elbow_world,
        yaw_delta,
        settings.operator_faces_robot,
    )
    upper_len = settings.robot_upper_arm_length
    forearm_len = settings.robot_forearm_length
    elbow_robot = shoulder_robot + upper_dir * upper_len
    wrist_robot = elbow_robot + forearm_dir * forearm_len
    return shoulder_robot, elbow_robot, wrist_robot


def _slerp_unit_direction(
    neutral_dir: np.ndarray, full_dir: np.ndarray, blend: float
) -> np.ndarray:
    """Interpolate unit bone directions (keeps FK chain valid after FK)."""
    t = float(np.clip(blend, 0.0, 1.0))
    mixed = (1.0 - t) * neutral_dir + t * full_dir
    norm = float(np.linalg.norm(mixed))
    if norm < 1e-6:
        return full_dir
    return mixed / norm


def _arm_direction_dot(upper_dir: np.ndarray, forearm_dir: np.ndarray) -> float:
    return float(np.dot(upper_dir, forearm_dir))


def _unit_direction(
    vector: np.ndarray, fallback: np.ndarray | None = None
) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        if fallback is not None:
            return _unit_direction(fallback)
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return vec / norm


def _elbow_interior_angle(upper_dir: np.ndarray, forearm_dir: np.ndarray) -> float:
    """Angle (rad) between upper-arm and forearm unit directions; 0 = fully extended."""
    dot = float(
        np.clip(
            _arm_direction_dot(
                _unit_direction(upper_dir), _unit_direction(forearm_dir)
            ),
            -1.0,
            1.0,
        )
    )
    return float(np.arccos(dot))


def _forearm_dir_from_elbow_angle(
    upper_dir: np.ndarray,
    forearm_hint: np.ndarray,
    elbow_angle: float,
) -> np.ndarray:
    """Rebuild a forearm unit vector with the given interior elbow angle."""
    upper = _unit_direction(upper_dir)
    hint = _unit_direction(forearm_hint, fallback=upper)
    perp = hint - upper * float(np.dot(hint, upper))
    perp_norm = float(np.linalg.norm(perp))
    if perp_norm < 1e-6:
        perp = np.cross(upper, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        perp_norm = float(np.linalg.norm(perp))
        if perp_norm < 1e-6:
            perp = np.cross(upper, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            perp_norm = float(np.linalg.norm(perp))
    perp = perp / max(perp_norm, 1e-8)
    angle = float(np.clip(elbow_angle, 0.0, np.pi - 1e-3))
    forearm = np.cos(angle) * upper + np.sin(angle) * perp
    return _unit_direction(forearm, fallback=upper)


def _human_robot_reach_scale(
    arm: _ArmCalibration, settings: NoitomRetargetingSettings
) -> float:
    """Shrink motion when the operator arm is longer than the fixed G1 chain."""
    human_reach = arm.upper_arm_length + arm.forearm_length
    robot_reach = settings.robot_upper_arm_length + settings.robot_forearm_length
    if human_reach <= robot_reach + 1e-4:
        return 1.0
    return float(
        np.clip(robot_reach * settings.human_reach_margin / human_reach, 0.25, 1.0)
    )


def _effective_motion_scale(
    base_scale: float, extension_dot: float, soft_limit: float
) -> float:
    """Shrink motion when the target arm chain is near full extension."""
    scale = float(np.clip(base_scale, 0.0, 1.0))
    if extension_dot <= soft_limit:
        return scale
    penalty = (extension_dot - soft_limit) / max(1e-3, 1.0 - soft_limit)
    return scale * (1.0 - 0.7 * float(np.clip(penalty, 0.0, 1.0)))


def _bend_forearm_direction(
    upper_dir: np.ndarray, forearm_dir: np.ndarray, target_dot: float = 0.55
) -> np.ndarray:
    """Pull forearm direction off a straight line to avoid elbow singularities."""
    upper = upper_dir / (np.linalg.norm(upper_dir) + 1e-8)
    forearm = forearm_dir / (np.linalg.norm(forearm_dir) + 1e-8)
    if _arm_direction_dot(upper, forearm) <= target_dot:
        return forearm
    axis = np.cross(upper, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    if float(np.linalg.norm(axis)) < 1e-4:
        axis = np.cross(upper, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    bent = forearm.copy()
    for _ in range(8):
        if _arm_direction_dot(upper, bent) <= target_dot:
            break
        bent = Rotation.from_rotvec(axis * 0.12).apply(bent)
        bent = bent / (np.linalg.norm(bent) + 1e-8)
    return bent


def _arm_chain_from_directions(
    shoulder_robot: np.ndarray,
    upper_dir: np.ndarray,
    forearm_dir: np.ndarray,
    settings: NoitomRetargetingSettings,
) -> tuple[np.ndarray, np.ndarray]:
    upper_len = settings.robot_upper_arm_length
    forearm_len = settings.robot_forearm_length
    elbow = shoulder_robot + upper_dir * upper_len
    wrist = elbow + forearm_dir * forearm_len
    wrist = _clamp_reach(shoulder_robot, wrist, upper_len, forearm_len)
    fore_vec = wrist - elbow
    fore_dist = float(np.linalg.norm(fore_vec))
    if fore_dist > 1e-6 and fore_dist > forearm_len:
        wrist = elbow + fore_vec * (forearm_len / fore_dist)
    return elbow, wrist


def _arm_fk_robot_blended(
    arm: _ArmCalibration,
    neutral_arm: _ArmCalibration,
    settings: NoitomRetargetingSettings,
    yaw_delta: float,
    is_left: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blend mocap posture (bone directions + elbow angle), then FK with G1 link lengths."""
    shoulder_robot = _shoulder_world_robot(settings, yaw_delta, is_left)
    upper_n = _map_mocap_direction_to_robot(
        neutral_arm.elbow_world - neutral_arm.shoulder_world,
        yaw_delta,
        settings.operator_faces_robot,
    )
    upper_f = _map_mocap_direction_to_robot(
        arm.elbow_world - arm.shoulder_world,
        yaw_delta,
        settings.operator_faces_robot,
    )
    fore_n = _map_mocap_direction_to_robot(
        neutral_arm.wrist_world - neutral_arm.elbow_world,
        yaw_delta,
        settings.operator_faces_robot,
    )
    fore_f = _map_mocap_direction_to_robot(
        arm.wrist_world - arm.elbow_world,
        yaw_delta,
        settings.operator_faces_robot,
    )
    scale = _effective_motion_scale(
        settings.motion_scale,
        _arm_direction_dot(upper_f, fore_f),
        settings.arm_extension_soft_limit,
    )
    yaw_factor = 1.0 - 0.45 * min(
        1.0, abs(yaw_delta) / max(settings.max_torso_yaw_delta, 1e-3)
    )
    scale *= max(0.25, yaw_factor)
    scale *= _human_robot_reach_scale(arm, settings)
    upper_d = _slerp_unit_direction(upper_n, upper_f, scale)

    theta_n = _elbow_interior_angle(upper_n, fore_n)
    theta_f = _elbow_interior_angle(upper_f, fore_f)
    theta_d = (1.0 - scale) * theta_n + scale * theta_f
    min_theta = max(
        settings.min_elbow_interior_angle,
        float(np.arccos(np.clip(settings.arm_extension_soft_limit, -1.0, 1.0))),
    )
    theta_d = float(np.clip(max(theta_d, min_theta), min_theta, np.pi - 1e-3))
    fore_d = _forearm_dir_from_elbow_angle(upper_d, fore_f, theta_d)

    if _arm_direction_dot(upper_d, fore_d) > settings.arm_extension_soft_limit:
        fore_d = _bend_forearm_direction(upper_d, fore_d)

    elbow_neutral, wrist_neutral = _arm_chain_from_directions(
        shoulder_robot, upper_n, fore_n, settings
    )
    elbow_full, wrist_full = _arm_chain_from_directions(
        shoulder_robot, upper_d, fore_d, settings
    )
    elbow_robot = elbow_neutral + scale * (elbow_full - elbow_neutral)
    wrist_robot = wrist_neutral + scale * (wrist_full - wrist_neutral)
    wrist_robot = _clamp_reach(
        shoulder_robot,
        wrist_robot,
        settings.robot_upper_arm_length,
        settings.robot_forearm_length,
    )
    return shoulder_robot, elbow_robot, wrist_robot


def _wrist_quat_from_forearm(
    forearm_dir: np.ndarray,
    nominal_quat: np.ndarray,
    blend: float,
) -> np.ndarray:
    """Derive a wrist quaternion consistent with the solved forearm direction."""
    forward = np.asarray(forearm_dir, dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        return _normalize_quat(nominal_quat)
    forward = forward / norm
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.92:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-6:
        return _normalize_quat(nominal_quat)
    right = right / right_norm
    up = np.cross(right, forward)
    aligned = Rotation.from_matrix(np.column_stack([right, forward, up]))
    blend_clamped = float(np.clip(blend, 0.0, 1.0))
    if blend_clamped <= 0.0:
        return _normalize_quat(nominal_quat)
    if blend_clamped >= 1.0:
        return _normalize_quat(aligned.as_quat())
    nominal_rot = Rotation.from_quat(nominal_quat)
    slerp = Slerp(
        [0.0, 1.0],
        Rotation.concatenate([nominal_rot, aligned]),
    )
    return _normalize_quat(slerp([blend_clamped]).as_quat()[0])


def _wrist_pose_from_posture(
    arm: _ArmCalibration,
    settings: NoitomRetargetingSettings,
    is_left: bool,
    yaw_delta: float,
    nominal_quat: np.ndarray,
) -> SE3Pose:
    _shoulder_robot, _elbow_robot, wrist_robot = _arm_fk_robot(
        arm, settings, yaw_delta, is_left
    )
    forearm_dir = _map_mocap_direction_to_robot(
        arm.wrist_world - arm.elbow_world,
        yaw_delta,
        settings.operator_faces_robot,
    )
    quat = _wrist_quat_for_ik(
        forearm_dir, nominal_quat, settings, track_orientation=False
    )
    return SE3Pose(wrist_robot, quat)


def _wrist_quat_for_ik(
    forearm_dir: np.ndarray,
    nominal_quat: np.ndarray,
    settings: NoitomRetargetingSettings,
    track_orientation: bool,
) -> np.ndarray:
    if track_orientation:
        return _normalize_quat(nominal_quat)
    return _wrist_quat_from_forearm(
        forearm_dir,
        nominal_quat,
        settings.wrist_orientation_forearm_blend,
    )


def compute_robot_reference_positions(
    frame: Any,
    settings: NoitomRetargetingSettings,
    calib: NoitomCalibrationView,
    current_yaw: float,
    neutral_left: _ArmCalibration,
    neutral_right: _ArmCalibration,
) -> dict[int, np.ndarray]:
    """Build a robot-proportioned reference skeleton (posture, not scaled joint dots)."""
    parsed = _parse_upper_body(frame)
    if parsed is None:
        return {}
    _torso, left, right, _pelvis_world = parsed
    yaw_delta = _resolve_yaw_delta(current_yaw - calib.body_yaw_isaac, settings)
    anchor = settings.robot_pelvis_world.astype(np.float64)
    positions: dict[int, np.ndarray] = {int(BodyJoint.PELVIS): anchor.copy()}

    _fill_torso_reference_positions(frame, positions, anchor, yaw_delta, settings)

    for arm, neutral_arm, is_left, shoulder_idx, elbow_idx, wrist_idx, hand_idx in (
        (
            left,
            neutral_left,
            True,
            BodyJoint.LEFT_SHOULDER,
            BodyJoint.LEFT_ELBOW,
            BodyJoint.LEFT_WRIST,
            BodyJoint.LEFT_HAND,
        ),
        (
            right,
            neutral_right,
            False,
            BodyJoint.RIGHT_SHOULDER,
            BodyJoint.RIGHT_ELBOW,
            BodyJoint.RIGHT_WRIST,
            BodyJoint.RIGHT_HAND,
        ),
    ):
        shoulder_robot, elbow_robot, wrist_robot = _arm_fk_robot_blended(
            arm, neutral_arm, settings, yaw_delta, is_left
        )
        positions[int(shoulder_idx)] = shoulder_robot
        positions[int(elbow_idx)] = elbow_robot
        positions[int(wrist_idx)] = wrist_robot
        forearm = wrist_robot - elbow_robot
        forearm_norm = float(np.linalg.norm(forearm))
        if forearm_norm > 1e-6:
            forearm_dir = forearm / forearm_norm
        else:
            forearm_dir = _map_mocap_direction_to_robot(
                arm.wrist_world - arm.elbow_world,
                yaw_delta,
                settings.operator_faces_robot,
            )
        positions[int(hand_idx)] = wrist_robot + forearm_dir * _ROBOT_HAND_EXTENSION

    spine3 = positions.get(int(BodyJoint.SPINE3))
    if spine3 is not None:
        for collar_idx, shoulder_idx in (
            (BodyJoint.LEFT_COLLAR, BodyJoint.LEFT_SHOULDER),
            (BodyJoint.RIGHT_COLLAR, BodyJoint.RIGHT_SHOULDER),
        ):
            shoulder_pos = positions.get(int(shoulder_idx))
            if shoulder_pos is not None:
                positions[int(collar_idx)] = 0.5 * (spine3 + shoulder_pos)

    return positions


def _fill_torso_reference_positions(
    frame: Any,
    positions: dict[int, np.ndarray],
    anchor: np.ndarray,
    yaw_delta: float,
    settings: NoitomRetargetingSettings,
) -> None:
    chain = (
        BodyJoint.PELVIS,
        BodyJoint.SPINE1,
        BodyJoint.SPINE2,
        BodyJoint.SPINE3,
        BodyJoint.NECK,
        BodyJoint.HEAD,
    )
    segment_lengths = {
        BodyJoint.SPINE1: _ROBOT_TORSO_SEGMENT_Z,
        BodyJoint.SPINE2: _ROBOT_TORSO_SEGMENT_Z,
        BodyJoint.SPINE3: _ROBOT_TORSO_SEGMENT_Z,
        BodyJoint.NECK: _ROBOT_NECK_SEGMENT,
        BodyJoint.HEAD: _ROBOT_HEAD_SEGMENT,
    }
    prev_robot = anchor.copy()
    prev_mocap = _joint_pose(frame, BodyJoint.PELVIS)
    if prev_mocap is None:
        return
    prev_mocap_pos = prev_mocap.position.copy()
    for joint in chain[1:]:
        mocap_joint = _joint_pose(frame, joint)
        if mocap_joint is None:
            continue
        direction = _map_mocap_direction_to_robot(
            mocap_joint.position - prev_mocap_pos,
            yaw_delta,
            settings.operator_faces_robot,
        )
        seg_len = segment_lengths.get(joint, _ROBOT_TORSO_SEGMENT_Z)
        robot_pos = prev_robot + direction * seg_len
        positions[int(joint)] = robot_pos
        prev_robot = robot_pos
        prev_mocap_pos = mocap_joint.position.copy()


def _wrist_pose_from_pelvis_relative(
    wrist_rel_pelvis: np.ndarray,
    arm_length_scale: float,
    settings: NoitomRetargetingSettings,
    nominal_quat_xyzw: np.ndarray,
) -> SE3Pose:
    anchor = settings.robot_pelvis_world.astype(np.float64)
    offset = _map_mocap_rel_to_robot_offset(
        wrist_rel_pelvis,
        arm_length_scale,
        settings.motion_scale,
        0.0,
        settings.operator_faces_robot,
    )
    position = anchor + offset
    quat = _normalize_quat(nominal_quat_xyzw)
    return SE3Pose(position, quat)


def _map_mocap_rel_to_robot_offset(
    rel_isaac: np.ndarray,
    arm_length_scale: float,
    motion_scale: float,
    yaw_delta: float,
    operator_faces_robot: bool,
) -> np.ndarray:
    """Map a pelvis-relative mocap vector into robot pelvis-relative offset."""
    rel = np.asarray(rel_isaac, dtype=np.float64) * arm_length_scale * motion_scale
    if operator_faces_robot:
        rel = Rotation.from_euler("z", np.pi).apply(rel)
    return Rotation.from_euler("z", yaw_delta).apply(rel)


def _shoulder_offset_robot(
    settings: NoitomRetargetingSettings,
    yaw_delta: float,
    is_left: bool,
) -> np.ndarray:
    offset = (
        settings.robot_left_shoulder_offset
        if is_left
        else settings.robot_right_shoulder_offset
    )
    return Rotation.from_euler("z", yaw_delta).apply(
        np.asarray(offset, dtype=np.float64)
    )


def _clamp_reach(
    shoulder_torso: np.ndarray,
    target_torso: np.ndarray,
    upper_len: float,
    forearm_len: float,
) -> np.ndarray:
    offset = target_torso - shoulder_torso
    distance = float(np.linalg.norm(offset))
    max_reach = (upper_len + forearm_len) * 0.98
    min_reach = abs(upper_len - forearm_len) * 1.02
    if distance < 1e-6:
        return shoulder_torso + np.array([max_reach * 0.5, 0.0, 0.0], dtype=np.float64)
    clamped = float(np.clip(distance, min_reach, max_reach))
    return shoulder_torso + offset * (clamped / distance)


def _calibration_view_from_state(calib: _CalibrationState) -> NoitomCalibrationView:
    return NoitomCalibrationView(
        pelvis_world=calib.pelvis_world.copy(),
        body_yaw_isaac=calib.body_yaw_isaac,
        arm_length_scale=calib.arm_length_scale,
        body_height_scale=calib.body_height_scale,
    )


def _calibration_view_from_scales(
    arm_length_scale: float,
    body_height_scale: float,
    pelvis_world: np.ndarray,
    body_yaw_isaac: float = 0.0,
) -> NoitomCalibrationView:
    return NoitomCalibrationView(
        pelvis_world=pelvis_world.copy(),
        body_yaw_isaac=body_yaw_isaac,
        arm_length_scale=arm_length_scale,
        body_height_scale=body_height_scale,
    )


def _aligned_skeleton_positions(
    frame: Any,
    settings: NoitomRetargetingSettings,
    calib_view: NoitomCalibrationView,
) -> dict[int, np.ndarray]:
    from .noitom_reference_draw import (
        ReferenceSkeletonLengths,
        aligned_reference_skeleton_from_frame,
    )

    return aligned_reference_skeleton_from_frame(
        frame,
        settings.robot_pelvis_world,
        draw_scale=1.0,
        calib_view=(None if settings.reference_use_robot_link_lengths else calib_view),
        use_robot_link_lengths=settings.reference_use_robot_link_lengths,
        link_lengths=ReferenceSkeletonLengths.from_retargeting_settings(settings),
        length_scale=settings.reference_length_scale,
        arm_length_scale=settings.reference_arm_length_scale,
        shoulder_span_scale=settings.reference_shoulder_span_scale,
    )


def _shoulder_se3_from_aligned_positions(
    positions: dict[int, np.ndarray],
    is_left: bool,
    nominal: SE3Pose,
    settings: NoitomRetargetingSettings,
) -> SE3Pose | None:
    shoulder_index = int(
        BodyJoint.LEFT_SHOULDER if is_left else BodyJoint.RIGHT_SHOULDER
    )
    elbow_index = int(BodyJoint.LEFT_ELBOW if is_left else BodyJoint.RIGHT_ELBOW)
    shoulder = positions.get(shoulder_index)
    if shoulder is None:
        return None
    elbow = positions.get(elbow_index)
    if elbow is not None:
        upper_arm = elbow - shoulder
        quat = _elbow_quat_for_ik(
            upper_arm,
            nominal.quaternion_xyzw,
            settings,
        )
    else:
        quat = _normalize_quat(nominal.quaternion_xyzw)
    return SE3Pose(shoulder.copy(), quat)


def _elbow_quat_for_ik(
    upper_arm_dir: np.ndarray,
    nominal_quat: np.ndarray,
    settings: NoitomRetargetingSettings,
) -> np.ndarray:
    return _wrist_quat_from_forearm(
        upper_arm_dir,
        nominal_quat,
        settings.wrist_orientation_forearm_blend,
    )


def _elbow_se3_from_aligned_positions(
    positions: dict[int, np.ndarray],
    is_left: bool,
    nominal: SE3Pose,
    settings: NoitomRetargetingSettings,
) -> SE3Pose | None:
    elbow_index = int(BodyJoint.LEFT_ELBOW if is_left else BodyJoint.RIGHT_ELBOW)
    shoulder_index = int(
        BodyJoint.LEFT_SHOULDER if is_left else BodyJoint.RIGHT_SHOULDER
    )
    elbow = positions.get(elbow_index)
    if elbow is None:
        return None
    shoulder = positions.get(shoulder_index)
    if shoulder is not None:
        upper_arm = elbow - shoulder
        quat = _elbow_quat_for_ik(
            upper_arm,
            nominal.quaternion_xyzw,
            settings,
        )
    else:
        quat = _normalize_quat(nominal.quaternion_xyzw)
    return SE3Pose(elbow.copy(), quat)


def _wrist_se3_from_aligned_positions(
    positions: dict[int, np.ndarray],
    is_left: bool,
    nominal: SE3Pose,
    settings: NoitomRetargetingSettings,
) -> SE3Pose | None:
    wrist_index = int(BodyJoint.LEFT_WRIST if is_left else BodyJoint.RIGHT_WRIST)
    elbow_index = int(BodyJoint.LEFT_ELBOW if is_left else BodyJoint.RIGHT_ELBOW)
    wrist = positions.get(wrist_index)
    if wrist is None:
        return None
    elbow = positions.get(elbow_index)
    if elbow is not None:
        forearm = wrist - elbow
        quat = _wrist_quat_for_ik(
            forearm,
            nominal.quaternion_xyzw,
            settings,
            track_orientation=False,
        )
    else:
        quat = _normalize_quat(nominal.quaternion_xyzw)
    return SE3Pose(wrist.copy(), quat)


def _nominal_shoulders_from_aligned_frame(
    frame: Any,
    settings: NoitomRetargetingSettings,
    arm_length_scale: float,
    body_height_scale: float,
    pelvis_world: np.ndarray,
) -> tuple[SE3Pose, SE3Pose] | None:
    calib_view = _calibration_view_from_scales(
        arm_length_scale, body_height_scale, pelvis_world
    )
    positions = _aligned_skeleton_positions(frame, settings, calib_view)
    if not positions:
        return None
    default_left = SE3Pose.from_nominal(
        np.zeros(3, dtype=np.float64),
        settings.nominal_left_wrist_quat_xyzw,
    )
    default_right = SE3Pose.from_nominal(
        np.zeros(3, dtype=np.float64),
        settings.nominal_right_wrist_quat_xyzw,
    )
    left = _shoulder_se3_from_aligned_positions(positions, True, default_left, settings)
    right = _shoulder_se3_from_aligned_positions(
        positions, False, default_right, settings
    )
    if left is None or right is None:
        return None
    return left, right


def _nominal_elbows_from_aligned_frame(
    frame: Any,
    settings: NoitomRetargetingSettings,
    arm_length_scale: float,
    body_height_scale: float,
    pelvis_world: np.ndarray,
) -> tuple[SE3Pose, SE3Pose] | None:
    calib_view = _calibration_view_from_scales(
        arm_length_scale, body_height_scale, pelvis_world
    )
    positions = _aligned_skeleton_positions(frame, settings, calib_view)
    if not positions:
        return None
    default_left = SE3Pose.from_nominal(
        np.zeros(3, dtype=np.float64),
        settings.nominal_left_wrist_quat_xyzw,
    )
    default_right = SE3Pose.from_nominal(
        np.zeros(3, dtype=np.float64),
        settings.nominal_right_wrist_quat_xyzw,
    )
    left = _elbow_se3_from_aligned_positions(positions, True, default_left, settings)
    right = _elbow_se3_from_aligned_positions(positions, False, default_right, settings)
    if left is None or right is None:
        return None
    return left, right


def _nominal_wrists_from_aligned_frame(
    frame: Any,
    settings: NoitomRetargetingSettings,
    arm_length_scale: float,
    body_height_scale: float,
    pelvis_world: np.ndarray,
) -> tuple[SE3Pose | None, SE3Pose | None]:
    calib_view = _calibration_view_from_scales(
        arm_length_scale, body_height_scale, pelvis_world
    )
    positions = _aligned_skeleton_positions(frame, settings, calib_view)
    if not positions:
        return None, None
    default_left = SE3Pose.from_nominal(
        settings.nominal_left_wrist_pos, settings.nominal_left_wrist_quat_xyzw
    )
    default_right = SE3Pose.from_nominal(
        settings.nominal_right_wrist_pos, settings.nominal_right_wrist_quat_xyzw
    )
    return (
        _wrist_se3_from_aligned_positions(positions, True, default_left, settings),
        _wrist_se3_from_aligned_positions(positions, False, default_right, settings),
    )


def _shoulder_target_from_aligned_skeleton(
    frame: Any,
    calib: _CalibrationState,
    settings: NoitomRetargetingSettings,
    is_left: bool,
) -> SE3Pose | None:
    calib_view = _calibration_view_from_state(calib)
    positions = _aligned_skeleton_positions(frame, settings, calib_view)
    if not positions:
        return None
    nominal = calib.nominal_left_shoulder if is_left else calib.nominal_right_shoulder
    return _shoulder_se3_from_aligned_positions(positions, is_left, nominal, settings)


def _solve_shoulder_target(
    arm: _ArmCalibration,
    neutral: _ArmCalibration,
    nominal: SE3Pose,
    settings: NoitomRetargetingSettings,
    yaw_delta: float,
    is_left: bool,
) -> SE3Pose:
    shoulder_robot, _elbow_robot, _wrist_robot = _arm_fk_robot_blended(
        arm, neutral, settings, yaw_delta, is_left
    )
    upper_arm = _elbow_robot - shoulder_robot
    upper_norm = float(np.linalg.norm(upper_arm))
    if upper_norm > 1e-6:
        upper_dir = upper_arm / upper_norm
    else:
        upper_dir = _map_mocap_direction_to_robot(
            arm.elbow_world - arm.shoulder_world,
            yaw_delta,
            settings.operator_faces_robot,
        )
    quat = _elbow_quat_for_ik(upper_dir, nominal.quaternion_xyzw, settings)
    return SE3Pose(shoulder_robot, quat)


def _elbow_target_from_aligned_skeleton(
    frame: Any,
    calib: _CalibrationState,
    settings: NoitomRetargetingSettings,
    is_left: bool,
) -> SE3Pose | None:
    calib_view = _calibration_view_from_state(calib)
    positions = _aligned_skeleton_positions(frame, settings, calib_view)
    if not positions:
        return None
    nominal = calib.nominal_left_elbow if is_left else calib.nominal_right_elbow
    return _elbow_se3_from_aligned_positions(positions, is_left, nominal, settings)


def _solve_elbow_target(
    arm: _ArmCalibration,
    neutral: _ArmCalibration,
    nominal: SE3Pose,
    settings: NoitomRetargetingSettings,
    yaw_delta: float,
    is_left: bool,
) -> SE3Pose:
    _shoulder_robot, elbow_robot, _wrist_robot = _arm_fk_robot_blended(
        arm, neutral, settings, yaw_delta, is_left
    )
    upper_arm = elbow_robot - _shoulder_world_robot(settings, yaw_delta, is_left)
    upper_norm = float(np.linalg.norm(upper_arm))
    if upper_norm > 1e-6:
        upper_dir = upper_arm / upper_norm
    else:
        upper_dir = _map_mocap_direction_to_robot(
            arm.elbow_world - arm.shoulder_world,
            yaw_delta,
            settings.operator_faces_robot,
        )
    quat = _elbow_quat_for_ik(upper_dir, nominal.quaternion_xyzw, settings)
    return SE3Pose(elbow_robot, quat)


def _wrist_target_from_aligned_skeleton(
    frame: Any,
    calib: _CalibrationState,
    settings: NoitomRetargetingSettings,
    is_left: bool,
) -> SE3Pose | None:
    calib_view = _calibration_view_from_state(calib)
    positions = _aligned_skeleton_positions(frame, settings, calib_view)
    if not positions:
        return None
    nominal = calib.nominal_left if is_left else calib.nominal_right
    return _wrist_se3_from_aligned_positions(positions, is_left, nominal, settings)


def _solve_wrist_target(
    torso: _TorsoFrame,
    arm: _ArmCalibration,
    pelvis_world: np.ndarray,
    neutral: _ArmCalibration,
    neutral_torso: _TorsoFrame,
    nominal: SE3Pose,
    calib_yaw: float,
    arm_length_scale: float,
    settings: NoitomRetargetingSettings,
    is_left: bool,
) -> SE3Pose:
    yaw_delta = _resolve_yaw_delta(_compute_torso_yaw(torso) - calib_yaw, settings)

    if settings.use_posture_based_arms:
        _shoulder_robot, elbow_robot, wrist_robot = _arm_fk_robot_blended(
            arm, neutral, settings, yaw_delta, is_left
        )
        forearm = wrist_robot - elbow_robot
        forearm_norm = float(np.linalg.norm(forearm))
        if forearm_norm > 1e-6:
            forearm_dir = forearm / forearm_norm
        else:
            forearm_dir = _map_mocap_direction_to_robot(
                arm.wrist_world - arm.elbow_world,
                yaw_delta,
                settings.operator_faces_robot,
            )
        if settings.track_wrist_orientation:
            wrist_world_rot = torso.rotation * arm.wrist_rot_torso
            wrist_neutral_world_rot = neutral_torso.rotation * neutral.wrist_rot_torso
            delta_rot_world = wrist_world_rot * wrist_neutral_world_rot.inv()
            nominal_rot = Rotation.from_quat(nominal.quaternion_xyzw)
            target_rot = delta_rot_world * nominal_rot
            quat = _normalize_quat(target_rot.as_quat())
        else:
            quat = _wrist_quat_for_ik(
                forearm_dir,
                nominal.quaternion_xyzw,
                settings,
                track_orientation=False,
            )
        return SE3Pose(wrist_robot, quat)

    rel_now = arm.wrist_world - pelvis_world
    anchor = settings.robot_pelvis_world.astype(np.float64)
    off_shoulder = _shoulder_offset_robot(settings, yaw_delta, is_left)
    off_wrist = _map_mocap_rel_to_robot_offset(
        rel_now,
        arm_length_scale,
        settings.motion_scale,
        yaw_delta,
        settings.operator_faces_robot,
    )
    clamped = _clamp_reach(
        off_shoulder,
        off_wrist,
        settings.robot_upper_arm_length,
        settings.robot_forearm_length,
    )
    target_pos = anchor + clamped

    if settings.track_wrist_orientation:
        wrist_world_rot = torso.rotation * arm.wrist_rot_torso
        wrist_neutral_world_rot = neutral_torso.rotation * neutral.wrist_rot_torso
        delta_rot_world = wrist_world_rot * wrist_neutral_world_rot.inv()
        nominal_rot = Rotation.from_quat(nominal.quaternion_xyzw)
        target_rot = delta_rot_world * nominal_rot
        quat = _normalize_quat(target_rot.as_quat())
    else:
        quat = _normalize_quat(nominal.quaternion_xyzw)

    return SE3Pose(target_pos, quat)


def _smooth_pose(
    current: SE3Pose,
    target: SE3Pose,
    position_alpha: float,
    rotation_alpha: float,
) -> SE3Pose:
    pos_alpha = float(np.clip(position_alpha, 0.0, 1.0))
    rot_alpha = float(np.clip(rotation_alpha, 0.0, 1.0))
    position = (1.0 - pos_alpha) * current.position + pos_alpha * target.position
    if rot_alpha <= 0.0:
        quaternion = current.quaternion_xyzw.copy()
    elif rot_alpha >= 1.0:
        quaternion = target.quaternion_xyzw.copy()
    else:
        slerp = Slerp(
            [0.0, 1.0],
            Rotation.from_quat(
                np.vstack([current.quaternion_xyzw, target.quaternion_xyzw])
            ),
        )
        quaternion = _normalize_quat(slerp([rot_alpha]).as_quat()[0])
    return SE3Pose(position, quaternion)


__all__ = [
    "ArmIkTargets",
    "NoitomCalibrationView",
    "NoitomG1Retargeter",
    "NoitomRetargetingSettings",
    "SE3Pose",
    "compute_robot_reference_positions",
    "map_point_to_robot_frame",
    "noitom_position_to_isaac",
    "noitom_quaternion_to_isaac",
]
