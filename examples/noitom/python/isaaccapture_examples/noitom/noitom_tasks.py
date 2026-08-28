# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External Isaac Lab task registration for Noitom-driven G1 teleop testing."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import registry

from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
    LocomanipulationG1EnvCfg,
)
from isaaccapture.schema import (
    BodyJoint,
    FullBodyPose,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    DeviceIOFullBodyPoseTracked,
    IDeviceIOSource,
)
from isaaccapture.retargeting_engine.interface import (
    ComputeContext,
    OutputCombiner,
    RetargeterIO,
    RetargeterIOType,
    TensorGroup,
    TensorGroupType,
)
from isaaccapture.retargeting_engine.tensor_types import DLDataType, NDArrayType
from isaaccapture.teleop_session_manager import PluginConfig

from .noitom_retargeting import (
    ArmIkTargets,
    NoitomG1Retargeter,
    NoitomRetargetingSettings,
    noitom_position_to_isaac,
)
from .noitom_reference_draw import (
    ReferenceSkeletonLengths,
    aligned_reference_skeleton_from_frame,
)

TASK_ID = "Isaac-PickPlace-Locomanipulation-G1-Noitom-Abs-v0"

_FULL_BODY_INPUT = "deviceio_full_body"
_ACTION_OUTPUT = "action"
_G1_ACTION_DIM_WRIST_ONLY = 32
_G1_ACTION_DIM_WITH_ARM_IK = 60
_PINK_PELVIS_LINK = "g1_29dof_with_hand_rev_1_0_pelvis"
_PINK_LEFT_ELBOW_LINK = "g1_29dof_with_hand_rev_1_0_left_elbow_link"
_PINK_RIGHT_ELBOW_LINK = "g1_29dof_with_hand_rev_1_0_right_elbow_link"
_PINK_LEFT_SHOULDER_LINK = "g1_29dof_with_hand_rev_1_0_left_shoulder_pitch_link"
_PINK_RIGHT_SHOULDER_LINK = "g1_29dof_with_hand_rev_1_0_right_shoulder_pitch_link"
_LineList = list[list[float]]
_ColorList = list[tuple[float, float, float, float]]

_LOCOMOTION_DEFAULT_HIP_HEIGHT = 0.72

_NOITOM_REFERENCE_COLOR = (0.15, 0.85, 1.0, 1.0)
_NOITOM_REFERENCE_LINE_THICKNESS = 4.0
_NOITOM_REFERENCE_JOINT_MARKER_SIZE = 0.018
# Approximate G1 pelvis height in the locomanipulation scene (meters, Isaac Z-up).
_ROBOT_PELVIS_ANCHOR = (0.0, 0.0, 0.72)
_NOITOM_REFERENCE_DEFAULT_OFFSET = (0.0, 0.0, 0.0)
_NOITOM_WRIST_TARGET_COLOR = (1.0, 0.65, 0.1, 1.0)
_NOITOM_WRIST_TARGET_MARKER_SIZE = 0.025
_NOITOM_ELBOW_TARGET_COLOR = (1.0, 0.2, 0.85, 1.0)
_NOITOM_ELBOW_TARGET_MARKER_SIZE = 0.022
_NOITOM_SHOULDER_TARGET_COLOR = (0.2, 1.0, 0.35, 1.0)
_NOITOM_SHOULDER_TARGET_MARKER_SIZE = 0.02
_NOITOM_PLUGIN_NAME = "noitom_mocap"
_NOITOM_PLUGIN_ROOT_ID = "noitom_mocap"
_NOITOM_VENDOR_ID = "body.noitom"


def g1_action_dim(*, use_arm_ik_frame_tasks: bool) -> int:
    """Flat teleop action size for the Noitom G1 locomanipulation task."""
    return (
        _G1_ACTION_DIM_WITH_ARM_IK
        if use_arm_ik_frame_tasks
        else _G1_ACTION_DIM_WRIST_ONLY
    )


@dataclass(frozen=True)
class NoitomG1Settings:
    """Release defaults for the Noitom-driven G1 locomanipulation example."""

    collection_id: str = "noitom_mocap"
    max_flatbuffer_size: int = 16 * 1024
    plugin_auto_launch: bool = True
    teleoperation_active_default: bool = True
    enable_motion: bool = True
    print_period_s: float = 0.5
    draw_reference: bool = True
    draw_scale: float = 1.0
    draw_offset: tuple[float, float, float] = _NOITOM_REFERENCE_DEFAULT_OFFSET
    # Anchor the cyan skeleton to the robot pelvis instead of a fixed world offset.
    draw_pelvis_relative: bool = True
    draw_pelvis_anchor: tuple[float, float, float] = _ROBOT_PELVIS_ANCHOR
    draw_wrist_targets: bool = True
    draw_elbow_targets: bool = True
    draw_shoulder_targets: bool = True
    # Wrist + elbow + shoulder LocalFrameTasks for Pink IK (60D action).
    use_arm_ik_frame_tasks: bool = True
    retargeting: NoitomRetargetingSettings = field(
        default_factory=lambda: NoitomRetargetingSettings(
            robot_pelvis_world=np.array(_ROBOT_PELVIS_ANCHOR, dtype=np.float64),
            motion_scale=0.75,
            track_aligned_mocap_wrists=True,
            track_elbow_ik_targets=True,
            track_shoulder_ik_targets=True,
        )
    )


_FULL_BODY_BONES = (
    (BodyJoint.PELVIS, BodyJoint.SPINE1),
    (BodyJoint.SPINE1, BodyJoint.SPINE2),
    (BodyJoint.SPINE2, BodyJoint.SPINE3),
    (BodyJoint.SPINE3, BodyJoint.NECK),
    (BodyJoint.NECK, BodyJoint.HEAD),
    (BodyJoint.SPINE3, BodyJoint.LEFT_COLLAR),
    (BodyJoint.LEFT_COLLAR, BodyJoint.LEFT_SHOULDER),
    (BodyJoint.LEFT_SHOULDER, BodyJoint.LEFT_ELBOW),
    (BodyJoint.LEFT_ELBOW, BodyJoint.LEFT_WRIST),
    (BodyJoint.LEFT_WRIST, BodyJoint.LEFT_HAND),
    (BodyJoint.SPINE3, BodyJoint.RIGHT_COLLAR),
    (BodyJoint.RIGHT_COLLAR, BodyJoint.RIGHT_SHOULDER),
    (BodyJoint.RIGHT_SHOULDER, BodyJoint.RIGHT_ELBOW),
    (BodyJoint.RIGHT_ELBOW, BodyJoint.RIGHT_WRIST),
    (BodyJoint.RIGHT_WRIST, BodyJoint.RIGHT_HAND),
    (BodyJoint.PELVIS, BodyJoint.LEFT_HIP),
    (BodyJoint.LEFT_HIP, BodyJoint.LEFT_KNEE),
    (BodyJoint.LEFT_KNEE, BodyJoint.LEFT_ANKLE),
    (BodyJoint.LEFT_ANKLE, BodyJoint.LEFT_FOOT),
    (BodyJoint.PELVIS, BodyJoint.RIGHT_HIP),
    (BodyJoint.RIGHT_HIP, BodyJoint.RIGHT_KNEE),
    (BodyJoint.RIGHT_KNEE, BodyJoint.RIGHT_ANKLE),
    (BodyJoint.RIGHT_ANKLE, BodyJoint.RIGHT_FOOT),
)
DEFAULT_NOITOM_G1_SETTINGS = NoitomG1Settings()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _noitom_settings_from_env() -> NoitomG1Settings:
    return replace(
        DEFAULT_NOITOM_G1_SETTINGS,
        plugin_auto_launch=_env_bool(
            "NOITOM_MOCAP_AUTO_LAUNCH",
            DEFAULT_NOITOM_G1_SETTINGS.plugin_auto_launch,
        ),
    )


def _plugin_search_paths() -> list[Path]:
    # Six levels up is the repo root -- or the install prefix, which has the
    # same shape under examples/<name>/python/isaaccapture_examples/<name>/.
    base = Path(__file__).resolve().parents[5]
    candidates = [
        base / "plugins",
        base / "install" / "plugins",
    ]
    return [path for path in candidates if path.exists()]


def _noitom_plugin_configs(settings: NoitomG1Settings) -> list[PluginConfig]:
    if not settings.plugin_auto_launch:
        return []
    search_paths = _plugin_search_paths()
    if not search_paths:
        raise RuntimeError(
            "Noitom plugin directory not found. Run `cmake --install build` "
            "or set NOITOM_MOCAP_AUTO_LAUNCH=0 for a manually started plugin."
        )
    return [
        # Noitom plugin launch arguments live in src/plugins/noitom_mocap/plugin.yaml.
        PluginConfig(
            plugin_name=_NOITOM_PLUGIN_NAME,
            plugin_root_id=_NOITOM_PLUGIN_ROOT_ID,
            search_paths=search_paths,
        )
    ]


# Waist joints stay in IK; clip targets slightly inside URDF hard stops (not fixed narrow ranges).
_WAIST_JOINT_NAMES = frozenset(
    {"waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"}
)
# Inset from hard joint limits (rad). Small enough to avoid IK deadlock at the stops.
_WAIST_HARD_LIMIT_MARGIN_RAD = 0.10


def _build_noitom_pink_ik_action_class():
    """Lazy import so unit tests can load noitom_tasks without Isaac Lab."""
    import torch
    from isaaclab.envs.mdp.actions.pink_task_space_actions import (
        PinkInverseKinematicsAction,
    )

    class NoitomPinkInverseKinematicsAction(PinkInverseKinematicsAction):
        """Pink IK with waist joint targets clamped to teleop-safe ranges."""

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            clip_ids: list[int] = []
            lows: list[float] = []
            highs: list[float] = []
            hard_limits = self._asset.data.joint_pos_limits.torch[0]
            name_to_joint_idx = {
                name: index for index, name in enumerate(self._asset.data.joint_names)
            }
            margin = _WAIST_HARD_LIMIT_MARGIN_RAD
            for ik_index, name in enumerate(self._isaaclab_controlled_joint_names):
                if name not in _WAIST_JOINT_NAMES:
                    continue
                joint_idx = name_to_joint_idx[name]
                lo_hard = float(hard_limits[joint_idx, 0])
                hi_hard = float(hard_limits[joint_idx, 1])
                lo = lo_hard + margin
                hi = hi_hard - margin
                if lo >= hi:
                    mid = 0.5 * (lo_hard + hi_hard)
                    half = 0.45 * (hi_hard - lo_hard)
                    lo, hi = mid - half, mid + half
                clip_ids.append(ik_index)
                lows.append(lo)
                highs.append(hi)
            self._waist_ik_indices = clip_ids
            if clip_ids:
                self._waist_low = torch.tensor(lows, device=self.device).view(1, -1)
                self._waist_high = torch.tensor(highs, device=self.device).view(1, -1)
                self._waist_ik_idx = torch.tensor(
                    clip_ids, device=self.device, dtype=torch.long
                )

        def _compute_ik_solutions(self) -> torch.Tensor:
            sol = super()._compute_ik_solutions()
            if self._waist_ik_indices:
                waist = sol.index_select(1, self._waist_ik_idx)
                sol.index_copy_(
                    1,
                    self._waist_ik_idx,
                    torch.clamp(waist, self._waist_low, self._waist_high),
                )
            return sol

    return NoitomPinkInverseKinematicsAction


def _configure_noitom_pink_ik(
    env_cfg: NoitomLocomanipulationG1EnvCfg, use_arm_frames: bool
) -> None:
    """Tune Pink IK for Noitom teleop; optionally add elbow/shoulder frame tasks."""
    from isaaclab.controllers.pink_ik import LocalFrameTaskCfg, NullSpacePostureTaskCfg

    env_cfg.actions.upper_body_ik.class_type = _build_noitom_pink_ik_action_class()
    controller = env_cfg.actions.upper_body_ik.controller
    controller.show_ik_warnings = False

    if use_arm_frames:
        tasks = list(controller.variable_input_tasks)
        wrist_task_count = sum(
            1 for task in tasks if isinstance(task, LocalFrameTaskCfg)
        )
        arm_frame_tasks = [
            LocalFrameTaskCfg(
                frame=_PINK_LEFT_ELBOW_LINK,
                base_link_frame_name=_PINK_PELVIS_LINK,
                position_cost=9.0,
                orientation_cost=0.0,
                lm_damping=30.0,
                gain=0.38,
            ),
            LocalFrameTaskCfg(
                frame=_PINK_RIGHT_ELBOW_LINK,
                base_link_frame_name=_PINK_PELVIS_LINK,
                position_cost=9.0,
                orientation_cost=0.0,
                lm_damping=30.0,
                gain=0.38,
            ),
            LocalFrameTaskCfg(
                frame=_PINK_LEFT_SHOULDER_LINK,
                base_link_frame_name=_PINK_PELVIS_LINK,
                position_cost=6.0,
                orientation_cost=0.0,
                lm_damping=30.0,
                gain=0.38,
            ),
            LocalFrameTaskCfg(
                frame=_PINK_RIGHT_SHOULDER_LINK,
                base_link_frame_name=_PINK_PELVIS_LINK,
                position_cost=6.0,
                orientation_cost=0.0,
                lm_damping=30.0,
                gain=0.38,
            ),
        ]
        controller.variable_input_tasks = (
            tasks[:wrist_task_count] + arm_frame_tasks + tasks[wrist_task_count:]
        )

    for task in controller.variable_input_tasks:
        if isinstance(task, LocalFrameTaskCfg):
            frame = task.frame
            task.gain = 0.38
            task.lm_damping = 30.0
            task.orientation_cost = 0.0
            if "wrist" in frame:
                task.position_cost = 18.0
            elif "elbow" in frame:
                task.position_cost = 9.0
            elif "shoulder" in frame:
                task.position_cost = 6.0
            else:
                task.position_cost = 14.0
        elif isinstance(task, NullSpacePostureTaskCfg):
            task.cost = 0.05
            task.gain = 0.25
            task.lm_damping = 50.0


def register_tasks() -> list[str]:
    """Register the Noitom G1 locomanipulation task with Gymnasium."""
    if TASK_ID not in registry:
        gym.register(
            id=TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": ("noitom_tasks:NoitomLocomanipulationG1EnvCfg"),
            },
            disable_env_checker=True,
        )
    return []


@configclass
class NoitomLocomanipulationG1EnvCfg(LocomanipulationG1EnvCfg):
    """G1 locomanipulation config using Noitom mocap as the IsaacTeleop source."""

    def __post_init__(self) -> None:
        """Use the base scene/action config and swap in the Noitom pipeline."""
        super().__post_init__()
        settings = _noitom_settings_from_env()
        self.isaac_teleop.pipeline_builder = lambda: (
            build_noitom_g1_locomanipulation_pipeline(settings)
        )
        self.isaac_teleop.plugins = _noitom_plugin_configs(settings)
        # Pelvis fixed: agile lower-body policy otherwise crouches and breaks IK reach.
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        # Pink IK: wrist primary, elbow/shoulder secondary frame tasks.
        # Waist stays in IK (base G1_UPPER_BODY_IK_ACTION_CFG) but is soft-limited
        # after solve and biased toward neutral via NullSpacePostureTask.
        _configure_noitom_pink_ik(
            self,
            use_arm_frames=settings.use_arm_ik_frame_tasks,
        )
        self.isaac_teleop.teleoperation_active_default = (
            settings.teleoperation_active_default
        )
        self.isaac_teleop.control_channel_uuid = None
        self.isaac_teleop.app_name = "IsaacLabNoitomG1"


def G1LocomanipulationAction(*, use_arm_ik_frame_tasks: bool = True) -> TensorGroupType:
    """G1 locomanipulation action tensor type."""
    return TensorGroupType(
        "g1_locomanipulation_action",
        [
            NDArrayType(
                "action",
                shape=(g1_action_dim(use_arm_ik_frame_tasks=use_arm_ik_frame_tasks),),
                dtype=DLDataType.FLOAT,
                dtype_bits=32,
            ),
        ],
    )


class NoitomG1ActionSource(IDeviceIOSource):
    """Convert Noitom mocap frames into G1 locomanipulation wrist actions."""

    def __init__(
        self,
        name: str = "noitom_g1_action",
        settings: NoitomG1Settings = DEFAULT_NOITOM_G1_SETTINGS,
    ) -> None:
        """Initialize the Noitom DeviceIO tracker and retargeter."""
        import isaaccapture.deviceio as deviceio

        self._tracker = deviceio.FullBodyTracker()
        vendor = deviceio.TrackerVendor(
            _NOITOM_VENDOR_ID,
            {
                "collection_id": settings.collection_id,
                "max_flatbuffer_size": str(settings.max_flatbuffer_size),
            },
        )
        self._collection_id = settings.collection_id
        self._enable_motion = settings.enable_motion
        self._print_period_s = max(0.0, settings.print_period_s)
        self._last_print_s = 0.0
        self._reference_viz = _NoitomReferenceVisualizer(settings)
        self._retargeter = NoitomG1Retargeter(settings.retargeting)
        self._use_arm_ik_frame_tasks = settings.use_arm_ik_frame_tasks
        self._hold_targets = self._retargeter.current_arm_targets
        self._frame_count = 0
        self._calibration_attempts = 0
        self._no_data_count = 0
        self._first_frame_printed = False
        self._calibration_fail_count = 0
        super().__init__(name, vendor=vendor)

    def get_tracker(self):
        """Return the Noitom mocap tracker used by this source."""
        return self._tracker

    def poll_tracker(self, deviceio_session: Any) -> RetargeterIO:
        """Poll Noitom data from the active DeviceIO session."""
        tracked = self._tracker.get_body_pose(deviceio_session)
        group = TensorGroup(self.input_spec()[_FULL_BODY_INPUT])
        group[0] = tracked
        return {_FULL_BODY_INPUT: group}

    def input_spec(self) -> RetargeterIOType:
        """Declare the raw full-body DeviceIO input."""
        return {_FULL_BODY_INPUT: DeviceIOFullBodyPoseTracked()}

    def output_spec(self) -> RetargeterIOType:
        """Declare the flattened G1 action output."""
        return {
            _ACTION_OUTPUT: G1LocomanipulationAction(
                use_arm_ik_frame_tasks=self._use_arm_ik_frame_tasks
            )
        }

    def _compute_fn(
        self,
        inputs: RetargeterIO,
        outputs: RetargeterIO,
        context: ComputeContext,
    ) -> None:
        """Convert a pushed full-body frame into the G1 action tensor."""
        self._frame_count += 1

        if context.execution_events.reset:
            self._retargeter.clear_calibration()
            self._calibration_attempts = 0
            self._no_data_count = 0
            self._calibration_fail_count = 0
            print(
                "NoitomG1ActionSource: cleared retargeting calibration "
                f"collection={self._collection_id}"
            )

        # Read the raw tracked data from DeviceIO
        frame: FullBodyPose | None = inputs[_FULL_BODY_INPUT][0]

        # --- First-frame diagnostic ---
        if not self._first_frame_printed:
            self._first_frame_printed = True
            print(
                "NoitomG1ActionSource: first frame "
                f"collection={self._collection_id} "
                f"has_data={frame is not None} "
                f"motion_enabled={self._enable_motion}"
            )

        # --- No data warning ---
        if frame is None:
            self._no_data_count += 1
            if self._no_data_count == 1 or self._no_data_count % 300 == 0:
                print(
                    f"NoitomG1ActionSource: WARNING no data from tracker "
                    f"collection={self._collection_id} "
                    f"no_data_frames={self._no_data_count}/{self._frame_count} "
                    f"(is the noitom_mocap plugin running with matching collection_id?)"
                )
            # Still output hold pose even without data
            body_yaw_delta = self._retargeter.body_yaw_delta
            action = _make_action(
                self._hold_targets,
                use_arm_ik_frame_tasks=self._use_arm_ik_frame_tasks,
            )
            outputs[_ACTION_OUTPUT][0] = np.ascontiguousarray(action, dtype=np.float32)
            return

        # Reset no-data counter when we get valid frames
        self._no_data_count = 0

        if not self._enable_motion:
            body_yaw_delta = self._retargeter.body_yaw_delta
            action = _make_action(
                self._hold_targets,
                use_arm_ik_frame_tasks=self._use_arm_ik_frame_tasks,
            )
            outputs[_ACTION_OUTPUT][0] = np.ascontiguousarray(action, dtype=np.float32)
            self._reference_viz.update(frame, self._retargeter)
            self._print_status(frame, body_yaw_delta, context)
            return

        # --- Calibration / retarget phase ---
        if self._retargeter.awaiting_calibration:
            self._calibration_attempts += 1
            success = self._retargeter.calibrate(frame)
            if success:
                self._hold_targets = self._retargeter.current_arm_targets
                print(
                    "NoitomG1ActionSource: calibrated neutral pose "
                    f"collection={self._collection_id} "
                    f"attempts={self._calibration_attempts}"
                )
            else:
                self._calibration_fail_count += 1
                if (
                    self._calibration_fail_count == 1
                    or self._calibration_fail_count % 150 == 0
                ):
                    # Diagnose WHY calibration is failing
                    diag = _calibration_diagnostics(frame)
                    print(
                        f"NoitomG1ActionSource: calibration attempt "
                        f"{self._calibration_attempts} failed {diag}"
                    )
        else:
            result = self._retargeter.retarget(frame)
            if result is not None:
                self._hold_targets = result

        body_yaw_delta = self._retargeter.body_yaw_delta
        action = _make_action(
            self._hold_targets,
            use_arm_ik_frame_tasks=self._use_arm_ik_frame_tasks,
        )
        outputs[_ACTION_OUTPUT][0] = np.ascontiguousarray(action, dtype=np.float32)

        self._reference_viz.update(frame, self._retargeter)
        self._print_status(frame, body_yaw_delta, context)

    def _print_status(
        self, frame: FullBodyPose, body_yaw_delta: float, context: ComputeContext
    ) -> None:
        if self._print_period_s <= 0.0:
            return
        now_s = context.graph_time.real_time_ns * 1.0e-9
        if now_s - self._last_print_s < self._print_period_s:
            return

        self._last_print_s = now_s
        motion = "on" if self._enable_motion else "off"
        calib = "ready" if self._retargeter.is_calibrated else "awaiting_neutral"
        left_pose = self._hold_targets.left_wrist.as_action_pose()
        right_pose = self._hold_targets.right_wrist.as_action_pose()
        frame_info = ""
        if self._use_arm_ik_frame_tasks:
            left_elbow_pose = self._hold_targets.left_elbow.as_action_pose()
            right_elbow_pose = self._hold_targets.right_elbow.as_action_pose()
            left_shoulder_pose = self._hold_targets.left_shoulder.as_action_pose()
            right_shoulder_pose = self._hold_targets.right_shoulder.as_action_pose()
            frame_info = (
                f" target_left_elbow={_fmt_pose(left_elbow_pose)}"
                f" target_right_elbow={_fmt_pose(right_elbow_pose)}"
                f" target_left_shoulder={_fmt_pose(left_shoulder_pose)}"
                f" target_right_shoulder={_fmt_pose(right_shoulder_pose)}"
            )
        print(
            "NoitomG1ActionSource: "
            f"joints={_valid_joint_count(frame)}/{int(BodyJoint.NUM_JOINTS)} "
            f"motion={motion} calibrated={calib} "
            f"yaw_delta={body_yaw_delta:+.3f} "
            f"motion_scale={self._retargeter.retargeting_settings.motion_scale:.2f} "
            f"torso_yaw_influence={self._retargeter.retargeting_settings.torso_yaw_arm_influence:.2f} "
            f"target_left={_fmt_pose(left_pose)} target_right={_fmt_pose(right_pose)}"
            f"{frame_info} "
            f"{_raw_full_body_status(frame)}"
        )


def build_noitom_g1_locomanipulation_pipeline(
    settings: NoitomG1Settings = DEFAULT_NOITOM_G1_SETTINGS,
) -> OutputCombiner:
    """Build a one-source IsaacTeleop pipeline for Noitom G1 testing."""
    source = NoitomG1ActionSource(settings=settings)
    return OutputCombiner({_ACTION_OUTPUT: source.output(_ACTION_OUTPUT)})


def _make_action(
    targets: ArmIkTargets,
    *,
    use_arm_ik_frame_tasks: bool,
) -> np.ndarray:
    action = np.zeros(
        g1_action_dim(use_arm_ik_frame_tasks=use_arm_ik_frame_tasks),
        dtype=np.float32,
    )
    action[0:7] = targets.left_wrist.as_action_pose()
    action[7:14] = targets.right_wrist.as_action_pose()
    hand_offset = 14
    if use_arm_ik_frame_tasks:
        action[14:21] = targets.left_elbow.as_action_pose()
        action[21:28] = targets.right_elbow.as_action_pose()
        action[28:35] = targets.left_shoulder.as_action_pose()
        action[35:42] = targets.right_shoulder.as_action_pose()
        hand_offset = 42
    action[hand_offset : hand_offset + 14] = 0.0
    action[-4] = 0.0
    action[-3] = 0.0
    action[-2] = 0.0
    action[-1] = _LOCOMOTION_DEFAULT_HIP_HEIGHT
    return action


class _NoitomReferenceVisualizer:
    """Draw the incoming Noitom frame as a Kit debug-draw stick figure."""

    def __init__(self, settings: NoitomG1Settings) -> None:
        self._enabled = settings.draw_reference
        self._draw: Any | None = None
        self._warned = False
        self._printed_first_draw = False
        self._scale = settings.draw_scale
        self._offset = np.array(settings.draw_offset, dtype=np.float32)
        self._pelvis_relative = settings.draw_pelvis_relative
        self._pelvis_anchor = np.array(settings.draw_pelvis_anchor, dtype=np.float32)
        self._draw_wrist_targets = settings.draw_wrist_targets
        self._draw_elbow_targets = (
            settings.draw_elbow_targets and settings.use_arm_ik_frame_tasks
        )
        self._draw_shoulder_targets = (
            settings.draw_shoulder_targets and settings.use_arm_ik_frame_tasks
        )
        self._retargeting = settings.retargeting

    def update(
        self,
        frame: FullBodyPose,
        retargeter: NoitomG1Retargeter,
    ) -> None:
        if not self._enabled:
            return
        draw = self._get_draw_interface()
        if draw is None:
            return

        calib_view = retargeter.calibration_view
        draw_positions = self._reference_positions(frame, calib_view)
        starts: list[list[float]] = []
        ends: list[list[float]] = []
        colors: list[tuple[float, float, float, float]] = []
        thicknesses: list[float] = []

        for parent_index, child_index in _FULL_BODY_BONES:
            parent = draw_positions.get(int(parent_index))
            child = draw_positions.get(int(child_index))
            if parent is None or child is None:
                continue
            starts.append(parent.tolist())
            ends.append(child.tolist())
            colors.append(_NOITOM_REFERENCE_COLOR)
            thicknesses.append(_NOITOM_REFERENCE_LINE_THICKNESS)

        marker_starts, marker_ends, marker_colors = self._joint_markers(draw_positions)
        starts.extend(marker_starts)
        ends.extend(marker_ends)
        colors.extend(marker_colors)
        thicknesses.extend([_NOITOM_REFERENCE_LINE_THICKNESS] * len(marker_starts))

        if self._draw_wrist_targets:
            wrist_starts, wrist_ends, wrist_colors = self._frame_highlight_markers(
                draw_positions,
                (
                    int(BodyJoint.LEFT_WRIST),
                    int(BodyJoint.RIGHT_WRIST),
                ),
                _NOITOM_WRIST_TARGET_COLOR,
                _NOITOM_WRIST_TARGET_MARKER_SIZE,
            )
            starts.extend(wrist_starts)
            ends.extend(wrist_ends)
            colors.extend(wrist_colors)
            thicknesses.extend([_NOITOM_REFERENCE_LINE_THICKNESS] * len(wrist_starts))

        if self._draw_elbow_targets:
            elbow_starts, elbow_ends, elbow_colors = self._frame_highlight_markers(
                draw_positions,
                (
                    int(BodyJoint.LEFT_ELBOW),
                    int(BodyJoint.RIGHT_ELBOW),
                ),
                _NOITOM_ELBOW_TARGET_COLOR,
                _NOITOM_ELBOW_TARGET_MARKER_SIZE,
            )
            starts.extend(elbow_starts)
            ends.extend(elbow_ends)
            colors.extend(elbow_colors)
            thicknesses.extend([_NOITOM_REFERENCE_LINE_THICKNESS] * len(elbow_starts))

        if self._draw_shoulder_targets:
            shoulder_starts, shoulder_ends, shoulder_colors = (
                self._frame_highlight_markers(
                    draw_positions,
                    (
                        int(BodyJoint.LEFT_SHOULDER),
                        int(BodyJoint.RIGHT_SHOULDER),
                    ),
                    _NOITOM_SHOULDER_TARGET_COLOR,
                    _NOITOM_SHOULDER_TARGET_MARKER_SIZE,
                )
            )
            starts.extend(shoulder_starts)
            ends.extend(shoulder_ends)
            colors.extend(shoulder_colors)
            thicknesses.extend(
                [_NOITOM_REFERENCE_LINE_THICKNESS] * len(shoulder_starts)
            )

        draw.clear_lines()
        if starts:
            draw.draw_lines(starts, ends, colors, thicknesses)
            if not self._printed_first_draw:
                anchor = (
                    _fmt_vec(self._pelvis_anchor)
                    if self._pelvis_relative
                    else "disabled"
                )
                print(
                    "NoitomG1ActionSource: drawing Noitom reference skeleton "
                    f"segments={len(starts)} joints={len(draw_positions)} "
                    f"pelvis_relative={self._pelvis_relative} "
                    f"robot_pelvis_anchor={anchor}"
                )
                self._printed_first_draw = True
        elif not self._warned:
            print(
                "NoitomG1ActionSource: reference visualizer found no drawable "
                "Noitom bones or joints"
            )
            self._warned = True

    def _get_draw_interface(self) -> Any | None:
        if self._draw is not None:
            return self._draw
        try:
            from isaacsim.core.experimental.utils.app import enable_extension

            enable_extension("isaacsim.util.debug_draw")
            from isaacsim.util.debug_draw import _debug_draw as omni_debug_draw

            self._draw = omni_debug_draw.acquire_debug_draw_interface()
        except (ImportError, AttributeError, RuntimeError, ModuleNotFoundError):
            try:
                import omni.isaac.debug_draw._debug_draw as omni_debug_draw

                self._draw = omni_debug_draw.acquire_debug_draw_interface()
            except (
                ImportError,
                AttributeError,
                RuntimeError,
                ModuleNotFoundError,
            ) as exc:
                if not self._warned:
                    print(
                        "NoitomG1ActionSource: reference visualizer disabled "
                        f"({type(exc).__name__}: {exc})"
                    )
                    self._warned = True
                return None
        except Exception as exc:
            if not self._warned:
                print(
                    "NoitomG1ActionSource: reference visualizer disabled "
                    f"({type(exc).__name__}: {exc})"
                )
                self._warned = True
            return None
        return self._draw

    def _reference_positions(
        self,
        frame: FullBodyPose,
        calib_view: Any | None,
    ) -> dict[int, np.ndarray]:
        if self._pelvis_relative:
            rt = self._retargeting
            positions = aligned_reference_skeleton_from_frame(
                frame,
                self._pelvis_anchor,
                draw_scale=self._scale,
                calib_view=(
                    calib_view if not rt.reference_use_robot_link_lengths else None
                ),
                use_robot_link_lengths=rt.reference_use_robot_link_lengths,
                link_lengths=ReferenceSkeletonLengths.from_retargeting_settings(rt),
                length_scale=rt.reference_length_scale,
                arm_length_scale=rt.reference_arm_length_scale,
                shoulder_span_scale=rt.reference_shoulder_span_scale,
            )
            return {
                index: (pos + self._offset).astype(np.float32)
                for index, pos in positions.items()
            }

        raw_positions = _joint_position_map(frame)
        return {
            index: (
                self._offset
                + noitom_position_to_isaac(point).astype(np.float32) * self._scale
            )
            for index, point in raw_positions.items()
        }

    def _joint_markers(
        self,
        positions: dict[int, np.ndarray],
    ) -> tuple[_LineList, _LineList, _ColorList]:
        starts: _LineList = []
        ends: _LineList = []
        colors: _ColorList = []
        marker_delta_x = np.array([_NOITOM_REFERENCE_JOINT_MARKER_SIZE, 0.0, 0.0])
        marker_delta_y = np.array([0.0, _NOITOM_REFERENCE_JOINT_MARKER_SIZE, 0.0])
        marker_delta_z = np.array([0.0, 0.0, _NOITOM_REFERENCE_JOINT_MARKER_SIZE])
        for position in positions.values():
            point = position.astype(np.float32)
            for marker_delta in (marker_delta_x, marker_delta_y, marker_delta_z):
                starts.append((point - marker_delta).tolist())
                ends.append((point + marker_delta).tolist())
                colors.append(_NOITOM_REFERENCE_COLOR)
        return starts, ends, colors

    def _frame_highlight_markers(
        self,
        draw_positions: dict[int, np.ndarray],
        joint_indices: tuple[int, ...],
        color: tuple[float, float, float, float],
        marker_size: float,
    ) -> tuple[_LineList, _LineList, _ColorList]:
        """Highlight selected joints on the cyan skeleton."""
        starts: _LineList = []
        ends: _LineList = []
        colors: _ColorList = []
        marker_delta_x = np.array([marker_size, 0.0, 0.0])
        marker_delta_y = np.array([0.0, marker_size, 0.0])
        marker_delta_z = np.array([0.0, 0.0, marker_size])
        for joint_index in joint_indices:
            position = draw_positions.get(joint_index)
            if position is None:
                continue
            point = position.astype(np.float32)
            for marker_delta in (marker_delta_x, marker_delta_y, marker_delta_z):
                starts.append((point - marker_delta).tolist())
                ends.append((point + marker_delta).tolist())
                colors.append(color)
        return starts, ends, colors


def _joint_position_map(frame: FullBodyPose) -> dict[int, np.ndarray]:
    positions: dict[int, np.ndarray] = {}
    if frame.joints is None:
        return positions
    for index in range(int(BodyJoint.NUM_JOINTS)):
        position = _joint_position(frame, index)
        if position is not None:
            positions[index] = position
    return positions


def _joint_position(frame: FullBodyPose, joint_index: int) -> np.ndarray | None:
    if frame.joints is None:
        return None
    joint = frame.joints.joints(int(joint_index))
    if not joint.is_valid:
        return None
    value = _point_to_array(joint.pose.position)
    if np.all(np.isfinite(value)):
        return value
    return None


def _valid_joint_count(frame: FullBodyPose) -> int:
    if frame.joints is None:
        return 0
    count = 0
    for index in range(int(BodyJoint.NUM_JOINTS)):
        if frame.joints.joints(index).is_valid:
            count += 1
    return count


def _raw_full_body_status(frame: FullBodyPose) -> str:
    left_wrist = _joint_position(frame, BodyJoint.LEFT_WRIST)
    right_wrist = _joint_position(frame, BodyJoint.RIGHT_WRIST)
    pelvis = _joint_position(frame, BodyJoint.PELVIS)
    spine3 = _joint_position(frame, BodyJoint.SPINE3)
    left_shoulder = _joint_position(frame, BodyJoint.LEFT_SHOULDER)
    right_shoulder = _joint_position(frame, BodyJoint.RIGHT_SHOULDER)
    left_elbow = _joint_position(frame, BodyJoint.LEFT_ELBOW)
    right_elbow = _joint_position(frame, BodyJoint.RIGHT_ELBOW)

    missing = []
    if pelvis is None:
        missing.append("pelvis")
    if spine3 is None:
        missing.append("spine3")
    if left_shoulder is None:
        missing.append("l_shoulder")
    if right_shoulder is None:
        missing.append("r_shoulder")
    if left_elbow is None:
        missing.append("l_elbow")
    if right_elbow is None:
        missing.append("r_elbow")
    if left_wrist is None:
        missing.append("l_wrist")
    if right_wrist is None:
        missing.append("r_wrist")

    if missing:
        return f"upper_body_missing={','.join(missing)}"
    left_isaac = noitom_position_to_isaac(left_wrist)
    right_isaac = noitom_position_to_isaac(right_wrist)
    pelvis_isaac = noitom_position_to_isaac(pelvis)
    return (
        f"left_wrist={_fmt_vec(left_wrist)} right_wrist={_fmt_vec(right_wrist)} "
        f"isaac_left={_fmt_vec(left_isaac)} isaac_right={_fmt_vec(right_isaac)} "
        f"isaac_pelvis={_fmt_vec(pelvis_isaac)}"
    )


def _calibration_diagnostics(frame: FullBodyPose) -> str:
    """Detailed calibration diagnostics: which joints are valid/invalid."""
    pelvis = _joint_position(frame, BodyJoint.PELVIS)
    spine3 = _joint_position(frame, BodyJoint.SPINE3)
    left_shoulder = _joint_position(frame, BodyJoint.LEFT_SHOULDER)
    right_shoulder = _joint_position(frame, BodyJoint.RIGHT_SHOULDER)
    left_elbow = _joint_position(frame, BodyJoint.LEFT_ELBOW)
    right_elbow = _joint_position(frame, BodyJoint.RIGHT_ELBOW)
    left_wrist = _joint_position(frame, BodyJoint.LEFT_WRIST)
    right_wrist = _joint_position(frame, BodyJoint.RIGHT_WRIST)

    required = {
        "pelvis": pelvis,
        "spine3": spine3,
        "L_shoulder": left_shoulder,
        "R_shoulder": right_shoulder,
        "L_elbow": left_elbow,
        "R_elbow": right_elbow,
        "L_wrist": left_wrist,
        "R_wrist": right_wrist,
    }
    valid = [k for k, v in required.items() if v is not None]
    missing = [k for k, v in required.items() if v is None]
    total_joints = int(BodyJoint.NUM_JOINTS)
    all_valid = sum(
        1
        for i in range(total_joints)
        if frame.joints is not None and frame.joints.joints(i).is_valid
    )
    return (
        f"total_valid={all_valid}/{total_joints} "
        f"required_valid={len(valid)}/8 "
        f"missing=[{','.join(missing)}]"
        if missing
        else f"all_required_ok={valid}"
    )


def _point_to_array(point: Any) -> np.ndarray:
    return np.array([point.x, point.y, point.z], dtype=np.float32)


def _fmt_vec(vec: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:+.3f}" for v in vec) + "]"


def _fmt_pose(pose: np.ndarray) -> str:
    return (
        f"pos=[{pose[0]:+.3f}, {pose[1]:+.3f}, {pose[2]:+.3f}] "
        f"quat=[{pose[3]:+.3f}, {pose[4]:+.3f}, {pose[5]:+.3f}, {pose[6]:+.3f}]"
    )


__all__ = [
    "TASK_ID",
    "NoitomG1ActionSource",
    "NoitomLocomanipulationG1EnvCfg",
    "build_noitom_g1_locomanipulation_pipeline",
    "g1_action_dim",
    "register_tasks",
]
