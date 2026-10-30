# SPDX-FileCopyrightText: Copyright (c) 2026 j-mex. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register the j-mex G1 locomanipulation task and wire it to the retargeting pipeline.

Subclasses ``LocomanipulationG1EnvCfg``, adds the elbows and shoulders to its Pink IK targets,
reads the robot's shoulders, link lengths and waist off the live pinocchio model for
``jmex_g1_retargeting``, and draws that module's reference overlay. README.md ("Teleop:
driving the G1") has the launch command.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import registry
from scipy.spatial.transform import Rotation

from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.schema import BodyJoint
from isaaccapture.teleop_session_manager import PluginConfig, RetargetingExecutionConfig
from isaaclab.controllers.pink_ik import FrameTaskCfg, LocalFrameTaskCfg
from isaaclab.utils.configclass import configclass

try:
    from isaaclab_tasks.contrib.locomanip_pick_place.locomanipulation_g1_env_cfg import (
        LocomanipulationG1EnvCfg,
    )
except ModuleNotFoundError as error:
    # Isaac Lab 3.0.0 beta2 keeps it under manager_based.
    if not (error.name or "").startswith("isaaclab_tasks.contrib"):
        raise
    from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
        LocomanipulationG1EnvCfg,
    )

from .jmex_g1_retargeting import (
    ACTION_OUTPUT,
    DEFAULT_JMEX_G1_SETTINGS,
    POSE_DIM,
    FrameTargetCfg,
    G1ActionLayout,
    JmexG1ActionSource,
    JmexG1Settings,
    PelvisPose,
    ReferenceOverlay,
    arm_frame_targets,
    forearm_frame,
    info_or_print,
    left_wrist_chain,
    right_wrist_chain,
)

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.mdp.actions.pink_actions_cfg import (
        PinkInverseKinematicsActionCfg,
    )
    from isaaclab.envs.mdp.actions.pink_task_space_actions import (
        PinkInverseKinematicsAction,
    )

logger = logging.getLogger("isaaccapture.jmex.jmex_g1_tasks")

TASK_ID = "Isaac-PickPlace-Locomanipulation-G1-Jmex-Abs-v0"

# Teleoperation runs one environment: one operator produces one action, which the action
# manager broadcasts to every environment there is.
_TELEOP_ENV_INDEX = 0

_PLUGIN_NAME = "jmex"
_PLUGIN_ROOT_ID = "jmex"

# Pink's frame names, which are the URDF's and carry the asset's prefix. Isaac Sim's body
# names for the same links do not -- the two namespaces are not interchangeable, and the
# prefix is not universal within the URDF either. The wrist names are the ones Isaac Lab's
# G1 pink controller config already targets (isaaclab_tasks/.../configs/pink_controller_cfg.py);
# the shoulder and elbow links follow the same naming.
_PINK_PELVIS_LINK = "g1_29dof_with_hand_rev_1_0_pelvis"
_PINK_LEFT_SHOULDER_LINK = "g1_29dof_with_hand_rev_1_0_left_shoulder_pitch_link"
_PINK_LEFT_ELBOW_LINK = "g1_29dof_with_hand_rev_1_0_left_elbow_link"
_PINK_LEFT_WRIST_LINK = "g1_29dof_with_hand_rev_1_0_left_wrist_yaw_link"
_PINK_RIGHT_SHOULDER_LINK = "g1_29dof_with_hand_rev_1_0_right_shoulder_pitch_link"
_PINK_RIGHT_ELBOW_LINK = "g1_29dof_with_hand_rev_1_0_right_elbow_link"
_PINK_RIGHT_WRIST_LINK = "g1_29dof_with_hand_rev_1_0_right_wrist_yaw_link"
# The waist joints the upper body tips on. The yaw joint only reports how far its axis passes
# from the point these two share -- a few millimeters on the G1, which the IK takes up.
_PINK_WAIST_ROLL_JOINT = "waist_roll_joint"
_PINK_WAIST_PITCH_JOINT = "waist_pitch_joint"
_PINK_WAIST_YAW_JOINT = "waist_yaw_joint"
# Further apart than this, the roll and pitch axes do not cross and there is no one point the
# upper body tips about.
_WAIST_AXES_CROSS_WITHIN_M = 1e-3
# Below this, |roll x pitch| says the two axes run parallel.
_WAIST_AXES_PARALLEL_BELOW = 1e-6

# Each arm's body slots and the Pink frames standing for them, one for one, shoulder first. The
# action is laid out in the controller's task order, which this table does not set.
#
# The wrist frame is wrist_yaw_link, after all three of the wrist's roll, pitch and yaw joints.
# The operator's WRIST slot observes all three, so Isaac Lab's scalar orientation_cost applies
# to every axis.
_ARM_SLOTS_AND_FRAMES = (
    (
        left_wrist_chain(),
        (_PINK_LEFT_SHOULDER_LINK, _PINK_LEFT_ELBOW_LINK, _PINK_LEFT_WRIST_LINK),
    ),
    (
        right_wrist_chain(),
        (_PINK_RIGHT_SHOULDER_LINK, _PINK_RIGHT_ELBOW_LINK, _PINK_RIGHT_WRIST_LINK),
    ),
)

# The frame tasks put behind the base task's wrists, in this order: the frame, the wrist task
# whose gain and damping it copies, and its share of that wrist's position cost, so the IK
# still places the hand first. Position only: the chain places these frames but does not
# turn them.
_ADDED_FRAME_TASKS = (
    (_PINK_LEFT_ELBOW_LINK, _PINK_LEFT_WRIST_LINK, 1.0 / 2.0),
    (_PINK_RIGHT_ELBOW_LINK, _PINK_RIGHT_WRIST_LINK, 1.0 / 2.0),
    (_PINK_LEFT_SHOULDER_LINK, _PINK_LEFT_WRIST_LINK, 1.0 / 3.0),
    (_PINK_RIGHT_SHOULDER_LINK, _PINK_RIGHT_WRIST_LINK, 1.0 / 3.0),
)

# The body slots' joint tree, as chains out from the root; consecutive slots are one bone.
_BODY_CHAINS = (
    (
        BodyJoint.PELVIS,
        BodyJoint.SPINE1,
        BodyJoint.SPINE2,
        BodyJoint.SPINE3,
        BodyJoint.NECK,
        BodyJoint.HEAD,
    ),
    (
        BodyJoint.SPINE3,
        BodyJoint.LEFT_COLLAR,
        BodyJoint.LEFT_SHOULDER,
        BodyJoint.LEFT_ELBOW,
        BodyJoint.LEFT_WRIST,
        BodyJoint.LEFT_HAND,
    ),
    (
        BodyJoint.SPINE3,
        BodyJoint.RIGHT_COLLAR,
        BodyJoint.RIGHT_SHOULDER,
        BodyJoint.RIGHT_ELBOW,
        BodyJoint.RIGHT_WRIST,
        BodyJoint.RIGHT_HAND,
    ),
    (
        BodyJoint.PELVIS,
        BodyJoint.LEFT_HIP,
        BodyJoint.LEFT_KNEE,
        BodyJoint.LEFT_ANKLE,
        BodyJoint.LEFT_FOOT,
    ),
    (
        BodyJoint.PELVIS,
        BodyJoint.RIGHT_HIP,
        BodyJoint.RIGHT_KNEE,
        BodyJoint.RIGHT_ANKLE,
        BodyJoint.RIGHT_FOOT,
    ),
)

# The overlay's two colors (RGBA): blue is the body pose as it arrived, orange the arms
# rebuilt from it. The README says how to read them.
_RECEIVED_COLOR = (0.25, 0.65, 1.0, 1.0)
_REBUILT_COLOR = (1.0, 0.55, 0.1, 1.0)
# Debug draw sizes are screen pixels, so they stay legible however far the camera is.
_LINE_WIDTH_PX = 2.0
_JOINT_POINT_PX = 6.0
_IK_TARGET_POINT_PX = 14.0

# How far along +Y, in meters, the packing table and the steering wheel on it are parked. Both
# move together, so the wheel still rests on the table.
_PARK_OFFSET_Y = 5.0


class _JmexRuntime:
    """What the pipeline needs from the running robot: its layout, env and live pelvis.

    The layout is only measurable once Isaac Lab has built the Pink IK controller, and the
    pipeline is built later still, when the teleop session starts.
    """

    def __init__(self) -> None:
        self._layout: G1ActionLayout | None = None
        self._env: Any | None = None
        self._pelvis = PelvisPose(
            position=np.zeros(3), orientation_xyzw=np.array([0.0, 0.0, 0.0, 1.0])
        )

    def publish_env(self, env: Any) -> None:
        self._env = env

    def env(self) -> Any | None:
        return self._env

    def env_origin(self) -> np.ndarray:
        """The teleoperated environment's origin in the simulation world."""
        if self._env is None:
            return np.zeros(3)
        origin = self._env.scene.env_origins[_TELEOP_ENV_INDEX]
        return origin.detach().cpu().numpy().astype(np.float64)

    def publish_layout(self, layout: G1ActionLayout) -> None:
        self._layout = layout

    def layout(self) -> G1ActionLayout:
        """The action layout, sized from the robot the environment actually built."""
        if self._layout is None:
            raise RuntimeError(
                "The j-mex pipeline was built before the Pink IK action term measured the "
                "robot. Launch through the registered task id so the environment is "
                f"constructed first: --task {TASK_ID}"
            )
        return self._layout

    def publish_pelvis(
        self, position: np.ndarray, orientation_xyzw: np.ndarray
    ) -> None:
        self._pelvis = PelvisPose(
            position=np.asarray(position, dtype=np.float64),
            orientation_xyzw=np.asarray(orientation_xyzw, dtype=np.float64),
        )

    def pelvis(self) -> PelvisPose:
        return self._pelvis


_RUNTIME = _JmexRuntime()


class _ReferenceOverlayDrawer:
    """Draws each frame's ``ReferenceOverlay`` with Kit's debug draw.

    Where there is no Kit debug draw -- a visualizer other than Kit, or no extension -- or
    drawing fails, it says so once and draws nothing more; the task runs the same either way.
    """

    def __init__(self) -> None:
        self._draw: Any | None = None
        self._off = False

    def __call__(self, overlay: ReferenceOverlay) -> None:
        draw = self._debug_draw()
        if draw is None:
            return
        # The overlay is in the per-env world frame; debug draw is in the simulation's.
        origin = _RUNTIME.env_origin()
        starts: list[list[float]] = []
        ends: list[list[float]] = []
        line_colors: list[tuple[float, ...]] = []
        points: list[list[float]] = []
        point_colors: list[tuple[float, ...]] = []
        point_sizes: list[float] = []

        def line(start: np.ndarray, end: np.ndarray, color: tuple[float, ...]) -> None:
            starts.append((start + origin).tolist())
            ends.append((end + origin).tolist())
            line_colors.append(color)

        def point(at: np.ndarray, color: tuple[float, ...], size: float) -> None:
            points.append((at + origin).tolist())
            point_colors.append(color)
            point_sizes.append(size)

        skeleton = overlay.skeleton
        for chain in _BODY_CHAINS:
            for parent, child in zip(chain, chain[1:]):
                if int(parent) in skeleton and int(child) in skeleton:
                    line(skeleton[int(parent)], skeleton[int(child)], _RECEIVED_COLOR)
        for at in skeleton.values():
            point(at, _RECEIVED_COLOR, _JOINT_POINT_PX)
        for chain in overlay.chains.values():
            for start, end in zip(chain, chain[1:]):
                line(start, end, _REBUILT_COLOR)
            for at in chain[:-1]:
                point(at, _REBUILT_COLOR, _JOINT_POINT_PX)
            point(chain[-1], _REBUILT_COLOR, _IK_TARGET_POINT_PX)

        try:
            draw.clear_lines()
            draw.clear_points()
            if starts:
                draw.draw_lines(
                    starts, ends, line_colors, [_LINE_WIDTH_PX] * len(starts)
                )
            draw.draw_points(points, point_colors, point_sizes)
        except Exception as exc:
            self._turn_off(f"drawing failed ({type(exc).__name__}: {exc})")

    def _debug_draw(self) -> Any | None:
        if self._draw is not None or self._off:
            return self._draw
        try:
            import omni.kit.app

            omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
                "isaacsim.util.debug_draw", True
            )
            from isaacsim.util.debug_draw import _debug_draw

            draw = _debug_draw.acquire_debug_draw_interface()
        # Outside Kit any of the three steps can fail, each in its own way.
        except Exception as exc:
            self._turn_off(f"no Kit debug draw ({type(exc).__name__}: {exc})")
            return None
        if draw is None:
            self._turn_off("no Kit debug draw")
            return None
        self._draw = draw
        info_or_print(
            logger,
            "drawing the reference overlay -- blue is the body pose as received, in the "
            "operator's proportions, and is not meant to line up with the robot; orange is each "
            "arm rebuilt in the robot's proportions, and its large dots are what the IK is told "
            "to reach -- the robot's wrist, elbow and shoulder should sit on them.",
        )
        return draw

    def _turn_off(self, reason: str) -> None:
        self._off = True
        self._draw = None
        logger.warning("%s; the reference overlay is off", reason)


@dataclass(frozen=True)
class JmexG1TaskSettings:
    """Everything the task reads at startup."""

    device: JmexG1Settings = DEFAULT_JMEX_G1_SETTINGS
    teleoperation_active_default: bool = True


DEFAULT_JMEX_G1_TASK_SETTINGS = JmexG1TaskSettings()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _settings_from_env() -> JmexG1TaskSettings:
    default = DEFAULT_JMEX_G1_TASK_SETTINGS
    device = replace(
        default.device,
        collection_id=os.environ.get(
            "JMEX_COLLECTION_ID", default.device.collection_id
        ),
        channel=int(os.environ.get("JMEX_CHANNEL", default.device.channel)),
        plugin_auto_launch=_env_bool(
            "JMEX_PLUGIN_AUTO_LAUNCH", default.device.plugin_auto_launch
        ),
    )
    return replace(default, device=device)


def _plugin_search_paths() -> list[Path]:
    # The repository root: this file sits in examples/jmex/python/isaaccapture_examples/jmex/.
    base = Path(__file__).resolve().parents[5]
    candidates = [base / "plugins", base / "install" / "plugins"]
    return [path for path in candidates if path.exists()]


def _plugin_configs(settings: JmexG1TaskSettings) -> list[PluginConfig]:
    if not settings.device.plugin_auto_launch:
        return []
    search_paths = _plugin_search_paths()
    if not search_paths:
        raise RuntimeError(
            "jmex plugin directory not found. Run `cmake --install build`, or set "
            "JMEX_PLUGIN_AUTO_LAUNCH=0 to drive a manually started plugin."
        )
    return [
        # The rest of the receiver's settings live in src/plugins/jmex/plugin.yaml. These
        # two are appended after it, and the last value of a flag wins.
        PluginConfig(
            plugin_name=_PLUGIN_NAME,
            plugin_root_id=_PLUGIN_ROOT_ID,
            search_paths=search_paths,
            plugin_args=[
                f"--channel={settings.device.channel}",
                f"--collection-id={settings.device.collection_id}",
            ],
            # Refuse to start rather than wait for a pairing that cannot happen: a build without
            # the SDK installs the other plugins but not this one.
            required=True,
        )
    ]


def _measure_frame_targets(
    pink_configuration: Any, frame_names: list[str]
) -> tuple[FrameTargetCfg, ...]:
    """Read each named frame task's arm off Pink's pinocchio model, in the order named.

    Reads a private ``Data`` at the model's rest configuration, leaving the controller's live
    state alone. Each shoulder read there is its arm's anchor before the operator's torso pose
    turns it about the waist (``_measure_waist_pivot``).
    """
    import pinocchio as pin

    model, data = _rest_kinematics(pink_configuration)

    def placement(name: str) -> pin.SE3:
        if not model.existFrame(name):
            raise RuntimeError(
                f"Frame '{name}' is not in the Pink kinematics model. Pink uses the URDF's "
                "prefixed names, which are not Isaac Sim's body names."
            )
        return data.oMf[model.getFrameId(name)]

    def axis_moving(name: str) -> np.ndarray:
        """World axis of the joint that moves frame ``name``: its Jacobian's angular column."""
        joint = model.frames[model.getFrameId(name)].parentJoint
        jacobian = pin.getJointJacobian(
            model, data, joint, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        return np.asarray(jacobian[3:, model.joints[joint].idx_v], dtype=np.float64)

    pelvis = placement(_PINK_PELVIS_LINK)
    by_frame: dict[str, FrameTargetCfg] = {}
    for chain, frames in _ARM_SLOTS_AND_FRAMES:
        placements = [placement(frame) for frame in frames]
        lengths = tuple(
            float(np.linalg.norm(nxt.translation - cur.translation))
            for cur, nxt in zip(placements, placements[1:])
        )
        # The action term expresses every target in the pelvis frame before handing it to
        # the task, so the anchor, the rest poses and the forearm are measured there too.
        at_rest = [pelvis.actInv(p) for p in placements]
        shoulder, elbow, wrist = at_rest
        forearm_in_task = Rotation.from_matrix(wrist.rotation).inv() * forearm_frame(
            shoulder.translation,
            elbow.translation,
            wrist.translation,
            elbow_axis=pelvis.rotation.T @ axis_moving(frames[1]),
        )
        arm = arm_frame_targets(
            chain,
            anchor_position_pelvis=shoulder.translation,
            robot_segment_lengths=lengths,
            rest_poses_pelvis=[
                (rest.translation, Rotation.from_matrix(rest.rotation).as_quat())
                for rest in at_rest
            ],
            forearm_frame_in_task_xyzw=forearm_in_task.as_quat(),
        )
        by_frame.update(zip(frames, arm))
        anchor_m = [round(float(v), 4) for v in shoulder.translation]
        axes = forearm_in_task.as_matrix()
        info_or_print(
            logger,
            "robot %s hangs from %s m (pelvis frame), chain %s m (reach %s m from there); in "
            "its task frame the forearm runs %s and the elbow hinges about %s",
            BodyJoint(chain[-1]).name.lower(),
            anchor_m,
            [round(v, 4) for v in lengths],
            round(sum(lengths), 4),
            np.round(axes[:, 0], 3).tolist(),
            np.round(axes[:, 1], 3).tolist(),
        )
    unknown = [name for name in frame_names if name not in by_frame]
    if unknown:
        raise RuntimeError(
            f"The Pink controller has frame tasks on {unknown}, which no body slot drives. "
            "Each frame task takes seven numbers of the action, so none can be left out."
        )
    return tuple(by_frame[name] for name in frame_names)


def _rest_kinematics(pink_configuration: Any) -> tuple[Any, Any]:
    """Pink's pinocchio model, and a private ``Data`` for it at the model's rest configuration."""
    import pinocchio as pin

    model = pink_configuration.full_model
    data = model.createData()
    q_rest = np.asarray(pink_configuration.robot_wrapper.q0, dtype=np.float64)
    pin.forwardKinematics(model, data, q_rest)
    pin.updateFramePlacements(model, data)
    pin.computeJointJacobians(model, data, q_rest)
    return model, data


def _measure_waist_pivot(pink_configuration: Any) -> tuple[float, float, float]:
    """Where the waist's roll and pitch axes cross, in the pelvis frame at rest.

    It is the point the upper body tips about, measured off the same rest configuration as the
    shoulders and the link lengths.
    """
    import pinocchio as pin

    model, data = _rest_kinematics(pink_configuration)
    pelvis = data.oMf[model.getFrameId(_PINK_PELVIS_LINK)]

    def axis_line(name: str) -> tuple[np.ndarray, np.ndarray]:
        """A point on joint ``name``'s axis and the axis's direction, in the pelvis frame."""
        if not model.existJointName(name):
            raise RuntimeError(
                f"Joint '{name}' is not in the Pink kinematics model, so there is no waist "
                "for the operator's torso to turn."
            )
        joint = model.getJointId(name)
        jacobian = pin.getJointJacobian(
            model, data, joint, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )
        direction = np.asarray(
            jacobian[3:, model.joints[joint].idx_v], dtype=np.float64
        )
        return (
            pelvis.actInv(data.oMi[joint]).translation,
            pelvis.rotation.T @ (direction / np.linalg.norm(direction)),
        )

    roll_at, roll = axis_line(_PINK_WAIST_ROLL_JOINT)
    pitch_at, pitch = axis_line(_PINK_WAIST_PITCH_JOINT)
    square = np.cross(roll, pitch)
    if np.linalg.norm(square) < _WAIST_AXES_PARALLEL_BELOW:
        raise RuntimeError(
            "The waist's roll and pitch axes are parallel, so they share no point to tip "
            "about."
        )
    # The two points, one on each axis, joined by a line square to both.
    along = np.linalg.solve(np.column_stack([roll, -pitch, square]), pitch_at - roll_at)
    on_roll = roll_at + along[0] * roll
    on_pitch = pitch_at + along[1] * pitch
    if np.linalg.norm(on_roll - on_pitch) > _WAIST_AXES_CROSS_WITHIN_M:
        raise RuntimeError(
            "The waist's roll and pitch axes pass "
            f"{np.linalg.norm(on_roll - on_pitch) * 1000.0:.1f} mm apart, so there is no one "
            "point the upper body tips about."
        )
    pivot = (on_roll + on_pitch) / 2.0
    yaw_at, yaw = axis_line(_PINK_WAIST_YAW_JOINT)
    off_yaw = (pivot - yaw_at) - np.dot(pivot - yaw_at, yaw) * yaw

    def limits_deg(name: str) -> str:
        """Joint ``name``'s range in the model the IK solves on, which is what bounds it."""
        index = model.joints[model.getJointId(name)].idx_q
        lower, upper = np.degrees(
            [model.lowerPositionLimit[index], model.upperPositionLimit[index]]
        )
        return f"[{lower:+.1f}, {upper:+.1f}] deg"

    info_or_print(
        logger,
        "robot upper body turns about %s m (pelvis frame), where the waist's roll and pitch "
        "axes cross; the yaw axis passes %.1f mm from it, which the IK takes up. The IK keeps "
        "the waist's pitch within %s, its roll within %s and its yaw within %s; a torso pose "
        "past them is not followed further",
        [round(float(v), 4) for v in pivot],
        np.linalg.norm(off_yaw) * 1000.0,
        limits_deg(_PINK_WAIST_PITCH_JOINT),
        limits_deg(_PINK_WAIST_ROLL_JOINT),
        limits_deg(_PINK_WAIST_YAW_JOINT),
    )
    return tuple(float(v) for v in pivot)


def _insert_arm_frame_tasks(controller: Any) -> None:
    """Put the elbow and shoulder tasks right behind the controller's own frame tasks."""
    tasks = list(controller.variable_input_tasks)
    leading = _leading_frame_tasks(tasks, (FrameTaskCfg, LocalFrameTaskCfg))
    by_frame = {task.frame: task for task in leading}
    added = []
    for frame, wrist_frame, share in _ADDED_FRAME_TASKS:
        wrist = by_frame.get(wrist_frame)
        if not isinstance(wrist, LocalFrameTaskCfg):
            raise RuntimeError(
                f"The base Pink controller has no leading LocalFrameTask on {wrist_frame}, "
                f"which the task on {frame} takes its settings from."
            )
        added.append(
            LocalFrameTaskCfg(
                frame=frame,
                base_link_frame_name=wrist.base_link_frame_name,
                position_cost=float(wrist.position_cost) * share,
                orientation_cost=0.0,
                lm_damping=wrist.lm_damping,
                gain=wrist.gain,
            )
        )
    controller.variable_input_tasks = leading + added + tasks[len(leading) :]


def _leading_frame_tasks(
    tasks: list[Any], frame_task_cls: type | tuple[type, ...]
) -> list[Any]:
    """The controller's frame tasks, which must be a prefix of its task list.

    The action term locates each task's slot with ``enumerate`` over the whole list and
    does not compress the index, so a frame task sitting behind a non-frame one reads the
    wrong seven numbers first and only raises later, if at all.
    """
    frame_indices = [
        index for index, task in enumerate(tasks) if isinstance(task, frame_task_cls)
    ]
    if frame_indices != list(range(len(frame_indices))):
        raise RuntimeError(
            "Pink frame tasks must occupy the front of variable_input_tasks; found them "
            f"at {frame_indices} of {len(tasks)} tasks."
        )
    return tasks[: len(frame_indices)]


def _assert_action_dim(layout: G1ActionLayout) -> None:
    """Check the pipeline's action length against the environment's, before the first frame.

    Unchecked, a mismatch surfaces on the first step as a bare shape ``ValueError`` from the
    action manager, which does not say which side is wrong. Skipped without a running
    environment, as in the tests.
    """
    env = _RUNTIME.env()
    if env is None:
        return
    total = env.action_manager.total_action_dim
    if layout.action_dim != total:
        raise RuntimeError(
            "j-mex action size disagrees with the environment: the pipeline produces "
            f"{layout.action_dim} from {len(layout.frame_targets)} frame tasks, "
            f"{layout.num_hand_joints} finger joints and {layout.num_locomotion} "
            f"locomotion elements, but the action manager expects {total}. A task added to "
            "the Pink controller without a matching frame target lands here."
        )


def _build_jmex_pink_ik_action_class() -> type[PinkInverseKinematicsAction]:
    """Built lazily, so importing this module does not pull in Isaac Lab's action stack."""
    from isaaclab.envs.mdp.actions.pink_task_space_actions import (
        PinkInverseKinematicsAction,
    )

    # The class the action term counts to give a task seven numbers of the action.
    from pink.tasks import FrameTask

    class JmexPinkInverseKinematicsAction(PinkInverseKinematicsAction):
        """Isaac Lab's Pink IK action, plus the two things the j-mex pipeline needs from it.

        It measures the robot once at startup and republishes the pelvis's live world pose
        every step. Both go to ``_RUNTIME``; neither changes how the IK behaves.
        """

        def __init__(
            self, cfg: PinkInverseKinematicsActionCfg, env: ManagerBasedEnv
        ) -> None:
            super().__init__(cfg, env)
            controller = self._ik_controllers[0]
            frame_tasks = _leading_frame_tasks(
                controller.cfg.variable_input_tasks, FrameTask
            )
            _RUNTIME.publish_env(env)
            _RUNTIME.publish_layout(
                G1ActionLayout(
                    frame_targets=_measure_frame_targets(
                        controller.pink_configuration,
                        [task.frame for task in frame_tasks],
                    ),
                    waist_pivot_pelvis=_measure_waist_pivot(
                        controller.pink_configuration
                    ),
                )
            )
            self._publish_pelvis()

        def _publish_pelvis(self) -> None:
            """Hand the pipeline the pelvis in the per-env world frame the action is read in.

            The action term subtracts the env origin from the pelvis before inverting it, so
            subtract it here too; otherwise every target is off by the env origin, which is
            zero for a single environment but not in general.
            """
            data = self._env.scene[self.cfg.controller.articulation_name].data
            pose = data.body_link_pose_w.torch[_TELEOP_ENV_INDEX, self._base_link_idx]
            pose = pose.detach().cpu().numpy()
            _RUNTIME.publish_pelvis(pose[:3] - _RUNTIME.env_origin(), pose[3:7])

        def process_actions(self, actions: torch.Tensor) -> None:
            # Read before the base class consumes the action, so the pose the pipeline
            # composes onto is at most one step behind the pose it is decomposed from.
            self._publish_pelvis()
            super().process_actions(actions)

    return JmexPinkInverseKinematicsAction


def build_jmex_g1_locomanipulation_pipeline(
    settings: JmexG1TaskSettings = DEFAULT_JMEX_G1_TASK_SETTINGS,
) -> OutputCombiner:
    """Build the one-source Isaac Teleop pipeline this task takes its action from."""
    layout = _RUNTIME.layout()
    _assert_action_dim(layout)
    info_or_print(
        logger,
        "the action is %d long -- %d for each of %s, then %d fingers and %d locomotion",
        layout.action_dim,
        POSE_DIM,
        ", ".join(target.name for target in layout.frame_targets),
        layout.num_hand_joints,
        layout.num_locomotion,
    )
    source = JmexG1ActionSource(
        layout=layout,
        pelvis_pose_provider=_RUNTIME.pelvis,
        settings=settings.device,
        overlay_sink=_ReferenceOverlayDrawer(),
    )
    return OutputCombiner({ACTION_OUTPUT: source.output(ACTION_OUTPUT)})


def register_tasks() -> None:
    """Register the j-mex G1 locomanipulation task with Gymnasium.

    It parses no arguments and leaves them all to the launcher.
    """
    if TASK_ID not in registry:
        gym.register(
            id=TASK_ID,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={"env_cfg_entry_point": f"{__name__}:JmexLocomanipulationG1EnvCfg"},
            disable_env_checker=True,
        )


@configclass
class JmexLocomanipulationG1EnvCfg(LocomanipulationG1EnvCfg):
    """Isaac Lab's G1 locomanipulation task, driven from a j-mex AgileMaster suit.

    The elbows and shoulders join the wrists as Pink IK tasks; the factory tasks and their weights,
    the finger channel and the lower-body policy stay as Isaac Lab ships them.
    """

    def __post_init__(self) -> None:
        """Take the base configuration first, then swap in the j-mex pipeline."""
        super().__post_init__()
        settings = _settings_from_env()

        # Without this the agile lower-body policy crouches over the first steps, and the
        # IK then cannot reach the targets it is being given.
        self.scene.robot.spawn.articulation_props.fix_root_link = True

        # The robot rests with its hands over the table, and hands following an operator's
        # lowered arms turn down into it and stay caught there. Park the table and the wheel
        # instead of deleting them: the observations read the wheel.
        for asset in (self.scene.packing_table, self.scene.object):
            x, y, z = asset.init_state.pos
            asset.init_state.pos = (x, y + _PARK_OFFSET_Y, z)
        # The wheel is now more than 1 m from the robot, which ends the episode on the first
        # step, and the pick-and-place checks have nothing left to judge. The time limit would
        # reset the robot every 20 s.
        self.terminations.object_dropping = None
        self.terminations.object_too_far = None
        self.terminations.success = None
        self.terminations.time_out = None

        # Seed the pelvis from the spawn pose, so the frames before the first physics step
        # already carry the robot's yaw instead of an assumed identity. `rot` is xyzw.
        _RUNTIME.publish_pelvis(
            np.asarray(self.scene.robot.init_state.pos, dtype=np.float64),
            np.asarray(self.scene.robot.init_state.rot, dtype=np.float64),
        )

        _insert_arm_frame_tasks(self.actions.upper_body_ik.controller)
        self.actions.upper_body_ik.class_type = _build_jmex_pink_ik_action_class()

        self.isaac_teleop.pipeline_builder = lambda: (
            build_jmex_g1_locomanipulation_pipeline(settings)
        )
        self.isaac_teleop.plugins = _plugin_configs(settings)
        self.isaac_teleop.retargeting_execution = RetargetingExecutionConfig(
            mode="sync"
        )
        self.isaac_teleop.teleoperation_active_default = (
            settings.teleoperation_active_default
        )
        self.isaac_teleop.control_channel_uuid = None
        self.isaac_teleop.app_name = "IsaacLabJmexG1"


__all__ = [
    "TASK_ID",
    "JmexG1TaskSettings",
    "JmexLocomanipulationG1EnvCfg",
    "build_jmex_g1_locomanipulation_pipeline",
    "register_tasks",
]
