# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""P7 bimanual teleoperation with RealHand L6, O6, or L20 hands.

The pipeline emits two absolute end-effector poses followed by the selected hand model's
joint commands. An Isaac Lab environment consumes the first 14 values with Pink IK and sends
the remaining values to the hand actuators.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeters import (
    ControllerTriggerRealHandRetargeter,
    ControllerTriggerRealHandRetargeterConfig,
    RealHandFFGGloveRetargeter,
    RealHandFFGGloveRetargeterConfig,
    P7ControllerPoseRetargeter,
    P7HandPoseRetargeter,
    P7WorkspacePoseConfig,
    RealHandHandTrackingRetargeter,
    RealHandHandTrackingRetargeterConfig,
    TensorReorderer,
)
from isaaccapture.retargeters.realhand import (
    REALHAND_FFG_GLOVE_SENSOR_NAMES,
    get_realhand_profile,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    HandsSource,
    JointStateSource,
)
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.teleop_session_manager import (
    PluginConfig,
    TeleopSession,
    TeleopSessionConfig,
)

_LEFT_EE = [
    "l_pos_x",
    "l_pos_y",
    "l_pos_z",
    "l_quat_x",
    "l_quat_y",
    "l_quat_z",
    "l_quat_w",
]
_RIGHT_EE = [
    "r_pos_x",
    "r_pos_y",
    "r_pos_z",
    "r_quat_x",
    "r_quat_y",
    "r_quat_z",
    "r_quat_w",
]

_HOME_POSES = {
    "l6": {
        "left": (
            (-0.41977179, -0.00000670, 0.72252082),
            (-0.70637390, 0.70783417, 0.00075116, 0.00247797),
        ),
        "right": (
            (0.41976724, -0.00000958, 0.71870925),
            (-0.70629016, 0.70773326, -0.01312369, 0.00977843),
        ),
    },
    "o6": {
        "left": (
            (-0.41990048, 0.00000132, 0.72069988),
            (-0.70637390, 0.70783417, 0.00075116, 0.00247797),
        ),
        "right": (
            (0.41989615, -0.00000103, 0.72070892),
            (-0.70635324, 0.70785451, -0.00077004, -0.00254952),
        ),
    },
    "l20": {
        "left": (
            (-0.41980995, 0.00017019, 0.68370038),
            (-0.70637390, 0.70783417, 0.00075116, 0.00247797),
        ),
        "right": (
            (0.41980285, -0.00017463, 0.68370944),
            (-0.70635324, 0.70785451, -0.00077004, -0.00254952),
        ),
    },
}

_INPUT_CENTERS = {
    "left": (-0.25, 0.0, 1.10),
    "right": (0.25, 0.0, 1.10),
}
_WORKSPACE_CENTERS = {
    "left": (-0.25, 0.0, 0.82),
    "right": (0.25, 0.0, 0.82),
}
_REALHAND_FFG_GLOVE_HAND_OFFSETS = {
    "l6": (0.115, 0.0, -0.075),
    "o6": (0.110, 0.0, -0.065),
    "l20": (0.115, 0.0, -0.090),
}


def _home_rpy(model: str, side: str) -> tuple[float, float, float]:
    quaternion = _HOME_POSES[model][side][1]
    return tuple(Rotation.from_quat(quaternion).as_euler("XYZ", degrees=True))


def _realhand_ffg_glove_rpy(model: str, side: str) -> tuple[float, float, float]:
    base = Rotation.from_euler("XYZ", _home_rpy(model, side), degrees=True)
    correction = Rotation.from_euler("XYZ", (-90.0, 0.0, 90.0), degrees=True)
    fingertip_correction = (
        90.0 if model in ("l6", "o6") else 180.0 if side == "right" else 0.0
    )
    rotation = (
        base * correction * Rotation.from_euler("X", fingertip_correction, degrees=True)
    )
    return tuple(rotation.as_euler("XYZ", degrees=True))


def _pose_config(model: str, side: str, mode: str) -> P7WorkspacePoseConfig:
    home_position, home_rotation = _HOME_POSES[model][side]
    uses_realhand_ffg_glove = mode == "ffg"
    return P7WorkspacePoseConfig(
        input_device=(
            ControllersSource.LEFT if side == "left" else ControllersSource.RIGHT
        )
        if mode != "handtracking"
        else (HandsSource.LEFT if side == "left" else HandsSource.RIGHT),
        fallback_position=home_position,
        fallback_rotation=home_rotation,
        input_center=(
            _WORKSPACE_CENTERS[side]
            if uses_realhand_ffg_glove
            else _INPUT_CENTERS[side]
        ),
        workspace_center=_WORKSPACE_CENTERS[side],
        position_scale=(1.0, 1.0, 1.0),
        max_delta=(0.65, 0.65, 0.75),
        rotation_offset_rpy_deg=(
            _realhand_ffg_glove_rpy(model, side)
            if uses_realhand_ffg_glove
            else _home_rpy(model, side)
        ),
        position_offset_local=(
            _REALHAND_FFG_GLOVE_HAND_OFFSETS[model]
            if uses_realhand_ffg_glove
            else (0.0, 0.0, 0.0)
        ),
        max_position_step_m=0.25,
        position_smoothing_alpha=1.0,
        orientation_smoothing_alpha=0.85,
        max_orientation_step_deg=90.0,
    )


def build_pipeline(mode: str, model: str, calibration: Path | None):
    """Build the official DeviceIO -> retargeters -> flat P7 action graph."""
    profile = get_realhand_profile(model)
    left_joint_names = profile.joint_names("left")
    right_joint_names = profile.joint_names("right")

    if mode == "handtracking":
        source = HandsSource(name="hands")
        left_arm = P7HandPoseRetargeter(
            _pose_config(model, "left", mode), name="p7_left_arm"
        )
        right_arm = P7HandPoseRetargeter(
            _pose_config(model, "right", mode), name="p7_right_arm"
        )
        left_hand = RealHandHandTrackingRetargeter(
            RealHandHandTrackingRetargeterConfig(
                input_device=HandsSource.LEFT,
                joint_names=left_joint_names,
                side="left",
                hand_model=model,
            ),
            name=f"{model}_left_hand",
        )
        right_hand = RealHandHandTrackingRetargeter(
            RealHandHandTrackingRetargeterConfig(
                input_device=HandsSource.RIGHT,
                joint_names=right_joint_names,
                side="right",
                hand_model=model,
            ),
            name=f"{model}_right_hand",
        )
        left_input = source.output(HandsSource.LEFT)
        right_input = source.output(HandsSource.RIGHT)
        left_arm_input_name = HandsSource.LEFT
        right_arm_input_name = HandsSource.RIGHT
        left_hand_input_name = HandsSource.LEFT
        right_hand_input_name = HandsSource.RIGHT
        hand_output = "hand_joints"
    else:
        source = ControllersSource(name="controllers")
        left_arm = P7ControllerPoseRetargeter(
            _pose_config(model, "left", mode), name="p7_left_arm"
        )
        right_arm = P7ControllerPoseRetargeter(
            _pose_config(model, "right", mode), name="p7_right_arm"
        )
        left_input = source.output(ControllersSource.LEFT)
        right_input = source.output(ControllersSource.RIGHT)
        left_arm_input_name = ControllersSource.LEFT
        right_arm_input_name = ControllersSource.RIGHT
        if mode == "controller":
            left_hand = ControllerTriggerRealHandRetargeter(
                ControllerTriggerRealHandRetargeterConfig(
                    input_device=ControllersSource.LEFT,
                    joint_names=left_joint_names,
                    side="left",
                    hand_model=model,
                ),
                name=f"{model}_left_hand",
            )
            right_hand = ControllerTriggerRealHandRetargeter(
                ControllerTriggerRealHandRetargeterConfig(
                    input_device=ControllersSource.RIGHT,
                    joint_names=right_joint_names,
                    side="right",
                    hand_model=model,
                ),
                name=f"{model}_right_hand",
            )
            left_hand_input_name = ControllersSource.LEFT
            right_hand_input_name = ControllersSource.RIGHT
        else:
            if calibration is None:
                raise ValueError("RealHand FFG Glove mode requires a calibration YAML")
            left_glove = JointStateSource(
                name="realhand_ffg_glove_left",
                collection_id="realhand_ffg_glove_left",
                joint_names=list(REALHAND_FFG_GLOVE_SENSOR_NAMES),
            )
            right_glove = JointStateSource(
                name="realhand_ffg_glove_right",
                collection_id="realhand_ffg_glove_right",
                joint_names=list(REALHAND_FFG_GLOVE_SENSOR_NAMES),
            )
            left_hand = RealHandFFGGloveRetargeter(
                RealHandFFGGloveRetargeterConfig(
                    input_device=JointStateSource.JOINTS,
                    joint_names=left_joint_names,
                    side="left",
                    hand_model=model,
                    calibration_path=str(calibration),
                ),
                name=f"{model}_left_hand",
            )
            right_hand = RealHandFFGGloveRetargeter(
                RealHandFFGGloveRetargeterConfig(
                    input_device=JointStateSource.JOINTS,
                    joint_names=right_joint_names,
                    side="right",
                    hand_model=model,
                    calibration_path=str(calibration),
                ),
                name=f"{model}_right_hand",
            )
            left_input = left_glove.output(JointStateSource.JOINTS)
            right_input = right_glove.output(JointStateSource.JOINTS)
            left_hand_input_name = JointStateSource.JOINTS
            right_hand_input_name = JointStateSource.JOINTS
        hand_output = "hand_joints"

    connected_left_arm = left_arm.connect(
        {left_arm_input_name: source.output(source.LEFT)}
    )
    connected_right_arm = right_arm.connect(
        {right_arm_input_name: source.output(source.RIGHT)}
    )
    connected_left_hand = left_hand.connect({left_hand_input_name: left_input})
    connected_right_hand = right_hand.connect({right_hand_input_name: right_input})

    action_order = _LEFT_EE + _RIGHT_EE + list(profile.action_joint_names)
    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": _LEFT_EE,
            "right_ee_pose": _RIGHT_EE,
            "left_hand_joints": left_joint_names,
            "right_hand_joints": right_joint_names,
        },
        output_order=action_order,
        name="p7_realhand_action",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_hand_joints": "scalar",
            "right_hand_joints": "scalar",
        },
    )
    connected_action = reorderer.connect(
        {
            "left_ee_pose": connected_left_arm.output("ee_pose"),
            "right_ee_pose": connected_right_arm.output("ee_pose"),
            "left_hand_joints": connected_left_hand.output(hand_output),
            "right_hand_joints": connected_right_hand.output(hand_output),
        }
    )
    return OutputCombiner({"action": connected_action.output("output")}), action_order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--mode", choices=("controller", "handtracking", "ffg"), default="handtracking"
    )
    parser.add_argument("--hand-model", choices=("l6", "o6", "l20"), default="l6")
    parser.add_argument(
        "--realhand-ffg-glove-calibration",
        type=Path,
        help="User-specific calibration YAML generated by realhand_ffg_glove_calibration.py",
    )
    parser.add_argument(
        "--plugin-path",
        type=Path,
        help="Directory containing realhand_ffg_glove/plugin.yaml",
    )
    parser.add_argument("--duration", type=float, default=360.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate the graph without OpenXR",
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    calibration = args.realhand_ffg_glove_calibration
    if args.mode == "ffg" and calibration is None:
        parser.error("--realhand-ffg-glove-calibration is required when --mode ffg")
    pipeline, action_order = build_pipeline(args.mode, args.hand_model, calibration)
    print(
        f"P7 + {args.hand_model.upper()} mode={args.mode}; action_dim={len(action_order)}"
    )
    print("action order:", action_order)
    if args.dry_run:
        return

    plugins = []
    if args.mode == "ffg" and args.plugin_path is not None:
        plugins.append(
            PluginConfig(
                plugin_name="realhand_ffg_glove_plugin",
                plugin_root_id="realhand_ffg_glove",
                search_paths=[args.plugin_path],
                required=True,
            )
        )
    config = TeleopSessionConfig(
        app_name="P7RealHandBimanual",
        trackers=[],
        pipeline=pipeline,
        plugins=plugins,
    )
    with CloudXRLauncher.launch_context(args), TeleopSession(config) as session:
        deadline = time.time() + args.duration
        while time.time() < deadline:
            result = session.step()
            if session.frame_count % 60 == 0:
                action = np.asarray(result["action"][0])
                print(
                    f"frame={session.frame_count} left_ee={np.round(action[:7], 3)} "
                    f"right_ee={np.round(action[7:14], 3)}"
                )
            time.sleep(0.016)


if __name__ == "__main__":
    main()
