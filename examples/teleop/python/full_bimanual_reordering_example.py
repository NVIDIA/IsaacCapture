# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Full Bimanual Retargeting Example with Tensor Reordering.

Demonstrates using:
1. Two Se3AbsRetargeters for arm end-effector control (Left/Right)
2. Two DexHandRetargeters for hand joint control (Left/Right)
3. TensorReorderer to flatten all outputs into a single action tensor for Isaac Lab.
"""

import sys
import time
import argparse
import contextlib
from types import SimpleNamespace
from pathlib import Path
import numpy as np

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeting_engine.deviceio_source_nodes import HandsSource
from isaaccapture.retargeters import (
    DexHandRetargeter,
    DexHandRetargeterConfig,
    Se3AbsRetargeter,
    Se3RetargeterConfig,
    TensorReorderer,
)
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig
from isaaccapture.retargeting_engine_ui import MultiRetargeterTuningUIImGui


def main():
    print("\n" + "=" * 80)
    print("  Full Bimanual Retargeting Example (Arms + Hands)")
    print("=" * 80)
    print("Demonstrates flattening 4 retargeters (2 arms, 2 hands) into one tensor.")
    print("=" * 80 + "\n")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Full Bimanual Retargeting Example")
    parser.add_argument("--left-urdf", type=str, help="Path to left hand URDF")
    parser.add_argument("--right-urdf", type=str, help="Path to right hand URDF")
    parser.add_argument(
        "--left-config", type=str, help="Path to left hand retargeting config (YAML)"
    )
    parser.add_argument(
        "--right-config", type=str, help="Path to right hand retargeting config (YAML)"
    )
    parser.add_argument(
        "--enable-tuning", action="store_true", help="Enable retargeting tuning UI"
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    # Config paths (similar to dex_bimanual_example)
    config_dir = Path(__file__).parent / "config" / "dex_retargeting"
    left_yaml = (
        Path(args.left_config)
        if args.left_config
        else config_dir / "hand_left_config.yml"
    )
    right_yaml = (
        Path(args.right_config)
        if args.right_config
        else config_dir / "hand_right_config.yml"
    )
    left_urdf_path = (
        Path(args.left_urdf) if args.left_urdf else config_dir / "left_robot_hand.urdf"
    )
    right_urdf_path = (
        Path(args.right_urdf)
        if args.right_urdf
        else config_dir / "right_robot_hand.urdf"
    )

    # ==================================================================
    # 1. Build Retargeting Pipeline
    # ==================================================================

    # Create source
    hands = HandsSource(name="hands")

    # --- A. Setup Hand Retargeters ---

    # Left Hand Config
    left_hand_cfg = DexHandRetargeterConfig(
        hand_retargeting_config=str(left_yaml),
        hand_urdf=str(left_urdf_path),
        hand_side="left",
        parameter_config_path="/tmp/left_hand_params.json",
    )
    left_hand_retargeter = DexHandRetargeter(left_hand_cfg, name="left_hand")

    # Right Hand Config
    right_hand_cfg = DexHandRetargeterConfig(
        hand_retargeting_config=str(right_yaml),
        hand_urdf=str(right_urdf_path),
        hand_side="right",
        parameter_config_path="/tmp/right_hand_params.json",
    )
    right_hand_retargeter = DexHandRetargeter(right_hand_cfg, name="right_hand")

    # --- B. Setup Arm (SE3) Retargeters ---

    # Left Arm (Absolute Pose)
    left_arm_cfg = Se3RetargeterConfig(
        input_device="hand_left",
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    left_arm_retargeter = Se3AbsRetargeter(left_arm_cfg, name="left_arm")

    # Right Arm (Absolute Pose)
    right_arm_cfg = Se3RetargeterConfig(
        input_device="hand_right",
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    right_arm_retargeter = Se3AbsRetargeter(right_arm_cfg, name="right_arm")

    # --- Connect Inputs ---

    # Connect Hands
    left_hand_connected = left_hand_retargeter.connect(
        {HandsSource.LEFT: hands.output(HandsSource.LEFT)}
    )
    right_hand_connected = right_hand_retargeter.connect(
        {HandsSource.RIGHT: hands.output(HandsSource.RIGHT)}
    )

    # Connect Arms
    left_arm_connected = left_arm_retargeter.connect(
        {"hand_left": hands.output(HandsSource.LEFT)}
    )
    right_arm_connected = right_arm_retargeter.connect(
        {"hand_right": hands.output(HandsSource.RIGHT)}
    )

    # ==================================================================
    # 2. Configure Tensor Reorderer
    # ==================================================================

    # Define the names for the 7 elements of the SE3 output (Pos + Rot)
    # Order: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w]
    left_arm_elements = [
        f"left_ee_{x}" for x in ["px", "py", "pz", "rx", "ry", "rz", "rw"]
    ]
    right_arm_elements = [
        f"right_ee_{x}" for x in ["px", "py", "pz", "rx", "ry", "rz", "rw"]
    ]

    # Define the full flattened order expected by "Isaac Lab" (Hypothetically)
    # Let's say the robot expects: [Left Arm Pose, Left Hand Joints, Right Arm Pose, Right Hand Joints]

    # 1. Left Arm
    full_action_order = []
    full_action_order.extend(left_arm_elements)

    # 2. Left Hand (using names from retargeter)
    # Note: These are usually like "joint_0", "joint_1" etc. from the URDF/Config
    full_action_order.extend(left_hand_retargeter._hand_joint_names)

    # 3. Right Arm
    full_action_order.extend(right_arm_elements)

    # 4. Right Hand
    full_action_order.extend(right_hand_retargeter._hand_joint_names)

    print(f"Total Action Dimension: {len(full_action_order)}")

    # Create Reorderer
    reorderer = TensorReorderer(
        input_config={
            "left_arm_pose": left_arm_elements,
            "right_arm_pose": right_arm_elements,
            "left_hand_joints": left_hand_retargeter._hand_joint_names,
            "right_hand_joints": right_hand_retargeter._hand_joint_names,
        },
        output_order=full_action_order,
        name="final_flatten",
        input_types={
            "left_arm_pose": "array",
            "right_arm_pose": "array",
        },
    )

    # Connect Reorderer to Retargeters
    reorderer_connected = reorderer.connect(
        {
            "left_arm_pose": left_arm_connected.output("ee_pose"),
            "right_arm_pose": right_arm_connected.output("ee_pose"),
            "left_hand_joints": left_hand_connected.output("hand_joints"),
            "right_hand_joints": right_hand_connected.output("hand_joints"),
        }
    )

    # Final Pipeline Output
    pipeline = OutputCombiner({"action": reorderer_connected.output("output")})

    # ==================================================================
    # 3. Run Session
    # ==================================================================

    with CloudXRLauncher.launch_context(args):
        session_config = TeleopSessionConfig(
            app_name="FullBimanualExample",
            trackers=[],
            pipeline=pipeline,
        )

        # UI setup
        retargeters_to_tune = [
            left_hand_retargeter,
            right_hand_retargeter,
            left_arm_retargeter,
            right_arm_retargeter,
        ]

        if args.enable_tuning:
            print("Opening Retargeting UI...")
            ui_context = MultiRetargeterTuningUIImGui(
                retargeters_to_tune, title="Full Bimanual Tuning"
            )
        else:
            ui_context = contextlib.nullcontext(
                SimpleNamespace(is_running=lambda: True)
            )

        with ui_context as ui:
            with TeleopSession(session_config) as session:
                start_time = time.time()

                print("\nStarting Loop. Press Ctrl+C to exit (or close UI window).")
                print("Outputting flattened 'action' tensor...")

                while time.time() - start_time < 360.0 and ui.is_running():
                    result = session.step()

                    # result["action"] is a TensorGroup containing ONE tensor (our array)
                    # Access it at index 0
                    action_tensor = result["action"][
                        0
                    ]  # This is a numpy array (float32)

                    if session.frame_count % 60 == 0:
                        elapsed = session.get_elapsed_time()
                        # Print summary of the tensor
                        # e.g. Left Arm Pos
                        l_pos = action_tensor[0:3]
                        # Right Arm Pos (index depends on length of left hand joints)
                        r_start_idx = 7 + len(left_hand_retargeter._hand_joint_names)
                        r_pos = action_tensor[r_start_idx : r_start_idx + 3]

                        # Left Hand Joints (first 3)
                        l_hand_start = 7
                        l_hand_joints = action_tensor[l_hand_start : l_hand_start + 3]

                        # Right Hand Joints (first 3)
                        r_hand_start = r_start_idx + 7
                        r_hand_joints = action_tensor[r_hand_start : r_hand_start + 3]

                        print(
                            f"[{elapsed:5.1f}s] Action Shape: {action_tensor.shape} | "
                            f"L_Arm: {np.round(l_pos, 3)} | L_Hand(3): {np.round(l_hand_joints, 3)} | "
                            f"R_Arm: {np.round(r_pos, 3)} | R_Hand(3): {np.round(r_hand_joints, 3)}"
                        )

                    time.sleep(0.016)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)
