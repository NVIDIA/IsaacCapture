# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Complete Gripper Retargeting Example

Demonstrates constructing a custom retargeting graph for gripper-based robots.
This example manually connects Controller and Hand sources to Gripper and SE3 retargeters,
combining their outputs into a unified action vector (Pose + Gripper) executed via TeleopSession.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    HandsSource,
)
from isaaccapture.retargeters import (
    GripperRetargeter,
    GripperRetargeterConfig,
    Se3AbsRetargeter,
    Se3RetargeterConfig,
)
from isaaccapture.retargeting_engine.interface import OutputCombiner

# Import TeleopSession to handle the loop correctly with new sources
from isaaccapture.teleop_session_manager import (
    TeleopSession,
    TeleopSessionConfig,
    PluginConfig,
)


# Six levels up is the tree root -- the checkout, or the install prefix
# when running from install/examples/teleop/.
PLUGIN_ROOT_DIR = Path(__file__).resolve().parents[5] / "plugins"
PLUGIN_NAME = "controller_synthetic_hands"
PLUGIN_ROOT_ID = "synthetic_hands"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    # Create controllers source (tracker is internal)
    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")

    # 1. Gripper Retargeters (Left & Right)
    gripper_left = GripperRetargeter(
        GripperRetargeterConfig(hand_side="left"), name="gripper_left"
    )
    connected_gripper_left = gripper_left.connect(
        {
            ControllersSource.LEFT: controllers.output(ControllersSource.LEFT),
            HandsSource.LEFT: hands.output(HandsSource.LEFT),
        }
    )

    gripper_right = GripperRetargeter(
        GripperRetargeterConfig(hand_side="right"), name="gripper_right"
    )
    connected_gripper_right = gripper_right.connect(
        {
            ControllersSource.RIGHT: controllers.output(ControllersSource.RIGHT),
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        }
    )

    # 2. SE3 Pose Retargeter (Left)
    se3_left = Se3AbsRetargeter(
        Se3RetargeterConfig(input_device=ControllersSource.LEFT), name="se3_left"
    )
    connected_se3_left = se3_left.connect(
        {ControllersSource.LEFT: controllers.output(ControllersSource.LEFT)}
    )

    # 3. SE3 Pose Retargeter (Right)
    se3_right = Se3AbsRetargeter(
        Se3RetargeterConfig(input_device=ControllersSource.RIGHT), name="se3_right"
    )
    connected_se3_right = se3_right.connect(
        {ControllersSource.RIGHT: controllers.output(ControllersSource.RIGHT)}
    )

    # 4. Combine outputs into a single pipeline
    pipeline = OutputCombiner(
        {
            "gripper_left": connected_gripper_left.output("gripper_command"),
            "gripper_right": connected_gripper_right.output("gripper_command"),
            "pose_left": connected_se3_left.output("ee_pose"),
            "pose_right": connected_se3_right.output("ee_pose"),
        }
    )

    # Configure Plugins
    plugins = []
    if PLUGIN_ROOT_DIR.exists():
        plugins.append(
            PluginConfig(
                plugin_name=PLUGIN_NAME,
                plugin_root_id=PLUGIN_ROOT_ID,
                search_paths=[PLUGIN_ROOT_DIR],
            )
        )

    # Create TeleopSessionConfig
    config = TeleopSessionConfig(
        app_name="GripperRetargetingExample",
        trackers=[],  # Auto-discovered
        pipeline=pipeline,
        plugins=plugins,
    )

    # Use TeleopSession to manage the loop and data injection
    with CloudXRLauncher.launch_context(args), TeleopSession(config) as session:
        run_loop(session)

    return 0


def run_loop(session):
    """Run the control loop using TeleopSession."""
    start_time = time.time()

    print("Starting gripper retargeting (20 seconds)...")
    print("=" * 60)

    while time.time() - start_time < 20.0:
        # Execute retargeting graph via session
        result = session.step()

        # Access output values

        # Left Hand: Combine Pose (7) + Gripper (1) -> Action (8)
        left_gripper = result["gripper_left"][0]
        left_pose = result["pose_left"][0]
        left_action = np.concatenate([left_pose, [left_gripper]])

        # Right Hand: Combine Pose (7) + Gripper (1) -> Action (8)
        right_gripper = result["gripper_right"][0]
        right_pose = result["pose_right"][0]
        right_action = np.concatenate([right_pose, [right_gripper]])

        # Print every 0.5 seconds
        if session.frame_count % 30 == 0:
            elapsed = session.get_elapsed_time()
            print(
                f"[{elapsed:5.1f}s] Left Action: {left_action[:3]}... G:{left_action[7]:.2f}"
            )
            print(
                f"         Right Action: {right_action[:3]}... G:{right_action[7]:.2f}"
            )

        time.sleep(0.016)  # ~60 FPS

    print("=" * 60)
    print(f"Completed {session.frame_count} frames in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
