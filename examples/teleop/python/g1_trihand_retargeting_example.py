# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
G1 TriHand Retargeting Example

Demonstrates using the TriHandMotionControllerRetargeter module to retarget motion controller
inputs to G1 robot hand joints.

This example shows:
1. TriHandMotionControllerRetargeter - Simple VR controller-based hand control for G1 TriHand
"""

import argparse
import sys
import time
from pathlib import Path

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeting_engine.deviceio_source_nodes import ControllersSource
from isaaccapture.retargeters import (
    TriHandMotionControllerRetargeter,
    TriHandMotionControllerConfig,
)
from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig
from isaaccapture.retargeting_engine.interface import OutputCombiner


PLUGIN_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "plugins"
PLUGIN_NAME = "controller_synthetic_hands"
PLUGIN_ROOT_ID = "synthetic_hands"


def example_trihand_motion_controller():
    """Run TriHandMotionControllerRetargeter example with VR controllers."""
    parser = argparse.ArgumentParser(description=__doc__)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  TriHandMotionControllerRetargeter Example")
    print("=" * 80)
    print("\nMapping VR controller inputs to G1 TriHand joints...")
    print("- Trigger: Controls index finger")
    print("- Squeeze: Controls middle finger")
    print("- Both: Controls thumb\n")

    # Create controllers source (tracker is internal)
    controllers = ControllersSource(name="controllers")

    # Configure TriHandMotionControllerRetargeter for G1 7-DOF hand
    hand_joint_names = [
        "thumb_rotation",
        "thumb_proximal",
        "thumb_distal",
        "index_proximal",
        "index_distal",
        "middle_proximal",
        "middle_distal",
    ]

    # Create left hand controller
    left_config = TriHandMotionControllerConfig(
        hand_joint_names=hand_joint_names,
        controller_side="left",
    )
    left_controller = TriHandMotionControllerRetargeter(
        left_config, name="trihand_motion_left"
    )

    # Connect left controller to source
    connected_left = left_controller.connect(
        {ControllersSource.LEFT: controllers.output(ControllersSource.LEFT)}
    )

    # Create right hand controller
    right_config = TriHandMotionControllerConfig(
        hand_joint_names=hand_joint_names,
        controller_side="right",
    )
    right_controller = TriHandMotionControllerRetargeter(
        right_config, name="trihand_motion_right"
    )

    # Connect right controller to source
    connected_right = right_controller.connect(
        {ControllersSource.RIGHT: controllers.output(ControllersSource.RIGHT)}
    )

    # ==================================================================
    # Create and run TeleopSession
    # ==================================================================

    combined_pipeline = OutputCombiner(
        {
            "left_hand": connected_left.output("hand_joints"),
            "right_hand": connected_right.output("hand_joints"),
        }
    )

    session_config = TeleopSessionConfig(
        app_name="TriHandMotionControllerRetargeterExample",
        trackers=[],  # Auto-discovered from pipeline
        pipeline=combined_pipeline,
    )

    # Configure Plugins
    plugins = []
    if PLUGIN_ROOT_DIR.exists():
        from isaaccapture.teleop_session_manager import PluginConfig

        plugins.append(
            PluginConfig(
                plugin_name=PLUGIN_NAME,
                plugin_root_id=PLUGIN_ROOT_ID,
                search_paths=[PLUGIN_ROOT_DIR],
            )
        )
    session_config.plugins = plugins

    with (
        CloudXRLauncher.launch_context(args),
        TeleopSession(session_config) as session,
    ):
        # No session injection needed

        run_motion_controller_loop(session)

    return 0


def run_motion_controller_loop(session):
    """Run the motion controller loop."""
    start_time = time.time()
    frame_count = 0

    print("Running for 20 seconds...")
    print("=" * 80)

    while time.time() - start_time < 20.0:
        # Execute retargeting graph
        result = session.step()

        # Access output joint angles
        joints_left = result["left_hand"]
        joints_right = result["right_hand"]

        # Print every 0.5 seconds
        if frame_count % 30 == 0:
            elapsed = session.get_elapsed_time()

            # Left hand proximal joints (thumb, index, middle)
            l_thumb = joints_left[1]
            l_index = joints_left[3]
            l_middle = joints_left[5]

            # Right hand proximal joints (thumb, index, middle)
            r_thumb = joints_right[1]
            r_index = joints_right[3]
            r_middle = joints_right[5]

            print(
                f"[{elapsed:5.1f}s] L: T={l_thumb:5.2f} I={l_index:5.2f} M={l_middle:5.2f} | R: T={r_thumb:5.2f} I={r_index:5.2f} M={r_middle:5.2f}"
            )

        frame_count += 1
        time.sleep(0.016)  # ~60 FPS

    print("=" * 80)
    print(f"✓ Completed {frame_count} frames in {time.time() - start_time:.1f}s")


def main():
    """Main entry point."""
    return example_trihand_motion_controller()


if __name__ == "__main__":
    sys.exit(main())
