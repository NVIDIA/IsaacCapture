# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simplified Gripper Retargeting Example

Demonstrates using TeleopSession with the new retargeting engine.
Minimal boilerplate - just configure and run!
"""

import argparse
import sys
from pathlib import Path
import isaaccapture.deviceio as deviceio

from _example_loop import add_duration_argument, run_example_loop

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeters import (
    GripperRetargeter,
    GripperRetargeterConfig,
)
from isaaccapture.teleop_session_manager import (
    TeleopSession,
    TeleopSessionConfig,
    PluginConfig,
    create_standard_inputs,
)


PLUGIN_ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "plugins"
PLUGIN_NAME = "controller_synthetic_hands"
PLUGIN_ROOT_ID = "synthetic_hands"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    CloudXRLauncher.add_launcher_arguments(parser)
    add_duration_argument(parser)
    args = parser.parse_args()

    # ==================================================================
    # Setup: Create standard inputs (trackers + sources)
    # ==================================================================

    hand_tracker = deviceio.HandTracker()
    controller_tracker = deviceio.ControllerTracker()
    trackers = [hand_tracker, controller_tracker]
    sources = create_standard_inputs(trackers)
    hands = sources["hands"]
    controllers = sources["controllers"]

    # ==================================================================
    # Build Retargeting Pipeline
    # ==================================================================

    retargeter_config = GripperRetargeterConfig()
    gripper = GripperRetargeter(retargeter_config, name="gripper")
    pipeline = gripper.connect(
        {
            "hand_right": hands.output(hands.RIGHT),
            "controller_right": controllers.output(controllers.RIGHT),
        }
    )

    # ==================================================================
    # Configure Plugins (optional)
    # ==================================================================

    plugins = []
    if PLUGIN_ROOT_DIR.exists():
        plugins.append(
            PluginConfig(
                plugin_name=PLUGIN_NAME,
                plugin_root_id=PLUGIN_ROOT_ID,
                search_paths=[PLUGIN_ROOT_DIR],
            )
        )

    # ==================================================================
    # Create and run TeleopSession
    # ==================================================================

    config = TeleopSessionConfig(
        app_name="GripperRetargetingSimple",
        trackers=[],  # Empty list if using new sources
        pipeline=pipeline,
        plugins=plugins,
    )

    with CloudXRLauncher.launch_context(args), TeleopSession(config) as session:
        # No session injection needed

        print("\n" + "=" * 60)
        print("Gripper Retargeting - Squeeze triggers to control grippers")
        print("=" * 60 + "\n")

        print("Gripper command: +1.00 = open, -1.00 = closed (trigger > 0.5).")

        def step():
            # Run one iteration (updates trackers + executes pipeline)
            result = session.step()

            # Get gripper values
            right = result["gripper_command"][0]

            # Print status every 0.5 seconds
            if session.frame_count % 30 == 0:
                elapsed = session.get_elapsed_time()
                print(f"[{elapsed:5.1f}s] Right: {right:.2f}")

        run_example_loop(step, args.duration)

    print("Session closed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
