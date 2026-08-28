# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Locomotion Retargeting Example

Demonstrates using LocomotionRootCmdRetargeter to generate robot base commands
from VR controller inputs.
"""

import argparse
import sys
import time
from pathlib import Path
import isaaccapture.deviceio as deviceio

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeters import (
    LocomotionRootCmdRetargeter,
    LocomotionRootCmdRetargeterConfig,
)
from isaaccapture.teleop_session_manager import (
    TeleopSession,
    TeleopSessionConfig,
    PluginConfig,
    create_standard_inputs,
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

    print("\n" + "=" * 80)
    print("  Locomotion Retargeting Example")
    print("=" * 80)
    print("Controls:")
    print("  Left Stick X/Y  : Linear Velocity (Forward/Backward/Left/Right)")
    print("  Right Stick X   : Angular Velocity (Turn)")
    print("  Right Stick Y   : Hip Height Adjustment")
    print("=" * 80 + "\n")

    # ==================================================================
    # Setup: Create standard inputs (trackers + sources)
    # ==================================================================

    controller_tracker = deviceio.ControllerTracker()
    trackers = [controller_tracker]
    sources = create_standard_inputs(trackers)
    controllers = sources["controllers"]

    # ==================================================================
    # Build Retargeting Pipeline
    # ==================================================================

    config = LocomotionRootCmdRetargeterConfig(
        initial_hip_height=0.72,
        movement_scale=1.0,  # Scale up for visibility
        rotation_scale=0.5,
    )

    locomotion = LocomotionRootCmdRetargeter(config, name="locomotion")

    pipeline = locomotion.connect(
        {
            "controller_left": controllers.output(controllers.LEFT),
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

    session_config = TeleopSessionConfig(
        app_name="LocomotionExample",
        trackers=[],  # Empty list if using new sources via create_standard_inputs
        pipeline=pipeline,
        plugins=plugins,
    )

    with (
        CloudXRLauncher.launch_context(args),
        TeleopSession(session_config) as session,
    ):
        # No need to inject session anymore

        start_time = time.time()

        while time.time() - start_time < 30.0:
            # Run one iteration
            result = session.step()

            # Get root command: [vel_x, vel_y, rot_vel_z, hip_height]
            cmd = result["root_command"][0]

            # Print status every 0.2 seconds
            if session.frame_count % 12 == 0:
                elapsed = session.get_elapsed_time()
                print(
                    f"[{elapsed:5.1f}s] Vel: ({cmd[0]:5.2f}, {cmd[1]:5.2f})  Rot: {cmd[2]:5.2f}  Height: {cmd[3]:.3f}"
                )

            time.sleep(0.016)  # ~60 FPS

        print("\nTime limit reached.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
