# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Foot Pedal Locomotion Example.

Demonstrates using FootPedalRootCmdRetargeter to generate robot base commands
from a 3-axis foot pedal (left/right pedals + rudder). Requires pedal data to
be pushed to OpenXR (e.g. run foot_pedal_reader or pedal_pusher with the same
collection_id as Generic3AxisPedalTracker).
"""

import argparse
import sys
import time
from pathlib import Path

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeters import (
    FootPedalRootCmdRetargeter,
    FootPedalRootCmdRetargeterConfig,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    Generic3AxisPedalSource,
)
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

    print("\n" + "=" * 80)
    print("  Foot Pedal Locomotion Example")
    print("=" * 80)
    print("Controls (3-axis pedal):")
    print("  Left/Right pedals : Forward/backward (horizontal mode)")
    print("  Left pedal       : Squat height (vertical mode)")
    print("  Rudder           : Yaw when no pedal pressed; strafe when pedal pressed")
    print("=" * 80)
    print("Note: Run foot_pedal_reader or pedal_pusher so pedal data is available.")
    print("=" * 80 + "\n")

    # ==================================================================
    # Setup: Create foot pedal source
    # ==================================================================
    pedals_source = Generic3AxisPedalSource(name="pedals")

    # ==================================================================
    # Build Retargeting Pipeline
    # ==================================================================

    config = FootPedalRootCmdRetargeterConfig(
        mode="horizontal",
    )

    foot_pedal_retargeter = FootPedalRootCmdRetargeter(config, name="foot_pedal")

    pipeline = foot_pedal_retargeter.connect(
        {
            "pedals": pedals_source.output("pedals"),
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
        app_name="FootPedalLocomotionExample",
        trackers=[],
        pipeline=pipeline,
        plugins=plugins,
    )

    with CloudXRLauncher.launch_context(args):
        with TeleopSession(session_config) as session:
            start_time = time.time()

            while time.time() - start_time < 30.0:
                result = session.step()

                # Get root command: [vel_x, vel_y, rot_vel_z, hip_height]
                cmd = result["root_command"][0]

                elapsed = session.get_elapsed_time()
                print(
                    f"[{elapsed:5.1f}s] Vel: ({cmd[0]:5.2f}, {cmd[1]:5.2f})  "
                    f"Rot: {cmd[2]:5.2f}  Height: {cmd[3]:.3f}",
                    end="\r",
                    flush=True,
                )

                time.sleep(0.01)  # ~100 FPS

            print("\nTime limit reached.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
