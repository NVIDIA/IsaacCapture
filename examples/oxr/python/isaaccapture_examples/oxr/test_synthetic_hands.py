# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test controller synthetic hands plugin using Plugin Manager.

This test:
1. Uses PluginManager to discover and launch the plugin
2. Queries available devices from the manifest
3. Starts the OpenXR session
4. Uses HandTracker to read the injected hands
5. Periodically polls plugin health

Note: Plugin crashes will raise pm.PluginCrashException
"""

import time
from pathlib import Path

import isaaccapture.deviceio as deviceio
import isaaccapture.oxr as oxr
import isaaccapture.plugin_manager as pm

# Paths
# The test will look for plugins in the install directory relative to this script
# This script is in install/examples/oxr/python
# Plugins are in install/plugins
# Six levels up is the tree root -- the checkout, or the install prefix
# when running from install/examples/.
PLUGIN_ROOT_DIR = Path(__file__).resolve().parents[5] / "plugins"


def run_test():
    print("=" * 80)
    print("Controller Synthetic Hands Plugin Test")
    print("=" * 80)
    print()

    if not PLUGIN_ROOT_DIR.exists():
        print(f"Error: Plugin directory not found at {PLUGIN_ROOT_DIR}")
        print("Please build and install the project first.")
        return

    # 1. Initialize Plugin Manager
    print("[Step 1] Initializing Plugin Manager...")
    print(f"  Search path: {PLUGIN_ROOT_DIR}")
    manager = pm.PluginManager([str(PLUGIN_ROOT_DIR)])

    plugins = manager.get_plugin_names()
    print(f"  Discovered plugins: {plugins}")

    plugin_name = "controller_synthetic_hands"
    plugin_root_id = "synthetic_hands"

    if plugin_name not in plugins:
        print(f"  ✗ {plugin_name} not found")
        return

    # 2. Query Plugin Devices
    print("[Step 2] Querying plugin devices...")
    devices = manager.query_devices(plugin_name)
    print(f"  ✓ Available devices: {devices}")
    print()

    # 3. Start Plugin and Create Reader Session
    print("[Step 3] Starting plugin and reader session...")
    extensions = [
        "XR_KHR_convert_timespec_time",
        "XR_MND_headless",
        "XR_EXT_hand_tracking",
    ]

    with (
        manager.start(plugin_name, plugin_root_id) as plugin,
        oxr.OpenXRSession("HandReader", extensions) as oxr_session,
    ):
        print("  ✓ Plugin started")
        print("  ✓ Reader session created")
        print()

        handles = oxr_session.get_handles()

        hand_tracker = deviceio.HandTracker()
        trackers = [hand_tracker]

        # run() throws exception on failure
        with deviceio.DeviceIOSession.run(trackers, handles) as deviceio_session:
            print("  ✓ DeviceIO session created")
            print()

            # 4. Loop and Read with periodic health checks
            print("[Step 4] Reading data (10 seconds)...")
            start_time = time.time()
            frame_count = 0

            while time.time() - start_time < 10.0:
                # Poll plugin health every ~1 second
                if frame_count % 60 == 0:
                    plugin.check_health()  # Throws PluginCrashException if plugin crashed

                deviceio_session.update()

                if frame_count % 60 == 0:
                    left_tracked = hand_tracker.get_left_hand(deviceio_session)
                    right_tracked = hand_tracker.get_right_hand(deviceio_session)

                    print(f"Frame {frame_count}:")
                    if left_tracked:
                        pos = left_tracked.joints.poses(
                            deviceio.JOINT_WRIST
                        ).pose.position
                        print(f"  Left wrist:  [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
                    else:
                        print("  Left hand:   inactive")
                    if right_tracked:
                        pos = right_tracked.joints.poses(
                            deviceio.JOINT_WRIST
                        ).pose.position
                        print(f"  Right wrist: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
                    else:
                        print("  Right hand:  inactive")
                    print()

                frame_count += 1
                time.sleep(0.016)

    # Plugin automatically stopped when exiting 'with' block
    # If plugin crashed, PluginCrashException will be raised during stop()
    print("[Cleanup]")
    print("  ✓ Plugin stopped")
    print("  ✓ Done")


if __name__ == "__main__":
    run_test()
