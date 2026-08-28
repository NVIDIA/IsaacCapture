# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test that HandTracker correctly reports inactive (data=None) after the synthetic
hands plugin is stopped.

Test sequence:
1. Start the controller_synthetic_hands plugin.
2. Open a reader session with HandTracker.
3. Wait until both hands are active (data is not None).
4. Stop the plugin (deinit the push hands extension).
5. Poll for a few seconds and assert that both hands become inactive (data is None).
"""

import time
from pathlib import Path

import isaaccapture.deviceio as deviceio
import isaaccapture.oxr as oxr
import isaaccapture.plugin_manager as pm

# Six levels up is the tree root -- the checkout, or the install prefix
# when running from install/examples/.
PLUGIN_ROOT_DIR = Path(__file__).resolve().parents[5] / "plugins"

ACTIVE_WAIT_S = 15.0  # max time to wait for hands to become active
INACTIVE_WAIT_S = 5.0  # max time to wait for hands to become inactive after stop
FRAME_SLEEP_S = 0.016


def poll_hands(hand_tracker, deviceio_session):
    """Return (left_active, right_active) for the current frame."""
    deviceio_session.update()
    left = hand_tracker.get_left_hand(deviceio_session)
    right = hand_tracker.get_right_hand(deviceio_session)
    return left is not None, right is not None


def run_test():
    print("=" * 80)
    print("Hand Inactive On Plugin Stop Test")
    print("=" * 80)
    print()

    if not PLUGIN_ROOT_DIR.exists():
        print(f"✗ Plugin directory not found at {PLUGIN_ROOT_DIR}")
        print("  Please build and install the project first.")
        return False

    manager = pm.PluginManager([str(PLUGIN_ROOT_DIR)])
    plugin_name = "controller_synthetic_hands"
    plugin_root_id = "synthetic_hands"

    if plugin_name not in manager.get_plugin_names():
        print(f"✗ Plugin '{plugin_name}' not found")
        return False

    extensions = [
        "XR_KHR_convert_timespec_time",
        "XR_MND_headless",
        "XR_EXT_hand_tracking",
    ]

    plugin = manager.start(plugin_name, plugin_root_id)
    print("✓ Plugin started")

    plugin_stopped = False
    try:
        with oxr.OpenXRSession("HandReader", extensions) as oxr_session:
            handles = oxr_session.get_handles()
            hand_tracker = deviceio.HandTracker()

            with deviceio.DeviceIOSession.run(
                [hand_tracker], handles
            ) as deviceio_session:
                print("✓ Reader session created")
                print()

                # ----------------------------------------------------------------
                # Phase 1: wait for hands to become active
                # ----------------------------------------------------------------
                print(
                    f"[Phase 1] Waiting up to {ACTIVE_WAIT_S:.0f}s for hands to become active..."
                )
                deadline = time.monotonic() + ACTIVE_WAIT_S
                left_active = right_active = False

                while time.monotonic() < deadline:
                    plugin.check_health()
                    left_active, right_active = poll_hands(
                        hand_tracker, deviceio_session
                    )

                    if left_active and right_active:
                        break
                    time.sleep(FRAME_SLEEP_S)

                if not (left_active and right_active):
                    print(
                        f"✗ Hands did not become active within {ACTIVE_WAIT_S:.0f}s "
                        f"(left={left_active}, right={right_active})"
                    )
                    return False

                print("✓ Both hands active")
                print()

                # ----------------------------------------------------------------
                # Phase 2: stop the plugin and wait for hands to become inactive
                # ----------------------------------------------------------------
                print(
                    "[Phase 2] Stopping plugin and waiting for hands to become inactive..."
                )
                plugin.stop()
                plugin_stopped = True
                print("✓ Plugin stopped")

                deadline = time.monotonic() + INACTIVE_WAIT_S
                left_inactive = right_inactive = False

                while time.monotonic() < deadline:
                    left_active, right_active = poll_hands(
                        hand_tracker, deviceio_session
                    )

                    left_inactive = not left_active
                    right_inactive = not right_active
                    if left_inactive and right_inactive:
                        break
                    time.sleep(FRAME_SLEEP_S)

                print()
                if left_inactive and right_inactive:
                    print(
                        "✓ PASS: both hands correctly reported as inactive after plugin stop"
                    )
                    return True
                else:
                    print(
                        f"✗ FAIL: hands still active after {INACTIVE_WAIT_S:.0f}s "
                        f"(left_inactive={left_inactive}, right_inactive={right_inactive})"
                    )
                    return False
    finally:
        if not plugin_stopped:
            plugin.stop()


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_test() else 1)
