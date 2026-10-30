#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test session sharing between multiple DeviceIOSession instances.

This demonstrates how to create one OpenXR session directly and share it
across multiple managers with different trackers.
"""

import time

import isaaccapture.deviceio as deviceio
import isaaccapture.oxr as oxr

print("=" * 80)
print("OpenXR Session Sharing Test (Python)")
print("=" * 80)
print()

# ============================================================================
# Step 1: Create OpenXR session directly with all required extensions
# ============================================================================
print("[Step 1] Creating standalone OpenXR session...")

# Define all extensions needed by our trackers
extensions = [
    "XR_KHR_convert_timespec_time",  # Required for time conversion
    "XR_MND_headless",  # Headless mode
    "XR_EXTX_overlay",  # Overlay mode
    "XR_EXT_hand_tracking",  # Hand tracking
]

print("  Required extensions:")
for ext in extensions:
    print(f"    - {ext}")

with oxr.OpenXRSession("SessionSharingExample", extensions) as oxr_session:
    print("  ✓ OpenXR session created")
    print()

    # ============================================================================
    # Step 2: Get handles from the session
    # ============================================================================
    print("[Step 2] Getting session handles...")
    handles = oxr_session.get_handles()

    print(f"  Instance: {handles.instance:#x}")
    print(f"  Session:  {handles.session:#x}")
    print(f"  Space:    {handles.space:#x}")
    print()

    # ============================================================================
    # Step 3: Create Manager 1 with HandTracker using the shared session
    # ============================================================================
    print("[Step 3] Creating Manager 1 with HandTracker...")
    hand_tracker = deviceio.HandTracker()

    # ============================================================================
    # Step 4: Create Manager 2 with HeadTracker using the SAME shared session
    # ============================================================================
    print("[Step 4] Creating Manager 2 with HeadTracker...")
    head_tracker = deviceio.HeadTracker()

    # run() throws exception on failure
    with (
        deviceio.DeviceIOSession.run([hand_tracker], handles) as session1,
        deviceio.DeviceIOSession.run([head_tracker], handles) as session2,
    ):
        print("  ✓ Manager 1 using shared session")
        print()
        print("  ✓ Manager 2 using shared session")
        print()

        # ============================================================================
        # Step 5: Update both sessions - they share the same OpenXR session!
        # ============================================================================
        print("[Step 5] Testing both managers with shared session (5 seconds)...")
        print()

        start_time = time.time()
        frame_count = 0
        while time.time() - start_time < 5.0:
            # Both sessions update using the same underlying OpenXR session
            session1.update()
            session2.update()

            # Print status every 60 frames
            if frame_count % 60 == 0:
                elapsed = time.time() - start_time

                # Get data from both trackers
                left_tracked = hand_tracker.get_left_hand(session1)
                head_tracked = head_tracker.get_head(session2)

                print(f"[{elapsed:4.1f}s] Frame {frame_count:3d}:")
                if left_tracked:
                    pos = left_tracked.joints.poses(deviceio.JOINT_WRIST).pose.position
                    print(f"  Left wrist: [{pos.x:6.3f}, {pos.y:6.3f}, {pos.z:6.3f}]")
                else:
                    print("  Left hand:  inactive")
                if head_tracked:
                    pos = head_tracked.pose.position
                    print(f"  Head pos:   [{pos.x:6.3f}, {pos.y:6.3f}, {pos.z:6.3f}]")
                else:
                    print("  Head:       inactive")
                print()

            frame_count += 1
            time.sleep(0.016)

        print(f"Processed {frame_count} frames")
        print()

    print("[Cleanup]")
    print("  Destroying Manager 2...")
    print("  ✓ Manager 2 destroyed")
    print("  Destroying Manager 1...")
    print("  ✓ Manager 1 destroyed")
    print("  Destroying shared OpenXR session...")
print("  ✓ OpenXR session destroyed")
print()

print("=" * 80)
print("✓ Session sharing test complete")
print("=" * 80)
print()
print("Summary:")
print("  ✓ One OpenXR session created")
print("  ✓ Two managers shared the same session")
print("  ✓ HandTracker (Manager 1) and HeadTracker (Manager 2)")
print("  ✓ Both updated successfully with shared session")
print()
