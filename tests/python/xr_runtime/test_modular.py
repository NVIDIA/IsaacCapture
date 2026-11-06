#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script for modular OpenXR tracking API
"""

import time

import isaaccapture.deviceio as deviceio
import isaaccapture.oxr as oxr
import isaaccapture.schema as schema

print("=" * 80)
print("OpenXR Modular Tracking API Test")
print("=" * 80)
print()

# Test 1: Create trackers
print("[Test 1] Creating trackers...")
hand_tracker = deviceio.HandTracker()
head_tracker = deviceio.HeadTracker()
print(f"✓ {hand_tracker.get_name()} created")
print(f"✓ {head_tracker.get_name()} created")
print()

# Test 2: Query required extensions
print("[Test 2] Querying required extensions...")
trackers = [hand_tracker, head_tracker]
required_extensions = deviceio.DeviceIOSession.get_required_extensions(trackers)
print(f"✓ Required extensions: {required_extensions}")
print()

# Test 3: Initialize
print("[Test 3] Creating OpenXR session and initializing...")

# Create OpenXR session
with oxr.OpenXRSession("ModularTest", required_extensions) as oxr_session:
    handles = oxr_session.get_handles()

    # Run deviceio session with trackers (throws exception on failure)
    with deviceio.DeviceIOSession.run(trackers, handles) as session:
        print("✅ OpenXR session initialized")
        print()

        # Test 4: Update and get data
        print("[Test 4] Testing data retrieval...")
        session.update()

        print("✓ Update successful")
        print()

        # Test 5: Check hand data
        print("[Test 5] Checking hand tracking data...")
        left_tracked: schema.HandPose | None = hand_tracker.get_left_hand(session)
        right_tracked: schema.HandPose | None = hand_tracker.get_right_hand(session)
        print(f"  Left hand: {'ACTIVE' if left_tracked else 'INACTIVE'}")
        print(f"  Right hand: {'ACTIVE' if right_tracked else 'INACTIVE'}")

        if left_tracked:
            pos = left_tracked.joints.poses(deviceio.JOINT_WRIST).pose.position
            print(f"  Left wrist position: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
        else:
            print("  Left hand: inactive")
        print()

        # Test 6: Check head data
        print("[Test 6] Checking head tracking data...")
        head_tracked: schema.HeadPose | None = head_tracker.get_head(session)
        if head_tracked:
            pos = head_tracked.pose.position
            ori = head_tracked.pose.orientation
            print(f"  Head position: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
            print(
                f"  Head orientation: [{ori.x:.3f}, {ori.y:.3f}, {ori.z:.3f}, {ori.w:.3f}]"
            )
        else:
            print("  Head: inactive")
        print()

        # Test 7: Run tracking loop
        print("[Test 7] Running tracking loop (5 seconds)...")
        frame_count = 0
        start_time = time.time()

        while time.time() - start_time < 5.0:
            session.update()

            if frame_count % 60 == 0:
                elapsed = time.time() - start_time
                left_tracked = hand_tracker.get_left_hand(session)
                head_tracked = head_tracker.get_head(session)
                print(f"  [{elapsed:4.1f}s] Frame {frame_count:3d}:")
                if left_tracked:
                    pos = left_tracked.joints.poses(deviceio.JOINT_WRIST).pose.position
                    print(f"    Left wrist: [{pos.x:6.3f}, {pos.y:6.3f}, {pos.z:6.3f}]")
                else:
                    print("    Left hand:  inactive")
                if head_tracked:
                    pos = head_tracked.pose.position
                    print(f"    Head pos:   [{pos.x:6.3f}, {pos.y:6.3f}, {pos.z:6.3f}]")
                else:
                    print("    Head:       inactive")

            frame_count += 1
            time.sleep(0.016)

        print(f"✓ Processed {frame_count} frames")
        print()

        # Cleanup
        print("[Test 8] Cleanup...")
        print("✓ Resources will be cleaned up when exiting 'with' blocks (RAII)")
        print()

print("=" * 80)
print("✅ All tests passed")
print("=" * 80)
