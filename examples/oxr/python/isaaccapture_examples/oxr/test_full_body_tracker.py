# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script for FullBodyTracker with XR_BD_body_tracking extension.

Demonstrates:
- Getting full body pose data (24 joints from pelvis to hands)
- Requires PICO device with body tracking support (default "body.pico-xr" vendor)
"""

import time

import isaaccapture.deviceio as deviceio
import isaaccapture.oxr as oxr
import isaaccapture.schema as schema

print("=" * 80)
print("Full Body Tracker Test (PICO XR_BD_body_tracking)")
print("=" * 80)
print()

# Test 1: Create full body tracker
print("[Test 1] Creating full body tracker...")
body_tracker = deviceio.FullBodyTracker()
print(f"✓ {body_tracker.get_name()} created")
print()

# Test 2: Query required extensions
print("[Test 2] Querying required extensions...")
trackers = [body_tracker]
required_extensions = deviceio.DeviceIOSession.get_required_extensions(trackers)
print(f"Required extensions: {required_extensions}")
print()

# Test 3: Show joint names from schema (BodyJoint enum)
print(f"[Test 3] Body joint names ({schema.BodyJoint.NUM_JOINTS} joints):")
for i in range(schema.BodyJoint.NUM_JOINTS):
    print(f"  [{i:2d}] {schema.BodyJoint(i).name}")
print()

# Test 4: Initialize
print("[Test 4] Creating OpenXR session and initializing...")
print("Note: This requires a PICO device with body tracking support")
print()

# Use context managers for proper RAII cleanup
with oxr.OpenXRSession("FullBodyTrackerTest", required_extensions) as oxr_session:
    handles = oxr_session.get_handles()

    # Run deviceio session with trackers (throws exception on failure)
    with deviceio.DeviceIOSession.run(trackers, handles) as session:
        print("✅ OpenXR session initialized with body tracking")
        print()

        # Test 5: Initial update
        print("[Test 5] Testing initial data retrieval...")
        session.update()

        print("✓ Update successful")
        print()

        # Test 6: Check initial body tracking state
        print("[Test 6] Checking body tracking state...")
        body_tracked = body_tracker.get_body_pose(session)
        print(f"  Body tracking active: {'YES' if body_tracked else 'NO'}")

        if body_tracked:
            valid_count = sum(
                1
                for i in range(schema.BodyJoint.NUM_JOINTS)
                if body_tracked.joints.joints(i).is_valid
            )
            print(f"  Valid joints: {valid_count}/{schema.BodyJoint.NUM_JOINTS}")
        print()

        # Test 7: Run tracking loop
        print("[Test 7] Running body tracking loop (10 seconds)...")
        print("Move around to see body tracking data!")
        print()

        frame_count = 0
        start_time = time.time()
        last_status_print = start_time

        while time.time() - start_time < 10.0:
            session.update()

            # Get current body pose
            current_time = time.time()
            if current_time - last_status_print >= 0.5:  # Print every 0.5 seconds
                elapsed = current_time - start_time
                body_tracked = body_tracker.get_body_pose(session)

                if body_tracked:
                    pelvis_pos = body_tracked.joints.joints(
                        int(schema.BodyJoint.PELVIS)
                    ).pose.position
                    head_pos = body_tracked.joints.joints(
                        int(schema.BodyJoint.HEAD)
                    ).pose.position
                    print(
                        f"  [{elapsed:5.2f}s] Frame {frame_count:4d}"
                        f" | Pelvis: [{pelvis_pos.x:+.2f}, {pelvis_pos.y:+.2f}, {pelvis_pos.z:+.2f}]"
                        f" | Head: [{head_pos.x:+.2f}, {head_pos.y:+.2f}, {head_pos.z:+.2f}]"
                    )
                else:
                    print(f"  [{elapsed:5.2f}s] Frame {frame_count:4d} | inactive")
                last_status_print = current_time

            frame_count += 1
            time.sleep(0.016)  # ~60 FPS

        print()
        print(f"✓ Processed {frame_count} frames")
        print()

        # Test 8: Show final body pose
        print("[Test 8] Final body pose state...")
        body_tracked = body_tracker.get_body_pose(session)

        print(f"  Body tracking active: {'YES' if body_tracked else 'NO'}")

        if body_tracked:
            print()
            print("  Joint positions:")
            for i in range(schema.BodyJoint.NUM_JOINTS):
                joint = body_tracked.joints.joints(i)
                name = schema.BodyJoint(i).name
                pos = joint.pose.position
                rot = joint.pose.orientation
                print(
                    f"    [{i:2d}] {name:15s}: pos=[{pos.x:+.3f}, {pos.y:+.3f}, {pos.z:+.3f}]"
                    f"  rot=[{rot.x:+.3f}, {rot.y:+.3f}, {rot.z:+.3f}, {rot.w:+.3f}]"
                )
        print()

        # Cleanup
        print("[Test 9] Cleanup...")
        print("✓ Resources will be cleaned up when exiting 'with' blocks (RAII)")
        print()

print("=" * 80)
print("✅ All tests passed")
print("=" * 80)
