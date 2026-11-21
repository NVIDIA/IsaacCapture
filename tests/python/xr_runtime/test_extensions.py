#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test DeviceIOSession.get_required_extensions() (session-level aggregate).

Query required extensions before creating an OpenXR session; individual trackers
do not expose get_required_extensions().
"""

import isaaccapture.deviceio as deviceio
import isaaccapture.oxr as oxr

print("=" * 80)
print("OpenXR Required Extensions Test")
print("=" * 80)
print()

# Test 1: Hand tracker only
print("[Test 1] HandTracker extensions")
hand_tracker = deviceio.HandTracker()

extensions1 = deviceio.DeviceIOSession.get_required_extensions([hand_tracker])
print(f"  Required extensions: {len(extensions1)}")
for ext in extensions1:
    print(f"    - {ext}")
print()

# Test 2: Head tracker only
print("[Test 2] HeadTracker extensions")
head_tracker = deviceio.HeadTracker()

extensions2 = deviceio.DeviceIOSession.get_required_extensions([head_tracker])
print(f"  Required extensions: {len(extensions2)}")
for ext in extensions2:
    print(f"    - {ext}")
print()

# Test 3: Both trackers
print("[Test 3] HandTracker + HeadTracker extensions")
hand_tracker3 = deviceio.HandTracker()
head_tracker3 = deviceio.HeadTracker()

extensions3 = deviceio.DeviceIOSession.get_required_extensions(
    [hand_tracker3, head_tracker3]
)
print(f"  Required extensions: {len(extensions3)}")
for ext in extensions3:
    print(f"    - {ext}")
print()

# Test 4: Use case - query before creating external session
print("[Test 4] Use case: Query before external session creation")
print()
print("Scenario: You want to create your own OpenXR session")
print("          and pass it to DeviceIOSession.run().")
print()

hand = deviceio.HandTracker()
head = deviceio.HeadTracker()

trackers = [hand, head]

# Query extensions BEFORE initializing
required_exts = deviceio.DeviceIOSession.get_required_extensions(trackers)

print("Step 1: Create trackers")
print("Step 2: Query required extensions:")
for ext in required_exts:
    print(f"  - {ext}")
print()
print("Step 3: Create your own OpenXR instance with these extensions")
print("        (in C++ or custom code)")
print()
print("Step 4: Pass trackers and handles to DeviceIOSession.run()")
print()

# Now initialize normally to show it works
print("[Test 5] Normal initialization with queried extensions (RAII)")

# Create OpenXR session with the queried extensions
with oxr.OpenXRSession("ExtensionTest", required_exts) as oxr_session:
    handles = oxr_session.get_handles()
    # run() throws exception on failure
    with deviceio.DeviceIOSession.run(trackers, handles) as session:
        print("  ✅ Initialized successfully")

        # Quick update test
        session.update()
        left_tracked = hand.get_left_hand(session)
        head_tracked = head.get_head(session)
        print("  ✅ Update successful")
        if left_tracked:
            pos = left_tracked.joints.poses(deviceio.JOINT_WRIST).pose.position
            print(f"    Left wrist: [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
        else:
            print("    Left hand:  inactive")
        if head_tracked:
            pos = head_tracked.pose.position
            print(f"    Head pos:   [{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]")
        else:
            print("    Head:       inactive")

        # Session will be cleaned up when exiting 'with' block (RAII)

print()
print("=" * 80)
print("✅ Extension query test complete")
print("=" * 80)
print()
