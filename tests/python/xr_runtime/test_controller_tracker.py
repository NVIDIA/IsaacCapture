# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script for ControllerTracker with XR_NVX1_action_context support.

Demonstrates:
- Getting left and right controller data via get_left_controller() and get_right_controller()
- Multiple ControllerTracker instances on the same session (action context isolation)
"""

import time

import isaaccapture.deviceio as deviceio
import isaaccapture.oxr as oxr

print("=" * 80)
print("Controller Tracker Test")
print("=" * 80)
print()

# Test 1: Create two controller trackers to verify action context isolation
print("[Test 1] Creating two controller trackers...")
controller_tracker_a = deviceio.ControllerTracker()
controller_tracker_b = deviceio.ControllerTracker()
print(f"✓ {controller_tracker_a.get_name()} A created")
print(f"✓ {controller_tracker_b.get_name()} B created")
print()

# Test 2: Query required extensions (should include XR_NVX1_action_context)
print("[Test 2] Querying required extensions...")
trackers = [controller_tracker_a, controller_tracker_b]
required_extensions = deviceio.DeviceIOSession.get_required_extensions(trackers)
print(f"Required extensions: {required_extensions}")
assert "XR_NVX1_action_context" in required_extensions, (
    "Expected XR_NVX1_action_context in required extensions"
)
print()

# Test 3: Initialize — both trackers on the same XrSession
print("[Test 3] Creating OpenXR session with two ControllerTrackers...")

with oxr.OpenXRSession("ControllerTrackerTest", required_extensions) as oxr_session:
    handles = oxr_session.get_handles()

    with deviceio.DeviceIOSession.run(trackers, handles) as session:
        print(
            "✅ OpenXR session initialized with two ControllerTrackers (action context isolation)"
        )
        print()

        # Test 4: Initial update
        print("[Test 4] Testing initial data retrieval...")
        session.update()

        print("✓ Update successful")
        print()

        # Test 5: Both trackers should see the same controller state
        print("[Test 5] Verifying both trackers report consistent data...")
        left_a = controller_tracker_a.get_left_controller(session)
        left_b = controller_tracker_b.get_left_controller(session)
        right_a = controller_tracker_a.get_right_controller(session)
        right_b = controller_tracker_b.get_right_controller(session)

        def assert_trackers_consistent(label, ta, tb):
            a_active = ta is not None and ta.grip_pose.is_valid
            b_active = tb is not None and tb.grip_pose.is_valid
            assert a_active == b_active, (
                f"{label}: A active={a_active} but B active={b_active}"
            )
            if a_active:
                pa = ta.grip_pose.pose.position
                pb = tb.grip_pose.pose.position
                tol = 0.01
                assert abs(pa.x - pb.x) < tol, f"{label} x: {pa.x} vs {pb.x}"
                assert abs(pa.y - pb.y) < tol, f"{label} y: {pa.y} vs {pb.y}"
                assert abs(pa.z - pb.z) < tol, f"{label} z: {pa.z} vs {pb.z}"
                print(f"  {label}: A and B match [{pa.x:.3f}, {pa.y:.3f}, {pa.z:.3f}]")
            else:
                print(f"  {label}: both inactive (consistent)")

        assert_trackers_consistent("Left", left_a, left_b)
        assert_trackers_consistent("Right", right_a, right_b)
        print("✓ Tracker A and B report consistent data")
        print()

        # Test 6: Available inputs
        print("[Test 6] Available controller inputs:")
        print("  Buttons: primary_click, secondary_click, thumbstick_click, menu_click")
        print("  Axes: thumbstick_x, thumbstick_y, squeeze_value, trigger_value")
        print()

        # Test 7: Run tracking loop
        print("[Test 7] Running controller tracking loop (10 seconds)...")
        print("Press buttons or move controls to see state!")
        print()

        frame_count = 0
        start_time = time.time()
        last_status_print = start_time

        while time.time() - start_time < 10.0:
            session.update()

            current_time = time.time()
            if current_time - last_status_print >= 0.5:
                elapsed = current_time - start_time
                left_tracked = controller_tracker_a.get_left_controller(session)
                right_tracked = controller_tracker_a.get_right_controller(session)

                print(f"  [{elapsed:5.2f}s] Frame {frame_count:4d}")

                left_data = left_tracked
                if left_data:
                    li = left_data.inputs
                    print(
                        f"    L: Trig={li.trigger_value:.2f} Sq={li.squeeze_value:.2f}"
                        f" Stick=({li.thumbstick_x:+.2f},{li.thumbstick_y:+.2f})"
                        f" Btn=[{int(li.primary_click)}{int(li.secondary_click)}{int(li.thumbstick_click)}{int(li.menu_click)}]"
                    )
                else:
                    print("    L: INACTIVE")

                right_data = right_tracked
                if right_data:
                    ri = right_data.inputs
                    print(
                        f"    R: Trig={ri.trigger_value:.2f} Sq={ri.squeeze_value:.2f}"
                        f" Stick=({ri.thumbstick_x:+.2f},{ri.thumbstick_y:+.2f})"
                        f" Btn=[{int(ri.primary_click)}{int(ri.secondary_click)}{int(ri.thumbstick_click)}{int(ri.menu_click)}]"
                    )
                else:
                    print("    R: INACTIVE")
                last_status_print = current_time

            frame_count += 1
            time.sleep(0.016)  # ~60 FPS

        print()
        print(f"✓ Processed {frame_count} frames")
        print()

        # Test 8: Show final statistics
        print("[Test 8] Final controller state...")

        def print_controller_summary(hand_name, tracked):
            print(f"  {hand_name} Controller:")
            if tracked:
                pos = tracked.grip_pose.pose.position
                print(f"    Grip position: [{pos.x:+.3f}, {pos.y:+.3f}, {pos.z:+.3f}]")
                pos = tracked.aim_pose.pose.position
                print(f"    Aim position:  [{pos.x:+.3f}, {pos.y:+.3f}, {pos.z:+.3f}]")
                inputs = tracked.inputs
                print(f"    Trigger: {inputs.trigger_value:.2f}")
                print(f"    Squeeze: {inputs.squeeze_value:.2f}")
                print(
                    f"    Thumbstick: ({inputs.thumbstick_x:+.2f}, {inputs.thumbstick_y:+.2f})"
                )
                print(
                    f"    Primary: {'PRESSED' if inputs.primary_click else 'released'}"
                )
                print(
                    f"    Secondary: {'PRESSED' if inputs.secondary_click else 'released'}"
                )
                print(f"    Menu: {'PRESSED' if inputs.menu_click else 'released'}")
            else:
                print("    inactive")

        left_tracked = controller_tracker_a.get_left_controller(session)
        right_tracked = controller_tracker_a.get_right_controller(session)
        print_controller_summary("Left", left_tracked)
        print()
        print_controller_summary("Right", right_tracked)
        print()

        # Cleanup
        print("[Test 9] Cleanup...")
        print("✓ Resources will be cleaned up when exiting 'with' blocks (RAII)")
        print()

print("=" * 80)
print("✅ All tests passed")
print("=" * 80)
