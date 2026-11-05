#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OpenXR Modular Tracking Example

Demonstrates the modular architecture where you can:
- Create independent trackers
- Add only the trackers you need
- Easily extend with new tracker types
"""

import argparse
import sys
import time
import isaaccapture.deviceio as deviceio
import isaaccapture.oxr as oxr
import isaaccapture.schema as schema
from isaaccapture.cloudxr import CloudXRLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    with CloudXRLauncher.launch_context(args):
        print("=" * 60)
        print("OpenXR Modular Tracking Example")
        print("=" * 60)
        print()

        # Create trackers independently
        print("Creating trackers...")
        hand_tracker = deviceio.HandTracker()
        head_tracker = deviceio.HeadTracker()
        print(f"✓ Created {hand_tracker.get_name()}")
        print(f"✓ Created {head_tracker.get_name()}")

        # Get required extensions
        print("\nQuerying required extensions...")
        trackers = [hand_tracker, head_tracker]
        required_extensions = deviceio.DeviceIOSession.get_required_extensions(trackers)
        print(f"✓ Required extensions: {required_extensions}")

        # Create OpenXR session
        print("\nCreating OpenXR session...")
        with oxr.OpenXRSession("ModularExample", required_extensions) as oxr_session:
            handles = oxr_session.get_handles()
            print("✓ OpenXR session created")

            # Run deviceio session with trackers (throws exception on failure)
            print("\nRunning deviceio session with trackers...")
            with deviceio.DeviceIOSession.run(trackers, handles) as session:
                print("✓ DeviceIO session initialized with all trackers!")
                print()

                # Main tracking loop
                print("=" * 60)
                print("Tracking (10 seconds)...")
                print("=" * 60)
                print()

                frame_count = 0
                start_time = time.time()

                while time.time() - start_time < 10.0:
                    # Update session and all trackers
                    session.update()

                    # Print every 60 frames (~1 second)
                    if frame_count % 60 == 0:
                        elapsed = time.time() - start_time
                        print(f"[{elapsed:4.1f}s] Frame {frame_count}")

                        # Get hand data
                        left_tracked: schema.HandPose | None = (
                            hand_tracker.get_left_hand(session)
                        )
                        right_tracked: schema.HandPose | None = (
                            hand_tracker.get_right_hand(session)
                        )

                        if left_tracked:
                            pos = left_tracked.joints.poses(
                                deviceio.JOINT_WRIST
                            ).pose.position
                            print(
                                f"  Left wrist:  [{pos.x:6.3f}, {pos.y:6.3f}, {pos.z:6.3f}]"
                            )
                        else:
                            print("  Left hand:   inactive")

                        if right_tracked:
                            pos = right_tracked.joints.poses(
                                deviceio.JOINT_WRIST
                            ).pose.position
                            print(
                                f"  Right wrist: [{pos.x:6.3f}, {pos.y:6.3f}, {pos.z:6.3f}]"
                            )
                        else:
                            print("  Right hand:  inactive")

                        # Get head data
                        head_tracked: schema.HeadPose | None = head_tracker.get_head(
                            session
                        )
                        if head_tracked:
                            pos = head_tracked.pose.position
                            print(
                                f"  Head pos:    [{pos.x:6.3f}, {pos.y:6.3f}, {pos.z:6.3f}]"
                            )
                        else:
                            print("  Head:        inactive")

                        print()

                    frame_count += 1
                    time.sleep(0.016)  # ~60 FPS

                # Cleanup
                print(f"\nProcessed {frame_count} frames")
                print("Cleaning up (RAII)...")
                print("✓ Resources will be cleaned up when exiting 'with' blocks")

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
