# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OpenXR Modular Tracking Example with MCAP Recording

Demonstrates the modular architecture with MCAP data capture:
- Create independent trackers
- Add only the trackers you need
- Record all tracker data to an MCAP file for playback/analysis
- Pass mcap_filename and mcap_channels to DeviceIOSession.run() to enable recording
"""

import argparse
import sys
import time
from datetime import datetime
import isaaccapture.deviceio as deviceio
import isaaccapture.oxr as oxr
from isaaccapture.cloudxr import CloudXRLauncher


RECORD_DURATION_S = 10.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    print("=" * 60)
    print("OpenXR Modular Tracking Example with MCAP Recording")
    print("=" * 60)
    print()

    # Generate timestamped filename for recording
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mcap_filename = f"tracking_recording_{timestamp}.mcap"

    with CloudXRLauncher.launch_context(args):
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
        with oxr.OpenXRSession(
            "ModularExampleWithMCAP", required_extensions
        ) as oxr_session:
            handles = oxr_session.get_handles()
            print("✓ OpenXR session created")

            # Run deviceio session with MCAP recording enabled.
            print("\nRunning deviceio session with MCAP recording...")
            recording_config = deviceio.McapRecordingConfig(
                mcap_filename, [(hand_tracker, "hands"), (head_tracker, "head")]
            )
            with deviceio.DeviceIOSession.run(
                trackers, handles, recording_config
            ) as session:
                print("✓ DeviceIO session initialized with all trackers!")
                print(f"✓ MCAP recording active → {mcap_filename}")
                print()

                # Main tracking loop
                print("=" * 60)
                print(f"Tracking ({RECORD_DURATION_S} seconds)...")
                print("=" * 60)
                print()

                frame_count = 0
                start_time = time.time()

                while time.time() - start_time < RECORD_DURATION_S:
                    session.update()

                    # Print every 60 frames (~1 second)
                    if frame_count % 60 == 0:
                        elapsed = time.time() - start_time
                        print(f"[{elapsed:4.1f}s] Frame {frame_count} (recording...)")
                        print()

                    frame_count += 1
                    time.sleep(0.016)  # ~60 FPS

                print(f"\nProcessed {frame_count} frames")

            print("✓ Recording stopped (MCAP file closed by session destructor)")

        print()
        print("=" * 60)
        print(f"✓ Recording saved to: {mcap_filename}")
        print("=" * 60)

    # ---- Replay the recorded MCAP file (no live OpenXR session required) ----
    print()
    print("=" * 60)
    print("Replaying recorded MCAP data")
    print("=" * 60)
    print()

    hand_tracker = deviceio.HandTracker()
    head_tracker = deviceio.HeadTracker()
    replay_config = deviceio.McapReplayConfig(
        mcap_filename, [(hand_tracker, "hands"), (head_tracker, "head")]
    )
    with deviceio.ReplaySession.run(
        replay_config,
    ) as replay_session:
        print(f"✓ Replay session opened: {mcap_filename}")
        print(f"  Replaying {RECORD_DURATION_S}s of recorded data...")
        print()

        replay_frame = 0
        start_time = time.time()

        while time.time() - start_time < RECORD_DURATION_S:
            replay_session.update()

            if replay_frame % 60 == 0:
                elapsed = time.time() - start_time
                head = head_tracker.get_head(replay_session)
                left = hand_tracker.get_left_hand(replay_session)
                right = hand_tracker.get_right_hand(replay_session)

                print(f"[{elapsed:4.1f}s] Replay frame {replay_frame}")
                if head is not None and head.pose:
                    p = head.pose.position
                    print(f"  Head  pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")
                else:
                    print("  Head  pos=N/A")
                print(f"  Left  hand={'present' if left is not None else 'None'}")
                print(f"  Right hand={'present' if right is not None else 'None'}")
                print()

            replay_frame += 1
            time.sleep(0.016)  # ~60 FPS

        print(f"\nReplay complete: {replay_frame} frames")

    print()
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
