# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SE3 Retargeting Example

Demonstrates using Se3AbsRetargeter and Se3RelRetargeter to generate end-effector
poses from hand tracking data.
"""

import argparse
import sys
import time
import numpy as np

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    HandsSource,
    ControllersSource,
)
from isaaccapture.retargeters import (
    Se3AbsRetargeter,
    Se3RelRetargeter,
    Se3RetargeterConfig,
)
from isaaccapture.teleop_session_manager import (
    TeleopSession,
    TeleopSessionConfig,
)


def run_abs_example(use_controller=False):
    print("\n" + "=" * 80)
    source_type = "Controller" if use_controller else "Hand"
    print(f"  SE3 Absolute Retargeting (Right {source_type})")
    print("=" * 80)
    print(f"Maps {source_type.lower()} pose directly to End-Effector pose.")
    if not use_controller:
        print("Using 'pinch' center (midpoint of thumb/index) for position.")
    print("=" * 80 + "\n")

    if use_controller:
        source = ControllersSource(name="controllers")
        input_device = ControllersSource.RIGHT
    else:
        source = HandsSource(name="hands")
        input_device = HandsSource.RIGHT

    config = Se3RetargeterConfig(
        input_device=input_device,
        use_wrist_position=False,
        zero_out_xy_rotation=False,
    )

    retargeter = Se3AbsRetargeter(config, name="se3_abs")

    pipeline = retargeter.connect({input_device: source.output(input_device)})

    session_config = TeleopSessionConfig(
        app_name="Se3AbsExample",
        trackers=[],  # Auto-discovered
        pipeline=pipeline,
    )

    with TeleopSession(session_config) as session:
        # No session injection needed

        start_time = time.time()
        while time.time() - start_time < 20.0:
            result = session.step()

            # Output: [x, y, z, qx, qy, qz, qw]
            pose = result["ee_pose"][0]
            pos = pose[:3]
            rot = pose[3:]  # x,y,z,w

            if session.frame_count % 30 == 0:
                elapsed = session.get_elapsed_time()
                print(
                    f"[{elapsed:5.1f}s] Pos: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  Rot: ({rot[0]:.2f}, {rot[1]:.2f}, {rot[2]:.2f}, {rot[3]:.2f})"
                )

            time.sleep(0.016)


def run_rel_example(use_controller=False):
    print("\n" + "=" * 80)
    source_type = "Controller" if use_controller else "Hand"
    print(f"  SE3 Relative Retargeting (Right {source_type})")
    print("=" * 80)
    print(f"Maps {source_type.lower()} movement DELTAS to End-Effector deltas.")
    print(f"Move your {source_type.lower()} to generate velocity commands.")
    print("=" * 80 + "\n")

    if use_controller:
        source = ControllersSource(name="controllers")
        input_device = ControllersSource.RIGHT
    else:
        source = HandsSource(name="hands")
        input_device = HandsSource.RIGHT

    config = Se3RetargeterConfig(
        input_device=input_device,
        use_wrist_position=use_controller,
        use_wrist_rotation=use_controller,
        zero_out_xy_rotation=True,
        delta_pos_scale_factor=5.0,
        delta_rot_scale_factor=2.0,
    )

    retargeter = Se3RelRetargeter(config, name="se3_rel")

    pipeline = retargeter.connect({input_device: source.output(input_device)})

    session_config = TeleopSessionConfig(
        app_name="Se3RelExample",
        trackers=[],  # Auto-discovered
        pipeline=pipeline,
    )

    with TeleopSession(session_config) as session:
        # No session injection needed

        start_time = time.time()
        while time.time() - start_time < 20.0:
            result = session.step()

            # Output: [dx, dy, dz, drx, dry, drz]
            delta = result["ee_delta"][0]
            dpos = delta[:3]
            drot = delta[3:]

            if session.frame_count % 30 == 0:
                elapsed = session.get_elapsed_time()
                # Calculate magnitude for easier reading
                vel_mag = np.linalg.norm(dpos)
                rot_mag = np.linalg.norm(drot)
                print(
                    f"[{elapsed:5.1f}s] Vel Mag: {vel_mag:.4f}  Rot Mag: {rot_mag:.4f} | dPos: ({dpos[0]:.3f}, ...)"
                )

            time.sleep(0.016)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    with CloudXRLauncher.launch_context(args):
        print("=" * 80)
        print("  SE3 Retargeting Examples")
        print("=" * 80)
        print("1. Absolute Positioning (Hand -> Pose)")
        print("2. Absolute Positioning (Controller -> Pose)")
        print("3. Relative Positioning (Hand Delta -> Delta)")
        print("4. Relative Positioning (Controller Delta -> Delta)")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            run_abs_example(use_controller=False)
        elif choice == "2":
            run_abs_example(use_controller=True)
        elif choice == "3":
            run_rel_example(use_controller=False)
        elif choice == "4":
            run_rel_example(use_controller=True)
        else:
            print("Invalid choice")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
