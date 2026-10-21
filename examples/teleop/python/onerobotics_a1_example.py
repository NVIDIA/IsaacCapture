# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OneRobotics A1 right-arm end-effector target example.

The A1 is the robot being teleoperated, not an operator-side input device. This
example maps the right XR controller pose to a 7-D ``right_ee_pose`` target:

    ControllersSource -> Se3AbsRetargeter -> OutputCombiner("right_ee_pose")

An Isaac Lab A1_2026 environment or another robot controller can consume that
target and solve it for the right arm. Robot assets, inverse kinematics, and the
robot controller intentionally remain with that consumer.

Example::

    python onerobotics_a1_example.py --duration 30
"""

from __future__ import annotations

import argparse
import sys

from _example_loop import add_duration_argument, run_example_loop

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeters import (
    Se3AbsRetargeter,
    Se3RetargeterConfig,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import ControllersSource
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.retargeting_engine.tensor_types import ControllerInputIndex
from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig


def build_pipeline() -> OutputCombiner:
    """Map the right XR controller to an A1 right-arm end-effector target."""
    controllers = ControllersSource(name="controllers")
    retargeter = Se3AbsRetargeter(
        Se3RetargeterConfig(
            input_device=ControllersSource.RIGHT,
            use_wrist_position=True,
            use_wrist_rotation=True,
            zero_out_xy_rotation=False,
        ),
        name="a1_right_ee_target",
    )
    mapped = retargeter.connect(
        {
            ControllersSource.RIGHT: controllers.output(ControllersSource.RIGHT),
        }
    )
    return OutputCombiner(
        {
            "right_ee_pose": mapped.output("ee_pose"),
            "right_controller": controllers.output(ControllersSource.RIGHT),
        }
    )


def run(duration: float) -> None:
    """Print valid right-arm pose targets for an external A1 controller."""
    config = TeleopSessionConfig(
        app_name="OneRoboticsA1RightArmTargetExample",
        trackers=[],
        pipeline=build_pipeline(),
        plugins=[],
    )
    valid_samples = 0
    with TeleopSession(config) as session:

        def step() -> None:
            nonlocal valid_samples
            result = session.step()
            controller = result["right_controller"]
            if controller.is_none or not bool(
                controller[ControllerInputIndex.GRIP_IS_VALID]
            ):
                return

            valid_samples += 1
            if valid_samples % 30 != 1:
                return
            pose = result["right_ee_pose"][0]
            print(
                f"sample {valid_samples:04d} | "
                f"position=({pose[0]:+.3f}, {pose[1]:+.3f}, {pose[2]:+.3f}) | "
                f"quaternion_xyzw=({pose[3]:+.3f}, {pose[4]:+.3f}, "
                f"{pose[5]:+.3f}, {pose[6]:+.3f})"
            )

        run_example_loop(step, duration)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_duration_argument(parser)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    with CloudXRLauncher.launch_context(args):
        run(args.duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
