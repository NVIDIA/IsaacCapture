# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive the isaacteleop mock the same way Guman's IsaacTeleopBackend does.

Mirrors ``GumanIsaacTeleopUsage.md`` §3.1-§3.4: build a CloudXRLauncher + session config, enter
TeleopSession, step() once per tick reading the result dict exactly like Guman's
``build_snapshot()`` does, then tear down. No headset, no browser tab, no CloudXR runtime.
"""

import time

import numpy as np

from isaacteleop.cloudxr import CloudXRLauncher
from isaacteleop.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    HandsSource,
    HeadSource,
)
from isaacteleop.retargeting_engine.interface import OutputCombiner
from isaacteleop.retargeting_engine.tensor_types import (
    ControllerInputIndex,
    HeadPoseIndex,
)
from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig


def default_pose_vec() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def head_pose_vec(head_data) -> np.ndarray:
    """Same decode pattern every Guman helper uses (GumanIsaacTeleopUsage.md §6)."""
    if head_data is None or head_data.is_none:
        return default_pose_vec()
    if not bool(head_data[HeadPoseIndex.IS_VALID]):
        return default_pose_vec()
    pos = np.asarray(head_data[HeadPoseIndex.POSITION])
    quat = np.asarray(head_data[HeadPoseIndex.ORIENTATION])
    return np.concatenate([pos, quat])


def main() -> int:
    launcher = CloudXRLauncher(
        install_dir="/tmp/mock-cloudxr-install",
        env_config=None,
        accept_eula=True,
        setup_oob=False,
        usb_local=False,
        run_embedded=True,
    )

    controllers = ControllersSource(name="controllers")
    head = HeadSource(name="head")
    hands = HandsSource(name="hands")
    pipeline = OutputCombiner(
        {
            "controller_left": controllers.output(ControllersSource.LEFT),
            "controller_right": controllers.output(ControllersSource.RIGHT),
            "head": head.output("head"),
            "hand_left": hands.output(HandsSource.LEFT),
            "hand_right": hands.output(HandsSource.RIGHT),
        }
    )
    config = TeleopSessionConfig(app_name="GumanShapedUsageExample", pipeline=pipeline)

    print("Press Ctrl+C to exit\n")
    with TeleopSession(config) as session:
        try:
            while True:
                result = session.step()
                head_vec = head_pose_vec(result.get("head"))
                left = result.get("controller_left")
                trigger = (
                    None
                    if left is None or left.is_none
                    else left[ControllerInputIndex.TRIGGER_VALUE]
                )
                print(f"head={head_vec.round(3)}  left_trigger={trigger}")
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            session.step()  # final harmless step, mirroring Guman's close()

    launcher.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
