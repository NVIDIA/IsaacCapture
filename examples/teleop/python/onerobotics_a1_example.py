# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OneRobotics A1 Isaac Teleop example.

The ``demo`` mode drives the packaged right-arm MJCF with a bounded sinusoid and
opens a MuJoCo viewer. The ``live`` mode consumes a JointState tracker through
the Isaac Teleop pipeline:

    JointStateSource -> JointStateRetargeter -> OutputCombiner("action")

The live mode expects a device/plugin publishing the collection
``onerobotics_a1_right`` with the seven joint names declared below. It does not
claim that the included asset is a hardware driver.

Examples::

    python onerobotics_a1_example.py --mode demo
    python onerobotics_a1_example.py --mode live --frames 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeters import (
    JointStateRetargeter,
    JointStateRetargeterConfig,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import JointStateSource
from isaaccapture.retargeting_engine.interface import OutputCombiner


A1_RIGHT_JOINTS = [
    "joint1-a1_r",
    "joint2-a1_r",
    "joint3-a1_r",
    "joint4-a1_r",
    "joint5-a1_r",
    "joint6-a1_r",
    "joint7-a1_r",
]
COLLECTION_ID = "onerobotics_a1_right"
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "onerobotics_a1"
MJCF_PATH = ASSET_DIR / "mjcf" / "A1" / "a1_right_position.xml"


def build_pipeline(source: JointStateSource):
    """Build the standard joint-space Isaac Teleop action graph for A1."""
    retargeter = JointStateRetargeter(
        name="onerobotics_a1_right",
        mode="joint",
        config=JointStateRetargeterConfig(
            device_joints=A1_RIGHT_JOINTS,
            target_joints=A1_RIGHT_JOINTS,
        ),
    )
    mapped = retargeter.connect(
        {JointStateRetargeter.JOINTS: source.output(JointStateSource.JOINTS)}
    )
    return OutputCombiner({"action": mapped.output("joint_targets")})


def run_demo(duration: float) -> None:
    """Exercise the packaged model without an OpenXR runtime or hardware."""
    import mujoco
    import mujoco.viewer

    if not MJCF_PATH.is_file():
        raise SystemExit(f"A1 MJCF not found: {MJCF_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    data = mujoco.MjData(model)
    ranges = np.array(model.jnt_range[: len(A1_RIGHT_JOINTS)], dtype=float)
    home = np.mean(ranges, axis=1)
    amplitude = 0.20 * (ranges[:, 1] - ranges[:, 0])
    started = time.monotonic()
    print(f"A1 demo: {MJCF_PATH}")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and (
            duration <= 0 or time.monotonic() - started < duration
        ):
            phase = time.monotonic() - started
            data.qpos[: len(A1_RIGHT_JOINTS)] = home + amplitude * np.sin(phase)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1 / 60)


def run_live(frames: int, timeout: float, args: argparse.Namespace) -> None:
    """Consume A1 joint states through TeleopSession and print mapped actions."""
    from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig

    source = JointStateSource(
        name="onerobotics_a1_right",
        collection_id=args.collection_id,
        joint_names=A1_RIGHT_JOINTS,
    )
    pipeline = build_pipeline(source)
    config = TeleopSessionConfig(
        app_name="OneRoboticsA1JointSpaceExample",
        trackers=[],
        pipeline=pipeline,
        plugins=[],
    )
    deadline = time.monotonic() + timeout
    received = 0
    with CloudXRLauncher.launch_context(args), TeleopSession(config) as session:
        while received < frames and time.monotonic() < deadline:
            result = session.step()
            action = result.get("action")
            if action is not None:
                values = np.asarray(action[0], dtype=float)
                print(
                    f"frame {received:04d} | "
                    + " ".join(
                        f"{name}={value:+.4f}"
                        for name, value in zip(A1_RIGHT_JOINTS, values)
                    )
                )
                received += 1
            time.sleep(0.016)
    if received < frames:
        raise SystemExit(
            f"received {received}/{frames} A1 actions before timeout; "
            f"check collection_id={args.collection_id!r} and the joint-state producer"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("demo", "live"), default="demo")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--collection-id", default=COLLECTION_ID)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()
    if args.duration < 0 or args.timeout <= 0 or args.frames <= 0:
        parser.error("duration must be >= 0; timeout and frames must be positive")
    if args.mode == "demo":
        run_demo(args.duration)
    else:
        run_live(args.frames, args.timeout, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
