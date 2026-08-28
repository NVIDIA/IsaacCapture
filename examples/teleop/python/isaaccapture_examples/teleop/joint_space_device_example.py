# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint-space device example: SO-101 leader arm -> Isaac Lab action.

This CLI is both an **API reference** and an **end-to-end test** for the generic joint-space
device path on the Isaac Teleop side:

    JointStateOutput schema  ->  JointStateSource  ->  JointStateRetargeter  ->  TensorReorderer
                                                                                       |
                                                                              OutputCombiner("action")

It consumes joint state streamed by the real ``so101_leader`` plugin over the OpenXR tensor
transport via a ``TeleopSession`` (the ``JointStateSource`` auto-discovers and polls the
``JointStateTracker``). ``CloudXRLauncher`` attaches to the CloudXR runtime and WSS proxy,
starting a background service if none is serving. Use ``--launch-plugin`` to spawn the synthetic plugin
process automatically; otherwise start ``so101_leader`` (or any device pushing the same
``collection_id``) separately.

Two modes:

* ``--mode joint`` -> 6-D joint mirror ``[shoulder_pan, ..., gripper]`` (no extra deps).
* ``--mode ee``   -> 8-D ``[pos_xyz, quat_xyzw, gripper]`` via URDF forward kinematics
                     (needs ``pinocchio`` and ``--urdf`` pointing at ``so101_new_calib.urdf``).

Examples::

    python joint_space_device_example.py --launch-plugin --mode joint --frames 8
    python joint_space_device_example.py --launch-plugin --mode ee --urdf /path/to/so101_new_calib.urdf
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeting_engine.deviceio_source_nodes import JointStateSource
from isaaccapture.retargeting_engine.interface import OutputCombiner
from isaaccapture.retargeters import (
    JointStateRetargeter,
    JointStateRetargeterConfig,
    TensorReorderer,
)

# Canonical SO-101 DOF names (match Simulation/SO101/so101_new_calib.urdf and the schema).
SO101_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

_COLLECTION_ID = "so101_leader"
_POSE_LABELS = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
_DEFAULT_PLUGIN_BIN = (
    # Six levels up is the repo root; see PLUGIN_ROOT_DIR in the sibling examples.
    Path(__file__).resolve().parents[5]
    / "build/src/plugins/so101_leader/so101_leader_plugin"
)


def build_pipeline(
    mode: str, source: JointStateSource, urdf_path: str | None, ee_link: str
):
    """Build the action pipeline from the live ``JointStateSource`` leaf.

    Returns ``(combiner, action_labels)``."""
    if mode == "joint":
        retargeter = JointStateRetargeter(
            name="leader",
            mode="joint",
            config=JointStateRetargeterConfig(
                device_joints=SO101_JOINTS, target_joints=SO101_JOINTS
            ),
        )
        head = retargeter.connect(
            {JointStateRetargeter.JOINTS: source.output(JointStateSource.JOINTS)}
        )
        reorderer = TensorReorderer(
            input_config={"joint_targets": SO101_JOINTS},
            output_order=SO101_JOINTS,
            name="action_reorderer",
            input_types={"joint_targets": "scalar"},
        )
        connected = reorderer.connect({"joint_targets": head.output("joint_targets")})
        combiner = OutputCombiner({"action": connected.output("output")})
        return combiner, list(SO101_JOINTS)

    if not urdf_path:
        raise SystemExit("--mode ee requires --urdf <so101_new_calib.urdf>")
    retargeter = JointStateRetargeter(
        name="leader",
        mode="ee_pose",
        config=JointStateRetargeterConfig(
            device_joints=SO101_JOINTS,
            urdf_path=urdf_path,
            ee_link=ee_link,
            gripper_joint="gripper",
        ),
    )
    head = retargeter.connect(
        {JointStateRetargeter.JOINTS: source.output(JointStateSource.JOINTS)}
    )
    action_labels = _POSE_LABELS + ["gripper_value"]
    reorderer = TensorReorderer(
        input_config={"ee_pose": _POSE_LABELS, "gripper_command": ["gripper_value"]},
        output_order=action_labels,
        name="action_reorderer",
        input_types={"ee_pose": "array", "gripper_command": "scalar"},
    )
    connected = reorderer.connect(
        {
            "ee_pose": head.output("ee_pose"),
            "gripper_command": head.output("gripper_command"),
        }
    )
    combiner = OutputCombiner({"action": connected.output("output")})
    return combiner, action_labels


def run_live(
    mode: str, num_frames: int, urdf_path: str | None, ee_link: str, timeout_s: float
) -> None:
    """Consume the live so101_leader plugin stream through a TeleopSession over OpenXR."""
    from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig

    source = JointStateSource(
        name="leader", collection_id=_COLLECTION_ID, joint_names=SO101_JOINTS
    )
    combiner, labels = build_pipeline(mode, source, urdf_path, ee_link)

    print(
        f"mode={mode}  action_dim={len(labels)}  layout={labels}  collection={_COLLECTION_ID!r}"
    )
    print("-" * 80)

    session_config = TeleopSessionConfig(
        app_name="JointSpaceDeviceLiveExample",
        trackers=[],
        pipeline=combiner,
        plugins=[],
    )
    actions: list[np.ndarray] = []
    with TeleopSession(session_config) as session:
        deadline = time.time() + timeout_s
        frame = 0
        while len(actions) < num_frames and time.time() < deadline:
            result = session.step()
            action = result.get("action")
            if action is not None:
                arr = np.asarray(action[0], dtype=np.float64)
                actions.append(arr)
                print(
                    f"step {frame:02d} | action = [ {'  '.join(f'{v:+.3f}' for v in arr)} ]"
                )
            frame += 1
            time.sleep(0.05)

    print("-" * 80)
    if len(actions) < num_frames:
        raise SystemExit(
            f"FAILED: only {len(actions)}/{num_frames} action(s) received from the live plugin "
            f"within {timeout_s:.0f}s (is the so101_leader plugin pushing?)"
        )
    # A single received frame can't be "stale" -- only flag multi-frame runs that never change.
    varied = len(actions) <= 1 or any(
        not np.allclose(actions[i], actions[0], atol=1e-4)
        for i in range(1, len(actions))
    )
    print(
        f"OK: received {len(actions)} live action(s) of width {len(labels)}; varying over time: {varied}"
    )
    if not varied:
        raise SystemExit(
            "FAILED: live actions did not vary -- stream may be stale (held-last only)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["joint", "ee"], default="joint")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument(
        "--urdf", default=None, help="Path to so101_new_calib.urdf (ee mode)"
    )
    parser.add_argument(
        "--ee-link",
        default="gripper_frame_link",
        help="URDF end-effector frame (ee mode)",
    )
    parser.add_argument(
        "--launch-plugin",
        action="store_true",
        help="Spawn the synthetic so101_leader plugin process automatically",
    )
    parser.add_argument(
        "--plugin-bin",
        default=str(_DEFAULT_PLUGIN_BIN),
        help="Path to so101_leader_plugin",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for plugin frames",
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    plugin_proc = None
    if args.launch_plugin:
        if not Path(args.plugin_bin).exists():
            raise SystemExit(
                f"plugin binary not found: {args.plugin_bin} (build it first)"
            )
        print(f"launching plugin: {args.plugin_bin}")
        # Empty device_path -> synthetic backend; collection id must match the source.
        plugin_proc = subprocess.Popen([args.plugin_bin, "", _COLLECTION_ID])
        time.sleep(1.5)  # let it create its OpenXR session and start pushing
    try:
        with CloudXRLauncher.launch_context(args):
            run_live(args.mode, args.frames, args.urdf, args.ee_link, args.timeout)
    finally:
        if plugin_proc is not None:
            plugin_proc.terminate()
            try:
                plugin_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                plugin_proc.kill()


if __name__ == "__main__":
    main()
