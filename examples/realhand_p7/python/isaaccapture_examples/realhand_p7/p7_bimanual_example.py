# SPDX-FileCopyrightText: Copyright (c) 2026 RealHand. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anchored P7 dual-arm end-effector targets using official retargeters."""

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.retargeters import (
    EePoseRateLimiter,
    Se3AbsRetargeter,
    Se3RetargeterConfig,
    TensorReorderer,
)
from isaaccapture.retargeting_engine.deviceio_source_nodes import (
    ControllersSource,
    HandsSource,
)
from isaaccapture.retargeting_engine.interface import (
    OutputCombiner,
    TensorGroup,
    ValueInput,
)
from isaaccapture.retargeting_engine.tensor_types import TransformMatrix
from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig

# Stage +X/right, +Y/up, -Z/forward maps to P7 +X/right, +Y/forward, +Z/up.
STAGE_TO_P7 = np.asarray(
    ((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)), dtype=np.float32
)
_TOOL_FRAME = (0, -1, 0, -1, 0, 0, 0, 0, -1)
_FIELDS = ("pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w")
ACTION_ORDER = tuple(f"{prefix}_{field}" for prefix in ("l", "r") for field in _FIELDS)


def build_pipeline(mode: str):
    """Require the consuming environment's world_T_anchor at every graph step."""
    if mode not in ("controller", "handtracking"):
        raise ValueError(f"Unknown input mode: {mode}")
    source = (
        ControllersSource(name="controllers")
        if mode == "controller"
        else HandsSource(name="hands")
    )
    anchor = ValueInput("world_T_anchor", TransformMatrix())
    transformed = source.transformed(anchor.output(ValueInput.VALUE))
    rpy = Rotation.from_matrix(np.asarray(_TOOL_FRAME).reshape(3, 3).T).as_euler(
        "XYZ", degrees=True
    )
    poses = {}
    for side in ("left", "right"):
        device = source.LEFT if side == "left" else source.RIGHT
        arm = Se3AbsRetargeter(
            Se3RetargeterConfig(
                input_device=device,
                zero_out_xy_rotation=False,
                use_wrist_position=True,
                use_wrist_rotation=True,
                target_offset_roll=float(rpy[0]),
                target_offset_pitch=float(rpy[1]),
                target_offset_yaw=float(rpy[2]),
            ),
            name=f"p7_{side}_pose",
        ).connect({device: transformed.output(device)})
        poses[side] = EePoseRateLimiter(name=f"p7_{side}_rate").connect(
            {"ee_pose": arm.output("ee_pose")}
        )
    action = TensorReorderer(
        input_config={
            "left_ee_pose": list(ACTION_ORDER[:7]),
            "right_ee_pose": list(ACTION_ORDER[7:]),
        },
        output_order=list(ACTION_ORDER),
        name="p7_action",
        input_types={"left_ee_pose": "array", "right_ee_pose": "array"},
    ).connect(
        {f"{side}_ee_pose": poses[side].output("ee_pose") for side in ("left", "right")}
    )
    return OutputCombiner({"action": action.output("output")}), list(ACTION_ORDER)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-mode", choices=("controller", "handtracking"), default="handtracking"
    )
    parser.add_argument(
        "--world-t-anchor",
        nargs=16,
        type=float,
        metavar="M",
        help="Row-major stage-to-P7-world matrix",
    )
    parser.add_argument("--duration", type=float, default=360.0)
    parser.add_argument(
        "--dry-run", action="store_true", help="Build the graph without OpenXR"
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()
    pipeline, order = build_pipeline(args.input_mode)
    print(f"P7 end-effector targets: {order}")
    if args.dry_run:
        return
    anchor = TensorGroup(TransformMatrix())
    anchor[0] = (
        np.asarray(args.world_t_anchor, dtype=np.float32).reshape(4, 4)
        if args.world_t_anchor
        else STAGE_TO_P7
    )
    config = TeleopSessionConfig(app_name="RealHandP7", trackers=[], pipeline=pipeline)
    with CloudXRLauncher.launch_context(args), TeleopSession(config) as session:
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            result = session.step(
                external_inputs={"world_T_anchor": {ValueInput.VALUE: anchor}}
            )
            if session.frame_count % 60 == 0:
                action = np.asarray(result["action"][0])
                print(
                    f"left_ee={np.round(action[:7], 3)} right_ee={np.round(action[7:], 3)}"
                )
            time.sleep(0.016)


if __name__ == "__main__":
    main()
