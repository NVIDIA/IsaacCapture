#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sharpa Hand Retargeter Demo — bimanual teleop sample app.

Two modes:

  Live (default):
    Reads OpenXR hand tracking from a Quest headset via TeleopSession and
    feeds both hands through SharpaHandRetargeter IK in real time.

  Synthetic (--synthetic):
    Animates a curl sequence on a single right hand with no headset required.

Usage:
    # Live bimanual from Quest headset:
    python sharpa_hand_retargeter_demo.py

    # Synthetic animation (no headset):
    python sharpa_hand_retargeter_demo.py --synthetic

    # Custom MJCF paths:
    python sharpa_hand_retargeter_demo.py --left-mjcf /path/to/left.xml --right-mjcf /path/to/right.xml
"""

import argparse
import sys
import time

from isaaccapture.cloudxr import CloudXRLauncher

import numpy as np

from isaaccapture.retargeters import SharpaHandRetargeter, SharpaHandRetargeterConfig


def _default_mjcf(name: str) -> str:
    """Resolve a Sharpa MJCF shipped inside the robotic_grounding wheel.

    The `[grounding]` extra requires `robotic_grounding` to be installed
    (see references/retargeting/sharpa.rst). If the user invokes this demo without
    it, the import of SharpaHandRetargeter above already raised a clear
    error long before this function is called.
    """
    from importlib.resources import files

    return str(files("robotic_grounding") / "assets" / "xmls" / "sharpawave" / name)


DEFAULT_LEFT_MJCF = _default_mjcf("left_sharpawave_nomesh.xml")
DEFAULT_RIGHT_MJCF = _default_mjcf("right_sharpawave_nomesh.xml")


# ---------------------------------------------------------------------------
# Synthetic mode helpers (no headset)
# ---------------------------------------------------------------------------


def _run_synthetic(mjcf_path: str) -> int:
    from isaaccapture.retargeting_engine.interface import (
        ComputeContext,
        ExecutionEvents,
        ExecutionState,
        OptionalTensorGroup,
        TensorGroup,
    )
    from isaaccapture.retargeting_engine.interface.retargeter_core_types import (
        GraphTime,
    )
    from isaaccapture.retargeting_engine.interface.tensor_group_type import (
        OptionalTensorGroupType,
    )
    from isaaccapture.retargeting_engine.tensor_types import (
        HandInput,
        HandInputIndex,
        HandJointIndex,
        NUM_HAND_JOINTS,
    )

    ID_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def _make_context() -> ComputeContext:
        return ComputeContext(
            graph_time=GraphTime(sim_time_ns=0, real_time_ns=0),
            execution_events=ExecutionEvents(
                reset=False, execution_state=ExecutionState.RUNNING
            ),
        )

    def _build_io(retargeter):
        inputs = {}
        for k, v in retargeter.input_spec().items():
            if isinstance(v, OptionalTensorGroupType):
                inputs[k] = OptionalTensorGroup(v)
            else:
                inputs[k] = TensorGroup(v)
        outputs = {}
        for k, v in retargeter.output_spec().items():
            if isinstance(v, OptionalTensorGroupType):
                outputs[k] = OptionalTensorGroup(v)
            else:
                outputs[k] = TensorGroup(v)
        return inputs, outputs

    def make_hand_pose(curl_factor: float = 0.0) -> TensorGroup:
        tg = TensorGroup(HandInput())
        positions = np.zeros((NUM_HAND_JOINTS, 3), dtype=np.float32)
        orientations = np.tile(ID_QUAT, (NUM_HAND_JOINTS, 1))
        valid = np.ones(NUM_HAND_JOINTS, dtype=np.uint8)

        positions[HandJointIndex.WRIST] = [0, 0, 0]
        positions[HandJointIndex.THUMB_METACARPAL] = [0.02, 0.02, 0]
        positions[HandJointIndex.THUMB_PROXIMAL] = [0.04, 0.04, 0]
        dx = -0.01 * curl_factor
        dz = -0.015 * curl_factor
        positions[HandJointIndex.THUMB_DISTAL] = [0.06 + dx, 0.06, dz]
        positions[HandJointIndex.THUMB_TIP] = [0.07 + 2 * dx, 0.07, 2 * dz]

        finger_specs = [
            (
                HandJointIndex.INDEX_PROXIMAL,
                HandJointIndex.INDEX_INTERMEDIATE,
                HandJointIndex.INDEX_DISTAL,
                HandJointIndex.INDEX_TIP,
                0.03,
            ),
            (
                HandJointIndex.MIDDLE_PROXIMAL,
                HandJointIndex.MIDDLE_INTERMEDIATE,
                HandJointIndex.MIDDLE_DISTAL,
                HandJointIndex.MIDDLE_TIP,
                0.01,
            ),
            (
                HandJointIndex.RING_PROXIMAL,
                HandJointIndex.RING_INTERMEDIATE,
                HandJointIndex.RING_DISTAL,
                HandJointIndex.RING_TIP,
                -0.01,
            ),
            (
                HandJointIndex.LITTLE_PROXIMAL,
                HandJointIndex.LITTLE_INTERMEDIATE,
                HandJointIndex.LITTLE_DISTAL,
                HandJointIndex.LITTLE_TIP,
                -0.03,
            ),
        ]
        for prox, inter, dist, tip, y_offset in finger_specs:
            positions[prox] = [0, y_offset, 0.04]
            positions[inter] = [0, y_offset, 0.07]
            cx = -0.02 * curl_factor
            positions[dist] = [cx, y_offset, 0.09 - 0.01 * curl_factor]
            positions[tip] = [2 * cx, y_offset, 0.09 - 0.03 * curl_factor]

        tg[HandInputIndex.JOINT_POSITIONS] = positions
        tg[HandInputIndex.JOINT_ORIENTATIONS] = orientations
        tg[HandInputIndex.JOINT_RADII] = (
            np.ones(NUM_HAND_JOINTS, dtype=np.float32) * 0.01
        )
        tg[HandInputIndex.JOINT_VALID] = valid
        return tg

    print(f"[1] Loading MJCF: {mjcf_path}")
    cfg = SharpaHandRetargeterConfig(
        robot_asset_path=mjcf_path,
        hand_side="right",
        input_convention="mano",
        max_iter=100,
        frequency=200.0,
    )
    retargeter = SharpaHandRetargeter(cfg, name="sharpa_demo")

    joint_names = [t.name for t in retargeter.output_spec()["hand_joints"].types]
    print(f"    {len(joint_names)} finger joints loaded.")
    print()

    n_frames = 20
    curl_sequence = np.concatenate(
        [
            np.linspace(0.0, 1.0, n_frames),
            np.linspace(1.0, 0.0, n_frames),
        ]
    )

    print(f"[2] Running {len(curl_sequence)} frames (open -> curled -> open)...")
    header = (
        f"{'frame':>5}  {'curl':>5}  {'dt':>7}  "
        + "  ".join(f"{n[:12]:>12}" for n in joint_names[:6])
        + "  ..."
    )
    print(header)
    print("-" * len(header))

    for idx, curl in enumerate(curl_sequence):
        inputs, outputs = _build_io(retargeter)
        inputs["hand_right"] = make_hand_pose(curl)
        t0 = time.perf_counter()
        retargeter.compute(inputs, outputs, _make_context())
        dt_ms = (time.perf_counter() - t0) * 1000.0
        vals = "  ".join(
            f"{float(outputs['hand_joints'][i]):12.4f}"
            for i in range(min(6, len(joint_names)))
        )
        print(f"{idx:5d}  {curl:5.2f}  {dt_ms:5.1f}ms  {vals}  ...")

    print("\n[3] Done.")
    return 0


# ---------------------------------------------------------------------------
# Live bimanual mode (Quest headset)
# ---------------------------------------------------------------------------


def _run_live(left_mjcf: str, right_mjcf: str, duration: float) -> int:
    from isaaccapture.retargeting_engine.deviceio_source_nodes import HandsSource
    from isaaccapture.retargeting_engine.interface import OutputCombiner
    from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig

    print("[1] Loading MJCFs...")
    print(f"    Left : {left_mjcf}")
    print(f"    Right: {right_mjcf}")

    hands = HandsSource(name="hands")

    left_cfg = SharpaHandRetargeterConfig(
        robot_asset_path=left_mjcf,
        hand_side="left",
        max_iter=100,
        frequency=200.0,
    )
    right_cfg = SharpaHandRetargeterConfig(
        robot_asset_path=right_mjcf,
        hand_side="right",
        max_iter=100,
        frequency=200.0,
    )
    left_retargeter = SharpaHandRetargeter(left_cfg, name="sharpa_left")
    right_retargeter = SharpaHandRetargeter(right_cfg, name="sharpa_right")

    left_joint_names = [
        t.name for t in left_retargeter.output_spec()["hand_joints"].types
    ]
    right_joint_names = [
        t.name for t in right_retargeter.output_spec()["hand_joints"].types
    ]
    print(f"    Left  joints ({len(left_joint_names)}): {left_joint_names[:3]} ...")
    print(f"    Right joints ({len(right_joint_names)}): {right_joint_names[:3]} ...")
    print()

    connected_left = left_retargeter.connect(
        {HandsSource.LEFT: hands.output(HandsSource.LEFT)}
    )
    connected_right = right_retargeter.connect(
        {HandsSource.RIGHT: hands.output(HandsSource.RIGHT)}
    )

    pipeline = OutputCombiner(
        {
            "left_hand_joints": connected_left.output("hand_joints"),
            "right_hand_joints": connected_right.output("hand_joints"),
        }
    )

    session_config = TeleopSessionConfig(
        app_name="SharpaHandBiManualDemo",
        trackers=[],
        pipeline=pipeline,
    )

    print(f"[2] Starting TeleopSession (duration={duration:.0f}s)...")
    print("    Waiting for Quest hand tracking...")
    print()

    with TeleopSession(session_config) as session:
        start_time = time.time()

        while time.time() - start_time < duration:
            result = session.step()

            left_vals = list(result["left_hand_joints"])
            right_vals = list(result["right_hand_joints"])

            if session.frame_count % 30 == 0:
                elapsed = session.get_elapsed_time()
                l_str = ", ".join(f"{v:.3f}" for v in left_vals[:4])
                r_str = ", ".join(f"{v:.3f}" for v in right_vals[:4])
                print(
                    f"[{elapsed:6.1f}s] frame {session.frame_count:5d}  "
                    f"L: [{l_str}, ...]  R: [{r_str}, ...]"
                )

            time.sleep(0.016)

    print("\n[3] Done.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Sharpa Hand Retargeter Demo (bimanual)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run synthetic curl animation instead of live Quest input.",
    )
    parser.add_argument(
        "--left-mjcf",
        type=str,
        default=DEFAULT_LEFT_MJCF,
        help="Path to the left-hand Sharpa MJCF XML file.",
    )
    parser.add_argument(
        "--right-mjcf",
        type=str,
        default=DEFAULT_RIGHT_MJCF,
        help="Path to the right-hand Sharpa MJCF XML file.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration in seconds for the live session (default: 60).",
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  Sharpa Hand Retargeter Demo (Bimanual)")
    print("=" * 70)
    print()

    if args.synthetic:
        print("  Mode: SYNTHETIC (no headset)")
        print()
        return _run_synthetic(args.right_mjcf)
    print("  Mode: LIVE (Quest hand tracking)")
    print()
    with CloudXRLauncher.launch_context(args):
        return _run_live(args.left_mjcf, args.right_mjcf, args.duration)


if __name__ == "__main__":
    sys.exit(main())
