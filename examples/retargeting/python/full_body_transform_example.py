#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Full Body Transform Example

Corrects a body-tracking skeleton for the headset that produced it, using
FullBodySource -> FullBodyTransform.

The CloudXR client converts a Meta Quest skeleton into the ByteDance 24-joint
layout, so the server receives XR_BD_body_tracking data whichever headset is
connected. Joint order and positions survive that conversion; per-joint
orientations do not, so anything driven by the quaternions needs correcting
first. Positions look fine either way, which is why this is easy to miss.

Run with --profile quest on a Quest, --profile pico on a PICO. The PICO profile
is identity and reports a zero delta; that is the expected result, not a
failure.

Usage:
    python full_body_transform_example.py --profile quest
    python full_body_transform_example.py --profile pico --duration 30
"""

import argparse
import sys
import time

import numpy as np

import isaacteleop.deviceio as deviceio
import isaacteleop.oxr as oxr
from isaacteleop.cloudxr import CloudXRLauncher
from isaacteleop.retargeting_engine.deviceio_source_nodes import FullBodySource
from isaacteleop.retargeting_engine.tensor_types import FullBodyInputIndex
from isaacteleop.retargeting_engine.utilities import (
    FullBodyTransform,
    SKELETON_PROFILES,
)

# Joint order is XR_BD_body_tracking; a few are named for the readout below.
REPORTED_JOINTS = {
    0: "PELVIS",
    3: "SPINE1",
    15: "HEAD",
    16: "L_SHOULDER",
    17: "R_SHOULDER",
    20: "L_WRIST",
    21: "R_WRIST",
}


def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-joint geodesic angle between two xyzw quaternion sets, sign-agnostic."""
    dot = np.abs(np.sum(a * b, axis=-1))
    return 2.0 * np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="quest",
        choices=sorted(SKELETON_PROFILES),
        help="Headset that produces the skeleton (default: quest)",
    )
    parser.add_argument(
        "--duration", type=float, default=20.0, help="Seconds to run (default: 20)"
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args()

    print(f"Full Body Transform Example  --  profile: {args.profile}")
    if args.profile == "pico":
        print("The PICO profile is identity; a zero delta below is correct.")

    # Source creates its own tracker; the transform node is a pure graph node.
    body_source = FullBodySource(name="full_body")
    body_fix = FullBodyTransform("body_fix", profile=args.profile)
    tracker = body_source.get_tracker()

    required_extensions = deviceio.DeviceIOSession.get_required_extensions([tracker])
    print(f"Required OpenXR extensions: {', '.join(required_extensions)}")

    with CloudXRLauncher.launch_context(args):
        with oxr.OpenXRSession(
            "FullBodyTransformExample", required_extensions
        ) as oxr_session:
            with deviceio.DeviceIOSession.run(
                [tracker], oxr_session.get_handles()
            ) as session:
                return _run_loop(session, body_source, body_fix, args.duration)


def _run_loop(session, body_source, body_fix, duration: float) -> int:
    print("\nWaiting for body tracking...\n")

    start = time.time()
    frame = 0
    reported = False
    next_report = 0.0

    while time.time() - start < duration:
        session.update()
        frame += 1

        # Poll the tracker, convert to tensors, then correct.
        source_out = body_source(body_source.poll_tracker(session))
        result = body_fix({"full_body": source_out["full_body"]})

        # Body tracking is optional: absent until the headset starts reporting.
        if source_out["full_body"].is_none:
            continue

        # Report about once a second, whatever rate the session runs at.
        elapsed = time.time() - start
        if elapsed < next_report:
            continue
        next_report = elapsed + 1.0

        raw = np.from_dlpack(
            source_out["full_body"][FullBodyInputIndex.JOINT_ORIENTATIONS]
        )
        fixed = np.from_dlpack(
            result["full_body"][FullBodyInputIndex.JOINT_ORIENTATIONS]
        )
        valid = np.from_dlpack(source_out["full_body"][FullBodyInputIndex.JOINT_VALID])
        delta = _quat_angle_deg(raw, fixed)

        print(
            f"[{elapsed:5.1f}s] frame {frame}  valid joints: {int(valid.sum())}/{len(valid)}"
        )
        print(
            f"          correction: mean {delta.mean():5.1f} deg, max {delta.max():5.1f} deg"
        )
        for index, name in REPORTED_JOINTS.items():
            print(f"            {name:<11} {delta[index]:5.1f} deg")
        print()
        reported = True

    if not reported:
        print("No body tracking data was received.")
        print("Check that body tracking is enabled on the headset and that the")
        print("client is connected, then run again.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
