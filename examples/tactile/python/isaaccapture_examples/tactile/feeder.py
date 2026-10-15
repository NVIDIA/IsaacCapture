# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish mock tactile frames and their TactileShape metadata, as a tactile device would.

Each frame pushes one TactileData sample on the data collection. The encoded TactileShape goes
on "<collection_id>/metadata" when the feeder starts and every ``--metadata-interval`` seconds
after that. Pair it with ``python -m isaaccapture_examples.tactile.printer``.
"""

import argparse
import logging
import math
import sys
import time
from contextlib import ExitStack

import numpy as np

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.deviceio_session import DeviceIOSession
from isaaccapture.deviceio_trackers import TensorPushTracker
from isaaccapture.oxr import OpenXRSession
from isaaccapture.schema import (
    TRACKER_METADATA_MAX_SIZE,
    TRACKER_METADATA_TENSOR_IDENTIFIER,
    TactileData,
    TactileShape,
    tracker_metadata_collection_id,
)

from .common import (
    DEFAULT_COLLECTION_ID,
    DEFAULT_NUM_JOINTS,
    DEFAULT_NUM_TAXELS,
    MAX_NUM_JOINTS,
    MIN_NUM_JOINTS,
    TACTILE_SAMPLE_SIZE,
)

logger = logging.getLogger("isaaccapture.examples.tactile.feeder")

WRIST = 1
# First and distal joint of each finger, thumb to little, as XrHandJointEXT ordinals.
FINGER_FIRST_JOINTS = (2, 6, 11, 16, 21)
FINGER_DISTAL_JOINTS = (4, 9, 14, 19, 24)


def mock_shape(num_joints: int, num_taxels: int) -> TactileShape:
    """Build a flat hand in hand-root space with taxels on its distal segments."""
    positions = np.zeros((num_joints, 3), np.float32)
    parents = np.full(num_joints, WRIST, np.uint32)
    parents[WRIST] = TactileShape.ROOT_PARENT
    positions[0] = (0.0, 0.05, 0.0)

    taxel_joints = []
    for finger, first in enumerate(FINGER_FIRST_JOINTS):
        if first >= num_joints:
            break
        finger_end = (
            FINGER_FIRST_JOINTS[finger + 1] - 1 if finger < 4 else MAX_NUM_JOINTS - 1
        )
        last = min(finger_end, num_joints - 1)
        for joint in range(first, last + 1):
            positions[joint] = (
                -0.04 + 0.02 * finger,
                0.02 + 0.025 * (joint - first),
                0.0,
            )
            if joint > first:
                parents[joint] = joint - 1
        taxel_joints.append(min(FINGER_DISTAL_JOINTS[finger], last))

    if not taxel_joints:
        taxel_joints.append(WRIST)

    taxel_positions = np.zeros((num_taxels, 3), np.float32)
    joint_indices = np.zeros((num_taxels, 4), np.uint32)
    joint_weights = np.zeros((num_taxels, 4), np.float32)
    per_segment = math.ceil(num_taxels / len(taxel_joints))
    for taxel in range(num_taxels):
        segment, slot = taxel % len(taxel_joints), taxel // len(taxel_joints)
        joint = taxel_joints[segment]
        taxel_positions[taxel] = positions[joint] + (
            0.0,
            0.02 * slot / per_segment,
            -0.005,
        )
        joint_indices[taxel] = joint
        joint_weights[taxel, 0] = 1.0

    return TactileShape(
        joint_bind_positions=positions,
        joint_bind_orientations=np.tile([0.0, 0.0, 0.0, 1.0], (num_joints, 1)),
        joint_parents=parents,
        taxel_positions=taxel_positions,
        taxel_normals=np.tile([0.0, 0.0, -1.0], (num_taxels, 1)),
        taxel_tangents=np.tile([1.0, 0.0, 0.0], (num_taxels, 1)),
        taxel_joint_indices=joint_indices,
        taxel_joint_weights=joint_weights,
    )


def mock_forces(num_taxels: int, t: float) -> np.ndarray:
    """Generate a positive 0.5 Hz Z-force wave, phase-shifted per finger."""
    finger = np.arange(num_taxels) % 5
    phase = 2.0 * math.pi * 0.5 * t - 1.2 * finger
    forces = np.zeros((num_taxels, 3), np.float32)
    forces[:, 2] = 1.25 + 0.75 * np.sin(phase)
    return forces


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    parser.add_argument("--joints", type=int, default=DEFAULT_NUM_JOINTS)
    parser.add_argument("--taxels", type=int, default=DEFAULT_NUM_TAXELS)
    parser.add_argument("--rate", type=float, default=60.0, help="Frames per second")
    parser.add_argument(
        "--metadata-interval",
        type=float,
        default=1.0,
        help="Seconds between pushes of the TactileShape",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds to run; 0 runs until Ctrl+C",
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args(argv[1:])
    if not MIN_NUM_JOINTS <= args.joints <= MAX_NUM_JOINTS:
        parser.error(f"--joints must be between {MIN_NUM_JOINTS} and {MAX_NUM_JOINTS}")
    if args.taxels <= 0:
        parser.error("--taxels must be positive")
    if not math.isfinite(args.rate) or args.rate <= 0:
        parser.error("--rate must be finite and positive")
    if not math.isfinite(args.metadata_interval) or args.metadata_interval <= 0:
        parser.error("--metadata-interval must be finite and positive")

    shape = mock_shape(args.joints, args.taxels).to_bytes()
    if len(shape) > TRACKER_METADATA_MAX_SIZE:
        parser.error(
            f"{args.joints} joints and {args.taxels} taxels make a {len(shape)} B TactileShape; "
            f"a tracker accepts at most {TRACKER_METADATA_MAX_SIZE} B"
        )
    logger.info(
        "Publishing %d joints and %d taxels on '%s'; TactileShape is %d B",
        args.joints,
        args.taxels,
        args.collection_id,
        len(shape),
    )

    frames = 0
    last_metadata_push = -math.inf
    with ExitStack() as stack:
        stack.enter_context(CloudXRLauncher.launch_context(args))
        data_tracker = TensorPushTracker(
            args.collection_id, "tactile", TACTILE_SAMPLE_SIZE
        )
        metadata_tracker = TensorPushTracker(
            tracker_metadata_collection_id(args.collection_id),
            TRACKER_METADATA_TENSOR_IDENTIFIER,
            len(shape),
        )
        trackers = [data_tracker, metadata_tracker]
        oxr_session = stack.enter_context(
            OpenXRSession(
                "TactileFeeder", DeviceIOSession.get_required_extensions(trackers)
            )
        )
        session = stack.enter_context(
            DeviceIOSession.run(trackers, oxr_session.get_handles())
        )

        start = time.monotonic()
        try:
            while args.duration <= 0 or time.monotonic() - start < args.duration:
                remaining = start + frames / args.rate - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                session.update()
                t = time.monotonic() - start
                data_tracker.push(
                    session, TactileData(mock_forces(args.taxels, t)).to_bytes()
                )
                if t - last_metadata_push >= args.metadata_interval:
                    metadata_tracker.push(session, shape)
                    last_metadata_push = t
                frames += 1
                if frames % max(1, int(args.rate * 5)) == 0:
                    logger.info("Published %d frames", frames)
        except KeyboardInterrupt:
            pass

    logger.info("Published %d frames in total", frames)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
