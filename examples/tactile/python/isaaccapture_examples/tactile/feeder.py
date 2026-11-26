# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish mock tactile frames and their TactileMetadata metadata, as a tactile device would.

Each frame pushes one TactileDataArray sample on the data collection. The encoded TactileMetadata goes
on "<collection_id>/metadata" when the feeder starts and every ``--metadata-interval`` seconds
after that. Pair it with ``python -m isaaccapture_examples.tactile.printer``.
"""

import argparse
import logging
import math
import sys
import time

import numpy as np

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.deviceio_session import DeviceIOSession
from isaaccapture.deviceio_trackers import TactileTracker, TensorPushTracker
from isaaccapture.oxr import OpenXRSession
from isaaccapture.schema import (
    TRACKER_METADATA_MAX_SIZE,
    TRACKER_METADATA_TENSOR_IDENTIFIER,
    TactileDataArray,
    TactileMetadata,
    HandJoint,
    HandSkeletonRef,
    LBS,
    tracker_metadata_collection_id,
)

from .common import (
    DEFAULT_COLLECTION_ID,
)

logger = logging.getLogger("isaaccapture.examples.tactile.feeder")

MIN_NUM_JOINTS = 2
MAX_NUM_JOINTS = int(HandJoint.NUM_JOINTS)
DEFAULT_NUM_JOINTS = MAX_NUM_JOINTS
DEFAULT_NUM_TAXELS = 200

WRIST = int(HandJoint.WRIST)
# First and distal joint of each finger, thumb to little, as XrHandJointEXT ordinals.
FINGER_FIRST_JOINTS = (
    int(HandJoint.THUMB_METACARPAL),
    int(HandJoint.INDEX_METACARPAL),
    int(HandJoint.MIDDLE_METACARPAL),
    int(HandJoint.RING_METACARPAL),
    int(HandJoint.LITTLE_METACARPAL),
)
FINGER_DISTAL_JOINTS = (
    int(HandJoint.THUMB_DISTAL),
    int(HandJoint.INDEX_DISTAL),
    int(HandJoint.MIDDLE_DISTAL),
    int(HandJoint.RING_DISTAL),
    int(HandJoint.LITTLE_DISTAL),
)


def mock_metadata(joint_ids: list[HandJoint], num_taxels: int) -> TactileMetadata:
    """Build metadata for a flat hand with taxels on its distal segments."""
    num_segments = len(joint_ids)
    # LBS indices address joint_ids rows, not HandJoint ordinals.
    segment_by_joint = {int(joint): segment for segment, joint in enumerate(joint_ids)}
    positions = np.zeros((num_segments, 3), np.float32)
    positions[segment_by_joint[int(HandJoint.PALM)]] = (0.0, 0.05, 0.0)

    taxel_segments = []
    for finger, first in enumerate(FINGER_FIRST_JOINTS):
        finger_end = (
            FINGER_FIRST_JOINTS[finger + 1]
            if finger < len(FINGER_FIRST_JOINTS) - 1
            else int(HandJoint.NUM_JOINTS)
        )
        finger_joints = [
            joint for joint in range(first, finger_end) if joint in segment_by_joint
        ]
        if not finger_joints:
            continue
        for joint in finger_joints:
            positions[segment_by_joint[joint]] = (
                -0.04 + 0.02 * finger,
                0.02 + 0.025 * (joint - first),
                0.0,
            )
        joint = min(
            finger_joints, key=lambda joint: abs(joint - FINGER_DISTAL_JOINTS[finger])
        )
        taxel_segments.append(segment_by_joint[joint])

    if not taxel_segments:
        taxel_segments.append(segment_by_joint[WRIST])

    taxel_positions = np.zeros((num_taxels, 3), np.float32)
    taxel_segment_indices = np.zeros((num_taxels, 4), np.uint32)
    taxel_segment_weights = np.zeros((num_taxels, 4), np.float32)
    per_segment = math.ceil(num_taxels / len(taxel_segments))
    for taxel in range(num_taxels):
        segment_index = taxel_segments[taxel % len(taxel_segments)]
        slot = taxel // len(taxel_segments)
        taxel_positions[taxel] = positions[segment_index] + (
            0.0,
            0.02 * slot / per_segment,
            -0.005,
        )
        taxel_segment_indices[taxel] = segment_index
        taxel_segment_weights[taxel, 0] = 1.0

    segment_poses = np.column_stack(
        (positions, np.tile([0.0, 0.0, 0.0, 1.0], (num_segments, 1)))
    )
    taxel_poses = np.column_stack(
        (taxel_positions, np.tile([1.0, 0.0, 0.0, 0.0], (num_taxels, 1)))
    )
    return TactileMetadata(
        geometry_mapping=LBS(
            segment_poses, taxel_poses, taxel_segment_indices, taxel_segment_weights
        ),
        segment_source=HandSkeletonRef(joint_ids),
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
        help="Seconds between pushes of the TactileMetadata",
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

    joint_ids = [HandJoint(i) for i in range(args.joints)]
    metadata_bytes = mock_metadata(joint_ids, args.taxels).to_bytes()
    if len(metadata_bytes) > TRACKER_METADATA_MAX_SIZE:
        parser.error(
            f"{args.joints} joints and {args.taxels} taxels make a {len(metadata_bytes)} B TactileMetadata; "
            f"a tracker accepts at most {TRACKER_METADATA_MAX_SIZE} B"
        )
    logger.info(
        "Publishing %d joints and %d taxels on '%s'; TactileMetadata is %d B",
        args.joints,
        args.taxels,
        args.collection_id,
        len(metadata_bytes),
    )

    frames = 0
    last_metadata_push = -math.inf
    data_tracker = TensorPushTracker(
        args.collection_id, "tactile", TactileTracker.DEFAULT_MAX_FLATBUFFER_SIZE
    )
    metadata_tracker = TensorPushTracker(
        tracker_metadata_collection_id(args.collection_id),
        TRACKER_METADATA_TENSOR_IDENTIFIER,
        len(metadata_bytes),
    )
    trackers = [data_tracker, metadata_tracker]
    with (
        CloudXRLauncher.launch_context(args),
        OpenXRSession(
            "TactileFeeder", DeviceIOSession.get_required_extensions(trackers)
        ) as oxr_session,
        DeviceIOSession.run(trackers, oxr_session.get_handles()) as session,
    ):
        start = time.monotonic()
        try:
            while args.duration <= 0 or time.monotonic() - start < args.duration:
                remaining = start + frames / args.rate - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                session.update()
                t = time.monotonic() - start
                data_tracker.push(
                    session, TactileDataArray(mock_forces(args.taxels, t)).to_bytes()
                )
                if t - last_metadata_push >= args.metadata_interval:
                    metadata_tracker.push(session, metadata_bytes)
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
