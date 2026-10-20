# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish mock tactile frames and their TactileMetadata, as a tactile device would.

The device is a left index fingertip pad: two segments bound to the OpenXR hand joints
INDEX_DISTAL and INDEX_TIP, with taxels in a line along the pad. Each frame pushes one
TactileData on the data collection; the TactileMetadata goes on "<collection_id>/metadata"
when the feeder starts and every ``--metadata-interval`` seconds after that.
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
    BodySide,
    SkeletonBinding,
    SkeletonProfileRef,
    TactileChannels,
    TactileData,
    TactileDevice,
    TactileMetadata,
    TactileShape,
    TaxelState,
    tracker_metadata_collection_id,
)

from .common import DEFAULT_COLLECTION_ID, TACTILE_MAX_PAYLOAD_SIZE

logger = logging.getLogger("isaaccapture.examples.tactile.feeder")

PAD_LENGTH_M = 0.025
DISTAL_TO_TIP_M = 0.022
# 180 degrees about X: the taxel +Z normal faces the palm side (-Z) of the finger.
PAD_ORIENTATION = (1.0, 0.0, 0.0, 0.0)


def mock_metadata(num_taxels: int, rate_hz: float) -> TactileMetadata:
    """Taxel i sits a fraction f = i / (N - 1) along the pad, skinned (1 - f, f) to the two segments."""
    along = np.linspace(0.0, 1.0, num_taxels, dtype=np.float32)
    taxel_positions = np.stack(
        [np.zeros_like(along), along * PAD_LENGTH_M, np.full_like(along, -0.008)],
        axis=1,
    )
    weights = np.zeros((num_taxels, 4), np.float32)
    weights[:, 0] = 1.0 - along
    weights[:, 1] = along
    segments = np.where(weights > 0.0, [0, 1, 0, 0], 0)

    shape = TactileShape(
        segment_bind_positions=[[0.0, 0.0, 0.0], [0.0, DISTAL_TO_TIP_M, 0.0]],
        segment_bind_orientations=[[0.0, 0.0, 0.0, 1.0]] * 2,
        taxel_positions=taxel_positions,
        taxel_orientations=np.tile(PAD_ORIENTATION, (num_taxels, 1)),
        taxel_segments=segments,
        taxel_weights=weights,
        taxel_areas=np.full(num_taxels, 4e-6, np.float32),
    )
    return TactileMetadata(
        shape=shape,
        channels=TactileChannels(normal_force=True, shear_force=True, taxel_state=True),
        device=TactileDevice(
            vendor="Example",
            model="Mock fingertip pad",
            serial_number="0001",
            side=BodySide.LEFT,
            nominal_rate_hz=rate_hz,
        ),
        binding=SkeletonBinding(
            ["INDEX_DISTAL", "INDEX_TIP"], profile=SkeletonProfileRef("openxr.hand", 1)
        ),
    )


def mock_frame(num_taxels: int, t: float, sequence: int) -> TactileData:
    """A 0.5 Hz press that travels along the pad, with a small shear along the pad."""
    along = np.linspace(0.0, 1.0, num_taxels)
    normal = 1.0 + np.sin(2.0 * math.pi * (0.5 * t - 0.25 * along))
    shear = np.stack([np.zeros(num_taxels), 0.1 * normal], axis=1)
    state = np.where(normal > 1.9, TaxelState.SATURATED.value, TaxelState.VALID.value)
    return TactileData(
        normal_force=normal,
        shear_force=shear,
        taxel_state=state,
        sequence=sequence,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    parser.add_argument("--taxels", type=int, default=16)
    parser.add_argument("--rate", type=float, default=60.0, help="Frames per second")
    parser.add_argument(
        "--metadata-interval",
        type=float,
        default=1.0,
        help="Seconds between pushes of the TactileMetadata",
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args(argv[1:])
    if args.taxels < 2:
        parser.error("--taxels must be at least 2")
    if not math.isfinite(args.rate) or args.rate <= 0:
        parser.error("--rate must be finite and positive")
    if not math.isfinite(args.metadata_interval) or args.metadata_interval <= 0:
        parser.error("--metadata-interval must be finite and positive")

    metadata = mock_metadata(args.taxels, args.rate).to_bytes()
    frame_size = len(mock_frame(args.taxels, 0.0, 0).to_bytes())
    if (
        len(metadata) > TRACKER_METADATA_MAX_SIZE
        or frame_size > TACTILE_MAX_PAYLOAD_SIZE
    ):
        parser.error(f"--taxels {args.taxels} does not fit the tactile tracker")
    logger.info(
        "Publishing %d taxels on '%s' at %.0f Hz",
        args.taxels,
        args.collection_id,
        args.rate,
    )

    frames = 0
    with ExitStack() as stack:
        stack.enter_context(CloudXRLauncher.launch_context(args))
        data_tracker = TensorPushTracker(
            args.collection_id, "tactile", TACTILE_MAX_PAYLOAD_SIZE
        )
        metadata_tracker = TensorPushTracker(
            tracker_metadata_collection_id(args.collection_id),
            TRACKER_METADATA_TENSOR_IDENTIFIER,
            len(metadata),
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
        last_metadata_push = -math.inf
        try:
            while True:
                remaining = start + frames / args.rate - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                session.update()
                t = time.monotonic() - start
                if t - last_metadata_push >= args.metadata_interval:
                    metadata_tracker.push(session, metadata)
                    last_metadata_push = t
                data_tracker.push(
                    session, mock_frame(args.taxels, t, frames).to_bytes()
                )
                frames += 1
        except KeyboardInterrupt:
            pass

    logger.info("Published %d frames", frames)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
