# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish synthetic three-axis tactile data and immutable metadata."""

from __future__ import annotations

import argparse
import math
import time

from isaaccapture.deviceio_session import DeviceIOSession
from isaaccapture.deviceio_trackers import TensorPushTracker
from isaaccapture.oxr import OpenXRSession
from isaaccapture.schema import (
    TRACKER_METADATA_TENSOR_IDENTIFIER,
    IsaacCapturePoseTactileBinding,
    OpenXRHandTactileBinding,
    Point,
    Pose,
    ProfileHandling,
    Quaternion,
    TactileBindingProfile,
    TactileData,
    TactileHandedness,
    TactileMeasurement,
    TactileMetadata,
    TactileNode,
    TactileQuantity,
    TactileSkinWeight,
    TactileTaxel,
    TactileUnit,
    TactileValueFrame,
    make_tactile_binding_profile,
    tracker_metadata_collection_id,
)

from .common import DEFAULT_COLLECTION_ID, TACTILE_MAX_SAMPLE_SIZE

TAXEL_COUNT = 8
PUBLISH_RATE_HZ = 30.0
METADATA_INTERVAL_S = 1.0


def _pose(x: float, y: float) -> Pose:
    return Pose(Point(x, y, 0.0), Quaternion(0.0, 0.0, 0.0, 1.0))


def _metadata() -> TactileMetadata:
    nodes = [
        TactileNode("thumb_distal", _pose(-0.02, 0.04)),
        TactileNode("index_distal", _pose(0.02, 0.06)),
    ]
    taxels = [
        TactileTaxel(
            f"taxel-{index}",
            _pose(-0.035 + index * 0.01, 0.08),
            [TactileSkinWeight(index % len(nodes), 1.0)],
        )
        for index in range(TAXEL_COUNT)
    ]
    openxr_binding = TactileBindingProfile(
        openxr_hand=OpenXRHandTactileBinding(
            TactileHandedness.LEFT,
            [4, 9],
        )
    )
    pose_binding = TactileBindingProfile(
        isaaccapture_pose=IsaacCapturePoseTactileBinding(
            "core.HandPose",
            tracker_name="hand",
            channel="left",
            subject_id="example-operator",
            device_instance_id="example-glove-left",
            skeleton_instance_id="example-left-hand",
            node_pose_indices=[4, 9],
        )
    )
    profiles = [
        make_tactile_binding_profile(openxr_binding, ProfileHandling.REQUIRED),
        make_tactile_binding_profile(pose_binding, ProfileHandling.OPTIONAL),
    ]
    measurement = TactileMeasurement(
        TactileQuantity.FORCE,
        TactileUnit.NEWTON,
        3,
        TactileValueFrame.TAXEL,
    )
    return TactileMetadata("example-glove-left", measurement, taxels, nodes, profiles)


def _frame(elapsed_s: float) -> TactileData:
    values: list[float] = []
    for taxel in range(TAXEL_COUNT):
        normal_force = 1.0 + 0.75 * math.sin(
            2.0 * math.pi * 0.5 * elapsed_s + taxel * 0.3
        )
        values.extend((0.0, 0.0, normal_force))
    return TactileData(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    args = parser.parse_args(argv)

    metadata = _metadata().to_bytes()
    data_pusher = TensorPushTracker(
        args.collection_id, "tactile", TACTILE_MAX_SAMPLE_SIZE
    )
    metadata_pusher = TensorPushTracker(
        tracker_metadata_collection_id(args.collection_id),
        TRACKER_METADATA_TENSOR_IDENTIFIER,
        len(metadata),
    )
    trackers = [data_pusher, metadata_pusher]

    print(
        f"Publishing {TAXEL_COUNT} taxels on {args.collection_id}; press Ctrl+C to stop."
    )
    with (
        OpenXRSession(
            "TactileExampleFeeder",
            DeviceIOSession.get_required_extensions(trackers),
        ) as oxr_session,
        DeviceIOSession.run(trackers, oxr_session.get_handles()) as session,
    ):
        started = time.monotonic()
        frame_index = 0
        last_metadata_push = -math.inf
        try:
            while True:
                deadline = started + frame_index / PUBLISH_RATE_HZ
                if (remaining := deadline - time.monotonic()) > 0:
                    time.sleep(remaining)
                session.update()
                elapsed_s = time.monotonic() - started
                data_pusher.push(session, _frame(elapsed_s).to_bytes())
                if elapsed_s - last_metadata_push >= METADATA_INTERVAL_S:
                    metadata_pusher.push(session, metadata)
                    last_metadata_push = elapsed_s
                frame_index += 1
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
