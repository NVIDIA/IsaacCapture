# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receive tactile data through TactileTracker and print one update per second."""

from __future__ import annotations

import argparse
import time

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.deviceio_session import DeviceIOSession
from isaaccapture.deviceio_trackers import TactileTracker
from isaaccapture.oxr import OpenXRSession
from isaaccapture.schema import TactileMetadata

from .common import DEFAULT_COLLECTION_ID

PRINT_INTERVAL_S = 1.0


def _print_metadata(metadata: TactileMetadata) -> None:
    measurement = metadata.measurement
    print("TactileMetadata")
    print(f"  schema_version: {metadata.schema_version}")
    print(f"  shape_frame_id: {metadata.shape_frame_id}")
    print(
        "  measurement: "
        f"{measurement.quantity.name}, {measurement.unit.name}, "
        f"components={measurement.component_count}, frame={measurement.frame.name}"
    )
    print(f"  nodes: {[node.node_id for node in metadata.nodes]}")
    print(f"  taxels: {[taxel.taxel_id for taxel in metadata.taxels]}")
    print(
        "  profiles: "
        f"{[f'{profile.profile_id}@{profile.version_major}.{profile.version_minor}' for profile in metadata.profiles]}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args(argv)

    tracker = TactileTracker(args.collection_id)
    trackers = [tracker]
    print(f"Waiting for tactile data on {args.collection_id}; press Ctrl+C to stop.")
    with (
        CloudXRLauncher.launch_context(args),
        OpenXRSession(
            "TactileExamplePrinter",
            DeviceIOSession.get_required_extensions(trackers),
        ) as oxr_session,
        DeviceIOSession.run(trackers, oxr_session.get_handles()) as session,
    ):
        metadata = None
        next_print = time.monotonic()
        try:
            while True:
                session.update()
                if (
                    metadata is None
                    and (metadata := tracker.get_metadata(session)) is not None
                ):
                    _print_metadata(metadata)

                data = tracker.get_data(session)
                now = time.monotonic()
                if metadata is not None and data is not None and now >= next_print:
                    component_count = metadata.measurement.component_count
                    values = list(data.values)
                    print(
                        f"first={values[:component_count]} "
                        f"last={values[-component_count:]}"
                    )
                    next_print = now + PRINT_INTERVAL_S
                time.sleep(0.005)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
