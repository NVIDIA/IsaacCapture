# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print what a TactileTracker receives from ``python -m isaaccapture_examples.tactile.feeder``.

Prints the TactileMetadata once it arrives, then the first and last taxel of the latest
TactileData once per second.
"""

import argparse
import sys
import time

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.deviceio_session import DeviceIOSession
from isaaccapture.deviceio_trackers import TactileTracker
from isaaccapture.oxr import OpenXRSession
from isaaccapture.schema import (
    TactileData,
    TactileMetadata,
    TaxelState,
    validate_tactile_data,
    validate_tactile_metadata,
)

from .common import DEFAULT_COLLECTION_ID

PRINT_INTERVAL_S = 1.0


def print_metadata(metadata: TactileMetadata) -> None:
    device, shape, binding, channels = (
        metadata.device,
        metadata.shape,
        metadata.binding,
        metadata.channels,
    )
    print("TactileMetadata")
    if device is not None:
        print(
            f"  device:   {device.vendor} {device.model} (serial {device.serial_number}), "
            f"side {device.side.name}, {device.nominal_rate_hz} Hz"
        )
    print(f"  shape:    {shape.segment_count} segments, {shape.taxel_count} taxels")
    if binding is not None:
        profile = binding.profile
        print(
            f"  binding:  {profile.id if profile else 'generic'} "
            f"v{profile.version if profile else '-'}: {', '.join(binding.segment_names)}"
        )
    print(
        f"  channels: normal_force={channels.normal_force}, shear_force={channels.shear_force}, "
        f"raw_values_per_taxel={channels.raw_values_per_taxel}, taxel_state={channels.taxel_state}"
    )
    print(f"  valid:    {validate_tactile_metadata(metadata) or 'yes'}")


def print_taxel(data: TactileData, index: int, raw_per_taxel: int) -> None:
    values = []
    if data.normal_force is not None:
        values.append(f"normal {data.normal_force[index]:6.3f} N")
    if data.shear_force is not None:
        x, y = data.shear_force[index]
        values.append(f"shear ({x:6.3f}, {y:6.3f}) N")
    if data.raw is not None:
        raw = data.raw[index * raw_per_taxel : (index + 1) * raw_per_taxel]
        values.append(f"raw {raw.tolist()}")
    if data.taxel_state is not None:
        values.append(TaxelState(int(data.taxel_state[index])).name)
    print(f"  taxel {index}: {', '.join(values)}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args(argv[1:])

    tracker = TactileTracker(args.collection_id)
    trackers = [tracker]
    try:
        with (
            CloudXRLauncher.launch_context(args),
            OpenXRSession(
                "TactilePrinter", DeviceIOSession.get_required_extensions(trackers)
            ) as oxr_session,
            DeviceIOSession.run(trackers, oxr_session.get_handles()) as session,
        ):
            print(f"Waiting for tactile data on '{args.collection_id}'. Ctrl+C stops.")
            metadata = None
            last_print = -PRINT_INTERVAL_S
            while True:
                session.update()
                if metadata is None:
                    metadata = tracker.get_metadata(session)
                    if metadata is not None:
                        print_metadata(metadata)
                data = tracker.get_data(session)
                now = time.monotonic()
                # The tracker withholds data until the producer's metadata has arrived.
                if data is not None and now - last_print >= PRINT_INTERVAL_S:
                    raw_per_taxel = metadata.channels.raw_values_per_taxel
                    print(f"TactileData sequence {data.sequence}")
                    print_taxel(data, 0, raw_per_taxel)
                    print_taxel(data, metadata.shape.taxel_count - 1, raw_per_taxel)
                    error = validate_tactile_data(data, metadata)
                    if error:
                        print(f"  invalid: {error}")
                    last_print = now
                time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
