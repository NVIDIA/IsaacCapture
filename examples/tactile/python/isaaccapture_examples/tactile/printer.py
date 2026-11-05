# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Print tactile frames from a TactileTracker and record them to MCAP.

The tracker discovers the TactileMetadata published by
``python -m isaaccapture_examples.tactile.feeder`` and derives the joint and taxel counts from
it. The session records the "tactile" channels to ``--output``. ``--replay FILE`` prints a
recording instead, with no OpenXR runtime.
"""

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.deviceio_session import (
    DeviceIOSession,
    McapRecordingConfig,
    McapReplayConfig,
    ReplaySession,
)
from isaaccapture.deviceio_trackers import TactileTracker
from isaaccapture.oxr import OpenXRSession

from .common import DEFAULT_COLLECTION_ID

GREEN = "\033[32m"
RESET = "\033[0m"

# Replay is unpaced, so use records rather than wall time to limit terminal output.
REPLAY_PRINT_EVERY_RECORDS = 30
REPLAY_EOF_STEPS = 30


def print_frame(elapsed: float | None, index: int, forces: np.ndarray) -> None:
    first, last = forces[0], forces[-1]
    prefix = (
        f"[replay] record {index:6d}"
        if elapsed is None
        else f"[{elapsed:6.2f}s] step {index:6d}"
    )
    print(
        f"{prefix}  forces: "
        f"({first[0]:.2f}, {first[1]:.2f}, {first[2]:.2f}) ... "
        f"({last[0]:.2f}, {last[1]:.2f}, {last[2]:.2f})"
    )


def run(session, tracker: TactileTracker, args: argparse.Namespace) -> int:
    start = time.monotonic()
    metadata_seen = False
    expected_taxels = None
    steps = empty = 0
    last_print = -1.0
    last_printed_step = 0
    last_replay_forces = None
    while args.duration <= 0 or time.monotonic() - start < args.duration:
        session.update()
        elapsed = time.monotonic() - start

        metadata = tracker.get_metadata(session)
        if not metadata_seen and metadata is not None:
            geometry_mapping, segment_source = (
                metadata.geometry_mapping,
                metadata.segment_source,
            )
            if geometry_mapping is None or segment_source is None:
                raise ValueError(
                    "TactileMetadata must specify geometry_mapping and segment_source"
                )
            expected_taxels = len(geometry_mapping.taxel_bind_poses)
            print(
                f"{GREEN}[{elapsed:6.2f}s] (metadata received) "
                f"{len(segment_source.joint_ids)} joints, "
                f"{expected_taxels} taxels{RESET}"
            )
            metadata_seen = True

        data = tracker.get_data(session)
        if data is None:
            empty += 1
            if args.replay and empty >= REPLAY_EOF_STEPS:
                break
        else:
            empty = 0
            steps += 1
            force_array = np.asarray(data.taxel_forces)
            if expected_taxels is not None and len(force_array) != expected_taxels:
                raise ValueError(
                    f"frame has {len(force_array)} taxels, "
                    f"TactileMetadata declares {expected_taxels}"
                )
            if args.replay:
                last_replay_forces = force_array.copy()
                if steps == 1 or steps % REPLAY_PRINT_EVERY_RECORDS == 0:
                    print_frame(None, steps, force_array)
                    last_printed_step = steps
            elif elapsed - last_print >= args.print_interval:
                print_frame(elapsed, steps, force_array)
                last_print = elapsed
        if not args.replay:
            time.sleep(0.005)

    if args.replay and last_replay_forces is not None and last_printed_step != steps:
        print_frame(None, steps, last_replay_forces)

    if args.replay:
        print(f"Replayed {steps} tactile records.")
    else:
        print(f"Processed {steps} session steps containing tactile data.")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-id", default=DEFAULT_COLLECTION_ID)
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Seconds to run; 0 runs until Ctrl+C",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help=f"Seconds between live prints; replay prints every {REPLAY_PRINT_EVERY_RECORDS} records",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="MCAP to record to (default: ./recordings/tactile_<timestamp>.mcap)",
    )
    parser.add_argument(
        "--replay", type=Path, help="Print this recording instead of live data"
    )
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args(argv[1:])
    if not math.isfinite(args.print_interval) or args.print_interval <= 0:
        parser.error("--print-interval must be finite and positive")

    # Shape sizes belong to producer metadata, not receiver configuration.
    tracker = TactileTracker(args.collection_id)

    if args.replay:
        config = McapReplayConfig(
            str(args.replay), tracker_names=[(tracker, "tactile")]
        )
        with ReplaySession.run(config) as session:
            return run(session, tracker, args)

    output = (
        args.output
        or Path.cwd() / "recordings" / f"tactile_{datetime.now():%Y%m%d_%H%M%S}.mcap"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    trackers = [tracker]
    recording = McapRecordingConfig(str(output), tracker_names=[(tracker, "tactile")])
    print(f"Recording to {output}. Start the feeder; Ctrl+C stops.")
    try:
        with (
            CloudXRLauncher.launch_context(args),
            OpenXRSession(
                "TactilePrinter",
                DeviceIOSession.get_required_extensions(trackers),
            ) as oxr_session,
            DeviceIOSession.run(
                trackers, oxr_session.get_handles(), recording
            ) as session,
        ):
            run(session, tracker, args)
    except KeyboardInterrupt:
        pass
    print(f"Recorded {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
