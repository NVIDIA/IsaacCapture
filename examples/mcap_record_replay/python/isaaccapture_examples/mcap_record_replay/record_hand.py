# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Record live OpenXR or SOMA hand tracking to an MCAP file.

``CloudXRLauncher`` starts the CloudXR runtime and WSS proxy automatically —
no separate terminal or pre-running headset daemon is needed. The pipeline in
``common.py`` wires only ``HandsSource``, so ``TeleopSession`` records exactly
the ``hands`` channel — no head, no controllers.
SOMA records its separate left/right collections without running FK.

Usage:
    python -m isaaccapture_examples.mcap_record_replay.record_hand [duration_seconds] [output.mcap] [--accept-eula]

Defaults: 5 seconds → ./recordings/hands_<timestamp>.mcap

See: https://nvidia.github.io/IsaacCapture/main/references/mcap_record_replay.html
"""

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.deviceio import McapRecordingConfig
from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig

from .common import (
    add_hand_arguments,
    validate_hand_arguments,
    build_hand_pipeline,
    record_trackers,
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "duration", nargs="?", type=float, default=5.0, help="Recording duration (s)"
    )
    parser.add_argument("output", nargs="?", help="Output .mcap path")
    add_hand_arguments(parser)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args(argv[1:])
    validate_hand_arguments(parser, args)
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be finite and positive")

    duration_s: float = args.duration

    if args.output:
        mcap_path = Path(args.output)
        mcap_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path.cwd() / "recordings"
        out_dir.mkdir(exist_ok=True)
        prefix = (
            f"soma_hand_{args.soma_hand_representation}"
            if args.hand_schema == "soma"
            else "hands"
        )
        mcap_path = out_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.mcap"

    print(f"[record] writing {mcap_path} for {duration_s:.1f}s")

    if args.hand_schema == "soma":
        from .soma_hand_common import collections, create_trackers

        ids = collections(
            args.soma_left_hand_collection_id, args.soma_right_hand_collection_id
        )
        trackers = create_trackers(ids, args.soma_hand_representation)
        record_trackers(
            args, mcap_path, [(tracker, ids[key]) for key, tracker in trackers.items()]
        )
        print(f"[record] done: {mcap_path}")
        return 0

    config = TeleopSessionConfig(
        app_name="McapHandRecordExample",
        pipeline=build_hand_pipeline(),
        mcap_config=McapRecordingConfig(str(mcap_path)),
    )

    with CloudXRLauncher.launch_context(args) as launcher:
        if launcher.owns_runtime:
            print(
                f"[record] CloudXR runtime started (WSS log: {launcher.wss_log_path})"
            )
        with TeleopSession(config) as session:
            start = time.time()
            while time.time() - start < duration_s:
                result = session.step()
                if session.frame_count % 60 == 0:
                    left = bool(result["left_valid"][0])
                    right = bool(result["right_valid"][0])
                    print(
                        f"[record] t={time.time() - start:5.2f}s  "
                        f"frame={session.frame_count}  L={'Y' if left else '-'} "
                        f"R={'Y' if right else '-'}"
                    )
                time.sleep(1 / 60)

    print(f"[record] done — {mcap_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
