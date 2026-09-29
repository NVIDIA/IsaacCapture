# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Record raw hand joint SE(3) poses to an MCAP file.

Reads the "manus_sensors_left" / "manus_sensors_right" tensor collections pushed by the
manus plugin (one core.JointSe3PoseTracker per collection) and records each to its own
MCAP channel pair (<cid>/joint_se3_pose and <cid>/joint_se3_pose_tracked). Replay with
replay_joint_se3_pose.py.

Any producer of JointSe3PoseOutput works; pass its collection ids with --collections.

Prerequisites (separate terminals):
  1. CloudXR runtime:  python -m isaaccapture.cloudxr
  2. the pusher:       ./install/plugins/manus/manus_hand_plugin --datasets=sensors

Usage:
    source ~/.cloudxr/run/cloudxr.env
    python -m isaaccapture_examples.mcap_record_replay.record_joint_se3_pose [duration_s] [output.mcap] [--collections a,b]
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaccapture.deviceio_session import DeviceIOSession, McapRecordingConfig
from isaaccapture.deviceio_trackers import JointSe3PoseTracker
from isaaccapture.oxr import OpenXRSession
from isaaccapture.schema import JointName, JointType

DEFAULT_COLLECTIONS = ["manus_sensors_left", "manus_sensors_right"]

EXPECTED_TYPE = JointType.HAND_RAW

TIPS = [
    JointName.HAND_RAW_THUMB_TIP,
    JointName.HAND_RAW_INDEX_TIP,
    JointName.HAND_RAW_MIDDLE_TIP,
    JointName.HAND_RAW_RING_TIP,
    JointName.HAND_RAW_LITTLE_TIP,
]


def summarize(data) -> str:
    """One-line view of a frame: which tips arrived, and where the thumb is."""
    if data is None:
        return "-"
    present = sum(1 for tip in TIPS if data.lookup(tip) is not None)
    # A frame names its family even when nothing is tracked, so report it either way.
    family = str(data.type).split(".")[-1]
    thumb = data.lookup(JointName.HAND_RAW_THUMB_TIP)
    if thumb is None:
        return f"{family} {present}/5 tips"
    p = thumb.pose.position
    return f"{family} {present}/5 tips  thumb=[{p.x:+.3f} {p.y:+.3f} {p.z:+.3f}]"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "duration", nargs="?", type=float, default=10.0, help="Recording duration (s)"
    )
    parser.add_argument("output", nargs="?", help="Output .mcap path")
    parser.add_argument(
        "--collections",
        default=",".join(DEFAULT_COLLECTIONS),
        help=f"Comma-separated collection ids (default: {','.join(DEFAULT_COLLECTIONS)})",
    )
    args = parser.parse_args(argv[1:])

    if args.output:
        mcap_path = Path(args.output)
        mcap_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path.cwd() / "recordings"
        out_dir.mkdir(exist_ok=True)
        mcap_path = out_dir / f"joint_se3_pose_{datetime.now():%Y%m%d_%H%M%S}.mcap"

    collections = [c.strip() for c in args.collections.split(",") if c.strip()]
    if not collections:
        print("[record-joints] no collections given", file=sys.stderr)
        return 1

    trackers = {cid: JointSe3PoseTracker(cid) for cid in collections}
    recording = McapRecordingConfig(
        str(mcap_path), tracker_names=[(t, cid) for cid, t in trackers.items()]
    )

    print(f"[record-joints] writing {mcap_path} for {args.duration:.1f}s")
    for cid in collections:
        print(f"[record-joints]   collection '{cid}'")

    tracker_list = list(trackers.values())
    extensions = DeviceIOSession.get_required_extensions(tracker_list)
    with OpenXRSession("McapJointSe3PoseRecord", extensions) as oxr_session:
        with DeviceIOSession.run(
            tracker_list, oxr_session.get_handles(), recording
        ) as session:
            start = time.time()
            frame = 0
            while time.time() - start < args.duration:
                session.update()
                if frame % 60 == 0:
                    parts = [
                        f"{cid}: {summarize(t.get_data(session))}"
                        for cid, t in trackers.items()
                    ]
                    print(
                        f"[record-joints] t={time.time() - start:5.2f}s  "
                        + "  ".join(parts)
                    )
                frame += 1
                time.sleep(1 / 90)

    print(f"[record-joints] done — {mcap_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
