# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualize live raw hand fingertip poses in real time with viser.

Polls the "manus_sensors_left" / "manus_sensors_right" tensor collections pushed by the
manus plugin and draws each tracked tip as a labelled coordinate frame. Each collection
carries exactly one hand, so a stream's side comes from which collection it is: tip labels
read "left thumb" / "right thumb", and the two hands are pushed apart along X so they stay
tellable apart when the streams share a frame.

Standalone: this reads JointSe3PoseTracker directly rather than going through the
retargeting engine, so it needs no source node. Open the URL viser prints to watch the
tips move; touch thumb to a fingertip and the two frames should meet.

Prerequisites (separate terminals):
  1. CloudXR runtime:  python -m isaaccapture.cloudxr
  2. the pusher:       ./install/plugins/manus/manus_hand_plugin --datasets=sensors

Usage:
    source ~/.cloudxr/run/cloudxr.env
    python -m isaaccapture_examples.mcap_record_replay.live_joint_se3_pose [--collections a,b] [--port 8080] [--separation M]

Press Ctrl+C to stop.
"""

import argparse
import sys
import time

from isaaccapture.deviceio_session import DeviceIOSession
from isaaccapture.deviceio_trackers import JointSe3PoseTracker
from isaaccapture.oxr import OpenXRSession

from .joint_se3_common import DEFAULT_SEPARATION_M, TipViz, make_server

DEFAULT_COLLECTIONS = ["manus_sensors_left", "manus_sensors_right"]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collections",
        default=",".join(DEFAULT_COLLECTIONS),
        help=f"Comma-separated collection ids (default: {','.join(DEFAULT_COLLECTIONS)})",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Viser HTTP bind address (default: 0.0.0.0, all interfaces; pass 127.0.0.1 to keep it local)",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser HTTP port")
    parser.add_argument(
        "--separation",
        type=float,
        default=DEFAULT_SEPARATION_M,
        help=f"Display-only gap between the two hands along X, in metres "
        f"(default: {DEFAULT_SEPARATION_M}); 0 draws them where they are tracked",
    )
    args = parser.parse_args(argv[1:])

    collections = [c.strip() for c in args.collections.split(",") if c.strip()]
    if not collections:
        print("[live-joints] no collections given", file=sys.stderr)
        return 1

    server = make_server(args.host, args.port, "[live-joints]")
    viz = {cid: TipViz(server, cid, args.separation) for cid in collections}

    trackers = {cid: JointSe3PoseTracker(cid) for cid in collections}
    tracker_list = list(trackers.values())
    extensions = DeviceIOSession.get_required_extensions(tracker_list)
    with OpenXRSession("LiveJointSe3Pose", extensions) as oxr_session:
        with DeviceIOSession.run(tracker_list, oxr_session.get_handles()) as session:
            print("[live-joints] streaming — Ctrl+C to stop")
            try:
                while True:
                    session.update()
                    for cid, tracker in trackers.items():
                        viz[cid].update(tracker.get_data(session))
                    time.sleep(1 / 90)
            except KeyboardInterrupt:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
