# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualize live FullBodyPose or SOMA body tracking in real time with viser.

``CloudXRLauncher`` starts the CloudXR runtime and WSS proxy automatically.
Open the URL viser prints in a browser (binds all interfaces, so another
machine can reach it at http://<this-host>:8080) to see
the selected skeleton updating live. FullBodyPose renders the 24-joint PICO
layout; SOMA renders its native 77 joints and hides unavailable joints.

Usage:
    python -m isaaccapture_examples.mcap_record_replay.live_full_body [--port 8080] [--host 127.0.0.1] [--accept-eula]

Press Ctrl+C to stop.

See: https://nvidia.github.io/IsaacCapture/main/references/mcap_record_replay.html
"""

import argparse
import sys
import time

import viser

from isaaccapture.cloudxr import CloudXRLauncher
from isaaccapture.teleop_session_manager import TeleopSession, TeleopSessionConfig

from .common import (
    add_body_arguments,
    validate_body_arguments,
    build_body_pipeline,
    create_body_visualizer,
    update_body_visualizer,
    setup_scene,
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Viser HTTP bind address (default: 0.0.0.0, all interfaces; pass 127.0.0.1 to keep it local)",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser HTTP port")
    add_body_arguments(parser)
    CloudXRLauncher.add_launcher_arguments(parser)
    args = parser.parse_args(argv[1:])
    validate_body_arguments(parser, args)

    pipeline = build_body_pipeline(args)

    server = viser.ViserServer(host=args.host, port=args.port)
    ground = setup_scene(server)

    config = TeleopSessionConfig(
        app_name="LiveFullBodyExample",
        pipeline=pipeline,
    )

    with CloudXRLauncher.launch_context(args) as launcher:
        if launcher.owns_runtime:
            print(f"[live] CloudXR runtime started (WSS log: {launcher.wss_log_path})")
        print("[live] waiting for headset connection… (Ctrl+C to stop)")

        with TeleopSession(config) as session:
            viz = create_body_visualizer(args, server, ground)
            print(
                f"[live] viser listening on {args.host}:{args.port} "
                f"(http://localhost:{args.port})"
            )
            try:
                while True:
                    result = session.step()
                    n_valid = update_body_visualizer(args, result, viz)

                    if session.frame_count % 60 == 0:
                        print(
                            f"[live] frame={session.frame_count}  joints={n_valid:02d}"
                        )
                    time.sleep(1 / 60)
            except KeyboardInterrupt:
                pass

    server.stop()
    print("[live] stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
