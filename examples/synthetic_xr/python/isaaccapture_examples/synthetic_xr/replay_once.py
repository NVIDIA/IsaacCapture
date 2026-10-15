# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay native controller MCAP once using the SDK example's replay pipeline."""

import argparse
import importlib
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mcap")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--hold-seconds", type=float, default=5)
    parser.add_argument(
        "--legacy-example-dir",
        type=Path,
        help="Explicit Isaac Teleop 1.5 examples/mcap_record_replay/python directory",
    )
    args = parser.parse_args()
    if not 0 <= args.hold_seconds <= 300:
        parser.error("--hold-seconds must be between 0 and 300")
    if args.legacy_example_dir is None:
        from isaaccapture_examples.mcap_record_replay import replay_controller as replay
    else:
        directory = args.legacy_example_dir.resolve()
        if not all(
            (directory / name).is_file()
            for name in ("replay_controller.py", "common.py")
        ):
            parser.error(
                "Legacy directory must contain replay_controller.py and common.py"
            )
        # The 1.5 example uses sibling imports; this explicit option selects that source.
        sys.path.insert(0, str(directory))
        replay = importlib.import_module("replay_controller")

    path = replay.resolve_mcap(args.mcap)
    duration = replay.mcap_duration_s(path)
    if duration <= 0:
        parser.error("MCAP has no positive-duration recording")
    server = replay.viser.ViserServer(host=args.host, port=args.port)
    try:
        if args.legacy_example_dir is None:
            replay.setup_scene(server)
        else:
            server.scene.set_up_direction("+y")
            server.scene.add_grid(
                name="/grid", width=2.0, height=2.0, plane="xz", cell_size=0.1
            )
        left = replay.ControllerViz(server, "controller_left", replay.LEFT_COLOR)
        right = replay.ControllerViz(server, "controller_right", replay.RIGHT_COLOR)
        print(f"Single replay viewer: http://{args.host}:{args.port}", flush=True)
        frames = replay.run_once(path, duration, left, right)
        if frames <= 0:
            raise RuntimeError("Replay produced no frames")
        print(f"Single replay complete: {frames} frames", flush=True)
        time.sleep(args.hold_seconds)
    finally:
        server.stop()


if __name__ == "__main__":
    main()
