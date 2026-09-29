# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Replay a recorded hand-tracking MCAP file and visualize it with viser.

``mode=SessionMode.REPLAY`` skips all OpenXR initialization, so this runs
headless on any machine. Open the URL viser prints (default
http://localhost:8080) in a browser to see the hands move.

Usage:
    python -m isaaccapture_examples.mcap_record_replay.replay_hand [path/to/file.mcap] [--port 8080]

If no path is given, the newest ``hands_*.mcap`` under ``./recordings/`` is
used.
Replay repeats until Ctrl+C.

See: https://nvidia.github.io/IsaacCapture/main/references/mcap_record_replay.html
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import viser
from mcap.reader import make_reader

from isaaccapture.deviceio import McapReplayConfig
from isaaccapture.teleop_session_manager import (
    SessionMode,
    TeleopSession,
    TeleopSessionConfig,
)

from .common import HandViz, LEFT_COLOR, RIGHT_COLOR, build_hand_pipeline, setup_scene


def mcap_duration_s(path: Path) -> float:
    """Read MCAP summary statistics and return wall-clock duration in seconds.

    The C++ replay session does not signal end-of-file — it just logs
    ``ReplayHandTrackerImpl: ... data not found`` and keeps spinning. We use
    this duration as the stop condition so playback exits cleanly.
    """
    with open(path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        if summary is None or summary.statistics is None:
            raise RuntimeError(f"{path}: MCAP file has no summary/statistics block")
        stats = summary.statistics
        if stats.message_count == 0:
            return 0.0
        return (stats.message_end_time - stats.message_start_time) / 1e9


def resolve_mcap(path_arg: str | None) -> Path:
    if path_arg:
        path = Path(path_arg)
        if not path.exists():
            sys.exit(f"[replay] error: {path} does not exist")
        return path

    recordings = Path.cwd() / "recordings"
    candidates = list(recordings.glob("hands_*.mcap"))
    if not candidates:
        sys.exit(
            f"[replay] error: no hands_*.mcap in {recordings}. "
            "Run record_hand first or pass a path."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_once(
    mcap_path: Path,
    duration_s: float,
    viz_left: HandViz,
    viz_right: HandViz,
) -> int:
    """Play the file once for ``duration_s`` wall-clock seconds. Returns frame count."""
    config = TeleopSessionConfig(
        app_name="McapHandReplayExample",
        pipeline=build_hand_pipeline(),
        mode=SessionMode.REPLAY,
        mcap_config=McapReplayConfig(str(mcap_path)),
    )

    frames = 0
    with TeleopSession(config) as session:
        start = time.time()
        while time.time() - start < duration_s:
            result = session.step()
            viz_left.update(
                np.asarray(result["left_positions"][0]),
                bool(result["left_valid"][0]),
            )
            viz_right.update(
                np.asarray(result["right_positions"][0]),
                bool(result["right_valid"][0]),
            )
            frames = session.frame_count

            if frames % 60 == 0:
                print(
                    f"[replay] t={time.time() - start:5.2f}s  "
                    f"frame={frames}  "
                    f"L={'Y' if bool(result['left_valid'][0]) else '-'} "
                    f"R={'Y' if bool(result['right_valid'][0]) else '-'}"
                )
            time.sleep(1 / 60)
    print(f"[replay] reached end of recording after {frames} frames")
    return frames


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mcap", nargs="?", help="Path to .mcap file")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Viser HTTP bind address (default: 0.0.0.0, all interfaces; pass 127.0.0.1 to keep it local)",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser HTTP port")
    args = parser.parse_args(argv[1:])

    mcap_path = resolve_mcap(args.mcap)
    duration_s = mcap_duration_s(mcap_path)

    server = viser.ViserServer(host=args.host, port=args.port)
    setup_scene(server)

    viz_left = HandViz(server, "hand_left", LEFT_COLOR)
    viz_right = HandViz(server, "hand_right", RIGHT_COLOR)

    print(
        f"[replay] viser listening on {args.host}:{args.port} "
        f"(http://localhost:{args.port})"
    )
    print(f"[replay] reading {mcap_path} (duration {duration_s:.2f}s)")

    try:
        while True:
            run_once(mcap_path, duration_s, viz_left, viz_right)
            print("[replay] looping…")
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
