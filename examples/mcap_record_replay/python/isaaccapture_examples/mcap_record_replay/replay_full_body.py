# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Replay a recorded full-body MCAP file and visualize it with viser.

``mode=SessionMode.REPLAY`` skips all OpenXR initialization, so this runs
headless on any machine. Open the URL viser prints (default
http://localhost:8080) in a browser to see the body skeleton.

Usage:
    python -m isaaccapture_examples.mcap_record_replay.replay_full_body [path/to/file.mcap] [--port 8080] [--loop]

If no path is given, the newest ``full_body_*.mcap`` under ``./recordings/`` is
used.
``--loop`` keeps replaying the file end-to-end until the process is killed.

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
from isaaccapture.retargeting_engine.tensor_types.indices import FullBodyInputIndex
from isaaccapture.teleop_session_manager import (
    SessionMode,
    TeleopSession,
    TeleopSessionConfig,
)

from .common import BODY_JOINT_NAMES, FullBodyViz, build_full_body_pipeline, setup_scene


def mcap_duration_s(path: Path) -> float:
    """Read MCAP summary statistics and return wall-clock duration in seconds.

    The C++ replay session does not signal end-of-file — it just logs
    ``Replay*TrackerImpl: ... data not found`` and keeps spinning. We use
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
    candidates = list(recordings.glob("full_body_*.mcap"))
    if not candidates:
        sys.exit(
            f"[replay] error: no full_body_*.mcap in {recordings}. "
            "Run record_full_body first or pass a path."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_once(
    mcap_path: Path,
    duration_s: float,
    viz: FullBodyViz,
) -> int:
    """Play the file once for ``duration_s`` wall-clock seconds. Returns frame count."""
    config = TeleopSessionConfig(
        app_name="McapFullBodyReplayExample",
        pipeline=build_full_body_pipeline(),
        mode=SessionMode.REPLAY,
        mcap_config=McapReplayConfig(str(mcap_path)),
    )

    frames = 0
    with TeleopSession(config) as session:
        start = time.time()
        while time.time() - start < duration_s:
            result = session.step()
            full_body = result["full_body"]

            if full_body.is_none:
                viz.update(None, None)
                n_valid = 0
            else:
                positions = np.asarray(
                    full_body[FullBodyInputIndex.JOINT_POSITIONS], dtype=np.float32
                )
                valid = np.asarray(
                    full_body[FullBodyInputIndex.JOINT_VALID], dtype=np.uint8
                )
                viz.update(positions, valid)
                n_valid = int(np.count_nonzero(valid))

            frames = session.frame_count
            if frames % 60 == 0:
                print(
                    f"[replay] t={time.time() - start:5.2f}s  "
                    f"frame={frames}  "
                    f"joints={n_valid:02d}/{len(BODY_JOINT_NAMES)}"
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
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Replay the file in a loop until Ctrl+C",
    )
    args = parser.parse_args(argv[1:])

    mcap_path = resolve_mcap(args.mcap)
    duration_s = mcap_duration_s(mcap_path)

    server = viser.ViserServer(host=args.host, port=args.port)
    ground = setup_scene(server)
    viz = FullBodyViz(server, ground)

    print(
        f"[replay] viser listening on {args.host}:{args.port} "
        f"(http://localhost:{args.port})"
    )
    print(f"[replay] reading {mcap_path} (duration {duration_s:.2f}s)")

    while True:
        run_once(mcap_path, duration_s, viz)
        if not args.loop:
            break
        print("[replay] looping…")

    print("[replay] done — viser server still up; Ctrl+C to exit")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
