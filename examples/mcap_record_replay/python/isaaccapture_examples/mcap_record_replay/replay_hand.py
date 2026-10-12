# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Replay recorded OpenXR or SOMA hand tracking and visualize it with viser.

``mode=SessionMode.REPLAY`` skips all OpenXR initialization, so this runs
headless on any machine. Open the URL viser prints (default
http://localhost:8080) in a browser to see the hands move.

Usage:
    python -m isaaccapture_examples.mcap_record_replay.replay_hand [path/to/file.mcap] [--port 8080]

If no path is given, the newest ``hands_*.mcap`` under ``./recordings/`` is
used.
SOMA mode selects ``soma_hand_<representation>_*.mcap`` instead.
Replay repeats until Ctrl+C.

See: https://nvidia.github.io/IsaacCapture/main/references/mcap_record_replay.html
"""

import argparse
import math
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

from .common import (
    HandViz,
    build_hand_pipeline,
    setup_scene,
    add_hand_arguments,
    validate_hand_arguments,
    build_selected_hand_pipeline,
    create_hand_visualizers,
    update_hand_visualizers,
    resolve_recording,
    replay_once,
)


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
    add_hand_arguments(parser)
    parser.add_argument(
        "--rate",
        type=float,
        help="SOMA playback Hz (default: busier hand's average capture rate); 0 = unpaced",
    )
    args = parser.parse_args(argv[1:])
    validate_hand_arguments(parser, args)
    if args.rate is not None:
        if args.hand_schema != "soma":
            parser.error("--rate is only supported with --hand-schema soma")
        if not math.isfinite(args.rate) or args.rate < 0:
            parser.error("rate must be finite and nonnegative")

    if args.hand_schema == "soma":
        from .soma_hand_common import collections, replay_info

        ids = collections(
            args.soma_left_hand_collection_id, args.soma_right_hand_collection_id
        )
        try:
            mcap_path = resolve_recording(
                args.mcap, f"soma_hand_{args.soma_hand_representation}"
            )
            counts, capture_rate = replay_info(
                mcap_path, ids, args.soma_hand_representation
            )
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        rate = capture_rate if args.rate is None else args.rate
        pipeline = build_selected_hand_pipeline(args)
        print(
            f"[replay] reading {mcap_path}: left={counts['hand_left']} right={counts['hand_right']} at {rate:.1f} Hz"
        )
    else:
        mcap_path = resolve_mcap(args.mcap)
        duration_s = mcap_duration_s(mcap_path)
        print(f"[replay] reading {mcap_path} (duration {duration_s:.2f}s)")

    server = viser.ViserServer(host=args.host, port=args.port)
    visualizers = create_hand_visualizers(args, server, setup_scene(server))

    print(
        f"[replay] viser listening on {args.host}:{args.port} "
        f"(http://localhost:{args.port})"
    )

    try:
        while True:
            if args.hand_schema == "soma":
                # Each tracker advances independently; exhausted hands become absent.
                replay_once(
                    mcap_path,
                    pipeline,
                    max(counts.values()),
                    lambda result: update_hand_visualizers(args, result, visualizers),
                    rate,
                )
            else:
                run_once(
                    mcap_path,
                    duration_s,
                    visualizers["hand_left"],
                    visualizers["hand_right"],
                )
            print("[replay] looping…")
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
