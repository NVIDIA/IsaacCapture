# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Replay recorded FullBodyPose or SOMA body data and visualize it with viser.

``mode=SessionMode.REPLAY`` skips all OpenXR initialization, so this runs
headless on any machine. Open the URL viser prints (default
http://localhost:8080) in a browser to see the body skeleton.

Usage:
    python -m isaaccapture_examples.mcap_record_replay.replay_full_body [path/to/file.mcap] [--port 8080]

If no path is given, the newest ``full_body_*.mcap`` under ``./recordings/`` is
used.
SOMA mode selects ``soma_body_<representation>_*.mcap`` instead.
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
from isaaccapture.retargeting_engine.tensor_types.indices import FullBodyInputIndex
from isaaccapture.teleop_session_manager import (
    SessionMode,
    TeleopSession,
    TeleopSessionConfig,
)

from .common import (
    BODY_JOINT_NAMES,
    FullBodyViz,
    build_full_body_pipeline,
    setup_scene,
    add_body_arguments,
    validate_body_arguments,
    build_body_pipeline,
    create_body_visualizer,
    update_body_visualizer,
    resolve_recording,
    replay_once,
)


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
    include_controllers: bool = True,
) -> int:
    """Play the file once for ``duration_s`` wall-clock seconds. Returns frame count."""
    config = TeleopSessionConfig(
        app_name="McapFullBodyReplayExample",
        pipeline=build_full_body_pipeline(include_controllers),
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
    add_body_arguments(parser)
    parser.add_argument(
        "--rate",
        type=float,
        help="SOMA playback Hz (default: average capture rate); 0 = unpaced",
    )
    args = parser.parse_args(argv[1:])
    validate_body_arguments(parser, args)
    if args.rate is not None:
        if args.body_schema != "soma":
            parser.error("--rate is only supported with --body-schema soma")
        if not math.isfinite(args.rate) or args.rate < 0:
            parser.error("rate must be finite and nonnegative")

    if args.body_schema == "soma":
        from .soma_body_common import replay_info

        try:
            mcap_path = resolve_recording(
                args.mcap, f"soma_body_{args.soma_body_representation}"
            )
            count, capture_rate = replay_info(
                mcap_path, args.soma_body_collection_id, args.soma_body_representation
            )
        except (FileNotFoundError, ValueError) as error:
            parser.error(str(error))
        rate = capture_rate if args.rate is None else args.rate
        pipeline = build_body_pipeline(args)
        print(f"[replay] reading {mcap_path}: {count} samples at {rate:.1f} Hz")
    else:
        mcap_path = resolve_mcap(args.mcap)
        duration_s = mcap_duration_s(mcap_path)
        print(f"[replay] reading {mcap_path} (duration {duration_s:.2f}s)")

    server = viser.ViserServer(host=args.host, port=args.port)
    ground = setup_scene(server)
    viz = create_body_visualizer(args, server, ground)

    print(
        f"[replay] viser listening on {args.host}:{args.port} "
        f"(http://localhost:{args.port})"
    )

    try:
        while True:
            if args.body_schema == "soma":
                replay_once(
                    mcap_path,
                    pipeline,
                    count,
                    lambda result: update_body_visualizer(args, result, viz),
                    rate,
                )
            else:
                run_once(mcap_path, duration_s, viz, args.controllers)
            print("[replay] looping…")
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
