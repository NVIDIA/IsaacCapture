# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Replay raw hand joint SE(3) poses from an MCAP file — no runtime needed.

Opens a recording made by record_joint_se3_pose.py and replays each per-collection channel
through ReplaySession + core.JointSe3PoseTracker. Replay needs no OpenXR runtime and no
hardware.

Per frame it prints one line per collection, each tip's distance from that hand's thumb
tip, which is what makes a thumb-to-fingertip pinch readable as text: the touched finger's
distance collapses toward zero while the others do not. Tip names carry the side ("left
index"), taken from the collection id.

The same frames are drawn in viser, as in live_joint_se3_pose.py: labelled tips, the two
hands pushed apart along X. Open the URL it prints.

Replay repeats until Ctrl+C.

Usage:
    python -m isaaccapture_examples.mcap_record_replay.replay_joint_se3_pose [recording.mcap] [--rate N] [--collections a,b]
                                    [--host H] [--port 8080] [--separation M]
"""

import argparse
import math
import sys
import time
from pathlib import Path

from mcap.reader import make_reader

from isaaccapture.deviceio_session import McapReplayConfig, ReplaySession
from isaaccapture.deviceio_trackers import JointSe3PoseTracker
from isaaccapture.schema import JointName

from .joint_se3_common import (
    DEFAULT_SEPARATION_M,
    FINGERS,
    TipViz,
    hand_side,
    make_server,
)

# Recordings write two channels per collection; strip either suffix for the collection id.
_CHANNEL_SUFFIXES = ("/joint_se3_pose_tracked", "/joint_se3_pose")


def _summary(mcap_path: Path):
    with open(mcap_path, "rb") as f:
        return make_reader(f).get_summary()


def resolve_mcap(path_arg: str | None) -> Path:
    """Use the given path, or the newest .mcap under ./recordings/."""
    if path_arg:
        return Path(path_arg)
    recordings = Path.cwd() / "recordings"
    candidates = list(recordings.glob("joint_se3_pose_*.mcap")) or list(
        recordings.glob("*.mcap")
    )
    if not candidates:
        sys.exit(
            f"[replay-joints] no .mcap files in {recordings}. "
            "Run python -m isaaccapture_examples.mcap_record_replay.record_joint_se3_pose first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def discover_collections(mcap_path: Path) -> list[str]:
    """Return the joint-pose collection ids present in an MCAP, in first-seen order."""
    seen = {}
    summary = _summary(mcap_path)
    channels = summary.channels.values() if summary else []
    for ch in channels:
        for suffix in _CHANNEL_SUFFIXES:
            if ch.topic.endswith(suffix):
                seen.setdefault(ch.topic[: -len(suffix)], None)
                break
    return list(seen)


def capture_rate_hz(mcap_path: Path, default: float = 30.0) -> float:
    """Estimate playback rate from the coalesced per-tick channel.

    Replay advances one frame per ReplaySession.update() and reads the "_tracked" channel
    (one message per recording tick), not the raw sample channel, whose count can exceed
    the tick count when the producer bursts several samples per tick.
    """
    summary = _summary(mcap_path)
    if not summary or not summary.statistics:
        return default
    stats = summary.statistics
    span_ns = stats.message_end_time - stats.message_start_time
    if span_ns <= 0:
        return default
    tracked_ids = [
        cid
        for cid, ch in summary.channels.items()
        if ch.topic.endswith("/joint_se3_pose_tracked")
    ]
    counts = stats.channel_message_counts
    best = max((counts.get(cid, 0) for cid in tracked_ids), default=0)
    if best <= 1:
        return default
    # best messages span best - 1 intervals; dividing the count by the duration would
    # replay a two-sample recording at twice its capture rate.
    return (best - 1) / (span_ns / 1e9)


def pinch_distances(data, side: str | None) -> str:
    """Thumb-to-fingertip distances in mm for one hand, or a reason there are none."""
    prefix = f"{side} " if side else ""
    if data is None:
        return "-"
    thumb = data.lookup(JointName.HAND_RAW_THUMB_TIP)
    if thumb is None:
        return f"no {prefix}thumb"
    t = thumb.pose.position
    parts = []
    for label, joint in FINGERS:
        found = data.lookup(joint)
        if found is None:
            parts.append(f"{prefix}{label}=-")
            continue
        p = found.pose.position
        mm = 1000.0 * math.dist((t.x, t.y, t.z), (p.x, p.y, p.z))
        parts.append(f"{prefix}{label}={mm:5.1f}")
    return " ".join(parts)


def run_once(session, trackers, viz, rate_hz: float) -> tuple[int, dict]:
    frames = 0
    samples = dict.fromkeys(trackers, 0)
    none_streak = 0
    tick = 1.0 / rate_hz if rate_hz > 0 else 0.0
    # update() is void; per the replay contract tracker data goes null at end-of-file,
    # so stop after a run of all-null frames.
    while none_streak < 30:
        session.update()
        frames += 1
        any_data = False
        lines = []
        for cid, tracker in trackers.items():
            data = tracker.get_data(session)
            if data is not None:
                any_data = True
                samples[cid] += 1
            viz[cid].update(data)
            lines.append(f"{cid}: {pinch_distances(data, hand_side(cid))}")
        none_streak = 0 if any_data else none_streak + 1
        if frames % 60 == 1:
            # One line per hand: sided tip names make a two-hand frame too wide for one.
            for line in lines:
                print(f"[replay-joints] frame={frames:5d}  {line}")
        if tick:
            time.sleep(tick)
    return frames, samples


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mcap",
        nargs="?",
        help="Recording to replay (default: newest under ./recordings/)",
    )
    parser.add_argument(
        "--collections",
        default=None,
        help="Comma-separated collection ids to replay (default: auto-discover from the MCAP)",
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
        f"(default: {DEFAULT_SEPARATION_M}); 0 draws them where they were recorded",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help="Playback tick rate in Hz (default: the recording's own capture rate); 0 = as fast as possible",
    )
    args = parser.parse_args(argv[1:])

    mcap_path = resolve_mcap(args.mcap)
    if not mcap_path.is_file():
        print(f"[replay-joints] no such file: {mcap_path}", file=sys.stderr)
        return 1

    if args.collections:
        collections = [c.strip() for c in args.collections.split(",") if c.strip()]
    else:
        collections = discover_collections(mcap_path)
        if not collections:
            print(
                f"[replay-joints] no joint-pose channels found in {mcap_path}",
                file=sys.stderr,
            )
            return 1
        print(f"[replay-joints] discovered collections: {', '.join(collections)}")

    rate = args.rate if args.rate is not None else capture_rate_hz(mcap_path)
    if args.rate is None:
        print(f"[replay-joints] playback rate {rate:.1f} Hz (from recording)")

    print(f"[replay-joints] replaying {mcap_path} (thumb-to-tip distance, mm)")

    server = make_server(args.host, args.port, "[replay-joints]")
    viz = {cid: TipViz(server, cid, args.separation) for cid in collections}

    try:
        while True:
            # Fresh trackers + session per pass (replay consumes the file front-to-back).
            trackers = {cid: JointSe3PoseTracker(cid) for cid in collections}
            config = McapReplayConfig(
                str(mcap_path), tracker_names=[(t, cid) for cid, t in trackers.items()]
            )
            with ReplaySession.run(config) as session:
                frames, samples = run_once(session, trackers, viz, rate)
            print(f"[replay-joints] done — {frames} frames")
            for cid in collections:
                side = hand_side(cid)
                named = f"{cid} ({side})" if side else cid
                print(f"[replay-joints]   {named}: {samples[cid]} frames with data")
            print("[replay-joints] looping…")
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
