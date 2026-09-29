# SPDX-FileCopyrightText: Copyright (c) 2026 HTC Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Replay VIVE Ultimate Tracker SE3 pose streams from an MCAP file — no runtime needed.

Opens a recording made by record_se3_vive, replays each per-tracker channel
through ReplaySession + core.Se3Tracker (ReplaySe3TrackerImpl), and shows every
tracker as a coordinate frame in a viser 3D view (browser). Replay needs no
OpenXR runtime and no hardware.

With no arguments it replays the most recent recording under ./recordings/,
auto-discovers its collections, and plays back at the recording's own capture
rate. Open the printed viser URL to see the trackers in 3D.

Replay repeats until Ctrl+C.

Usage:
    python -m isaaccapture_examples.mcap_record_replay.replay_se3_vive [recording.mcap] [--rate N] [--no-viz] \
        [--collections a,b,c]
"""

import argparse
import sys
import time
from pathlib import Path

from mcap.reader import make_reader

from isaaccapture.deviceio_session import McapReplayConfig, ReplaySession
from isaaccapture.deviceio_trackers import Se3Tracker

# SE3 recordings write two channels per collection: "<cid>/se3_tracker" and
# "<cid>/se3_tracker_tracked". Strip either suffix to recover the collection id.
_CHANNEL_SUFFIXES = ("/se3_tracker_tracked", "/se3_tracker")


def _summary(mcap_path: Path):
    with open(mcap_path, "rb") as f:
        return make_reader(f).get_summary()


def resolve_mcap(path_arg: str | None) -> Path:
    """Use the given path, or the newest .mcap under ./recordings/."""
    if path_arg:
        return Path(path_arg)
    recordings = Path.cwd() / "recordings"
    candidates = list(recordings.glob("se3_vive_*.mcap")) or list(
        recordings.glob("*.mcap")
    )
    if not candidates:
        sys.exit(
            f"[replay-se3] no .mcap files in {recordings}. Run record_se3_vive first."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def discover_collections(mcap_path: Path) -> list[str]:
    """Return the SE3 collection ids present in an MCAP, in first-seen order."""
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
    """Estimate playback rate from the per-tick channel replay actually consumes.

    Replay advances one frame per ReplaySession.update() and reads the coalesced
    "<cid>/se3_tracker_tracked" channel (one message per recording tick). Rate is
    derived from that channel's message count -- not the raw "/se3_tracker" sample
    channel, whose count can exceed the tick count when the producer bursts
    several samples per tick (which would make playback run too fast).
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
        if ch.topic.endswith("/se3_tracker_tracked")
    ]
    if not tracked_ids:
        return default
    counts = stats.channel_message_counts
    best = max((counts.get(cid, 0) for cid in tracked_ids), default=0)
    if best <= 1:
        return default
    return best / (span_ns / 1e9)


# Fixed distinct axis-label colors per tracker slot (viser label background).
TRACKER_COLORS = [
    (230, 60, 60),
    (60, 180, 75),
    (65, 105, 225),
    (240, 160, 30),
    (150, 90, 200),
]


class Se3Viz:
    """One coordinate frame + name label per tracker in a viser scene."""

    def __init__(self, server, collections: list[str]):
        import viser  # noqa: F401  (import here so --no-viz runs without viser)

        # Also deferred: common imports viser at module level, so a top-level
        # import here would defeat --no-viz.
        from .common import setup_scene

        setup_scene(server)
        self._frames = {}
        self._labels = {}
        for cid in collections:
            self._frames[cid] = server.scene.add_frame(
                f"/trackers/{cid}", axes_length=0.15, axes_radius=0.006, visible=False
            )
            self._labels[cid] = server.scene.add_label(
                f"/trackers/{cid}/label", text=cid.removeprefix("vive_tracker_")
            )

    def update(self, cid: str, pose, valid: bool) -> None:
        frame = self._frames[cid]
        if not valid or pose is None:
            frame.visible = False
            return
        p, q = pose.position, pose.orientation
        frame.position = (p.x, p.y, p.z)
        # schema quaternion is (x, y, z, w); viser expects (w, x, y, z).
        frame.wxyz = (q.w, q.x, q.y, q.z)
        frame.visible = True


def run_once(session, trackers, viz, rate_hz: float) -> tuple[int, dict]:
    frames = 0
    samples = dict.fromkeys(trackers, 0)
    none_streak = 0
    tick = 1.0 / rate_hz if rate_hz > 0 else 0.0
    # update() is void; per the replay contract tracker data goes null at
    # end-of-file, so stop after a run of all-null frames.
    while none_streak < 30:
        session.update()
        frames += 1
        any_data = False
        parts = []
        for cid, tracker in trackers.items():
            data = tracker.get_data(session)
            if data.data is None:
                parts.append(f"{cid}: -")
                if viz:
                    viz.update(cid, None, False)
                continue
            any_data = True
            samples[cid] += 1
            valid = bool(data.data.is_valid)
            if viz:
                viz.update(cid, data.data.pose, valid)
            if not valid:
                parts.append(f"{cid}: lost")
            else:
                p = data.data.pose.position
                parts.append(f"{cid}: [{p.x:+.2f} {p.y:+.2f} {p.z:+.2f}]")
        none_streak = 0 if any_data else none_streak + 1
        if frames % 60 == 1:
            print(f"[replay-se3] frame={frames:5d}  " + "  ".join(parts))
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
        "--rate",
        type=float,
        default=None,
        help="Playback tick rate in Hz (default: the recording's own capture rate); 0 = as fast as possible",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Text output only, no viser"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Viser HTTP bind address (default: 0.0.0.0, all interfaces; pass 127.0.0.1 to keep it local)",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viser HTTP port")
    args = parser.parse_args(argv[1:])

    mcap_path = resolve_mcap(args.mcap)
    if not mcap_path.is_file():
        print(f"[replay-se3] no such file: {mcap_path}", file=sys.stderr)
        return 1

    if args.collections:
        collections = [c.strip() for c in args.collections.split(",") if c.strip()]
    else:
        collections = discover_collections(mcap_path)
        if not collections:
            print(
                f"[replay-se3] no SE3 tracker channels found in {mcap_path}",
                file=sys.stderr,
            )
            return 1
        print(f"[replay-se3] discovered collections: {', '.join(collections)}")

    rate = args.rate if args.rate is not None else capture_rate_hz(mcap_path)
    if args.rate is None:
        print(f"[replay-se3] playback rate {rate:.1f} Hz (from recording)")

    viz = None
    if not args.no_viz:
        import viser

        server = viser.ViserServer(host=args.host, port=args.port)
        viz = Se3Viz(server, collections)
        print(
            f"[replay-se3] viser listening on {args.host}:{args.port} "
            f"(http://localhost:{args.port})"
        )

    print(f"[replay-se3] replaying {mcap_path}")

    try:
        while True:
            # Fresh trackers + session per pass (replay consumes the file front-to-back).
            trackers = {cid: Se3Tracker(cid) for cid in collections}
            config = McapReplayConfig(
                str(mcap_path), tracker_names=[(t, cid) for cid, t in trackers.items()]
            )
            with ReplaySession.run(config) as session:
                frames, samples = run_once(session, trackers, viz, rate)
            print(f"[replay-se3] done — {frames} frames")
            for cid in collections:
                print(f"[replay-se3]   {cid}: {samples[cid]} frames with data")
            print("[replay-se3] looping…")
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
