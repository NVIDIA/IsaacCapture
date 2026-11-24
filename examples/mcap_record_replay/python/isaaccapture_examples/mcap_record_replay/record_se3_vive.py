# SPDX-FileCopyrightText: Copyright (c) 2026 HTC Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Record live VIVE Ultimate Tracker SE3 pose streams to an MCAP file — headset-free.

Reads the per-tracker "vive_tracker_<serial>" tensor collections pushed by the
vive_se3_tracker plugin (one core.Se3Tracker per collection) and records each
one to its own MCAP channel pair (<cid> / <cid>_tracked) via the standard
Se3TrackerRecordingTraits. Replay with replay_se3_vive.

Prerequisites (separate terminals):
  1. CloudXR runtime:  python -m isaaccapture.cloudxr
  2. the pusher:       ./vive_se3_tracker_plugin
     (VIVEHub tracker_server running, or VIVE_SE3_SYNTHETIC=1 for a smoke test)

With no arguments it records 10 s of every collection the pusher currently
advertises (see --collections), to a timestamped file under ./recordings/.

Usage:
    source ~/.cloudxr/run/cloudxr.env
    python -m isaaccapture_examples.mcap_record_replay.record_se3_vive [duration_s] [output.mcap] [--collections a,b,c]
"""

import argparse
import os
import stat
import sys
import time
from datetime import datetime
from pathlib import Path


def _collections_file() -> str | None:
    """Match the publisher's trusted existing discovery directories."""
    return os.environ.get("VIVE_SE3_COLLECTIONS_FILE") or _automatic_collections_file()


def _automatic_collections_file() -> str | None:
    if os.name != "posix":
        return None
    from isaaccapture.logging_config._core import default_temp_root, is_private_dir

    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg and is_private_dir(Path(xdg)):
        return str(Path(xdg) / "vive_se3_collections.txt")
    base = default_temp_root() / f"vive_se3_tracker-{os.geteuid()}"
    # Only the publisher creates its directory; discovery validates existing paths.
    if is_private_dir(base):
        return str(base / "vive_se3_collections.txt")
    return None


def discover_collections() -> list[str]:
    """Read trusted collection ids in the order advertised by the pusher."""
    override = os.environ.get("VIVE_SE3_COLLECTIONS_FILE")
    selected = override or _automatic_collections_file()
    if selected is None:
        return []
    path = Path(selected)
    try:
        if override:  # Explicit paths remain the operator's responsibility.
            if not path.is_file():
                return []
            text = path.read_text()
        else:
            fd = os.open(
                path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
            )
            try:
                info = os.fstat(fd)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
                ):
                    return []
                with os.fdopen(fd, closefd=False) as stream:
                    text = stream.read()
            finally:
                os.close(fd)
    except (OSError, UnicodeError):
        return []
    return list(
        dict.fromkeys(line.strip() for line in text.splitlines() if line.strip())
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "duration", nargs="?", type=float, default=10.0, help="Recording duration (s)"
    )
    parser.add_argument("output", nargs="?", help="Output .mcap path")
    parser.add_argument(
        "--collections",
        default=None,
        help="Comma-separated collection ids (default: auto-discover from the running pusher)",
    )
    args = parser.parse_args(argv[1:])

    if args.output:
        mcap_path = Path(args.output)
        mcap_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path.cwd() / "recordings"
        out_dir.mkdir(exist_ok=True)
        mcap_path = out_dir / f"se3_vive_{datetime.now():%Y%m%d_%H%M%S}.mcap"

    if args.collections:
        collections = [c.strip() for c in args.collections.split(",") if c.strip()]
    else:
        collections = discover_collections()
        if not collections:
            print(
                "[record-se3] no live collections found "
                f"({_collections_file() or 'no trusted advertisement path'}; missing, empty, or unsafe). "
                "Is the vive_se3_tracker pusher running and a tracker streaming? "
                "Or pass --collections explicitly.",
                file=sys.stderr,
            )
            return 1
        print(f"[record-se3] discovered collections: {', '.join(collections)}")

    from isaaccapture.deviceio_session import DeviceIOSession, McapRecordingConfig
    from isaaccapture.deviceio_trackers import Se3Tracker
    from isaaccapture.oxr import OpenXRSession

    trackers = {cid: Se3Tracker(cid) for cid in collections}

    # MCAP channel base name = collection id (channels "<cid>" + "<cid>_tracked").
    recording = McapRecordingConfig(
        str(mcap_path), tracker_names=[(t, cid) for cid, t in trackers.items()]
    )

    print(f"[record-se3] writing {mcap_path} for {args.duration:.1f}s")
    for cid in collections:
        print(f"[record-se3]   collection '{cid}'")

    tracker_list = list(trackers.values())
    extensions = DeviceIOSession.get_required_extensions(tracker_list)
    with OpenXRSession("McapSe3ViveRecord", extensions) as oxr_session:
        with DeviceIOSession.run(
            tracker_list, oxr_session.get_handles(), recording
        ) as session:
            start = time.time()
            frame = 0
            while time.time() - start < args.duration:
                session.update()
                if frame % 60 == 0:
                    parts = []
                    for cid, tracker in trackers.items():
                        data = tracker.get_data(session)
                        if data.data is None:
                            parts.append(f"{cid}: -")
                        elif not data.data.is_valid:
                            parts.append(f"{cid}: lost")
                        else:
                            p = data.data.pose.position
                            parts.append(f"{cid}: [{p.x:+.2f} {p.y:+.2f} {p.z:+.2f}]")
                    print(
                        f"[record-se3] t={time.time() - start:5.2f}s  "
                        + "  ".join(parts)
                    )
                frame += 1
                time.sleep(1 / 90)

    print(f"[record-se3] done — {mcap_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
