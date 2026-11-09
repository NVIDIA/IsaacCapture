#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Time-align one kept episode and publish the aligned copy beside its session.

The raw session is only read. The episode's stored continuity evidence comes from its manifest, and which
sources have a qualified event time from the episode's own ``timing`` attribute; an episode that lacks either is
refused with the reason, never assumed good.

Usage (from any directory outside ``egocap`` itself)::

    <egocap>/.venv/bin/python recording/tools/align_episode.py SESSION_DIR --episode N --index-source NAME \
        [--manifest-session DIR] [--aligned-session DIR] [--downsample K]

``--manifest-session`` is where the episode's manifest lives when that is not ``SESSION_DIR`` (a routed
episode's manifest stays in the session it was recorded in), and ``--manifest-episode`` its index there when that
differs from ``--episode`` (a routed episode sits under its slot). The manifest is used only if it describes
exactly this finalized, kept episode. Exit status 0 means the aligned episode was
published; 1 means it was refused or failed, with the reason on stderr.
"""

import argparse
import sys
from pathlib import Path

from egocap.recording.aligned_writer import AlignedWriteError, align_kept_episode
from egocap.recording.alignment import AlignmentError
from egocap.recording.episode_layout import manifest_path, read_manifest


def main(argv: list[str] | None = None) -> int:
    """Align the named episode and report the result; returns the process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("session", type=Path, help="the raw session directory holding episode_N")
    parser.add_argument("--episode", type=int, required=True, help="the episode's index in that directory")
    parser.add_argument("--index-source", required=True, help="the camera whose frames define the timeline")
    parser.add_argument("--manifest-session", type=Path, help="the session whose manifest describes the episode")
    parser.add_argument(
        "--manifest-episode",
        type=int,
        help="the episode's index in that session's manifests, when it differs from --episode (a routed episode)",
    )
    parser.add_argument("--aligned-session", type=Path, help="where to publish (default: <session>_aligned)")
    parser.add_argument("--downsample", type=int, default=1, help="keep every K-th index frame (default 1)")
    args = parser.parse_args(argv)

    # The manifest carries the continuity evidence; without a readable one the episode cannot be vouched for.
    try:
        manifest_episode = args.episode if args.manifest_episode is None else args.manifest_episode
        manifest = read_manifest(manifest_path(args.manifest_session or args.session, manifest_episode))
    except (OSError, ValueError) as exc:
        print(f"cannot read the episode's manifest: {exc}", file=sys.stderr)
        return 1
    try:
        aligned = align_kept_episode(
            args.session,
            args.episode,
            args.index_source,
            manifest,
            aligned_session=args.aligned_session,
            downsample=args.downsample,
        )
    except (AlignedWriteError, AlignmentError) as exc:
        print(f"not aligned: {exc}", file=sys.stderr)
        return 1
    kept, total = int(aligned.valid_indices.sum()), len(aligned.valid_indices)
    print(f"aligned episode_{args.episode}: {kept} of {total} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
