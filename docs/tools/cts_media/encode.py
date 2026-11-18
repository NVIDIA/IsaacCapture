# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encode the PNG frames capture.mjs wrote into looping GIFs, one per step.

    python3 docs/tools/cts_media/encode.py <frames dir> <output dir>

Each GIF plays at the page's fps and holds its last frame for PAUSE_S before looping.
``--stats full`` builds the palette from every pixel rather than from changes, so a
colour that appears mid-GIF and then stays (a contact ring) keeps its own entry.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

PAUSE_S = 0.6
COLOURS = 64
# Steps whose camera circles redraw the whole hand every frame; 32 colours look the same
# there and keep the hand suite under 3 MB.
FEWER_COLOURS = {"fist": 32, "palms_together": 32}


def encode(
    frames: Path, fps: float, out: Path, colours: int, stats: str = "diff"
) -> None:
    graph = (
        f"format=rgb24,tpad=stop_mode=clone:stop_duration={PAUSE_S},split[a][b];"
        f"[a]palettegen=max_colors={colours}:stats_mode={stats}:reserve_transparent=0[p];"
        "[b][p]paletteuse=dither=none:diff_mode=rectangle"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames / "%04d.png"),
            "-vf",
            graph,
            "-loop",
            "0",
            str(out),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("frames", type=Path, help="a directory capture.mjs wrote")
    parser.add_argument("out", type=Path)
    parser.add_argument("--colours", type=int, default=COLOURS)
    parser.add_argument("--stats", choices=("diff", "full"), default="diff")
    args = parser.parse_args()
    meta = json.loads((args.frames / "meta.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for step in meta["steps"]:
        name = Path(step["dir"]).name
        gif = args.out / f"{name}.gif"
        colours = min(args.colours, FEWER_COLOURS.get(step["label"], args.colours))
        encode(args.frames / name, meta["fps"], gif, colours, args.stats)
        size = gif.stat().st_size
        total += size
        print(f"{gif}  {size / 1024:.0f} KiB  {step['frames']} frames")
    print(f"total {total / 1024:.0f} KiB ({total / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
