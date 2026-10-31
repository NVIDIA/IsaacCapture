# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-frame cost of the SENSING CPU path at native size: H.264 decode, YUV->RGB, JPEG encode.

Run on the target (e.g. the Orin) to learn what delivery rate one stream can sustain per core:

    uv run python tests/capture_server/bench_sensing_decode.py [IMAGE.png] [--quality 80] [--frames 30]

With no image it uses synthetic noise (a worst case for both the H.264 unit size and JPEG size); pass a real
scene capture for realistic numbers. The units here are intra-only fixtures; an inter-coded stream decodes
cheaper per frame. Each iteration decodes exactly one unit, the cost the reader pays per frame.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from capture_server.acquisition.sensing_h264 import SensingStreamReader  # noqa: E402
from capture_server.acquisition.sensing_jpeg import encode_rgb_jpeg  # noqa: E402

WIDTH, HEIGHT = 2560, 1984


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--quality", type=int, default=80)
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args()

    if args.image:
        rgb = np.asarray(Image.open(args.image).convert("RGB").resize((WIDTH, HEIGHT)))
    else:
        rgb = np.random.default_rng(0).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    # encode_units takes solid colours; encode the real picture the same way through PyAV directly.
    from fractions import Fraction

    import av

    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    enc.framerate, enc.time_base, enc.gop_size = 30, Fraction(1, 30), 1
    enc.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "20", "bframes": "0"}
    units = []
    for i in range(3):
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(
            format="yuv420p", dst_colorspace="ITU601", dst_color_range="MPEG"
        )
        frame.pts = i
        units += [bytes(p) for p in enc.encode(frame)]
    units += [bytes(p) for p in enc.encode(None)]
    print(f"intra-only unit: {len(units[0]) / 1e6:.2f} MB; JPEG q{args.quality} cap 2.00 MB")

    path = Path("/tmp/bench_sensing.h264")
    decode_ms, jpeg_ms, sizes = [], [], []
    for _ in range(args.frames):
        path.write_bytes(b"".join(units[:2]))  # the first unit is complete; the parser holds the second back
        reader = SensingStreamReader(str(path), width=WIDTH, height=HEIGHT)
        reader.connect(1.0)
        t = time.perf_counter()
        out, _ = reader.read_latest(5.0)
        decode_ms.append((time.perf_counter() - t) * 1000)
        reader.close()
        t = time.perf_counter()
        sizes.append(len(encode_rgb_jpeg(out, args.quality, max_bytes=1 << 30)))  # measure, do not enforce
        jpeg_ms.append((time.perf_counter() - t) * 1000)
    d, j = statistics.median(decode_ms), statistics.median(jpeg_ms)
    size = statistics.median(sizes)
    verdict = "OVER the 2 MiB wire ceiling" if size > 2 * 1024 * 1024 else "within the 2 MiB wire ceiling"
    print(f"decode+yuv->rgb: {d:.1f} ms   jpeg: {j:.1f} ms   JPEG size: {size / 1e6:.2f} MB ({verdict})")
    total = d + j
    print(f"per frame {total:.1f} ms  =>  ~{1000 / total:.1f} fps per stream per core; two streams need 2x that")


if __name__ == "__main__":
    main()
