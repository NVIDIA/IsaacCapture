# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Real H.264 for SENSING reader tests, encoded with libx264 (fixtures, not native output).

Encoded with PyAV/libx264 (IDR frames carry in-band SPS/PPS; no B-frames), so the reader and adapter
are exercised against genuinely decodable Annex-B data rather than hand-made bytes.
"""

from __future__ import annotations

import os
import threading
from fractions import Fraction

import numpy as np


def encode_units(width: int, height: int, colors: list[tuple[int, int, int]], *, gop: int = 1) -> list[bytes]:
    """One Annex-B access unit per solid-colour frame; ``gop=1`` makes each independently decodable, a larger GOP
    makes later units P-frames that reference earlier ones."""
    import av

    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = width, height, "yuv420p"
    enc.framerate, enc.time_base, enc.gop_size = 30, Fraction(1, 30), gop
    enc.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "18", "bframes": "0"}
    packets = []
    for i, (r, g, b) in enumerate(colors):
        # BT.601 limited-range YUV, the producer's own conversion, so decoded RGB should match the input.
        rgb = np.empty((height, width, 3), np.uint8)
        rgb[...] = (r, g, b)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(
            format="yuv420p", dst_colorspace="ITU601", dst_color_range="MPEG"
        )
        frame.pts = i
        packets += enc.encode(frame)
    packets += enc.encode(None)
    return [bytes(p) for p in packets]


class FileProducer:
    """Appends units to ``path`` like the native ``output=`` writer: buffered, so a unit's tail lags."""

    def __init__(self, path: str, units: list[bytes], *, interval_s: float = 0.02, loop: bool = False) -> None:
        self.path, self.units, self.interval_s, self.loop = path, units, interval_s, loop
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "FileProducer":
        self._thread.start()
        return self

    def _run(self) -> None:
        with open(self.path, "wb") as f:
            while not self._stop.is_set():
                for unit in self.units:
                    f.write(unit)
                    f.flush()
                    if self._stop.wait(self.interval_s):
                        return
                if not self.loop:
                    # Like the encoder tail: a final unit only becomes complete when another begins.
                    f.write(self.units[-1])
                    f.flush()
                    self._stop.wait()
                    return

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def allocated_bytes(path: str) -> int:
    return os.stat(path).st_blocks * 512
