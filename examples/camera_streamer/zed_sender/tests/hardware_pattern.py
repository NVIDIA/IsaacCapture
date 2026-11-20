#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in VIC/NVENC/NVDEC test of color, eye order, scaling and buffer reuse."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    directory = args.output_dir or Path(tempfile.mkdtemp(prefix="zed-pattern-"))
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / "pattern"
    binary = Path(__file__).resolve().parents[1] / "build/zed_encoder_pattern"
    subprocess.run([str(binary), str(prefix)], check=True, timeout=20)
    colors = np.array(
        [
            [16, 128, 128],
            [126, 128, 128],
            [235, 128, 128],
            [81, 90, 240],
            [145, 54, 34],
            [41, 240, 110],
            [210, 16, 146],
            [170, 166, 16],
        ],
        float,
    )
    y = (colors[:, 0] - 16) / 219
    u, v = (colors[:, 1] - 128) / 224, (colors[:, 2] - 128) / 224
    rgb = np.clip(
        np.stack(
            (y + 1.402 * v, y - 0.344136 * u - 0.714136 * v, y + 1.772 * u), axis=-1
        ),
        0,
        1,
    )
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
    expected = np.stack(
        (
            16 + 219 * luma,
            128 + 224 * (rgb[:, 2] - luma) / 1.8556,
            128 + 224 * (rgb[:, 0] - luma) / 1.5748,
        ),
        axis=-1,
    )
    summary = {}
    for eye in ("left", "right"):
        raw_path = Path(f"{prefix}-{eye}.nv12")
        result = subprocess.run(
            [
                "gst-launch-1.0",
                "-v",
                "filesrc",
                f"location={prefix}-{eye}.h265",
                "!",
                "h265parse",
                "!",
                "nvv4l2decoder",
                "!",
                "nvvidconv",
                "!",
                "video/x-raw,format=NV12",
                "!",
                "filesink",
                f"location={raw_path}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        (directory / f"decode-{eye}.log").write_text(result.stdout + result.stderr)
        assert "colorimetry=(string)bt709" in result.stdout, (
            "incorrect HEVC color signaling"
        )
        raw = np.fromfile(raw_path, dtype=np.uint8).reshape(-1, 640 * 360 * 3 // 2)
        assert len(raw) == 8, f"decoded {len(raw)} frames; expected 8"
        worst = 0.0
        for f, frame in enumerate(raw):
            yp, uv = (
                frame[: 640 * 360].reshape(360, 640),
                frame[640 * 360 :].reshape(180, 320, 2),
            )
            observed = np.array(
                [[yp[180, 80 * i + 40], *uv[90, 40 * i + 20]] for i in range(8)], float
            )
            indices = [((i if eye == "left" else 7 - i) + f) % 8 for i in range(8)]
            worst = max(worst, float(np.abs(observed - expected[indices]).max()))
        assert worst < 4, f"{eye}: color/order/reuse error {worst:.2f} levels"
        summary[eye] = dict(frames=len(raw), max_color_error_levels=worst)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
