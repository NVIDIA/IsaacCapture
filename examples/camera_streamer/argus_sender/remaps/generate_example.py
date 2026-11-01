#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the example wrist RMAPs from a standalone fisheye calibration JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import cv2
import numpy as np

HEADER = struct.Struct("<4sIIf")
CAMERAS = {"wrist_left": 4, "wrist_right": 5}


def generate(calibration: Path, output: Path) -> dict:
    calib = json.loads(calibration.read_text())
    width, height = calib["image_size"]
    if (width, height) != (640, 496):
        raise ValueError("This example requires intrinsics at 640x496")
    left = calib["cameras"]["wrist_left"]
    target = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        np.array(left["K"], dtype=np.float64),
        np.array(left["D"], dtype=np.float64).reshape(4, 1),
        (width, height),
        np.eye(3),
        balance=0.0,
        new_size=(width, height),
        fov_scale=1.0,
    )
    target[1, 2] -= (height - 480) // 2
    scale = np.diag([2.0, 2.0, 1.0])
    target = scale @ target
    manifest = {
        "format": "RMAP v1",
        "example_only": True,
        "source_calibration": calibration.name,
        "source_size": [1280, 992],
        "dest_size": [1280, 960],
        "output_size": [640, 480],
        "supersample": 2,
        "target_K": target.tolist(),
        "rotation": 0,
        "cameras": [],
    }
    output.mkdir(parents=True, exist_ok=True)
    for role, device in CAMERAS.items():
        entry = calib["cameras"][role]
        mx, my = cv2.fisheye.initUndistortRectifyMap(
            scale @ np.array(entry["K"], dtype=np.float64),
            np.array(entry["D"], dtype=np.float64).reshape(4, 1),
            np.eye(3),
            target,
            (1280, 960),
            cv2.CV_32FC1,
        )
        outside = (mx < 0) | (mx >= 1279) | (my < 0) | (my >= 991)
        if not np.isfinite(mx).all() or not np.isfinite(my).all():
            raise ValueError(f"Non-finite map coordinates for {role}")
        if outside.mean() > 0.01:
            raise ValueError(f"More than 1% out-of-source coordinates for {role}")
        data = (
            HEADER.pack(b"RMAP", 1280, 960, 0.0)
            + mx.astype("<f4").tobytes()
            + my.astype("<f4").tobytes()
        )
        filename = f"cam{device}.remap"
        (output / filename).write_bytes(data)
        manifest["cameras"].append(
            {
                "role": role,
                "argus_id": device,
                "file": filename,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "K": entry["K"],
                "D": entry["D"],
            }
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--calibration", type=Path, default=here / "example_calibration.json"
    )
    parser.add_argument("--output", type=Path, default=here)
    args = parser.parse_args()
    manifest = generate(args.calibration, args.output)
    for camera in manifest["cameras"]:
        print(f"{camera['file']}: {camera['bytes']} bytes, SHA256 {camera['sha256']}")


if __name__ == "__main__":
    main()
