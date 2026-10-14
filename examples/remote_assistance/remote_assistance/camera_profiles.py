# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate matching camera sender/viewer profiles without changing credentials."""

import argparse
import copy
import os
from pathlib import Path

import yaml

OAK_MODES = {
    "720p60": (1280, 720, 60),
    "1080p60": (1920, 1080, 60),
    "4k30": (3840, 2160, 30),
    "2016x1520-60": (2016, 1520, 60),
}
S56_MODES = {
    "720p60": (1280, 720, 60, 0),
    "1080p30": (1920, 1080, 30, 0),
    "1080p60": (1920, 1080, 60, 0),
    "2560x1984-30": (2560, 1984, 30, 1),
}


def _write_profile(site, original, filename, spec, encoder):
    cfg = copy.deepcopy(original)
    cfg["livekit"].update(
        codec="h264", video_encoder=encoder, video_decoder="auto", bitrate_mbps=15
    )
    cfg["cameras"] = [spec]
    path = Path(site) / filename
    if path.is_symlink():
        raise ValueError(f"Refusing symlink: {path}")
    with open(
        path, "w", opener=lambda name, flags: os.open(name, flags, 0o600)
    ) as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False)
    path.chmod(0o600)


def _camera(original):
    cameras = [c for c in original["cameras"] if c.get("enabled", True)]
    if len(cameras) != 1:
        raise ValueError("Camera profiles require exactly one enabled camera")
    return cameras[0]


def write_oak_profiles(site, original):
    camera = _camera(original)
    for label, (width, height, fps) in OAK_MODES.items():
        spec = dict(
            name=camera["name"], type="depthai", width=width, height=height, fps=fps
        )
        if camera.get("mxid"):
            spec["mxid"] = camera["mxid"]
        _write_profile(site, original, f"camera-oak-{label}.yaml", spec, "depthai")


def write_s56_profiles(site, original):
    camera = _camera(original)
    for label, (width, height, fps, mode) in S56_MODES.items():
        spec = dict(
            name=camera["name"],
            type="argus",
            sensor_id=0,
            sensor_mode=mode,
            width=width,
            height=height,
            fps=fps,
            flip_method=2,
        )
        _write_profile(site, original, f"camera-s56-{label}.yaml", spec, "gstreamer")


def write_camera_profiles(site, original):
    write_oak_profiles(site, original)
    write_s56_profiles(site, original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, default=Path.home() / "remote-assistance-network"
    )
    args = parser.parse_args()
    configs = {
        role: yaml.safe_load((args.bundle / role / "camera.yaml").read_text())
        for role in ("edge", "robot")
    }
    for role, cfg in configs.items():
        write_camera_profiles(args.bundle / role, cfg)
    print(
        f"Created OAK and S56 profiles in each site under {args.bundle}; credentials unchanged."
    )


if __name__ == "__main__":
    main()
