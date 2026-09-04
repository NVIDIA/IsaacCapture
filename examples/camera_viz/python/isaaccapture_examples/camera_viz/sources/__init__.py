# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera / video sources for camera_viz.

Each source emits GPU-resident RGBA8 frames via the ``FrameSource``
contract — the teleop hot path never round-trips through host memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from ..pipeline import FrameSource

from ._helpers import PairedFrameSource, set_notify_sink, set_verbose
from .argus import ArgusSource
from .cuda_ipc import CudaIpcSource
from .oakd import OakdSource
from .rtp_h264 import RtpH264Source
from .synthetic import SyntheticSource, SyntheticStereoSource
from .v4l2 import V4l2Source
from .video_file import VideoFileSource
from .zed import ZedSource

__all__ = [
    "ArgusSource",
    "CudaIpcSource",
    "OakdSource",
    "PairedFrameSource",
    "RtpH264Source",
    "SyntheticSource",
    "SyntheticStereoSource",
    "V4l2Source",
    "VideoFileSource",
    "ZedSource",
    "build_local_camera",
    "resolve_video_paths",
    "set_notify_sink",
    "set_verbose",
]


def resolve_video_paths(cfg: dict, base_dir) -> None:
    """Anchor relative ``path:`` values of ``type: video`` cameras to the
    YAML file's directory (in place), so playback doesn't depend on the
    process CWD. Call right after loading the config."""
    for cam in cfg.get("cameras", []):
        if cam.get("type") == "video" and "path" in cam:
            p = Path(str(cam["path"])).expanduser()
            if not p.is_absolute():
                p = Path(base_dir) / p
            cam["path"] = str(p)


def build_local_camera(spec: dict) -> List[FrameSource]:
    """Build local FrameSource(s) for one ``cameras:`` entry.

    Mono → [source]; ``stereo: true`` → [PairedFrameSource]. v4l2 rejects
    stereo (UVC is mono). Shared by camera_viz + camera_streamer.
    """
    kind = spec["type"]
    stereo = bool(spec.get("stereo", False))
    name = spec["name"]
    if kind == "synthetic":
        if stereo:
            return [
                SyntheticStereoSource(
                    name=name,
                    width=int(spec["width"]),
                    height=int(spec["height"]),
                    fps=float(spec.get("fps", 60.0)),
                    hue_speed_hz=float(spec.get("hue_speed_hz", 0.25)),
                    disparity_px=int(spec.get("disparity_px", 20)),
                )
            ]
        return [
            SyntheticSource(
                name=name,
                width=int(spec["width"]),
                height=int(spec["height"]),
                fps=float(spec.get("fps", 60.0)),
                hue_speed_hz=float(spec.get("hue_speed_hz", 0.25)),
            )
        ]
    if kind == "v4l2":
        if stereo:
            raise ValueError(
                f"build_local_camera: v4l2 camera {name!r} cannot be stereo "
                "(single-stream USB / UVC). Use type: oakd or zed."
            )
        return [
            V4l2Source(
                name=name,
                device=spec.get("device", "/dev/video0"),
                width=int(spec["width"]),
                height=int(spec["height"]),
                fps=float(spec.get("fps", 30.0)),
                fourcc=spec.get("fourcc"),
            )
        ]
    if kind == "argus":

        def make_argus(source_name: str, sensor_ids: list[int]) -> ArgusSource:
            return ArgusSource(
                name=source_name,
                sensor_ids=sensor_ids,
                width=int(spec["width"]),
                height=int(spec["height"]),
                sensor_mode=int(spec.get("sensor_mode", 0)),
                fps=float(spec.get("fps", 30.0)),
                gpu_id=int(spec.get("gpu_id", 0)),
                full_range=bool(spec.get("full_range", False)),
                swap_uv=bool(spec.get("swap_uv", False)),
                acquire_timeout_ms=int(spec.get("acquire_timeout_ms", 0xFFFFFFFF)),
                repeat_capture=bool(spec.get("repeat_capture", True)),
            )

        if stereo:
            left_id = int(spec.get("sensor_id_left", spec.get("left_sensor_id", 0)))
            right_id = int(spec.get("sensor_id_right", spec.get("right_sensor_id", 1)))
            sync_session = bool(spec.get("sync_session", False))
            if sync_session:
                source = make_argus(name, [left_id, right_id])
                left = source.eye_source("left", f"{name}_left")
                right = source.eye_source("right", f"{name}_right")
            else:
                left = make_argus(f"{name}_left", [left_id])
                right = make_argus(f"{name}_right", [right_id])
            emit_mode = spec.get("pair_emit_mode", "both")
            return [
                PairedFrameSource(
                    name=name, left=left, right=right, emit_mode=emit_mode
                )
            ]

        sensor_id = int(spec.get("sensor_id", spec.get("sensor_id_left", 0)))
        return [make_argus(name, [sensor_id])]
    if kind == "cuda_ipc":
        if stereo:
            # Each publisher serves one sensor, so a stereo rig is two
            # cameras: entries here, paired by the caller.
            raise ValueError(
                f"build_local_camera: cuda_ipc camera {name!r} cannot be stereo — "
                "declare one entry per socket."
            )
        return [
            CudaIpcSource(
                name=name,
                socket_path=spec["socket"],
                width=int(spec["width"]),
                height=int(spec["height"]),
            )
        ]
    if kind == "oakd":
        # ``stereo: true`` shorthand for ``mode: stereo``; explicit mode wins.
        mode = spec.get("mode", "stereo" if stereo else "mono")
        eyes = list(
            OakdSource.build(
                base_name=name,
                mode=mode,
                device_id=spec.get("device_id", ""),
                width=int(spec["width"]),
                height=int(spec["height"]),
                fps=int(spec.get("fps", 30)),
                camera_socket=spec.get("camera_socket", "RGB"),
                rgb_width=int(spec.get("rgb_width", 0)),
                rgb_height=int(spec.get("rgb_height", 0)),
                rgb_fps=int(spec.get("rgb_fps", 0)),
            )
        )
        if stereo or mode in ("stereo", "stereo_rgb"):
            # stereo_rgb's third stream is intentionally dropped here.
            if len(eyes) < 2:
                raise ValueError(
                    f"build_local_camera: oakd {name!r} stereo mode produced {len(eyes)} "
                    "source(s); expected at least 2"
                )
            return [PairedFrameSource(name=name, left=eyes[0], right=eyes[1])]
        return eyes
    if kind == "video":
        # Stereo (side-by-side file) emits both eyes from one source, like
        # SyntheticStereoSource — viewer-only; camera_streamer's
        # _eye_sources rejects it because there are no per-eye streams.
        return [
            VideoFileSource(
                name=name,
                path=spec["path"],
                width=int(spec.get("width", 0)),
                height=int(spec.get("height", 0)),
                fps=float(spec.get("fps", 0.0)),
                loop=bool(spec.get("loop", True)),
                stereo=stereo,
            )
        ]
    if kind == "zed":
        eyes = list(
            ZedSource.build(
                base_name=name,
                width=int(spec["width"]),
                height=int(spec["height"]),
                fps=int(spec.get("fps", 30)),
                serial_number=int(spec.get("serial_number", 0)),
                bus_type=spec.get("bus_type", "usb"),
                stereo=stereo,
            )
        )
        if stereo:
            if len(eyes) != 2:
                raise ValueError(
                    f"build_local_camera: zed {name!r} stereo produced {len(eyes)} "
                    "source(s); expected 2"
                )
            return [PairedFrameSource(name=name, left=eyes[0], right=eyes[1])]
        return eyes
    raise ValueError(
        f"build_local_camera: unknown camera type {kind!r} "
        "(known: synthetic, v4l2, argus, cuda_ipc, oakd, zed, video)"
    )
