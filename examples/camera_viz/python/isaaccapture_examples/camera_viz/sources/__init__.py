# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera / video sources for camera_viz.

Each source emits GPU-resident RGBA8 frames via the ``FrameSource``
contract — the teleop hot path never round-trips through host memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..pipeline import FrameSource

from ._helpers import PairedFrameSource, set_notify_sink, set_verbose
from .argus import ArgusSource
from .oakd import OakdSource
from .oakd_stereo import OakdStereoSource
from .rtp_h264 import RtpH264Source
from .synthetic import SyntheticSource, SyntheticStereoSource
from .v4l2 import V4l2Source
from .video_file import VideoFileSource
from .zed import ZedSource, ZedStereoSource
from .zed import _resolution_for_dims as _zed_resolution
from .zed import video_settings as _zed_video_settings

__all__ = [
    "ArgusSource",
    "OakdSource",
    "OakdStereoSource",
    "PairedFrameSource",
    "RtpH264Source",
    "SyntheticSource",
    "SyntheticStereoSource",
    "V4l2Source",
    "VideoFileSource",
    "ZedSource",
    "ZedStereoSource",
    "build_local_camera",
    "build_stereo_camera",
    "StereoCameraTraits",
    "check_stereo_camera",
    "describe_stereo_camera",
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
                settings=_zed_video_settings(spec),
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
        "(known: synthetic, v4l2, argus, oakd, zed, video)"
    )


def _zed_stereo_arguments(spec: dict, name: str) -> dict:
    """What a stereo ``type: zed`` entry passes to its source, checked."""
    try:
        arguments = {
            "width": int(spec["width"]),
            "height": int(spec["height"]),
            "fps": int(spec.get("fps", 30)),
            "serial_number": int(spec.get("serial_number", 0)),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"camera {name!r}: width, height, fps and serial_number are whole "
            f"numbers ({exc})"
        ) from exc
    bus_type = str(spec.get("bus_type", "usb")).lower()
    if bus_type not in ("usb", "gmsl"):
        raise ValueError(f"camera {name!r}: bus_type is usb or gmsl, got {bus_type!r}")
    _zed_resolution(arguments["width"], arguments["height"])
    return {**arguments, "bus_type": bus_type, "settings": _zed_video_settings(spec)}


def _video_stereo_arguments(spec: dict, name: str) -> dict:
    """What a side-by-side ``type: video`` entry passes to its source,
    checked."""
    try:
        return {
            "path": str(spec["path"]),
            "width": int(spec.get("width", 0)),
            "height": int(spec.get("height", 0)),
            "fps": float(spec.get("fps", 0.0)),
            "loop": bool(spec.get("loop", True)),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"camera {name!r}: width, height and fps are numbers ({exc})"
        ) from exc


def _oakd_stereo_arguments(spec: dict, name: str, mode: str) -> dict:
    """What a stereo ``type: oakd`` entry passes to its source, checked."""
    try:
        arguments = {
            "width": int(spec["width"]),
            "height": int(spec["height"]),
            "fps": int(spec.get("fps", 30)),
            "picture_width": int(spec.get("rgb_width", 0)),
            "picture_height": int(spec.get("rgb_height", 0)),
            "exposure_limit_ms": float(spec.get("exposure_limit_ms", 0.0)),
        }
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"camera {name!r}: width, height, fps, rgb_width, rgb_height and "
            f"exposure_limit_ms are numbers ({exc})"
        ) from exc
    if min(arguments["width"], arguments["height"], arguments["fps"]) <= 0:
        raise ValueError(f"camera {name!r}: width, height and fps must be above 0")
    if arguments["picture_width"] < 0 or arguments["picture_height"] < 0:
        raise ValueError(
            f"camera {name!r}: rgb_width and rgb_height cannot be negative"
        )
    limit = arguments["exposure_limit_ms"]
    if not (math.isfinite(limit) and limit >= 0.0):
        raise ValueError(
            f"camera {name!r}: exposure_limit_ms is a finite number, 0 or above"
        )
    return {
        **arguments,
        "device_id": str(spec.get("device_id", "")),
        "picture": mode == "stereo_rgb",
    }


def _stereo_arguments(spec: dict) -> tuple:
    """``(type, name, arguments)`` of a stereo camera entry. Raises
    ValueError, with the reason, for any other entry."""
    if not isinstance(spec, dict):
        raise ValueError(f"a camera entry is a mapping, got {type(spec).__name__}")
    kind = spec.get("type")
    name = spec.get("name", kind)
    stereo = bool(spec.get("stereo", False))

    def refuse(reason: str):
        raise ValueError(f"camera {name!r} is not a stereo camera: {reason}")

    def require(*keys: str) -> None:
        missing = [key for key in keys if key not in spec]
        if missing:
            raise ValueError(
                f"camera {name!r}: a `type: {kind}` entry needs {' and '.join(missing)}"
            )

    if kind == "zed":
        if not stereo:
            refuse("a zed entry opens one eye unless it sets `stereo: true`")
        require("width", "height")
        return kind, name, _zed_stereo_arguments(spec, name)
    if kind == "video":
        if not stereo:
            refuse("it is one view; a side-by-side recording sets `stereo: true`")
        require("path")
        return kind, name, _video_stereo_arguments(spec, name)
    if kind == "synthetic":
        if not stereo:
            refuse("it is one view")
        refuse(
            "the test pattern is one drawing shifted between the eyes, with "
            "nothing behind it"
        )
    if kind == "v4l2":
        refuse("a v4l2 camera gives one image")
    if kind == "argus":
        if stereo and spec.get("sync_session", False):
            refuse("an argus camera is not offered as a stereo camera yet")
        refuse(
            "its eyes are not one capture unless it sets `stereo: true` and "
            "`sync_session: true`, one session for both sensors"
        )
    if kind == "oakd":
        # As build_local_camera reads it: `stereo: true` is short for
        # `mode: stereo`, and an explicit mode wins.
        mode = str(spec.get("mode", "stereo" if stereo else "mono")).lower()
        if mode == "mono":
            refuse(
                "an oakd entry opens one camera unless it sets `stereo: true` or "
                "`mode: stereo_rgb`, the pair with the colour camera's picture"
            )
        if mode not in ("stereo", "stereo_rgb"):
            raise ValueError(
                f"camera {name!r}: unknown oakd mode {mode!r} "
                "(stereo cameras: stereo, stereo_rgb)"
            )
        require("width", "height")
        return kind, name, _oakd_stereo_arguments(spec, name, mode)
    raise ValueError(
        f"camera {name!r}: unknown camera type {kind!r} "
        "(stereo cameras: zed, oakd, video)"
    )


def check_stereo_camera(spec: dict) -> None:
    """Raise ValueError unless ``spec`` describes a stereo camera: one whose
    every frame holds both eyes of one capture. Opens nothing."""
    _stereo_arguments(spec)


@dataclass(frozen=True)
class StereoCameraTraits:
    """What is known of a stereo camera from its entry, before it is
    opened.

    ``calibrated``: it stores its stereo calibration, for ``calibration()``;
    one that does not needs it from elsewhere, such as a file.
    ``rectified``: its images are already rectified.
    ``picture``: it delivers a picture of its own beside the pair, in
    ``Frame.picture``.
    """

    calibrated: bool
    rectified: bool
    picture: bool = False


def describe_stereo_camera(spec: dict) -> StereoCameraTraits:
    """The traits of the stereo camera ``spec`` describes. Raises
    ValueError, as ``check_stereo_camera`` does, for an entry that is not
    one. Opens nothing."""
    kind, _name, arguments = _stereo_arguments(spec)
    return StereoCameraTraits(
        calibrated=kind in ("zed", "oakd"),
        rectified=kind == "zed",
        picture=kind == "oakd" and arguments["picture"],
    )


def build_stereo_camera(spec: dict, *, imu: bool = False) -> FrameSource:
    """Open one ``cameras:`` entry as a stereo camera.

    Every frame of the returned source holds both eyes of one capture,
    ``image`` and ``image_right``, in capture order and never twice. A
    camera stamps the frame with its capture time; a recording stamps it
    when the frame is read. A reader that falls behind loses the captures
    it did not read, and a source that counts them does so in
    ``dropped_captures``.

    ``calibration()`` gives the stereo calibration of a camera that stores
    one, and a camera whose pair is not what one wants to look at also
    fills ``picture``; ``describe_stereo_camera`` says which cameras do
    either. ``imu=True`` asks a camera with an IMU to read it, for
    ``imu()``. Raises ValueError for an entry that is not a stereo camera.
    """
    kind, name, arguments = _stereo_arguments(spec)
    if kind == "zed":
        return ZedStereoSource.build(name=name, imu=imu, **arguments)
    if kind == "oakd":
        return OakdStereoSource(name=name, imu=imu, **arguments)
    # video: a side-by-side file already emits both eyes in one frame.
    return VideoFileSource(name=name, stereo=True, hold=True, **arguments)
