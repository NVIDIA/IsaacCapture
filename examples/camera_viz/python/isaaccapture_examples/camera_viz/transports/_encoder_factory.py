# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encoder backend selection for camera_streamer.

Two H.264 encoders share the same ``encode(rgba) -> List[bytes]`` API:

  * :class:`NvH264Encoder` — desktop, native NVENC at
    ``examples/camera_viz/codec/``. Zero-copy from CuPy into NVENC.
  * :class:`GstNvH264Encoder` — Jetson / portable, GStreamer's
    ``nvv4l2h264enc`` (or ``nvh264enc`` / ``x264enc`` fallbacks).
    ~1-3 ms D2H download at 720p.

``"auto"`` tries the native codec first; falls back to GStreamer.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("isaaccapture.camera_viz.transports._encoder_factory")


def _has_native_codec() -> bool:
    # Both ImportError (no package) and RuntimeError (raised by
    # codec/__init__.py on missing .so) fall through to GStreamer.
    try:
        from .. import codec  # noqa: F401

        return True
    except (ImportError, RuntimeError):
        return False


def make_encoder(
    backend: str,
    *,
    width: int,
    height: int,
    bitrate: int,
    fps: int,
    gop: int,
    gpu_id: int,
):
    """Build a sender-side H.264 encoder.

    ``backend`` is ``"auto"``, ``"native"``, or ``"gstreamer"``. Returns
    an object with ``encode(rgba_cupy_array) -> List[bytes]`` +
    ``end_of_stream() -> List[bytes]`` + ``reset() -> None``.
    """
    chosen = backend.lower() if isinstance(backend, str) else "auto"
    if chosen == "auto":
        chosen = "native" if _has_native_codec() else "gstreamer"
        logger.info("encoder backend (auto): %s", chosen)

    if chosen == "native":
        from ._nv_encode import NvH264Encoder

        return NvH264Encoder(
            width=width, height=height, bitrate=bitrate, fps=fps, gop=gop, gpu_id=gpu_id
        )
    if chosen == "gstreamer":
        from ._nv_encode_gst import GstNvH264Encoder

        return GstNvH264Encoder(
            width=width, height=height, bitrate=bitrate, fps=fps, gop=gop, gpu_id=gpu_id
        )
    raise ValueError(
        f"unknown encoder backend {backend!r} (known: auto | native | gstreamer)"
    )
