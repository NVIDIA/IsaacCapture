# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVENC H.264 encoder — wrapper around the native ``codec`` module."""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger("isaaccapture.camera_viz.transports._nv_encode")


class NvH264Encoder:
    """RGBA8 GPU buffer → Annex-B H.264 packets.

    Encoder is built lazily on the first ``encode()`` so a never-firing
    sender doesn't hold an NVENC session.
    """

    def __init__(
        self,
        width: int,
        height: int,
        bitrate: int = 15_000_000,
        fps: int = 30,
        gop: Optional[int] = None,
        gpu_id: int = 0,
    ) -> None:
        if width % 2 != 0 or height % 2 != 0:
            raise ValueError(
                f"NvH264Encoder: width and height must be even for NV12, "
                f"got {width}x{height}."
            )

        self._width = width
        self._height = height
        self._bitrate = bitrate
        self._fps = fps
        self._gop = gop or 0  # 0 → fps*5
        self._gpu_id = gpu_id
        self._encoder = None

    def _ensure_initialized(self, rgba_device_id: int) -> None:
        if self._encoder is not None:
            return

        try:
            from .. import codec
        except ImportError as e:
            raise RuntimeError(
                "NvH264Encoder requires the native codec. Run "
                "`examples/camera_viz/codec/build.sh`."
            ) from e

        # Follow the input's GPU when the caller didn't pin one.
        target_gpu_id = self._gpu_id if self._gpu_id != 0 else rgba_device_id
        self._gpu_id = target_gpu_id

        cfg = codec.EncoderConfig()
        cfg.width = self._width
        cfg.height = self._height
        cfg.bitrate_bps = self._bitrate
        cfg.fps = self._fps
        cfg.gop = self._gop
        cfg.gpu_id = target_gpu_id
        cfg.pixel_format = codec.PixelFormat.RGBA8
        self._encoder = codec.H264Encoder(cfg)

    def encode(self, rgba) -> List[bytes]:
        """Encode one RGBA8 frame. Returns 0 or 1 Annex-B packet."""
        rgba_device_id = int(rgba.device.id)
        self._ensure_initialized(rgba_device_id)
        if rgba_device_id != self._gpu_id:
            raise RuntimeError(
                f"NvH264Encoder: input on GPU {rgba_device_id}, "
                f"encoder built for GPU {self._gpu_id}"
            )

        packet = self._encoder.encode(rgba)
        return [packet] if packet else []

    def end_of_stream(self) -> List[bytes]:
        if self._encoder is None:
            return []
        flushed = self._encoder.end_of_stream()
        return [flushed] if flushed else []

    def reset(self) -> None:
        self._encoder = None
