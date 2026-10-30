# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVDEC video decoder wrapper for camera_receiver."""

from __future__ import annotations

import logging

logger = logging.getLogger("isaaccapture.camera_streamer." + __name__)


class NvVideoDecoder:
    """Annex-B H.264 / HEVC access-unit decoder to GPU buffers."""

    def __init__(
        self,
        width: int,
        height: int,
        full_range: bool = False,
        gpu_id: int = 0,
        low_latency: bool = True,
        codec: str = "h264",
    ) -> None:
        del low_latency

        codec_norm = str(codec).lower()
        if codec_norm in ("h265", "hevc"):
            codec_norm = "hevc"
        elif codec_norm in ("h264", "avc"):
            codec_norm = "h264"
        else:
            raise ValueError(f"NvVideoDecoder: codec must be h264|hevc, got {codec!r}")

        self._width = width
        self._height = height
        self._full_range = full_range
        self._gpu_id = gpu_id
        self._codec = codec_norm
        self._decoder = None

    def _ensure_initialized(self) -> None:
        if self._decoder is not None:
            return
        try:
            import codec
        except ImportError as e:
            raise RuntimeError(
                "NvVideoDecoder requires the native codec. Follow "
                "the host receiver build instructions in examples/camera_streamer/README.md."
            ) from e

        cfg = codec.DecoderConfig()
        cfg.width = self._width
        cfg.height = self._height
        cfg.full_range = self._full_range
        cfg.gpu_id = self._gpu_id
        try:
            cfg.codec = codec.Codec.HEVC if self._codec == "hevc" else codec.Codec.H264
        except AttributeError as e:
            raise RuntimeError(
                "The native codec module is stale (no Codec enum). Rebuild it "
                "using the host receiver build instructions in examples/camera_streamer/README.md."
            ) from e
        self._decoder = codec.NvVideoDecoder(cfg)

    def decode(self, packet: bytes, rgba_out, rotation: int = 0) -> bool:
        """Feed one Annex-B AU. Returns True iff a frame was written."""
        self._ensure_initialized()
        return self._decoder.decode(
            packet, rgba_out, self._width, self._height, rotation
        )

    def decode_nv12(self, packet: bytes, nv12_out, rotation: int = 0) -> bool:
        """Feed one Annex-B AU into a tight NV12 GPU buffer.

        ``nv12_out`` is expected to be a ``uint8`` array shaped
        ``((height * 3) // 2, width)``. Returns True iff a frame was written.
        """
        self._ensure_initialized()
        return self._decoder.decode_nv12(
            packet, nv12_out, self._width, self._height, rotation
        )

    def reset(self) -> None:
        """Tear down NVDEC state. Use after stream timeout or disconnect."""
        if self._decoder is not None:
            self._decoder.reset()
