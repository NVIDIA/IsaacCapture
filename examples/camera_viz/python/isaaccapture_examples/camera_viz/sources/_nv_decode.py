# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVDEC H.264 decoder — wrapper around the native ``codec`` module."""

from __future__ import annotations

import logging

logger = logging.getLogger("isaaccapture.camera_viz.sources._nv_decode")


class NvH264Decoder:
    """Annex-B H.264 packet → RGBA8 GPU buffer.

    Decoder is created lazily on the first packet. Resolution is fixed
    at construction; streams that don't match drop frames with a warning.
    """

    def __init__(
        self,
        width: int,
        height: int,
        full_range: bool = False,
        gpu_id: int = 0,
        low_latency: bool = True,
    ) -> None:
        del low_latency  # native codec is always zero-latency

        self._width = width
        self._height = height
        self._full_range = full_range
        self._gpu_id = gpu_id
        self._decoder = None

    def _ensure_initialized(self) -> None:
        if self._decoder is not None:
            return
        try:
            from .. import codec
        except ImportError as e:
            raise RuntimeError(
                "NvH264Decoder requires the native codec. Run "
                "`examples/camera_viz/codec/build.sh`."
            ) from e

        cfg = codec.DecoderConfig()
        cfg.width = self._width
        cfg.height = self._height
        cfg.full_range = self._full_range
        cfg.gpu_id = self._gpu_id
        self._decoder = codec.H264Decoder(cfg)

    def decode(self, packet: bytes, rgba_out) -> bool:
        """Feed one Annex-B AU. Returns True iff a frame was written to ``rgba_out``."""
        self._ensure_initialized()
        return self._decoder.decode(packet, rgba_out, self._width, self._height)

    def reset(self) -> None:
        """Tear down NVDEC state. Use after stream-timeout / disconnect."""
        if self._decoder is not None:
            self._decoder.reset()
