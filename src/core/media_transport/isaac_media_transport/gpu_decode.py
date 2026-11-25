# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional NVDEC → owned CUDA RGBA frames, with no decoded CPU staging."""

import time

from .transport import FrameMetadata, ReceivedFrame, VideoFrame
from .latency import get_recorder


class CudaDecoder:
    codec = "NVDEC/CUDA (PyNvVideoCodec)"

    def __init__(self, settings):
        self.timing = get_recorder(settings)
        self.track = settings.get("latency_track", "")
        try:
            import cupy as cp
            import PyNvVideoCodec as nvc
        except ImportError as exc:
            raise RuntimeError(
                "Install the gpu-receiver extra in the receiver/viewer environment"
            ) from exc
        self.cp, self.nvc = cp, nvc
        self.device = int(settings.get("decode_gpu", 0))
        self.context = None
        self.reset()

    def reset(self):
        self.context = None

    def decode(self, payload, stamp, index):
        self.timing.record("decode_begin", self.track, stamp * 1000)
        import numpy as np

        cp, nvc = self.cp, self.nvc
        with cp.cuda.Device(self.device):
            if self.context is None:
                self.stream = cp.cuda.Stream(non_blocking=True)
                self.context = nvc.CreateDecoder(
                    gpuid=self.device,
                    codec=nvc.cudaVideoCodec.H264,
                    usedevicememory=True,
                    cudastream=self.stream.ptr,
                    outputColorType=nvc.OutputColorType.RGB,
                    latency=nvc.DisplayDecodeLatencyType.ZERO,
                )
            packet = nvc.PacketData()
            encoded = np.frombuffer(payload, dtype=np.uint8)
            packet.bsl, packet.bsl_data = encoded.size, encoded.ctypes.data
            packet.pts = packet.dts = stamp
            packet.decode_flag = 2 | 8  # CUVID_PKT_TIMESTAMP | CUVID_PKT_ENDOFPICTURE
            result = []
            with self.stream:
                frames = self.context.Decode(packet)
                # ZERO in PyNvVideoCodec 2.0.2 omits PTS. Our no-B-frame,
                # one-access-unit packets must produce exactly one current image.
                # LOW buffers nine frames and bursts at each IDR on S56 2K streams.
                if len(frames) != 1:
                    raise RuntimeError(
                        "Zero-latency CUDA decoding requires one complete H.264 "
                        "picture per packet and no B-frames"
                    )
                for decoded in frames:
                    rgb = cp.from_dlpack(decoded)
                    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != cp.uint8:
                        raise ValueError("Expected NVDEC RGB8 output")
                    height, width = rgb.shape[:2]
                    rgba = cp.empty((height, width, 4), dtype=cp.uint8)
                    rgba[..., :3] = rgb
                    rgba[..., 3] = 255
                    # Complete before returning decoder surfaces or sharing across threads.
                    self.stream.synchronize()
                    received_ns = time.monotonic_ns()
                    self.timing.record(
                        "decode_ready", self.track, stamp * 1000, at_ns=received_ns
                    )
                    result.append(
                        ReceivedFrame(
                            VideoFrame(width, height, rgba, "cuda", self.device),
                            received_ns,
                            FrameMetadata(index, stamp * 1000),
                        )
                    )
            return result
