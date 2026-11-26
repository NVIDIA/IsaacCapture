# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared RGBA → NV12 CuPy kernel.

Used by both NVENC paths (native + GStreamer). BT.601 full-range matrix.
"""

from __future__ import annotations

_RGBA_NV12_KERNEL_SRC = r"""
extern "C" __global__ void rgba_to_nv12(
    const unsigned char* __restrict__ rgba,
    int rgba_row_bytes,
    int width,
    int height,
    unsigned char* __restrict__ y_plane,
    int y_pitch,
    unsigned char* __restrict__ uv_plane,
    int uv_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int rgba_idx = y * rgba_row_bytes + x * 4;
    float R = (float)rgba[rgba_idx + 0];
    float G = (float)rgba[rgba_idx + 1];
    float B = (float)rgba[rgba_idx + 2];

    // BT.601 full-range Y. Identity with ITU-T T.871 § 7.
    float Yf = 0.299f * R + 0.587f * G + 0.114f * B;
    int Y = (int)(Yf + 0.5f);
    if (Y < 0) Y = 0; else if (Y > 255) Y = 255;
    y_plane[y * y_pitch + x] = (unsigned char)Y;

    // UV: one sample per 2x2 block, taken at the top-left corner.
    if ((x & 1) == 0 && (y & 1) == 0) {
        float Cbf = -0.168736f * R - 0.331264f * G + 0.500000f * B + 128.f;
        float Crf =  0.500000f * R - 0.418688f * G - 0.081312f * B + 128.f;
        int Cb = (int)(Cbf + 0.5f);
        int Cr = (int)(Crf + 0.5f);
        if (Cb < 0) Cb = 0; else if (Cb > 255) Cb = 255;
        if (Cr < 0) Cr = 0; else if (Cr > 255) Cr = 255;
        int uv_idx = (y >> 1) * uv_pitch + x;
        uv_plane[uv_idx + 0] = (unsigned char)Cb;
        uv_plane[uv_idx + 1] = (unsigned char)Cr;
    }
}
"""


def build_rgba_to_nv12_kernel():
    """Lazy-compile the RGBA → NV12 RawKernel. Caller invokes as
    ``kernel(grid, block, args)``."""
    import cupy as cp

    return cp.RawKernel(_RGBA_NV12_KERNEL_SRC, "rgba_to_nv12")


def launch_rgba_to_nv12(
    kernel, rgba, nv12_y, nv12_uv, *, stream, width, height
) -> None:
    """Run the kernel inside ``stream``'s context. ``nv12_y`` and ``nv12_uv``
    are the two planes of the destination NV12 surface (Y is HxW, UV is
    (H/2)xW, each contiguous within itself). ``rgba`` is HxWx4 uint8."""
    import cupy as cp

    block = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16, 1)
    rgba_row_bytes = rgba.strides[0] if hasattr(rgba, "strides") else width * 4
    with stream:
        kernel(
            grid,
            block,
            (
                rgba,
                cp.int32(rgba_row_bytes),
                cp.int32(width),
                cp.int32(height),
                nv12_y,
                cp.int32(width),
                nv12_uv,
                cp.int32(width),
            ),
        )
