// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// NV12 → RGBA8 colour conversion in a single kernel launch.
// BT.709 limited-range (H.264 default) or BT.601 full-range (T.871).
// Alpha is always 255.

#include "nv12_to_rgba.cuh"

#include <cstdint>

namespace
{

__global__ void nv12_to_rgba_kernel(const unsigned char* __restrict__ y_plane,
                                    const unsigned char* __restrict__ uv_plane,
                                    int y_pitch,
                                    int uv_pitch,
                                    int width,
                                    int height,
                                    unsigned char* __restrict__ rgba_out,
                                    int rgba_row_bytes,
                                    int full_range)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    const int Y = y_plane[y * y_pitch + x];
    // UV is half-resolution, interleaved Cb,Cr per 2-pixel block.
    const int uv_x = (x & ~1);
    const int uv_y = y >> 1;
    const int Cb = uv_plane[uv_y * uv_pitch + uv_x + 0];
    const int Cr = uv_plane[uv_y * uv_pitch + uv_x + 1];

    float R, G, B;
    if (full_range)
    {
        // BT.601 full-range (ITU-T T.871).
        const float yf = (float)Y;
        const float u = (float)Cb - 128.f;
        const float v = (float)Cr - 128.f;
        R = yf + 1.402f * v;
        G = yf - 0.344136f * u - 0.714136f * v;
        B = yf + 1.772f * u;
    }
    else
    {
        // BT.709 limited-range (luma 16-235).
        const float yf = ((float)Y - 16.f) * 1.16438f;
        const float u = (float)Cb - 128.f;
        const float v = (float)Cr - 128.f;
        R = yf + 1.79274f * v;
        G = yf - 0.21325f * u - 0.53291f * v;
        B = yf + 2.11240f * u;
    }

    const int idx = y * rgba_row_bytes + x * 4;
    rgba_out[idx + 0] = (unsigned char)(R < 0.f ? 0 : (R > 255.f ? 255 : R));
    rgba_out[idx + 1] = (unsigned char)(G < 0.f ? 0 : (G > 255.f ? 255 : G));
    rgba_out[idx + 2] = (unsigned char)(B < 0.f ? 0 : (B > 255.f ? 255 : B));
    rgba_out[idx + 3] = 255;
}

} // namespace

namespace camera_viz::codec
{

void launch_nv12_to_rgba(const uint8_t* y_plane,
                         const uint8_t* uv_plane,
                         int y_pitch,
                         int uv_pitch,
                         int width,
                         int height,
                         uint8_t* rgba_out,
                         int rgba_row_bytes,
                         bool full_range,
                         cudaStream_t stream)
{
    const dim3 block(16, 16, 1);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
    nv12_to_rgba_kernel<<<grid, block, 0, stream>>>(
        y_plane, uv_plane, y_pitch, uv_pitch, width, height, rgba_out, rgba_row_bytes, full_range ? 1 : 0);
}

} // namespace camera_viz::codec
