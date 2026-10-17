// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rgba_to_nv12.cuh"

namespace
{

__device__ __forceinline__ uint8_t clamp_u8(float v)
{
    return static_cast<uint8_t>(fminf(fmaxf(v, 0.0f), 255.0f) + 0.5f);
}

// BT.601 RGB->YCbCr, matching the coefficients yuv_to_rgba.cu inverts on the
// capture side so a round trip through RGBA is close to identity.
__device__ __forceinline__ float rgb_to_y(float r, float g, float b, bool full_range)
{
    return full_range ? (0.299f * r + 0.587f * g + 0.114f * b) : (16.0f + 0.257f * r + 0.504f * g + 0.098f * b);
}

__device__ __forceinline__ float rgb_to_u(float r, float g, float b, bool full_range)
{
    return full_range ? (128.0f - 0.168736f * r - 0.331264f * g + 0.5f * b) :
                        (128.0f - 0.148f * r - 0.291f * g + 0.439f * b);
}

__device__ __forceinline__ float rgb_to_v(float r, float g, float b, bool full_range)
{
    return full_range ? (128.0f + 0.5f * r - 0.418688f * g - 0.081312f * b) :
                        (128.0f + 0.439f * r - 0.368f * g - 0.071f * b);
}

__global__ void rgba_to_nv12_kernel(const uint8_t* __restrict__ rgba,
                                    int rgba_pitch,
                                    uint8_t* __restrict__ y_plane,
                                    int y_pitch,
                                    uint8_t* __restrict__ uv_plane,
                                    int uv_pitch,
                                    int width,
                                    int height,
                                    bool full_range)
{
    // One thread per 2x2 block: writes four luma samples and one chroma pair.
    const int bx = blockIdx.x * blockDim.x + threadIdx.x;
    const int by = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = bx * 2;
    const int y = by * 2;
    if (x >= width || y >= height)
        return;

    for (int dy = 0; dy < 2; ++dy)
    {
        for (int dx = 0; dx < 2; ++dx)
        {
            const int px = x + dx;
            const int py = y + dy;
            if (px >= width || py >= height)
                continue;

            const uint8_t* pixel = rgba + static_cast<size_t>(py) * rgba_pitch + static_cast<size_t>(px) * 4;
            const float r = pixel[0];
            const float g = pixel[1];
            const float b = pixel[2];
            y_plane[static_cast<size_t>(py) * y_pitch + px] = clamp_u8(rgb_to_y(r, g, b, full_range));
        }
    }

    const uint8_t* top_left = rgba + static_cast<size_t>(y) * rgba_pitch + static_cast<size_t>(x) * 4;
    const float r = top_left[0];
    const float g = top_left[1];
    const float b = top_left[2];

    uint8_t* uv = uv_plane + static_cast<size_t>(by) * uv_pitch + static_cast<size_t>(bx) * 2;
    uv[0] = clamp_u8(rgb_to_u(r, g, b, full_range));
    uv[1] = clamp_u8(rgb_to_v(r, g, b, full_range));
}

} // namespace

namespace plugins
{
namespace sensing
{

void launch_rgba_to_nv12(const uint8_t* rgba,
                         int rgba_pitch,
                         uint8_t* y_plane,
                         int y_pitch,
                         uint8_t* uv_plane,
                         int uv_pitch,
                         int width,
                         int height,
                         bool full_range,
                         cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid((width / 2 + block.x - 1) / block.x, (height / 2 + block.y - 1) / block.y);
    rgba_to_nv12_kernel<<<grid, block, 0, stream>>>(
        rgba, rgba_pitch, y_plane, y_pitch, uv_plane, uv_pitch, width, height, full_range);
}

} // namespace sensing
} // namespace plugins
