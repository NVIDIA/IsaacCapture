// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mock_nv12_source.cuh"

namespace plugins
{
namespace sensing
{

namespace
{

constexpr int kCounterCell = 16;
constexpr int kCounterBits = 16;

__global__ void paint_luma(uint8_t* y_plane, int y_pitch, int width, int height, uint32_t frame_index)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const float u = static_cast<float>(x) / width;
    const float v = static_cast<float>(y) / height;
    const float t = static_cast<float>(frame_index % 240u) / 240.0f;

    // Diagonal gradient scrolling with the frame index.
    float luma = 255.0f * (0.25f + 0.5f * fabsf(__sinf(6.2831f * (u + v + t))));

    // A bright square orbiting the centre: a repeated frame freezes it.
    const float cx = 0.5f + 0.3f * __cosf(6.2831f * t);
    const float cy = 0.5f + 0.3f * __sinf(6.2831f * t);
    if (fabsf(u - cx) < 0.04f && fabsf(v - cy) < 0.04f * width / height)
        luma = 255.0f;

    if (y < kCounterCell && x < kCounterBits * kCounterCell)
    {
        const unsigned bit = static_cast<unsigned>(x / kCounterCell);
        luma = ((frame_index >> (kCounterBits - 1u - bit)) & 1u) ? 255.0f : 0.0f;
    }

    y_plane[static_cast<size_t>(y) * y_pitch + x] = static_cast<uint8_t>(luma);
}

__global__ void paint_chroma(uint8_t* uv_plane, int uv_pitch, int width, int height, uint32_t sensor_id)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // chroma sample column
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width / 2 || y >= height / 2)
        return;

    // Even sensors lean blue-green, odd sensors red-orange. The counter strip stays neutral so its
    // cells decode as pure black and white.
    const bool odd = (sensor_id & 1u) != 0;
    const bool counter = y < kCounterCell / 2 && x < kCounterBits * kCounterCell / 2;
    uint8_t* sample = uv_plane + static_cast<size_t>(y) * uv_pitch + static_cast<size_t>(x) * 2;
    sample[0] = counter ? 128 : (odd ? 100 : 160); // U
    sample[1] = counter ? 128 : (odd ? 180 : 100); // V
}

} // namespace

void launch_mock_nv12_pattern(uint8_t* y_plane,
                              int y_pitch,
                              uint8_t* uv_plane,
                              int uv_pitch,
                              int width,
                              int height,
                              uint32_t sensor_id,
                              uint32_t frame_index,
                              cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 luma_grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    paint_luma<<<luma_grid, block, 0, stream>>>(y_plane, y_pitch, width, height, frame_index);

    const dim3 chroma_grid((width / 2 + block.x - 1) / block.x, (height / 2 + block.y - 1) / block.y);
    paint_chroma<<<chroma_grid, block, 0, stream>>>(uv_plane, uv_pitch, width, height, sensor_id);
}

} // namespace sensing
} // namespace plugins
