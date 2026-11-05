// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rgba_box_downscale.cuh"

namespace plugins
{
namespace sensing
{

namespace
{

__global__ void box_downscale(
    const uint8_t* src, int src_pitch, uint8_t* dst, int dst_pitch, int width, int height, int divisor)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int sum[4] = { 0, 0, 0, 0 };
    for (int j = 0; j < divisor; ++j)
    {
        const uint8_t* row =
            src + static_cast<size_t>(y * divisor + j) * src_pitch + static_cast<size_t>(x * divisor) * 4;
        for (int i = 0; i < divisor; ++i)
        {
            for (int c = 0; c < 4; ++c)
                sum[c] += row[i * 4 + c];
        }
    }

    const int count = divisor * divisor;
    uint8_t* out = dst + static_cast<size_t>(y) * dst_pitch + static_cast<size_t>(x) * 4;
    for (int c = 0; c < 4; ++c)
        out[c] = static_cast<uint8_t>((sum[c] + count / 2) / count);
}

} // namespace

void launch_rgba_box_downscale(const uint8_t* src,
                               int src_pitch,
                               uint8_t* dst,
                               int dst_pitch,
                               int dst_width,
                               int dst_height,
                               int divisor,
                               cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);
    box_downscale<<<grid, block, 0, stream>>>(src, src_pitch, dst, dst_pitch, dst_width, dst_height, divisor);
}

} // namespace sensing
} // namespace plugins
