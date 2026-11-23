// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/gpu_ops/box_downscale.hpp"

namespace camera_bridge
{

namespace
{

constexpr uint32_t kMaxDivisor = 8;

__global__ void downscale_luma(const uint8_t* src,
                               size_t src_pitch,
                               uint8_t* dst,
                               size_t dst_pitch,
                               uint32_t dst_width,
                               uint32_t dst_height,
                               uint32_t divisor)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_width || y >= dst_height)
        return;
    uint32_t sum = 0;
    for (uint32_t j = 0; j < divisor; ++j)
    {
        const uint8_t* row = src + static_cast<size_t>(y * divisor + j) * src_pitch + x * divisor;
        for (uint32_t i = 0; i < divisor; ++i)
            sum += row[i];
    }
    dst[static_cast<size_t>(y) * dst_pitch + x] =
        static_cast<uint8_t>((sum + divisor * divisor / 2) / (divisor * divisor));
}

// One thread per destination chroma sample pair (U and V), averaging the same block of source pairs.
__global__ void downscale_chroma(const uint8_t* src,
                                 size_t src_pitch,
                                 uint8_t* dst,
                                 size_t dst_pitch,
                                 uint32_t dst_pairs_x,
                                 uint32_t dst_rows,
                                 uint32_t divisor)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_pairs_x || y >= dst_rows)
        return;
    uint32_t sum_u = 0;
    uint32_t sum_v = 0;
    for (uint32_t j = 0; j < divisor; ++j)
    {
        const uint8_t* row =
            src + static_cast<size_t>(y * divisor + j) * src_pitch + static_cast<size_t>(x) * divisor * 2;
        for (uint32_t i = 0; i < divisor; ++i)
        {
            sum_u += row[2 * i];
            sum_v += row[2 * i + 1];
        }
    }
    const uint32_t count = divisor * divisor;
    uint8_t* out = dst + static_cast<size_t>(y) * dst_pitch + static_cast<size_t>(x) * 2;
    out[0] = static_cast<uint8_t>((sum_u + count / 2) / count);
    out[1] = static_cast<uint8_t>((sum_v + count / 2) / count);
}

} // namespace

bool box_downscale_supported(uint32_t width, uint32_t height, uint32_t divisor)
{
    if (divisor == 0 || divisor > kMaxDivisor)
        return false;
    if (width % divisor != 0 || height % divisor != 0)
        return false;
    return (width / divisor) % 2 == 0 && (height / divisor) % 2 == 0 && width / divisor > 0 && height / divisor > 0;
}

cudaError_t box_downscale_nv12(const Nv12Planes& source,
                               uint32_t width,
                               uint32_t height,
                               const Nv12PlanesOut& destination,
                               uint32_t divisor,
                               cudaStream_t stream)
{
    if (!box_downscale_supported(width, height, divisor))
        return cudaErrorInvalidValue;
    const uint32_t dst_width = width / divisor;
    const uint32_t dst_height = height / divisor;
    const dim3 block(32, 8);
    const dim3 luma_grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);
    downscale_luma<<<luma_grid, block, 0, stream>>>(
        source.y, source.y_pitch, destination.y, destination.y_pitch, dst_width, dst_height, divisor);
    const uint32_t pairs = dst_width / 2;
    const uint32_t rows = dst_height / 2;
    const dim3 chroma_grid((pairs + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    downscale_chroma<<<chroma_grid, block, 0, stream>>>(
        source.uv, source.uv_pitch, destination.uv, destination.uv_pitch, pairs, rows, divisor);
    return cudaGetLastError();
}

} // namespace camera_bridge
