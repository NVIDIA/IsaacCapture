// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/gpu_ops/sine_warp.hpp"

#include <cmath>

namespace camera_bridge
{

namespace
{

constexpr float kTwoPi = 6.28318530717958647692F;
constexpr unsigned kBlockX = 32;
constexpr unsigned kBlockY = 8;

struct WarpArgs
{
    float amplitude;
    float cycles_per_row;
    float phase;
};

__device__ __forceinline__ float displaced_x(float x, float y, const WarpArgs& a)
{
    return x + a.amplitude * sinf(kTwoPi * y * a.cycles_per_row + a.phase);
}

// Bilinear sample of one byte channel; `stride` is the byte distance between neighbouring samples of the channel
// (1 for luma, 2 for interleaved chroma). The caller has checked that sx lies inside [0, w-1]; sy is a row.
__device__ __forceinline__ float bilinear(const uint8_t* plane, size_t pitch, int stride, int w, int h, float sx, float sy)
{
    const int x0 = static_cast<int>(floorf(sx));
    const int y0 = static_cast<int>(floorf(sy));
    const int x1 = min(x0 + 1, w - 1);
    const int y1 = min(y0 + 1, h - 1);
    const float fx = sx - static_cast<float>(x0);
    const float fy = sy - static_cast<float>(y0);
    const uint8_t* r0 = plane + static_cast<size_t>(y0) * pitch;
    const uint8_t* r1 = plane + static_cast<size_t>(y1) * pitch;
    const float top = (1.0F - fx) * r0[x0 * stride] + fx * r0[x1 * stride];
    const float bottom = (1.0F - fx) * r1[x0 * stride] + fx * r1[x1 * stride];
    return (1.0F - fy) * top + fy * bottom;
}

__device__ __forceinline__ uint8_t to_byte(float v)
{
    return static_cast<uint8_t>(fminf(fmaxf(v + 0.5F, 0.0F), 255.0F));
}

__global__ void warp_luma(const uint8_t* in, size_t in_pitch, uint8_t* out, size_t out_pitch, int w, int h, WarpArgs args)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const float sx = displaced_x(static_cast<float>(x), static_cast<float>(y), args);
    // A positive predicate: a NaN coordinate takes the fill path instead of reaching the sampler.
    const bool inside = sx >= 0.0F && sx <= static_cast<float>(w - 1);
    out[static_cast<size_t>(y) * out_pitch + x] =
        inside ? to_byte(bilinear(in, in_pitch, 1, w, h, sx, static_cast<float>(y))) : 16;
}

// Chroma sample (cx, cy) sits at luma position (2*cx, 2*cy + 0.5); the mapping is applied there and converted back
// to chroma-plane coordinates.
__global__ void warp_chroma(const uint8_t* in, size_t in_pitch, uint8_t* out, size_t out_pitch, int cw, int ch, WarpArgs args)
{
    const int cx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= cw || cy >= ch)
        return;
    const float luma_y = 2.0F * static_cast<float>(cy) + 0.5F;
    const float scx = 0.5F * displaced_x(2.0F * static_cast<float>(cx), luma_y, args);
    const float scy = static_cast<float>(cy);
    uint8_t* dst = out + static_cast<size_t>(cy) * out_pitch + 2 * cx;
    if (!(scx >= 0.0F && scx <= static_cast<float>(cw - 1)))
    {
        dst[0] = 128;
        dst[1] = 128;
        return;
    }
    dst[0] = to_byte(bilinear(in, in_pitch, 2, cw, ch, scx, scy));
    dst[1] = to_byte(bilinear(in + 1, in_pitch, 2, cw, ch, scx, scy));
}

} // namespace

bool sine_warp_supported(uint32_t width, uint32_t height, const SineWarpParams& params)
{
    const bool finite =
        std::isfinite(params.amplitude_px) && std::isfinite(params.wavelength_px) && std::isfinite(params.phase_rad);
    return finite && params.wavelength_px > 0.0F && std::isfinite(1.0F / params.wavelength_px) && width != 0 &&
           height != 0 && width % 2 == 0 && height % 2 == 0;
}

cudaError_t sine_warp_nv12(const Nv12Planes& source,
                           uint32_t width,
                           uint32_t height,
                           const Nv12PlanesOut& destination,
                           const SineWarpParams& params,
                           cudaStream_t stream)
{
    if (!sine_warp_supported(width, height, params))
        return cudaErrorInvalidValue;
    const int w = static_cast<int>(width);
    const int h = static_cast<int>(height);
    const WarpArgs args{ params.amplitude_px, 1.0F / params.wavelength_px, params.phase_rad };
    const dim3 block(kBlockX, kBlockY);
    warp_luma<<<dim3((w + kBlockX - 1) / kBlockX, (h + kBlockY - 1) / kBlockY), block, 0, stream>>>(
        source.y, source.y_pitch, destination.y, destination.y_pitch, w, h, args);
    if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess)
        return status;
    const int cw = w / 2;
    const int ch = h / 2;
    warp_chroma<<<dim3((cw + kBlockX - 1) / kBlockX, (ch + kBlockY - 1) / kBlockY), block, 0, stream>>>(
        source.uv, source.uv_pitch, destination.uv, destination.uv_pitch, cw, ch, args);
    return cudaGetLastError();
}

} // namespace camera_bridge
