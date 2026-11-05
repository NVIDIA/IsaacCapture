// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/video_stage/sine_warp.hpp"

#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace video_stage
{

namespace
{

constexpr float kTwoPi = 6.28318530717958647692F;
constexpr unsigned kBlockX = 32;
constexpr unsigned kBlockY = 8;

struct WarpArgs
{
    float amplitude;
    float cycles_per_row; // 1 / wavelength
    float phase;
};

__device__ __forceinline__ float displaced_x(float x, float y, const WarpArgs& a)
{
    return x + a.amplitude * sinf(kTwoPi * y * a.cycles_per_row + a.phase);
}

// Bilinear sample of one byte channel. `stride` is the byte distance between neighbouring samples of the channel
// (1 for luma, 2 for interleaved chroma). The caller has checked that (sx, sy) lies inside [0,w-1]x[0,h-1].
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

__global__ void warp_luma(
    const uint8_t* in, size_t in_pitch, uint8_t* out, size_t out_pitch, int w, int h, WarpArgs args, uint8_t fill)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    const float sx = displaced_x(static_cast<float>(x), static_cast<float>(y), args);
    const float sy = static_cast<float>(y);
    const bool inside = sx >= 0.0F && sx <= static_cast<float>(w - 1);
    out[static_cast<size_t>(y) * out_pitch + x] = inside ? to_byte(bilinear(in, in_pitch, 1, w, h, sx, sy)) : fill;
}

// Chroma sample (cx, cy) sits at luma position (2*cx, 2*cy + 0.5): co-sited with even luma columns, half way between
// luma rows. The mapping is applied there and the result is converted back to chroma-plane coordinates.
__global__ void warp_chroma(const uint8_t* in, size_t in_pitch, uint8_t* out, size_t out_pitch, int cw, int ch, WarpArgs args)
{
    const int cx = blockIdx.x * blockDim.x + threadIdx.x;
    const int cy = blockIdx.y * blockDim.y + threadIdx.y;
    if (cx >= cw || cy >= ch)
        return;
    const float luma_x = 2.0F * static_cast<float>(cx);
    const float luma_y = 2.0F * static_cast<float>(cy) + 0.5F;
    const float src_luma_x = displaced_x(luma_x, luma_y, args);
    const float scx = 0.5F * src_luma_x;
    const float scy = 0.5F * (luma_y - 0.5F);
    uint8_t* dst = out + static_cast<size_t>(cy) * out_pitch + 2 * cx;
    // A positive predicate: a NaN coordinate fails it and takes the fill path instead of reaching the sampler.
    if (!(scx >= 0.0F && scx <= static_cast<float>(cw - 1)))
    {
        dst[0] = 128;
        dst[1] = 128;
        return;
    }
    dst[0] = to_byte(bilinear(in, in_pitch, 2, cw, ch, scx, scy));
    dst[1] = to_byte(bilinear(in + 1, in_pitch, 2, cw, ch, scx, scy));
}

void check(cudaError_t status, const char* what)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("SineWarp: ") + what + ": " + cudaGetErrorString(status));
}

} // namespace

SineWarp::SineWarp(const SineWarpParams& params) : m_params(params)
{
    const bool finite =
        std::isfinite(params.amplitude_px) && std::isfinite(params.wavelength_px) && std::isfinite(params.phase_rad);
    if (!finite || !(params.wavelength_px > 0.0F) || !std::isfinite(1.0F / params.wavelength_px))
        throw std::invalid_argument(
            "SineWarp: amplitude, phase and wavelength must be finite and the wavelength positive");
}

Geometry SineWarp::output_geometry(const Geometry& input) const
{
    return input;
}

void SineWarp::prepare(const Geometry& input)
{
    if (input.width == 0 || input.height == 0 || (input.width % 2) != 0 || (input.height % 2) != 0)
        throw std::invalid_argument("SineWarp: NV12 needs non-zero even dimensions");
    m_prepared = input;
}

void SineWarp::enqueue(const Nv12View& in, const Nv12View& out, cudaStream_t stream)
{
    if (m_prepared.width == 0 || !(in.geometry == m_prepared))
        throw std::logic_error("SineWarp: not prepared for this input geometry");
    if (!(out.geometry == output_geometry(in.geometry)))
        throw std::invalid_argument("SineWarp: output geometry does not match");
    const auto bad = [](const Nv12View& v)
    { return v.y == nullptr || v.uv == nullptr || v.y_pitch < v.geometry.width || v.uv_pitch < v.geometry.width; };
    if (bad(in) || bad(out))
        throw std::invalid_argument("SineWarp: surface has a null plane or a pitch below the row size");

    const int w = static_cast<int>(in.geometry.width);
    const int h = static_cast<int>(in.geometry.height);
    const WarpArgs args{ m_params.amplitude_px, 1.0F / m_params.wavelength_px, m_params.phase_rad };
    const uint8_t fill = in.color.full_range ? 0 : 16;

    const dim3 block(kBlockX, kBlockY);
    const dim3 luma_grid((w + kBlockX - 1) / kBlockX, (h + kBlockY - 1) / kBlockY);
    warp_luma<<<luma_grid, block, 0, stream>>>(in.y, in.y_pitch, out.y, out.y_pitch, w, h, args, fill);
    check(cudaGetLastError(), "luma launch");

    const int cw = w / 2;
    const int ch = h / 2;
    const dim3 chroma_grid((cw + kBlockX - 1) / kBlockX, (ch + kBlockY - 1) / kBlockY);
    warp_chroma<<<chroma_grid, block, 0, stream>>>(in.uv, in.uv_pitch, out.uv, out.uv_pitch, cw, ch, args);
    check(cudaGetLastError(), "chroma launch");
}

} // namespace video_stage
