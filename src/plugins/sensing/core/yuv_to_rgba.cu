// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// YUV420 -> RGBA8 conversion for the SIPL ISP0 output.
//
// BT.601 and BT.709, selected per call. SIPL's ISP0 output on this rig is
// REC709_ER and the request is not negotiable -- GetImageAttributes rejects a
// BT.601 attribute list with NVSIPL_STATUS_BAD_ARGUMENT -- so the 709 matrix is
// the one the SIPL path actually uses. rgba_to_nv12.cu on the encode side is
// still 601, so an encode round trip is not identity; that only matters if the
// H.264 output is ever compared against the RGBA it came from.

#include "yuv_to_rgba.cuh"

namespace
{

__device__ __forceinline__ unsigned char clamp_u8(float v)
{
    return static_cast<unsigned char>(v < 0.f ? 0.f : (v > 255.f ? 255.f : v));
}

__device__ __forceinline__ void yuv_to_rgb(int Y,
                                           int Cb,
                                           int Cr,
                                           bool full_range,
                                           bool bt709,
                                           unsigned char& r,
                                           unsigned char& g,
                                           unsigned char& b)
{
    // Full range uses Y as-is; limited range expands 16..235 first.
    const float yf = full_range ? static_cast<float>(Y) : (static_cast<float>(Y) - 16.f) * 1.16438f;
    const float u = static_cast<float>(Cb) - 128.f;
    const float v = static_cast<float>(Cr) - 128.f;

    // Limited-range coefficients already fold in the 255/219 luma gain applied
    // above, so they differ from the full-range pair by more than rounding.
    float kr, kgu, kgv, kb;
    if (bt709)
    {
        if (full_range) { kr = 1.5748f;  kgu = 0.18733f; kgv = 0.46813f; kb = 1.8556f; }
        else            { kr = 1.79274f; kgu = 0.21325f; kgv = 0.53291f; kb = 2.11240f; }
    }
    else
    {
        if (full_range) { kr = 1.402f; kgu = 0.344136f; kgv = 0.714136f; kb = 1.772f; }
        else            { kr = 1.596f; kgu = 0.391f;    kgv = 0.813f;    kb = 2.018f; }
    }

    r = clamp_u8(yf + kr * v);
    g = clamp_u8(yf - kgu * u - kgv * v);
    b = clamp_u8(yf + kb * u);
}

__global__ void yuv420_pitch_to_rgba_kernel(const uint8_t* __restrict__ y_plane,
                                            const uint8_t* __restrict__ uv_or_u_plane,
                                            const uint8_t* __restrict__ v_plane,
                                            int y_pitch,
                                            int uv_pitch,
                                            int v_pitch,
                                            int width,
                                            int height,
                                            uint8_t* __restrict__ rgba_out,
                                            int rgba_row_bytes,
                                            int layout_value,
                                            int full_range_value,
                                            int bt709_value)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    const auto layout = static_cast<plugins::sensing::YuvLayout>(layout_value);
    const int Y = y_plane[y * y_pitch + x];
    int Cb = 128;
    int Cr = 128;

    if (layout == plugins::sensing::YuvLayout::YUV420SemiPlanar || layout == plugins::sensing::YuvLayout::YVU420SemiPlanar)
    {
        const int uv_x = x & ~1;
        const int uv_y = y >> 1;
        const uint8_t a = uv_or_u_plane[uv_y * uv_pitch + uv_x + 0];
        const uint8_t b = uv_or_u_plane[uv_y * uv_pitch + uv_x + 1];
        if (layout == plugins::sensing::YuvLayout::YUV420SemiPlanar)
        {
            Cb = a;
            Cr = b;
        }
        else
        {
            Cr = a;
            Cb = b;
        }
    }
    else
    {
        const int uv_x = x >> 1;
        const int uv_y = y >> 1;
        const uint8_t a = uv_or_u_plane[uv_y * uv_pitch + uv_x];
        const uint8_t b = v_plane[uv_y * v_pitch + uv_x];
        if (layout == plugins::sensing::YuvLayout::YUV420Planar)
        {
            Cb = a;
            Cr = b;
        }
        else
        {
            Cr = a;
            Cb = b;
        }
    }

    const int idx = y * rgba_row_bytes + x * 4;
    yuv_to_rgb(Y, Cb, Cr, full_range_value != 0, bt709_value != 0, rgba_out[idx + 0], rgba_out[idx + 1],
               rgba_out[idx + 2]);
    rgba_out[idx + 3] = 255;
}

__global__ void yuv420_array_to_rgba_kernel(cudaTextureObject_t y_tex,
                                            cudaTextureObject_t uv_or_u_tex,
                                            cudaTextureObject_t v_tex,
                                            int width,
                                            int height,
                                            uint8_t* __restrict__ rgba_out,
                                            int rgba_row_bytes,
                                            int layout_value,
                                            int full_range_value,
                                            int bt709_value)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    const auto layout = static_cast<plugins::sensing::YuvLayout>(layout_value);
    const int Y = tex2D<unsigned char>(y_tex, x, y);
    int Cb = 128;
    int Cr = 128;

    if (layout == plugins::sensing::YuvLayout::YUV420SemiPlanar || layout == plugins::sensing::YuvLayout::YVU420SemiPlanar)
    {
        const int uv_x = x >> 1;
        const int uv_y = y >> 1;
        const uchar2 uv = tex2D<uchar2>(uv_or_u_tex, uv_x, uv_y);
        if (layout == plugins::sensing::YuvLayout::YUV420SemiPlanar)
        {
            Cb = uv.x;
            Cr = uv.y;
        }
        else
        {
            Cr = uv.x;
            Cb = uv.y;
        }
    }
    else
    {
        const int uv_x = x >> 1;
        const int uv_y = y >> 1;
        const uint8_t a = tex2D<unsigned char>(uv_or_u_tex, uv_x, uv_y);
        const uint8_t b = tex2D<unsigned char>(v_tex, uv_x, uv_y);
        if (layout == plugins::sensing::YuvLayout::YUV420Planar)
        {
            Cb = a;
            Cr = b;
        }
        else
        {
            Cr = a;
            Cb = b;
        }
    }

    const int idx = y * rgba_row_bytes + x * 4;
    yuv_to_rgb(Y, Cb, Cr, full_range_value != 0, bt709_value != 0, rgba_out[idx + 0], rgba_out[idx + 1],
               rgba_out[idx + 2]);
    rgba_out[idx + 3] = 255;
}

} // namespace

namespace plugins
{
namespace sensing
{

void launch_yuv420_pitch_to_rgba(const uint8_t* y_plane,
                                 const uint8_t* uv_or_u_plane,
                                 const uint8_t* v_plane,
                                 int y_pitch,
                                 int uv_pitch,
                                 int v_pitch,
                                 int width,
                                 int height,
                                 uint8_t* rgba_out,
                                 int rgba_row_bytes,
                                 YuvLayout layout,
                                 bool full_range,
                                 bool bt709,
                                 cudaStream_t stream)
{
    const dim3 block(16, 16, 1);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
    yuv420_pitch_to_rgba_kernel<<<grid, block, 0, stream>>>(y_plane, uv_or_u_plane, v_plane, y_pitch, uv_pitch, v_pitch,
                                                            width, height, rgba_out, rgba_row_bytes,
                                                            static_cast<int>(layout), full_range ? 1 : 0, bt709 ? 1 : 0);
}

void launch_yuv420_array_to_rgba(cudaTextureObject_t y_tex,
                                 cudaTextureObject_t uv_or_u_tex,
                                 cudaTextureObject_t v_tex,
                                 int width,
                                 int height,
                                 uint8_t* rgba_out,
                                 int rgba_row_bytes,
                                 YuvLayout layout,
                                 bool full_range,
                                 bool bt709,
                                 cudaStream_t stream)
{
    const dim3 block(16, 16, 1);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
    yuv420_array_to_rgba_kernel<<<grid, block, 0, stream>>>(
        y_tex, uv_or_u_tex, v_tex, width, height, rgba_out, rgba_row_bytes, static_cast<int>(layout), full_range ? 1 : 0, bt709 ? 1 : 0);
}

} // namespace sensing
} // namespace plugins
