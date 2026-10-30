// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Argus YUV420 -> RGBA8 conversion for CUDA EGLStream consumers.

#include "yuv_to_rgba.cuh"

namespace
{

__device__ __forceinline__ unsigned char clamp_u8(float v)
{
    return static_cast<unsigned char>(v < 0.f ? 0.f : (v > 255.f ? 255.f : v));
}

__device__ __forceinline__ void yuv_to_rgb(
    int Y, int Cb, int Cr, bool full_range, unsigned char& r, unsigned char& g, unsigned char& b)
{
    float R, G, B;
    if (full_range)
    {
        const float yf = static_cast<float>(Y);
        const float u = static_cast<float>(Cb) - 128.f;
        const float v = static_cast<float>(Cr) - 128.f;
        R = yf + 1.402f * v;
        G = yf - 0.344136f * u - 0.714136f * v;
        B = yf + 1.772f * u;
    }
    else
    {
        const float yf = (static_cast<float>(Y) - 16.f) * 1.16438f;
        const float u = static_cast<float>(Cb) - 128.f;
        const float v = static_cast<float>(Cr) - 128.f;
        R = yf + 1.79274f * v;
        G = yf - 0.21325f * u - 0.53291f * v;
        B = yf + 2.11240f * u;
    }
    r = clamp_u8(R);
    g = clamp_u8(G);
    b = clamp_u8(B);
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
                                            int full_range_value)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    const auto layout = static_cast<camera_viz::argus::YuvLayout>(layout_value);
    const int Y = y_plane[y * y_pitch + x];
    int Cb = 128;
    int Cr = 128;

    if (layout == camera_viz::argus::YuvLayout::YUV420SemiPlanar ||
        layout == camera_viz::argus::YuvLayout::YVU420SemiPlanar)
    {
        const int uv_x = x & ~1;
        const int uv_y = y >> 1;
        const uint8_t a = uv_or_u_plane[uv_y * uv_pitch + uv_x + 0];
        const uint8_t b = uv_or_u_plane[uv_y * uv_pitch + uv_x + 1];
        if (layout == camera_viz::argus::YuvLayout::YUV420SemiPlanar)
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
        if (layout == camera_viz::argus::YuvLayout::YUV420Planar)
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
    yuv_to_rgb(Y, Cb, Cr, full_range_value != 0, rgba_out[idx + 0], rgba_out[idx + 1], rgba_out[idx + 2]);
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
                                            int full_range_value)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    const auto layout = static_cast<camera_viz::argus::YuvLayout>(layout_value);
    const int Y = tex2D<unsigned char>(y_tex, x, y);
    int Cb = 128;
    int Cr = 128;

    if (layout == camera_viz::argus::YuvLayout::YUV420SemiPlanar ||
        layout == camera_viz::argus::YuvLayout::YVU420SemiPlanar)
    {
        const int uv_x = x >> 1;
        const int uv_y = y >> 1;
        const uchar2 uv = tex2D<uchar2>(uv_or_u_tex, uv_x, uv_y);
        if (layout == camera_viz::argus::YuvLayout::YUV420SemiPlanar)
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
        if (layout == camera_viz::argus::YuvLayout::YUV420Planar)
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
    yuv_to_rgb(Y, Cb, Cr, full_range_value != 0, rgba_out[idx + 0], rgba_out[idx + 1], rgba_out[idx + 2]);
    rgba_out[idx + 3] = 255;
}

} // namespace

namespace camera_viz::argus
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
                                 cudaStream_t stream)
{
    const dim3 block(16, 16, 1);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
    yuv420_pitch_to_rgba_kernel<<<grid, block, 0, stream>>>(y_plane, uv_or_u_plane, v_plane, y_pitch, uv_pitch, v_pitch,
                                                            width, height, rgba_out, rgba_row_bytes,
                                                            static_cast<int>(layout), full_range ? 1 : 0);
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
                                 cudaStream_t stream)
{
    const dim3 block(16, 16, 1);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
    yuv420_array_to_rgba_kernel<<<grid, block, 0, stream>>>(
        y_tex, uv_or_u_tex, v_tex, width, height, rgba_out, rgba_row_bytes, static_cast<int>(layout), full_range ? 1 : 0);
}

} // namespace camera_viz::argus
