// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// NV12 → RGBA8 colour conversion in a single kernel launch.
// BT.709 limited-range (H.264 default) or BT.601 full-range (T.871).
// Alpha is always 255.

#include "nv12_to_rgba.cuh"

#include <cstdint>
#include <sstream>
#include <stdexcept>

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
                                    int full_range,
                                    int rotation)
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

    const int output_x = rotation == 180 ? width - 1 - x : x;
    const int output_y = rotation == 180 ? height - 1 - y : y;
    const int idx = output_y * rgba_row_bytes + output_x * 4;
    rgba_out[idx + 0] = (unsigned char)(R < 0.f ? 0 : (R > 255.f ? 255 : R));
    rgba_out[idx + 1] = (unsigned char)(G < 0.f ? 0 : (G > 255.f ? 255 : G));
    rgba_out[idx + 2] = (unsigned char)(B < 0.f ? 0 : (B > 255.f ? 255 : B));
    rgba_out[idx + 3] = 255;
}

__global__ void nv12_rotate_180_kernel(const unsigned char* __restrict__ y_plane,
                                       const unsigned char* __restrict__ uv_plane,
                                       int y_pitch,
                                       int uv_pitch,
                                       int width,
                                       int height,
                                       unsigned char* __restrict__ nv12_out,
                                       int output_row_bytes)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        nv12_out[(height - 1 - y) * output_row_bytes + width - 1 - x] = y_plane[y * y_pitch + x];
    }

    // One thread per interleaved Cb/Cr pair. Rotate pair locations while
    // preserving byte order within each pair.
    if (x < width / 2 && y < height / 2)
    {
        const int source_x = x * 2;
        const int destination_x = width - 2 - source_x;
        const int destination_y = height / 2 - 1 - y;
        unsigned char* destination_uv = nv12_out + output_row_bytes * height;
        destination_uv[destination_y * output_row_bytes + destination_x] = uv_plane[y * uv_pitch + source_x];
        destination_uv[destination_y * output_row_bytes + destination_x + 1] = uv_plane[y * uv_pitch + source_x + 1];
    }
}

} // namespace

namespace camera_receiver::codec
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
                         int rotation,
                         cudaStream_t stream)
{
    const dim3 block(16, 16, 1);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
    nv12_to_rgba_kernel<<<grid, block, 0, stream>>>(
        y_plane, uv_plane, y_pitch, uv_pitch, width, height, rgba_out, rgba_row_bytes, full_range ? 1 : 0, rotation);
    const cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        std::ostringstream os;
        os << "NV12->RGBA kernel launch failed: " << cudaGetErrorString(launch_err);
        throw std::runtime_error(os.str());
    }
}

void launch_nv12_rotate_180(const uint8_t* y_plane,
                            const uint8_t* uv_plane,
                            int y_pitch,
                            int uv_pitch,
                            int width,
                            int height,
                            uint8_t* nv12_out,
                            int output_row_bytes,
                            cudaStream_t stream)
{
    const dim3 block(16, 16, 1);
    const dim3 grid((width + 15) / 16, (height + 15) / 16, 1);
    nv12_rotate_180_kernel<<<grid, block, 0, stream>>>(
        y_plane, uv_plane, y_pitch, uv_pitch, width, height, nv12_out, output_row_bytes);
    const cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess)
    {
        std::ostringstream os;
        os << "NV12 rotate-180 kernel launch failed: " << cudaGetErrorString(launch_err);
        throw std::runtime_error(os.str());
    }
}

} // namespace camera_receiver::codec
