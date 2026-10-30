// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cudaEGL.h>
#include <cuda_runtime.h>

namespace camera_viz::argus
{

enum class YuvLayout
{
    YUV420Planar,
    YVU420Planar,
    YUV420SemiPlanar,
    YVU420SemiPlanar,
};

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
                                 cudaStream_t stream);

void launch_yuv420_array_to_rgba(cudaTextureObject_t y_tex,
                                 cudaTextureObject_t uv_or_u_tex,
                                 cudaTextureObject_t v_tex,
                                 int width,
                                 int height,
                                 uint8_t* rgba_out,
                                 int rgba_row_bytes,
                                 YuvLayout layout,
                                 bool full_range,
                                 cudaStream_t stream);

} // namespace camera_viz::argus
