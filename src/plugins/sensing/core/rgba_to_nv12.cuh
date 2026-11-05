// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace plugins
{
namespace sensing
{

/**
 * @brief Convert packed RGBA8 to NV12 (Y plane + interleaved UV plane).
 *
 * Chroma is point-sampled from the top-left pixel of each 2x2 block rather
 * than averaged: the ISP already delivered 4:2:0, so the RGBA the
 * capture path produced was upsampled from it and averaging would only blur
 * chroma that was never independent. Width and height must be even.
 *
 * @param full_range  true for [0,255] luma, false for broadcast [16,235].
 */
void launch_rgba_to_nv12(const uint8_t* rgba,
                         int rgba_pitch,
                         uint8_t* y_plane,
                         int y_pitch,
                         uint8_t* uv_plane,
                         int uv_pitch,
                         int width,
                         int height,
                         bool full_range,
                         cudaStream_t stream);

} // namespace sensing
} // namespace plugins
