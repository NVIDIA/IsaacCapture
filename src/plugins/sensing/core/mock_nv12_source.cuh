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
 * @brief Paint one synthetic ISP0 frame from a GIF frame already in device memory.
 *
 * The source is scaled to cover the full width x height and centre-cropped, so its aspect ratio is
 * never distorted: a uniform scale of max(width / src_width, height / src_height), bilinear when
 * enlarging and area-averaged when reducing. The result is full-range BT.709 NV12, the inverse of
 * the matrix the real conversion applies, so the colours come back out of that conversion.
 *
 * @param src_rgba   Device pointer to the GIF frame, src_width x src_height RGBA8.
 * @param src_pitch  Source row stride in bytes.
 */
void launch_mock_gif_nv12(const uint8_t* src_rgba,
                          int src_pitch,
                          int src_width,
                          int src_height,
                          uint8_t* y_plane,
                          int y_pitch,
                          uint8_t* uv_plane,
                          int uv_pitch,
                          int width,
                          int height,
                          cudaStream_t stream);

} // namespace sensing
} // namespace plugins
