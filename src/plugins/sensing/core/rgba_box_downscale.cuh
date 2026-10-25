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
 * @brief Reduce RGBA8 by an integer divisor, averaging each divisor x divisor block.
 *
 * The aspect ratio is kept exactly: the source size must divide evenly, so nothing is stretched,
 * cropped or rounded.
 *
 * @param dst_width  src_width / divisor.
 * @param dst_height src_height / divisor.
 */
void launch_rgba_box_downscale(const uint8_t* src,
                               int src_pitch,
                               uint8_t* dst,
                               int dst_pitch,
                               int dst_width,
                               int dst_height,
                               int divisor,
                               cudaStream_t stream);

} // namespace sensing
} // namespace plugins
