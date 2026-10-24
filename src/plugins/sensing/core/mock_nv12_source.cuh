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
 * @brief Paint one synthetic ISP0 frame: full-range NV12 in pitch-linear device memory.
 *
 * Stands in for the buffer SIPL would fill, so the real YUV-to-RGBA kernel runs
 * on it unchanged. A 16-bit frame counter is drawn top-left as 16 px cells (white
 * for a set bit, black for clear, most significant bit first), so a test can tell
 * a fresh frame from a stale or torn one by reading pixels. Chroma is tinted by
 * sensor so the two eyes are distinguishable.
 */
void launch_mock_nv12_pattern(uint8_t* y_plane,
                              int y_pitch,
                              uint8_t* uv_plane,
                              int uv_pitch,
                              int width,
                              int height,
                              uint32_t sensor_id,
                              uint32_t frame_index,
                              cudaStream_t stream);

} // namespace sensing
} // namespace plugins
