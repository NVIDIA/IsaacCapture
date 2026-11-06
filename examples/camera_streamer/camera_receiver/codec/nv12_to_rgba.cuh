// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace camera_receiver::codec
{

// Launch the NV12 → RGBA8 kernel.
//   full_range = true  → BT.601 full-range (T.871)
//              = false → BT.709 limited-range (H.264 default)
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
                         cudaStream_t stream);

// Rotate pitched NV12 into the caller's output buffer while copying it out of
// NVDEC. UV pairs retain their Cb/Cr byte order.
void launch_nv12_rotate_180(const uint8_t* y_plane,
                            const uint8_t* uv_plane,
                            int y_pitch,
                            int uv_pitch,
                            int width,
                            int height,
                            uint8_t* nv12_out,
                            int output_row_bytes,
                            cudaStream_t stream);

} // namespace camera_receiver::codec
