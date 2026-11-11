// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace camera_bridge
{

/// A device NV12 image: a luma plane and one interleaved chroma plane at half resolution in both directions.
struct Nv12Planes
{
    const uint8_t* y = nullptr;
    const uint8_t* uv = nullptr;
    size_t y_pitch = 0;
    size_t uv_pitch = 0;
};

struct Nv12PlanesOut
{
    uint8_t* y = nullptr;
    uint8_t* uv = nullptr;
    size_t y_pitch = 0;
    size_t uv_pitch = 0;
};

/// True when an image of this size shrinks by `divisor` to a valid NV12 size (even width and height).
bool box_downscale_supported(uint32_t width, uint32_t height, uint32_t divisor);

/// Shrinks an NV12 image by an integer `divisor` (1 copies), averaging each divisor x divisor block of luma and of
/// chroma, on `stream`. The destination is width/divisor x height/divisor. Returns the CUDA status of the launch.
cudaError_t box_downscale_nv12(const Nv12Planes& source,
                               uint32_t width,
                               uint32_t height,
                               const Nv12PlanesOut& destination,
                               uint32_t divisor,
                               cudaStream_t stream);

} // namespace camera_bridge
