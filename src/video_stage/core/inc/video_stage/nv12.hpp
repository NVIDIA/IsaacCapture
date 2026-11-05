// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace video_stage
{

/// Image size in luma pixels. Both dimensions are even, as NV12 requires.
struct Geometry
{
    uint32_t width = 0;
    uint32_t height = 0;

    bool operator==(const Geometry& other) const
    {
        return width == other.width && height == other.height;
    }
};

/// Color description carried unchanged from input to output. Only the quantisation range matters to an
/// operation here: it decides the value of a border fill (black is Y=16 limited-range, Y=0 full-range).
struct ColorInfo
{
    bool full_range = false;
};

/// A device-memory NV12 image: a luma plane followed by one interleaved chroma (U,V) plane at half
/// resolution in both directions. Pitches are in bytes and may exceed the row size.
///
/// A view does not own its memory. The owner decides how long the planes stay valid; an operation only
/// reads or writes them on the stream it is given.
struct Nv12View
{
    uint8_t* y = nullptr;
    uint8_t* uv = nullptr;
    size_t y_pitch = 0;
    size_t uv_pitch = 0;
    Geometry geometry;
    ColorInfo color;
};

} // namespace video_stage
