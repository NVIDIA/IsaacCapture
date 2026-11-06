// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace video_stage
{

/// One access unit of an Annex-B stream, as a range of the input.
struct AccessUnitRange
{
    std::size_t offset = 0;
    std::size_t size = 0;
    /// The unit contains an IDR slice.
    bool idr = false;
};

/**
 * @brief Split an Annex-B H.264 stream into access units.
 *
 * A unit starts at the first NAL after a picture's slices, or at a picture whose first slice follows another
 * picture directly; further slices of one picture stay with it. End-of-sequence, end-of-stream and filler NALs
 * belong to the picture before them. Parameter sets and SEI stay with the picture that follows.
 */
std::vector<AccessUnitRange> split_access_units(const std::uint8_t* data, std::size_t size);

} // namespace video_stage
