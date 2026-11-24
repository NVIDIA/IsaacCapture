// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <ctime>
#include <vector>

namespace camera_bridge
{

struct Geometry
{
    uint32_t width = 0;
    uint32_t height = 0;

    bool operator==(const Geometry& other) const
    {
        return width == other.width && height == other.height;
    }
};

/// One access unit out of the encoder.
struct EncodedPacket
{
    std::vector<uint8_t> data;
    uint64_t key = 0;
    bool keyframe = false;
};

/// Quantisation range of the picture; it decides the value of a border fill (black is Y=16 limited, Y=0 full).
struct ColorInfo
{
    bool full_range = false;
};

/// A device-memory NV12 image: a luma plane followed by one interleaved chroma (U,V) plane at half resolution in both
/// directions. Pitches are in bytes and may exceed the row size. A view does not own its memory.
struct Nv12View
{
    uint8_t* y = nullptr;
    uint8_t* uv = nullptr;
    size_t y_pitch = 0;
    size_t uv_pitch = 0;
    Geometry geometry;
    ColorInfo color;
};

struct EncodedVideoFrame
{
    /// Annex-B bytes: an IDR carries its SPS/PPS in front of it.
    const uint8_t* data = nullptr;
    size_t size = 0;
    /// The unit is an IDR, decodable on its own.
    bool keyframe = false;
    /// CLOCK_MONOTONIC microseconds when the frame was published: the host timeline, not sensor time.
    uint64_t timestamp_us = 0;
    /// The frame's application timestamp in microseconds, on the clock its stream declares. Zero means the publisher
    /// cannot vouch for a time, and the frame carries none.
    uint64_t user_timestamp_us = 0;
    /// The frame's sequence on its track, from 1.
    uint32_t frame_id = 0;
    /// The sensor's capture time on its own clock, carried as user data.
    uint64_t capture_tsc = 0;
};

inline int64_t monotonic_ns()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

} // namespace camera_bridge
