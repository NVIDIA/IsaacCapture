// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <ctime>
#include <vector>

namespace video_stage
{

/// Picture size in luma pixels.
struct Geometry
{
    uint32_t width = 0;
    uint32_t height = 0;

    bool operator==(const Geometry& other) const
    {
        return width == other.width && height == other.height;
    }
};

/// One H.264 access unit leaving the stage and the key of the input it came from.
struct EncodedPacket
{
    std::vector<uint8_t> data;
    uint64_t key = 0;
    bool keyframe = false;
};

/// Source identity carried with a picture from the access unit that produced it to the packet that leaves.
struct FrameMeta
{
    uint64_t frame_number = 0;
    uint64_t source_timestamp_ns = 0;
    int64_t received_ns = 0; ///< CLOCK_MONOTONIC when the intake finished reading the record
    uint64_t key = ~uint64_t{ 0 }; ///< the key this entry was written for; a mismatch means the entry was overwritten
    int64_t submitted_ns = 0; ///< when the access unit was handed to a decoder, if there is one
};

/// Where finished access units go. Called on the stage's processing thread; it must not block.
class PacketSink
{
public:
    virtual ~PacketSink() = default;
    virtual void write(const EncodedPacket& packet, const FrameMeta& meta, const Geometry& geometry) = 0;
};

/// CLOCK_MONOTONIC in nanoseconds.
inline int64_t monotonic_ns()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

} // namespace video_stage
