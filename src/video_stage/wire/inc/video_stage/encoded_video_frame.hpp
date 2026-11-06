// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace video_stage
{

/// One complete encoded H.264 access unit, with what accompanies it.
struct EncodedVideoFrame
{
    /// Annex-B bytes: an IDR carries its SPS/PPS in front of it.
    const uint8_t* data = nullptr;
    size_t size = 0;
    /// The unit is an IDR, decodable on its own.
    bool keyframe = false;
    /// CLOCK_MONOTONIC microseconds at capture: the host timeline, not sensor time.
    uint64_t timestamp_us = 0;
    /// The frame's application timestamp in microseconds, on the clock its stream declares. Zero means the publisher
    /// cannot vouch for a time, and the frame carries none.
    uint64_t user_timestamp_us = 0;
    /// The camera's own frame sequence.
    uint32_t frame_id = 0;
    /// The sensor's capture time on its own clock, carried as user data so a consumer can pair
    /// frames from different sensors.
    uint64_t capture_tsc = 0;
};

} // namespace video_stage
