// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>

namespace plugins
{
namespace sensing
{

/// What accompanies one published frame.
struct VideoFrameInfo
{
    /// CLOCK_MONOTONIC microseconds at capture: the host timeline, not sensor time.
    uint64_t timestamp_us = 0;
    /// The camera's own frame sequence.
    uint32_t frame_id = 0;
    /// The sensor's capture time on its own clock, carried as user data so a consumer can pair
    /// frames from different sensors.
    uint64_t capture_tsc_ns = 0;
};

/**
 * @brief One published video track, as the camera side sees it.
 *
 * The caller fills frame_buffer() with a frame and then submits it. A single thread must drive one
 * sink: the buffer is reused for every frame.
 */
class VideoTrackSink
{
public:
    virtual ~VideoTrackSink() = default;

    virtual uint32_t width() const = 0;
    virtual uint32_t height() const = 0;

    /// Tightly packed NV12: width*height luma, then width*height/2 interleaved chroma, BT.601 limited range.
    virtual uint8_t* frame_buffer() = 0;

    /// Publishes the current contents of frame_buffer(). Returns once the SDK has copied it, so the
    /// buffer may be rewritten immediately.
    /// @throws std::runtime_error when the frame cannot be published.
    virtual void submit(const VideoFrameInfo& info) = 0;

    /// One line naming the encoder in use and its achieved size and rate, or empty if unavailable.
    virtual std::string describe() = 0;
};

} // namespace sensing
} // namespace plugins
