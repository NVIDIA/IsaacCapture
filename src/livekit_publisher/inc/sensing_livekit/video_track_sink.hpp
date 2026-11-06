// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <video_stage/encoded_video_frame.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace plugins
{
namespace sensing
{

using video_stage::EncodedVideoFrame;

/// What the receiving side asked of the encoder since the last call.
struct EncoderFeedback
{
    /// The next frame must be an IDR with its parameter sets.
    bool keyframe_requested = false;
    /// Latest bitrate target from congestion control, if one was requested.
    std::optional<uint32_t> target_bitrate_bps;
};

/**
 * @brief One published video track that carries frames already encoded by hardware.
 *
 * The SDK forwards each access unit without encoding it again. A single thread must drive one sink.
 */
class VideoTrackSink
{
public:
    virtual ~VideoTrackSink() = default;

    /// Size of the encoded pictures this track was published for.
    virtual uint32_t width() const = 0;
    virtual uint32_t height() const = 0;

    /// Publishes one access unit; the bytes are copied before this returns.
    /// @return false when the SDK did not accept the frame, which can break the reference chain:
    ///         the caller must follow with a keyframe.
    /// @throws std::runtime_error when the frame cannot be published at all.
    virtual bool submit(const EncodedVideoFrame& frame) = 0;

    /// Keyframe and rate requests accumulated since the previous call.
    virtual EncoderFeedback take_feedback() = 0;

    /// One line naming the transport's view of the stream, or empty if unavailable.
    virtual std::string describe() = 0;
};

} // namespace sensing
} // namespace plugins
